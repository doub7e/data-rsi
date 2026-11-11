"""Production Harbor evaluation on native RunAI CPU sandboxes."""
import concurrent.futures
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request
import native_common as c
from auth_guard import require_auth, AuthenticationUnavailable, recoverable_result


def request(base, path, data=None, timeout=30):
    req=urllib.request.Request(base.removesuffix('/v1')+path,
        data=json.dumps(data).encode() if data is not None else None,
        headers={'Content-Type':'application/json','Authorization':'Bearer EMPTY'})
    with urllib.request.urlopen(req,timeout=timeout) as response:
        body=response.read();return json.loads(body) if body else None


def health(config):
    models=request(os.environ['NATIVE_MODEL_API_BASE'],'/v1/models',timeout=10)
    if not any(m['id']==config['model_name'] for m in models['data']):raise RuntimeError('Unexpected served model')
    request(os.environ['NATIVE_MODEL_API_BASE'],'/health',timeout=10)


def load_probe(config, run):
    """Representative long prompts, before any model evaluation sandbox starts."""
    health(config);base=os.environ['NATIVE_MODEL_API_BASE'];measurements=[]
    for concurrency in [2,8,16,32]:
        def one(index):
            text=f'Native capacity probe {concurrency}/{index}/{time.time_ns()}.\n'+('Inspect this build log and report any failure. Test passed.\n'*1500)
            tokenized=request(base,'/tokenize',{'model':config['model_name'],'prompt':text},timeout=30)
            tokens=tokenized['tokens'][:16384]
            begin=time.monotonic()
            result=request(base,'/v1/completions',{'model':config['model_name'],'prompt':tokens,
                'max_tokens':512,'ignore_eos':True,'temperature':0.6},timeout=180)
            usage=result.get('usage',{})
            if not result.get('choices') or usage.get('completion_tokens',0)<=0:raise RuntimeError('Empty generation probe')
            return {'elapsed_sec':time.monotonic()-begin,**usage}
        begin=time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            rows=list(pool.map(one,range(concurrency)))
        elapsed=time.monotonic()-begin
        measurements.append({'concurrency':concurrency,'wall_sec':elapsed,
            'output_tokens_per_sec':sum(x['completion_tokens'] for x in rows)/elapsed,'requests':rows})
        c.write(run/'inference-scaling.json',{'measurements':measurements,'diagnostic_only':True})
        if max(x['elapsed_sec'] for x in rows)>120:raise RuntimeError('Inference latency too high; no evaluation sandboxes started')


def cleanup(config, run):
    label=config['run_id'];owned=set()
    for p in (run/'jobs').glob('*/*/attempts/*/native-events.jsonl'):
        for line in p.read_text().splitlines():
            try:entry=json.loads(line)
            except ValueError:continue
            if entry.get('event')=='start_requested':owned.add(entry['job'])
    pods=json.loads(c.call(['kubectl','get','pods','-n',c.NS,'-l','data-rsi-native='+label,'-o','json']).stdout)['items']
    deleted=[]
    for pod in pods:
        job=pod['metadata']['labels'].get('workloadName')
        if job not in owned:raise RuntimeError('Labeled sandbox is absent from this run ownership log')
        c.archive_delete(job,run/'sandbox-cleanup-logs');deleted.append(job)
    for _ in range(45):
        remaining=json.loads(c.call(['kubectl','get','pods','-n',c.NS,'-l','data-rsi-native='+label,'-o','json']).stdout)['items']
        if not remaining:break
        time.sleep(1)
    c.write(run/'native-cleanup.json',{'label':label,'deleted':deleted,'remaining':[p['metadata']['name'] for p in remaining],'checked_at':c.stamp()})
    if remaining:raise RuntimeError('Native sandboxes remain after cleanup')


def run_harbor(config, command, log):
    require_auth()
    process=subprocess.Popen(command,cwd=c.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    failed_since=None
    try:
        while process.poll() is None:
            time.sleep(5)
            require_auth()
            try:health(config);failed_since=None
            except Exception:
                failed_since=failed_since or time.monotonic()
                if time.monotonic()-failed_since>=60:raise RuntimeError('Model API unavailable; stopping native evaluation')
        return process.returncode
    finally:
        if process.poll() is None:
            os.killpg(process.pid,signal.SIGINT)
            try:process.wait(timeout=120)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()


def wait_for_auth(run):
    waiting=False
    while True:
        try:require_auth();return
        except AuthenticationUnavailable:
            if not waiting:c.state(run,'blocked','authentication','Cluster login expired; evaluation suspended until credentials refresh')
            waiting=True;time.sleep(15)


def needs_recovery(out):
    return any(recoverable_result(c.read(p,{}) or {}) for p in out.glob('*/result.json') if (p.parent/'config.json').is_file())


def refresh_resume_endpoint(out, endpoint, model_name):
    path=out/'config.json'
    if not path.exists():return
    cfg=c.read(path)
    for agent in cfg['agents']:
        if agent['model_name']!='hosted_vllm/'+model_name:
            raise RuntimeError('Resume model identity differs from serving model')
        agent.setdefault('kwargs',{})['api_base']=endpoint
    c.write(path,cfg)


def execute_batch(config, run, name, cfg_path, out):
    while True:
        wait_for_auth(run)
        c.state(run,'running','evaluation',name+'; up to 32 native sandboxes')
        if (out/'config.json').exists():refresh_resume_endpoint(out,os.environ['NATIVE_MODEL_API_BASE'],config['model_name'])
        command=[sys.executable,str(c.ROOT/'scripts/resume_native_job.py'),str(out)] if (out/'config.json').exists() else ['harbor','run','--config',str(cfg_path)]
        try:
            with (run/(name+'.log')).open('a') as log:code=run_harbor(config,command,log)
        except AuthenticationUnavailable:
            c.state(run,'blocked','authentication','Cluster authentication lost; preserving trials for recovery')
            wait_for_auth(run)
            cleanup(config,run)
            continue
        if needs_recovery(out):
            c.state(run,'restarting','auth_recovery','Archiving authentication/interruption failures before retrying affected tasks')
            wait_for_auth(run)
            cleanup(config,run)
            continue
        if code:raise RuntimeError('Harbor command failed: '+name)
        return


def job_config(config, run, name, dataset=None):
    template=c.read(c.ROOT/'production/harbor-native-template.json')
    template.update(job_name=name,jobs_dir=str(run/'jobs'),n_attempts=1,n_concurrent_trials=32,
                    datasets=[{'path':str(dataset)}] if dataset else [],tasks=[])
    template['agents'][0]['model_name']='hosted_vllm/'+config['model_name']
    template['agents'][0]['kwargs']['api_base']=os.environ['NATIVE_MODEL_API_BASE']
    template['agents'][0]['kwargs']['parser_name']='json'
    return template


def smoke(config, run):
    name='native-terminus-smoke';cfg=job_config(config,run,name)
    cfg.update(n_concurrent_trials=1,tasks=[{'path':str(c.ROOT/'production/smoke-task')}])
    cfg['retry']['max_retries']=0;cfg['agents'][0]['max_timeout_sec']=180
    c.write(run/(name+'.json'),cfg)
    with (run/(name+'.log')).open('a') as log:
        code=run_harbor(config,['harbor','run','--config',str(run/(name+'.json'))],log)
    score=c.trial_summary(run/'jobs'/name/'result.json',1)
    if code or score['passed']!=1 or score['errors']:raise RuntimeError('Native Terminus smoke failed')
    c.write(run/'native-smoke-result.json',score)


def main(config):
    if not Path('/var/run/secrets/kubernetes.io/serviceaccount').exists():raise RuntimeError('Use the RunAI CPU worker')
    node,run,train=c.paths(config)
    os.environ['NATIVE_RUN_LABEL']=config['run_id']
    if not c.training_complete(train):raise RuntimeError('Training checkpoint is incomplete')
    results={};done=0
    try:
        wait_for_auth(run)
        c.state(run,'starting','inference_validation','Long-context concurrency test before native sandbox allocation')
        if any((run/'jobs').glob('*repeat*/config.json')):
            health(config)  # Resume the same checkpoint without overwriting prior probes.
        else:
            load_probe(config,run)
            smoke(config,run)
        for benchmark,(expected,reference,ref_se) in c.BENCHMARKS.items():
            while not c.benchmark_ready(benchmark):
                c.state(run,'planned','waiting_for_images','Waiting for the complete '+benchmark+' task set; no tasks are dropped')
                time.sleep(30)
            catalog=c.read(c.ROOT/'production/catalog.json')
            c.write(run/'configs'/(benchmark+'-images.json'),[x for x in catalog['tasks'] if x['benchmark']==benchmark])
            repeats=[]
            for repeat in range(1,4):
                name=f'{benchmark}-repeat-{repeat}';out=run/'jobs'/name
                cfg=job_config(config,run,name,c.ROOT/'production/tasks'/benchmark)
                cfg_path=run/'configs'/(name+'.json');c.write(cfg_path,cfg)
                if not c.read(out/'result.json',{}).get('finished_at') or needs_recovery(out):
                    execute_batch(config,run,name,cfg_path,out)
                repeats.append(c.trial_summary(out/'result.json',expected));done+=1
                with (run/'metrics.jsonl').open('a') as f:f.write(json.dumps({'step':done,'total_steps':9,'accuracy':repeats[-1]['accuracy_pct']})+'\n')
                c.write(run/'partial-results.json',{**results,benchmark:{'repeats':repeats}})
            results[benchmark]={**c.aggregate(repeats),'paper_10k_reference_pct':reference,'paper_10k_reference_se_pct':ref_se}
        cleanup(config,run)
        metrics={k+'_accuracy_pct':v['accuracy_pct'] for k,v in results.items()}
        c.write(run/'result.json',{'status':'succeeded','training_rows':config['training_rows'],'backend':'runai-native',
                'metrics':metrics,'benchmarks':results,'completed_at':c.stamp()})
        lines=['# Native RunAI evaluation','',f"Training data: {config['training_rows']} rows. Three independent repeats; fixed denominators; errors count as zero.",'',
               '| Benchmark | Mean ± SE (%) | Paper 10K reference (%) |','|---|---:|---:|']
        for name,row in results.items():lines.append(f"| {name} | {row['accuracy_pct']:.2f} ± {row['se_pct']:.2f} | {row['paper_10k_reference_pct']:.1f} |")
        lines+=['','Reference: https://arxiv.org/pdf/2606.24855 Table 11. For a 1K model this is a different-data-scale reference, not a matched 1K target.','',
                'Adaptations: native RunAI CPU sandboxes, four-GPU training with gradient accumulation, adjacent-user message normalization, pinned current author Harbor fork, JSON parser and temperature 0.6.']
        (run/'comparison.md').write_text('\n'.join(lines)+'\n')
        c.state(run,'succeeded','complete','All 867 native trials completed; results and comparison saved')
    except BaseException as exc:
        c.state(run,'failed','worker_error',str(exc))
        try:cleanup(config,run)
        except Exception as cleanup_error:c.write(run/'cleanup-error.json',{'error':str(cleanup_error),'updated_at':c.stamp()})
        raise


if __name__=='__main__':
    def interrupted(signum,frame):raise KeyboardInterrupt('Native evaluation interrupted')
    signal.signal(signal.SIGTERM,interrupted)
    main(c.read(Path(sys.argv[1])))
