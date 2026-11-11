"""Post-training pipeline; does not build environments without recorded approval.

Run as a user service after reviewing build-authorization.json. Keeps queued H200
jobs, evaluates three independent repeats, and writes all evidence in this node.
"""
from pathlib import Path
import argparse,datetime,json,math,os,statistics,subprocess,sys,time,urllib.request
from gates import require_training_complete, require_build_ready, TrainingNotReady
ROOT=Path(__file__).resolve().parent
NODE=ROOT.parent
TRAIN=NODE/'runs/repro-10k-4h200-20261030'
RUN=NODE/'runs/eval-repro-10k-20261030'
NS='runai-cvlab-shuli'
TRAIN_JOB='ota-qwen3-8b-10k-4h200-1030'
SERVE_JOB='ota-qwen3-8b-repro-eval-1030'
BENCHMARKS={'swebench100':(100,24.3,1.9),'tblite':(100,15.6,1.5),'tb2':(89,7.9,1.4)}

def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(path,default=None):
 try:return json.loads(path.read_text())
 except (OSError,ValueError):return default

def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,path)

def state(status,stage,message):
 value={'state':status,'stage':stage,'message':message,'updated_at':stamp()}
 if status in ('succeeded','failed','cancelled'):value['finished_at']=stamp()
 write(RUN/'state.json',value)
 print(stamp(),stage,message,flush=True)

def call(args,timeout=60,check=True,env=None):
 p=subprocess.run(args,capture_output=True,text=True,timeout=timeout,env=env)
 if check and p.returncode:raise RuntimeError((p.stderr or p.stdout)[-2000:])
 return p

def pods(job):
 p=call(['kubectl','get','pods','-n',NS,'-l','workloadName='+job,'--request-timeout=12s','-o','json'],timeout=20)
 return json.loads(p.stdout)['items']

def wait_training():
 last=None
 while True:
  complete=False
  try:
   require_training_complete(ROOT)
   complete=True
  except TrainingNotReady:
   pass
  status=read(TRAIN/'state.json',{})
  try:
   active=pods(TRAIN_JOB)
   phases=[p['status']['phase'] for p in active]
   if complete and (not phases or ('Succeeded' in phases and all(p in ('Succeeded','Failed') for p in phases))):
    return
   if len(active)>=3 and all(p=='Failed' for p in phases):raise RuntimeError('Training exhausted its retries; evaluation cannot proceed')
   current=(status.get('state','queued'),tuple(phases))
  except (RuntimeError,OSError,ValueError,subprocess.TimeoutExpired) as exc:
   if 'exhausted' in str(exc):raise
   current=('cluster_unavailable',str(exc))
  if current!=last:state('planned','waiting_for_training',str(current));last=current
  time.sleep(45)

def archive_and_delete(job,folder):
 folder.mkdir(parents=True,exist_ok=True)
 for pod in pods(job):
  name=pod['metadata']['name']
  p=call(['kubectl','logs','-n',NS,name],check=False,timeout=120)
  (folder/(name+'.log')).write_text(p.stdout+p.stderr)
 p=call(['runai','delete','job',job],check=False)
 if p.returncode and not any(marker in (p.stderr+p.stdout).lower() for marker in ('not found','could not find any job')):raise RuntimeError(p.stderr+p.stdout)

def prepare_environments():
 require_build_ready(ROOT)
 summary=read(ROOT/'builds/summary.json',{})
 if summary.get('status')=='succeeded' and summary.get('total')==289:return True
 approval=read(ROOT/'build-authorization.json',{})
 if approval.get('allowed') is not True:
  state('blocked','environment_approval','Missing-image CPU builds await explicit approval; no build controller started.')
  return False
 state('running','environment_preparation','Reuse published images; RunAI/Kaniko builds for missing images, maximum 2 concurrent CPU jobs.')
 p=subprocess.run([sys.executable,str(ROOT/'prebuild.py')],cwd=ROOT)
 if p.returncode:raise RuntimeError('Environment preparation incomplete; see eval/builds/summary.json')
 return True

def launch_serving():
 image=read(ROOT/'serving-image.json')['immutable_image']
 existing=call(['kubectl','get','runaijob','-n',NS,SERVE_JOB,'-o','json'],check=False)
 if existing.returncode:
  if 'NotFound' not in existing.stderr:raise RuntimeError(existing.stderr)
  call(['runai','submit','--name',SERVE_JOB,'--image',image,'--node-pool','h200','--gpu','4','--cpu','16','--memory','128G','--run-as-uid','223974','--run-as-gid','11166','--environment','USER=shuli','--environment','LOGNAME=shuli','--environment','VLLM_CACHE_ROOT=/tmp/ota-vllm','--environment','VLLM_CONFIG_ROOT=/tmp/ota-vllm-config','--environment','XDG_CACHE_HOME=/tmp/ota-cache','--environment','TRITON_CACHE_DIR=/tmp/ota-triton','--environment','TORCHINDUCTOR_CACHE_DIR=/tmp/ota-torchinductor','--large-shm','--backoff-limit','0','--existing-pvc','claimname=cvlab-scratch,path=/scratch','--command','--','/bin/bash','/scratch/cvlab/home/shuli/data-rsi/002-openthoughts-qwen3-8b/eval/serve_checkpoint.sh'])
 else:
  spec=json.loads(existing.stdout)['spec']['template']['spec']
  if spec['containers'][0]['image']!=image:raise RuntimeError('Existing serve job does not match this experiment image')
  allocated=spec['containers'][0].get('resources',{}).get('limits',{}).get('nvidia.com/gpu',0)
  if float(allocated)!=4:raise RuntimeError('Existing serving job must request exactly 4 GPUs')
 meta=read(RUN/'run.json',{})
 meta['job']={'name':SERVE_JOB,'namespace':NS,'provider':'runai'};write(RUN/'run.json',meta)
 state('queued','serving_queue','Waiting for 4 H200 to serve the trained checkpoint; queue is preserved. No Daytona sandboxes started.')
 while True:
  current=pods(SERVE_JOB)
  if current:
   pod=max(current,key=lambda p:p['metadata'].get('creationTimestamp',''))
   if pod['status']['phase']=='Failed':raise RuntimeError('vLLM serving job failed')
   if pod['status']['phase']=='Running':return pod['metadata']['name']
  time.sleep(20)

def http(path,data=None):
 req=urllib.request.Request('http://127.0.0.1:18080'+path,data=json.dumps(data).encode() if data else None,headers={'Content-Type':'application/json','Authorization':'Bearer EMPTY'})
 with urllib.request.urlopen(req,timeout=120) as response:return json.load(response)

def ensure_ready(pod):
 # Probe inside the pod: opening kubectl port-forward before vLLM binds can
 # terminate forwarding with connection-refused while model loading is healthy.
 model='ota-qwen3-8b'
 probe="import json,urllib.request; base='http://127.0.0.1:8000'; models=json.load(urllib.request.urlopen(base+'/v1/models',timeout=5)); assert any(m['id']=='ota-qwen3-8b' for m in models['data']); req=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps({'model':'ota-qwen3-8b','messages':[{'role':'user','content':'Reply with OK.'}],'max_tokens':64,'temperature':0.6}).encode(),headers={'Content-Type':'application/json'}); print(urllib.request.urlopen(req,timeout=60).read().decode())"
 deadline=time.monotonic()+1800
 while time.monotonic()<deadline:
  current=pods(SERVE_JOB)
  if any(x['metadata']['name']==pod and x['status']['phase']=='Failed' for x in current):raise RuntimeError('vLLM pod failed during initialization; see serving logs')
  result=call(['kubectl','exec','-n',NS,pod,'--','python3','-c',probe],timeout=90,check=False)
  if result.returncode==0:
   response=json.loads(result.stdout)
   if not response.get('choices'):raise RuntimeError('Model readiness probe returned no choices')
   write(RUN/'serving-probe.json',response)
   return None,None
  time.sleep(15)
 raise RuntimeError('vLLM did not become ready within 30 minutes')

def trial_summary(path,expected):
 result=read(path,{})
 if not result.get('finished_at'):raise RuntimeError(f'Job not complete: {path}')
 if result.get('n_total_trials')!=expected:raise RuntimeError('Evaluation task denominator changed')
 stats=result['stats']
 if stats.get('n_pending_trials',0) or stats.get('n_running_trials',0) or stats.get('n_cancelled_trials',0):raise RuntimeError('Incomplete or cancelled trials remain')
 passes=0;errors={}
 for value in stats.get('evals',{}).values():
  rewards=value.get('reward_stats',{}).get('reward',{})
  passes+=sum(len(ids) for reward,ids in rewards.items() if float(reward)==1.0)
  for key,ids in value.get('exception_stats',{}).items():errors[key]=errors.get(key,0)+len(ids)
 if passes>expected:raise RuntimeError('Pass count exceeds task denominator')
 return {'passed':passes,'tasks':expected,'accuracy_pct':100*passes/expected,'errors':errors,'result_path':str(path)}

def aggregate(repeats):
 if len(repeats)!=3:raise RuntimeError('Exactly three independent repeats are required')
 scores=[x['accuracy_pct'] for x in repeats]
 return {'accuracy_pct':statistics.mean(scores),'se_pct':statistics.stdev(scores)/math.sqrt(len(scores)) if len(scores)>1 else None,'repeats':repeats}

def evaluate():
 if os.getenv('DATA_RSI_EVAL_EXECUTION_LOCATION')!='runai-cpu':raise RuntimeError('Harbor evaluation must run in the allocated CPU worker')
 env=os.environ.copy();env.update({'PYTHONPATH':str(ROOT),'HARBOR_DISABLE_TELEMETRY':'1','OTEL_SDK_DISABLED':'true','OPENAI_API_KEY':'EMPTY','HOSTED_VLLM_API_KEY':'EMPTY'})
 configs=RUN/'configs';configs.mkdir(exist_ok=True);results={};completed=0
 for benchmark,(expected,reference,ref_se) in BENCHMARKS.items():
  task_root=ROOT/'benchmarks/raw'/benchmark
  if len(list(task_root.glob('*/task.toml')))!=expected:raise RuntimeError('Prepared benchmark task set is incomplete')
  repeats=[]
  for repeat in range(1,4):
   name=f'{benchmark}-repeat-{repeat}';out=RUN/'jobs'/name;config=read(ROOT/'harbor-template.json')
   config.update(job_name=name,jobs_dir=str(RUN/'jobs'),datasets=[{'path':str(task_root)}])
   config['agents'][0]['kwargs']['api_base']=os.environ['OTA_MODEL_API_BASE']
   from daytona_worker import benchmark_concurrency
   config['n_concurrent_trials']=benchmark_concurrency(task_root,int(os.environ.get('OTA_EVAL_CONCURRENCY','32')))
   # Released SFT trajectories use JSON command objects, including SWE.
   config['agents'][0]['kwargs']['parser_name']='json'
   write(configs/(name+'.json'),config)
   existing=read(out/'result.json',{})
   if not existing.get('finished_at'):
    state('running','evaluation',f'{benchmark}: independent repeat {repeat}/3')
    command=['harbor','jobs','resume','--job-path',str(out)] if (out/'config.json').exists() else ['harbor','run','--config',str(configs/(name+'.json'))]
    with (RUN/(name+'.log')).open('a') as log:
     from daytona_worker import run_harbor
     code=run_harbor(command,env,log)
    if code:raise RuntimeError(f'Harbor command failed for {name}; original outputs preserved')
   repeats.append(trial_summary(out/'result.json',expected));completed+=1
   with (RUN/'metrics.jsonl').open('a') as f:f.write(json.dumps({'step':completed,'total_steps':9,'accuracy':repeats[-1]['accuracy_pct']})+'\n')
   write(RUN/'partial-results.json',{**results,benchmark:{'repeats':repeats}})
  results[benchmark]={**aggregate(repeats),'paper_accuracy_pct':reference,'paper_se_pct':ref_se}
  results[benchmark]['delta_pp']=results[benchmark]['accuracy_pct']-reference
 return results

def write_comparison(results):
 metrics={}
 lines=['# Qwen3-8B / 10K reproduction comparison','', 'Reference: https://arxiv.org/pdf/2606.24855 (Table 11).','', '| Benchmark | Our mean ± SE (%) | Paper (%) | Difference (pp) |','| --- | ---: | ---: | ---: |']
 for name,r in results.items():
  metrics[name+'_accuracy_pct']=r['accuracy_pct'];metrics[name+'_paper_pct']=r['paper_accuracy_pct']
  lines.append(f"| {name} | {r['accuracy_pct']:.2f} ± {r['se_pct']:.2f} | {r['paper_accuracy_pct']:.1f} | {r['delta_pp']:+.2f} |")
 lines+=['','Three independent repeats; fixed task denominators; errored/timeout trials count as zero. SE is across repeat accuracies.','', 'Execution: Daytona task sandboxes; RunAI CPU evaluation controller; 4×H200 vLLM inference with TP=4; target concurrency 32 (actual configuration retained per repeat). Adaptations: 4×H200 training with gradient accumulation; adjacent user-message normalization (10,000 rows retained); pinned current author Harbor fork; JSON parser for all benchmarks, matching released SFT trajectories; temperature 0.6 from Qwen generation defaults. These prevent claiming bit-identical replication.','', 'Per-trial trajectories, verifier outcomes, errors, configs, image/data revisions and repeat-level scores are retained beside this report.']
 (RUN/'comparison.md').write_text('\n'.join(lines)+'\n')
 write(RUN/'result.json',{'status':'succeeded','metrics':metrics,'benchmarks':results,'completed_at':stamp(),'comparison_reference':'arXiv:2606.24855 Table 11'})
 meta=read(RUN/'run.json',{});meta['artifacts']=[{'label':'论文结果对比','path':str((RUN/'comparison.md').relative_to(NODE))},{'label':'完整评测指标','path':str((RUN/'result.json').relative_to(NODE))}];write(RUN/'run.json',meta)

def main():
 # Cluster lifecycle only. Harbor and all bulk work run in a RunAI CPU worker.
 from daytona_controller import main as controller_main
 return controller_main()

if __name__=='__main__':
 try:sys.exit(main())
 except Exception as exc:
  state('failed','pipeline_error',str(exc));raise
