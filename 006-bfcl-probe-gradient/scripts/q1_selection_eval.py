"""Fresh Q1 repeated evaluation, same sampling and strict history for all models."""
import argparse,concurrent.futures,os,shutil,signal,subprocess,sys,time,urllib.request,math
from common import *
STUDY=ROOT/'runs/q1-selection-20261104';PLAN=json.loads((STUDY/'protocol.json').read_text())
def model_view(run,v):
 source=MODEL if v=='base' else BASE/'runs/random100-s42-20261102/checkpoint' if v=='random100' else ROOT/f'runs/q1-{v}-train-20261104/checkpoint'
 if v!='base':assert (source/'training-complete.txt').exists()
 view=run/'model-64k';view.mkdir(exist_ok=True)
 for p in source.iterdir():
  if p.is_file() and p.name!='config.json':
   dest=view/p.name
   if not dest.exists():dest.symlink_to(p)
 config=json.loads((source/'config.json').read_text());base=json.loads((ROOT/'runs/probe-rollouts/model-64k/config.json').read_text());config['max_position_embeddings']=base['max_position_embeddings'];config['rope_scaling']=base['rope_scaling'];write(view/'config.json',config)
 write(run/'model-provenance.json',{'source':str(source),'view_config':config,'source_config_sha256':sha(source/'config.json'),'weights':[{'file':p.name,'bytes':p.stat().st_size} for p in source.glob('*.safetensors')]})
 return view

def main(v):
 assert v in ['base','random100']+PLAN['variants'];run=ROOT/f'runs/q1-{v}-eval-20261104';total=PLAN['evaluation']['rollouts_per_model'];start=time.monotonic()
 os.environ.update(BFCL_PROJECT_ROOT=str(run/'preflight-runtime'),BFCL_DATA_DIR='/opt/bfcl-data',BFCL_HISTORY_PROTOCOL='strict-v2',BFCL_DROP_HISTORY_THINK='1',NO_PROXY='localhost,127.0.0.1',no_proxy='localhost,127.0.0.1',BFCL_MODEL_LABEL='Qwen3-4B '+v)
 import scaling_probe_rollouts as shared
 shared.RUN=run;shared.preflight()
 from bfcl_board_export import normalize
 model=model_view(run,v);port=25194
 env={**os.environ,'MODEL':'Qwen/Qwen3-4B-FC','MODEL_PATH':str(model),'BFCL_DATA_DIR':str(run/'data'),'PATH':str(BASE/'scripts/bin')+os.pathsep+os.environ['PATH'],'LOCAL_SERVER_PORT':str(port),'HF_HOME':str(ROOT/'cache/huggingface')}
 (run/'logs').mkdir(exist_ok=True);clients=[]
 write(run/'jobs/rollouts/config.json',{'job_name':'Q1 '+v+' fresh evaluation','agents':[{'name':'BFCL Qwen3-FC'}]})
 state(run,'server_starting',started_at=now());attempt=now().replace(':','')
 with (run/f'logs/server-{attempt}.log').open('a') as log:
  server=subprocess.Popen(['vllm','serve',str(model),'--port',str(port),'--dtype','bfloat16','--max-model-len','65536','--tensor-parallel-size','1','--gpu-memory-utilization','0.85','--trust-remote-code','--max-num-seqs','32'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:
   ready=False;opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
   for _ in range(600):
    if server.poll() is not None:raise RuntimeError('vLLM startup failed')
    try:
     with opener.open(f'http://127.0.0.1:{port}/v1/models',timeout=3) as response:ready=any(d['id']==str(model) for d in json.load(response)['data'])
    except Exception:pass
    if ready:break
    time.sleep(2)
   if not ready:raise TimeoutError('vLLM not ready')
   state(run,'evaluating',total=total)
   def rollout(rep):
    out=run/f'bfcl/r{rep:03d}';marker=out/'complete.json'
    if marker.exists():return rep
    if out.exists():
     archive=run/f'attempts/{attempt}/r{rep:03d}';archive.parent.mkdir(parents=True,exist_ok=True);out.rename(archive)
    out.mkdir(parents=True);local={**env,'PROBE_REPEAT':str(rep),'PROBE_SEED_OFFSET':str(PLAN['evaluation']['seed_offset']),'BFCL_PROJECT_ROOT':str(run/f'harness/r{rep:03d}')}
    cmds=[[sys.executable,ROOT/'scripts/probe22_cli.py','generate','--model','Qwen/Qwen3-4B-FC','--test-category','multi_turn_miss_func','--temperature','1.0','--backend','vllm','--num-gpus','1','--num-threads','1','--local-model-path',model,'--result-dir',out,'--skip-server-setup','--include-input-log'],['bfcl','evaluate','--model','Qwen/Qwen3-4B-FC','--test-category','multi_turn_miss_func','--result-dir',out,'--score-dir',out/'score']]
    with (run/f'logs/r{rep:03d}.log').open('a') as f:
     for cmd in cmds:
      p=subprocess.Popen(list(map(str,cmd)),env=local,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);clients.append(p)
      if p.wait():raise RuntimeError(f'Trial {rep} failed, no replacement')
    trace=shared.extract(rep,out);write(run/f'traces/r{rep:03d}.json',trace)
    raw=rows(out/'Qwen_Qwen3-4B-FC/multi_turn/BFCL_v4_multi_turn_miss_func_result.json')[0]
    write(run/f'jobs/rollouts/r{rep:03d}/agent/trajectory.json',normalize(raw));write(run/f'jobs/rollouts/r{rep:03d}/result.json',{'task_name':PLAN['probe_id']+f' / r{rep:03d}','finished_at':now(),'verifier_result':{'rewards':{'reward':trace['reward']}},'exception_info':None,'provenance':{'history_protocol':'strict-v2','scoring_verified':True}})
    write(marker,{'finished_at':now(),'reward':trace['reward'],'calls':len(trace['steps'])});return rep
   outcomes={}
   def refresh(rep):
    outcomes[rep]=json.loads((run/f'bfcl/r{rep:03d}/complete.json').read_text())['reward'];done=len(outcomes)
    write(run/'progress.json',{'done':done,'total':total,'successes':sum(outcomes.values()),'updated_at':now()})
    write(run/'jobs/rollouts/result.json',{'n_total_trials':total,'finished_at':now() if done==total else None,'stats':{'n_completed_trials':done,'n_running_trials':0,'n_errored_trials':0,'evals':{'bfcl':{'reward_stats':{'reward':{str(r):[f'r{i:03d}' for i,y in sorted(outcomes.items()) if y==r] for r in [0,1]}}}}},'provenance':{'scoring_verified':True}})
    print('EVALUATED',v,done,sum(outcomes.values()),flush=True)
   rollout(1);refresh(1)
   with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
    futures=[pool.submit(rollout,rep) for rep in range(2,total+1)]
    for f in concurrent.futures.as_completed(futures):refresh(f.result())
   assert len(outcomes)==total
   calls=output=0;seeds=set()
   for rep in range(1,total+1):
    t=json.loads((run/f'traces/r{rep:03d}.json').read_text());assert t['reward']==outcomes[rep]
    for s in t['steps']:
     expected=PLAN['evaluation']['seed_offset']+rep*1000+s['call_index'];assert s['seed']==expected and expected not in seeds;seeds.add(expected);calls+=1;output+=len(s['completion_token_ids'])
   success=sum(outcomes.values());p=success/total;z=1.959963984540054;den=1+z*z/total;center=(p+z*z/(2*total))/den;rad=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
   result={'variant':v,'finished_at':now(),'trials':total,'successes':success,'accuracy':p,'wilson95':[center-rad,center+rad],'outcomes':[outcomes[i] for i in range(1,total+1)],'calls':calls,'output_tokens':output,'elapsed_seconds':time.monotonic()-start,'metrics':{'q1_success_percent':100*p,'evaluated_trials':total}}
   write(run/'result.json',result);write(run/'complete.json',{'finished_at':now()});state(run,'complete','succeeded')
  finally:
   for p in clients+[server]:
    if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
   for p in clients+[server]:
    try:p.wait(timeout=20)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('variant');a=p.parse_args()
 try:main(a.variant)
 except BaseException as e:state(ROOT/f'runs/q1-{a.variant}-eval-20261104','failed','failed',error=str(e));raise
