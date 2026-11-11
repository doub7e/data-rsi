"""Durable sequential dot -> gated projected influence controller. No local model work."""
import fcntl,subprocess,time
from common import *
from q1_selection_controller import ENV,NS,call,archive
DATE='20261105';ACTIVE_RUN=None;HOME_RUN=ROOT/f'runs/q1-dot-{DATE}';CONTAINER='/scratch/cvlab/home/shuli/data-rsi/006-bfcl-probe-gradient'
def plan(method):return json.loads((ROOT/f'runs/q1-{method}-{DATE}/protocol.json').read_text())
def tasks(cfg):
 prefix=cfg['prefix'];method=cfg['method'];out=[{'job':cfg.get('prepare_job',f'{prefix}-prepare-1105'),'directory':f'{prefix}-prepare-{DATE}','stage':'prepare','gpus':0 if method=='dot' or cfg.get('prepare_mode')=='cached-basic-checks' else 1}]
 for stage in ['train','eval']:
  for v in cfg['variants']:out.append({'job':f'{prefix}-{v}-{stage}-1105','directory':f'{prefix}-{v}-{stage}-{DATE}','stage':stage,'variant':v,'gpus':2 if stage=='train' else 1})
 out.append({'job':f'{prefix}-compare-1105','directory':cfg['study'],'stage':'compare','gpus':0});return out

def submit(t,cfg):
 stage=t['stage'];method=cfg['method'];gpus=t['gpus'];run=ROOT/'runs'/t['directory']
 script=f"{cfg['prefix']}_{stage}.py" if stage in ['train','eval'] else 'q1_projected_influence.py' if method=='pif' and stage=='prepare' else 'q1_attribution_worker.py'
 if method=='pif' and stage=='prepare' and cfg.get('prepare_mode')=='cached-basic-checks':script='q1_use_cached_pif.py'
 args=[t['variant']] if 'variant' in t else [] if method=='pif' and stage=='prepare' else [f'prepare-{method}' if stage=='prepare' else f'compare-{method}']
 image=cfg['eval_image'] if stage=='eval' else cfg['train_image'];cpu='16' if gpus else '4';memory='512G' if stage=='prepare' and method=='pif' and cfg.get('prepare_mode')!='cached-basic-checks' else '128G' if gpus else '16G'
 cmd=['runai','submit','--name',t['job'],'--image',image,'--gpu',str(gpus),'--cpu',cpu,'--memory',memory,'--backoff-limit','0','--large-shm','--existing-pvc','claimname=cvlab-scratch,path=/scratch']
 if gpus:cmd+=['--node-pool','h200']
 cmd+=['--command','--','/opt/datasense/bin/python',CONTAINER+'/scripts/'+script]+args
 write(run/'submission.json',{'at':now(),'command':cmd});print(call(cmd),flush=True);m=json.loads((run/'run.json').read_text());m['job']={'provider':'runai','name':t['job'],'namespace':NS};write(run/'run.json',m);state(run,'submitted','queued')

def main():
 global ACTIVE_RUN
 with (HOME_RUN/'controller.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for method in ['dot','pif']:
   cfg=plan(method);run=ROOT/'runs'/cfg['study'];ACTIVE_RUN=run;todo=tasks(cfg)
   if (run/'pipeline-complete.json').exists():continue
   while True:
    if method=='pif':
     assert (ROOT/cfg['gate']).exists()
     if not (run/'implementation-ready.json').exists():
      state(run,'waiting_for_projection_implementation','queued');time.sleep(20);continue
    try:
     pods=json.loads(call(['kubectl','get','pods','-n',NS,'-o','json']))['items'];jobs={x['metadata']['name'] for x in json.loads(call(['kubectl','get','runaijobs','-n',NS,'-o','json']))['items']}
    except RuntimeError as e:write(run/'controller-state.json',{'stage':'connection_retry','error':str(e),'at':now()});state(run,'waiting_cluster_access','blocked',message='Cluster access unavailable; awaiting credential refresh. Existing remote jobs may still be running.');time.sleep(20);continue
    for old_job in cfg.get('superseded_prepare_jobs',[]):
     if old_job not in jobs:continue
     items=[p for p in pods if p['metadata']['name'].startswith(old_job+'-')]
     archive({'job':old_job,'directory':f"{cfg['prefix']}-prepare-{DATE}"},items)
     print(call(['runai','delete','job',old_job]),flush=True);jobs.discard(old_job)
     write(run/f'stopped-{old_job}.json',{'at':now(),'reason':'User requested simplified checks and training from completed scores; stop redundant gradient recomputation.','prior_pod_phases':[p['status']['phase'] for p in items]})
    for t in todo:
     items=[p for p in pods if p['metadata']['name'].startswith(t['job']+'-')]
     if not items:continue
     latest=max(items,key=lambda p:p['metadata']['creationTimestamp']);phase=latest['status']['phase']
     if phase in ['Succeeded','Failed']:
      archive(t,items)
      if phase=='Failed':raise RuntimeError(t['job']+' failed; no blind retries; artifacts preserved')
      assert (ROOT/'runs'/t['directory']/'complete.json').exists(),t['job']+' missing completion marker'
      print(call(['runai','delete','job',t['job']]),flush=True);jobs.discard(t['job'])
    def done(t):return (ROOT/'runs'/t['directory']/'complete.json').exists()
    active={s:sum(t['job'] in jobs for t in todo if t['stage']==s) for s in ['train','eval']}
    for t in todo:
     if done(t) or t['job'] in jobs:continue
     s=t['stage'];ready=False
     if s=='prepare':ready=True
     if s=='train':ready=done(todo[0]) and (ROOT/'runs'/t['directory']/'preparation-complete.json').exists() and active[s]<cfg['execution']['max_parallel_train']
     if s=='eval':ready=(ROOT/f"runs/{cfg['prefix']}-{t['variant']}-train-{DATE}/complete.json").exists() and active[s]<cfg['execution']['max_parallel_eval']
     if s=='compare':ready=all(done(x) for x in todo if x['stage']=='eval')
     if ready:
      submit(t,cfg);jobs.add(t['job'])
      if s in active:active[s]+=1
    write(run/'controller-state.json',{'stage':'monitoring','at':now(),'active_jobs':[t['job'] for t in todo if t['job'] in jobs],'complete':[t['directory'] for t in todo if done(t)]})
    if all(done(t) and t['job'] not in jobs for t in todo):write(run/'pipeline-complete.json',{'finished_at':now()});state(run,'complete','succeeded');print('PIPELINE COMPLETE',method,flush=True);break
    state(run,'preparation_training_evaluation','running');time.sleep(20)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  write((ACTIVE_RUN or HOME_RUN)/'controller-failure.json',{'error':str(e),'at':now()});state(ACTIVE_RUN or HOME_RUN,'controller_failed','failed',error=str(e),finished_at=now());raise
