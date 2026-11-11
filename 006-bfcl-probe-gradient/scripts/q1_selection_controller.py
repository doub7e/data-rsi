"""One lightweight host controller; all experiment work remains in RunAI."""
import fcntl,importlib.util,os,subprocess,time
from common import *
RUN=ROOT/'runs/q1-selection-20261104';PLAN=json.loads((RUN/'protocol.json').read_text());NS='runai-cvlab-shuli'
ENV={**os.environ,'KUBECONFIG':'/mnt/cvlab/scratch/cvlab/home/shuli/.codex/secrets/native-runai-kubeconfig.json'}
CONTAINER='/scratch/cvlab/home/shuli/data-rsi/006-bfcl-probe-gradient'
POLICY=json.loads((RUN/'execution-policy.json').read_text()) if (RUN/'execution-policy.json').exists() else PLAN['execution']
def call(cmd):
 p=subprocess.run(cmd,env=ENV,text=True,capture_output=True,timeout=90)
 if p.returncode:raise RuntimeError(' '.join(cmd[:5])+': '+p.stderr[-1500:]+p.stdout[-500:])
 return p.stdout

def tasks():
 out=[{'job':'q1-scores-1104','directory':'q1-scores-20261104','stage':'scores','gpus':2}]
 for stage in ['train','eval']:
  for v in (PLAN['variants'] if stage=='train' else ['base','random100']+PLAN['variants']):
   out.append({'job':f'q1-{v}-{stage}-1104','directory':f'q1-{v}-{stage}-20261104','stage':stage,'variant':v,'gpus':2 if stage=='train' else 1})
 out.append({'job':'q1-compare-1104','directory':RUN.name,'stage':'compare','gpus':0})
 return out

def submit(t):
 d=ROOT/'runs'/t['directory'];stage=t['stage'];script=f'q1_selection_{stage}.py';image=PLAN['eval_image'] if stage=='eval' else PLAN['train_image']
 cmd=['runai','submit','--name',t['job'],'--image',image,'--gpu',str(t['gpus']),'--cpu','16' if t['gpus'] else '2','--memory','128G' if t['gpus'] else '8G','--backoff-limit','0','--large-shm','--existing-pvc','claimname=cvlab-scratch,path=/scratch']
 if t['gpus']:cmd+=['--node-pool','h200']
 cmd+=['--command','--','/opt/datasense/bin/python',CONTAINER+'/scripts/'+script]
 if 'variant' in t:cmd.append(t['variant'])
 write(d/'submission.json',{'command':cmd,'at':now()});print(call(cmd),flush=True)
 if not (d/'state.json').exists() or json.loads((d/'state.json').read_text()).get('state') in ['planned','failed']:state(d,'submitted','queued')
 if stage=='compare':
  m=json.loads((d/'run.json').read_text());m['job']={'provider':'runai','name':t['job'],'namespace':NS};write(d/'run.json',m)

def archive(t,items):
 d=ROOT/'runs'/t['directory'];hist=[]
 for p in items:
  name=p['metadata']['name'];write(d/f'cluster/{name}.json',p)
  log=d/f'logs/{name}.log';log.parent.mkdir(exist_ok=True)
  with log.open('w') as f:subprocess.run(['kubectl','logs','-n',NS,name],env=ENV,stdout=f,stderr=subprocess.PIPE,timeout=60)
  hist.append({'name':name,'phase':p['status']['phase'],'resources':[x.get('resources') for x in p['spec']['containers']],'container_statuses':p['status'].get('containerStatuses'),'created_at':p['metadata']['creationTimestamp']})
 write(d/'cluster-execution.json',hist)

def main():
 with (RUN/'controller.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  todo=tasks();state(RUN,'selection_training_evaluation',started_at=now())
  while True:
   try:
    pods=json.loads(call(['kubectl','get','pods','-n',NS,'-o','json']))['items'];jobs={x['metadata']['name'] for x in json.loads(call(['kubectl','get','runaijobs','-n',NS,'-o','json']))['items']}
   except RuntimeError as e:write(RUN/'controller-state.json',{'stage':'cluster_connection_retry','error':str(e),'at':now()});time.sleep(20);continue
   live={t['job']:[p for p in pods if p['metadata']['name'].startswith(t['job']+'-')] for t in todo}
   for t in todo:
    items=live[t['job']];d=ROOT/'runs'/t['directory']
    if not items:continue
    latest=max(items,key=lambda p:p['metadata']['creationTimestamp']);phase=latest['status']['phase']
    if phase in ['Succeeded','Failed']:
     archive(t,items)
     if phase=='Failed':raise RuntimeError(t['job']+' failed; retained artifacts and unrelated active jobs')
     assert (d/'complete.json').exists(),t['job']+' exited without verified completion'
     print(call(['runai','delete','job',t['job']]),flush=True);jobs.discard(t['job']);live[t['job']]=[]
   def done(t):return (ROOT/'runs'/t['directory']/'complete.json').exists()
   active={s:sum(t['job'] in jobs for t in todo if t['stage']==s) for s in ['train','eval']}
   for t in todo:
    if done(t) or t['job'] in jobs:continue
    s=t['stage'];ready=False
    if s=='scores':continue # Submitted explicitly after code review.
    if s=='train':ready=(ROOT/f"runs/q1-{t['variant']}-train-20261104/preparation-complete.json").exists() and (ROOT/'runs/q1-scores-20261104/complete.json').exists() and active[s]<POLICY['max_parallel_train'] and (not POLICY.get('training_gate') or (ROOT/POLICY['training_gate']).exists())
    if s=='eval':ready=(t['variant'] in ['base','random100'] or (ROOT/f"runs/q1-{t['variant']}-train-20261104/complete.json").exists()) and active[s]<PLAN['execution']['max_parallel_eval']
    if s=='compare':ready=all(done(x) for x in todo if x['stage']=='eval')
    if ready:
     submit(t);jobs.add(t['job'])
     if s in active:active[s]+=1
   if not (RUN/'complete.json').exists():
    gated=bool(POLICY.get('training_gate')) and not (ROOT/POLICY['training_gate']).exists()
    state(RUN,'waiting_for_continuation_training' if gated else 'training_and_evaluation','queued' if gated else 'running')
   write(RUN/'controller-state.json',{'stage':'monitoring','at':now(),'active_jobs':[t['job'] for t in todo if t['job'] in jobs],'complete':[t['directory'] for t in todo if done(t)]})
   if all(done(t) and t['job'] not in jobs for t in todo):
    write(RUN/'pipeline-complete.json',{'finished_at':now()});state(RUN,'complete','succeeded');print('ALL Q1 STUDY TASKS VERIFIED AND CLEANED',flush=True);return
   time.sleep(20)
if __name__=='__main__':
 try:main()
 except BaseException as e:write(RUN/'controller-state.json',{'stage':'failed','error':str(e),'at':now()});raise
