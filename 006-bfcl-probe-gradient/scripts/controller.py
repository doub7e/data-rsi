"""Host control only: GPU and bulk CPU steps are submitted to RunAI."""
import fcntl,importlib.util,json,os,sys,time
from common import *
spec=importlib.util.spec_from_file_location('cluster',BASE/'scripts/random100_controller.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
CONTAINER=str(ROOT).replace('/mnt/cvlab/scratch','/scratch',1)
PROBE_JOB=CFG.get('probe_job','pg-probe-1103')
TRAIN_JOBS=CFG.get('training_jobs',{'top100':'pg-top-train-1103','bottom100':'pg-bottom-train-1103'})

def submit(job,run,script,args,gpus,image='train',cpu=8,ram='128G'):
 if (run/'complete.json').exists() or (run/'training-complete.json').exists() or (run/'evaluation-complete.json').exists():return
 if c.pods(job):return
 if any(x['metadata']['name']==job for x in json.loads(c.call(['kubectl','get','runaijobs','-n',c.NS,'-o','json']))['items']):return
 cmd=['runai','submit','--name',job,'--image',CFG[image+'_image'],'--gpu',str(gpus),'--cpu',str(cpu),'--memory',ram,'--large-shm','--backoff-limit','0','--existing-pvc','claimname=cvlab-scratch,path=/scratch']
 if gpus:cmd+=['--node-pool','h200']
 cmd+=['--command','--','/opt/datasense/bin/python',CONTAINER+'/scripts/'+script]+list(args)
 write(run/'submission.json',{'command':cmd,'created_at':now()})
 print(c.call(cmd),flush=True);state(run,'submitted','queued')
def wait(job,run,marker='complete.json'):
 if (run/marker).exists() and not c.pods(job):return
 c.wait_job(job,run,run/marker)
def main():
 with (ROOT/'.controller.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  probe=ROOT/'runs/probe-rollouts'
  while not (ROOT/'data/candidates.jsonl').exists():
   items=c.pods(PROBE_JOB)
   if any(p['status']['phase']=='Failed' for p in items):raise RuntimeError('Probe preparation failed')
   time.sleep(20)
  smoke=ROOT/'runs/gradient-smoke';submit('pg-gsmoke-1103',smoke,'gradients.py',['smoke'],1);wait('pg-gsmoke-1103',smoke)
  wait(PROBE_JOB,probe)
  for rank in (0,1):submit(f'pg-gprobe{rank}-1103',ROOT/f'runs/gradient-probe-{rank}','gradients.py',['probe','--rank',str(rank)],1)
  for rank in (0,1):wait(f'pg-gprobe{rank}-1103',ROOT/f'runs/gradient-probe-{rank}')
  for rank in (0,1):submit(f'pg-gcand{rank}-1103',ROOT/f'runs/gradient-candidates-{rank}','gradients.py',['candidates','--rank',str(rank)],1)
  for rank in (0,1):wait(f'pg-gcand{rank}-1103',ROOT/f'runs/gradient-candidates-{rank}')
  material=ROOT/'runs/materialize';material.mkdir(exist_ok=True)
  submit('pg-materialize-1103',material,'materialize.py',[],0,cpu=4,ram='32G')
  c.wait_job('pg-materialize-1103',material,ROOT/'data/selection-complete.json')
  for group in ('top100','bottom100'):
   short=group.replace('100','');submit(TRAIN_JOBS[group],ROOT/f'runs/train-{group}','model_worker.py',['train',group],2,cpu=16)
  pending={'top100','bottom100'}
  while pending:
   for group in sorted(pending):
    short=group.replace('100','');job=TRAIN_JOBS[group];run=ROOT/f'runs/train-{group}'
    items=c.pods(job)
    if any(p['status']['phase']=='Failed' for p in items):wait(job,run,'training-complete.json')
    if (run/'training-complete.json').exists() and all(p['status']['phase']=='Succeeded' for p in items):
     wait(job,run,'training-complete.json')
     submit(f'pg-{short}-eval-1103',ROOT/f'runs/eval-{group}','model_worker.py',['eval',group],1,image='eval',cpu=16,ram='64G')
     pending.remove(group)
   if pending:time.sleep(20)
  for group in ('top100','bottom100'):
   short=group.replace('100','');wait(f'pg-{short}-eval-1103',ROOT/f'runs/eval-{group}','evaluation-complete.json')
  while not (BASE/'runs/eval-random100-s42-20261102/evaluation-complete.json').exists():time.sleep(30)
  compare=ROOT/'runs/comparison';submit('pg-compare-1103',compare,'compare.py',[],0,cpu=4,ram='32G');wait('pg-compare-1103',compare)
  jobs=[(PROBE_JOB,'probe-rollouts'),('pg-gsmoke-1103','gradient-smoke'),
    ('pg-materialize-1103','materialize'),('pg-compare-1103','comparison')]
  jobs += [(f'pg-gprobe{i}-1103',f'gradient-probe-{i}') for i in (0,1)]
  jobs += [(f'pg-gcand{i}-1103',f'gradient-candidates-{i}') for i in (0,1)]
  jobs += [(TRAIN_JOBS[g],f'train-{g}') for g in ('top100','bottom100')]
  jobs += [(f'pg-{g.replace("100","")}-eval-1103',f'eval-{g}') for g in ('top100','bottom100')]
  resources=[]
  for job,name in jobs:
   items=c.pods(job)
   if not items:continue
   assert all(p['status']['phase']=='Succeeded' for p in items),(job,'not terminal success')
   for pod in items:
    resources.append({'job':job,'pod':pod['metadata']['name'],'uid':pod['metadata']['uid'],
      'created_at':pod['metadata']['creationTimestamp'],'node':pod['spec'].get('nodeName'),
      'resources':[x.get('resources') for x in pod['spec']['containers']],
      'termination':[x.get('state',{}).get('terminated') for x in pod['status'].get('containerStatuses',[])]})
   state(ROOT/'runs'/name,'complete','succeeded')
   print(c.call(['runai','delete','job',job]),flush=True)
  write(compare/'runai-executions.json',{'note':'Successful phase executions; prior startup failures retained separately. Not a billing ledger.','pods':resources})
  write(ROOT/'complete.json',{'finished_at':now()});print('Pilot comparison complete',flush=True)
if __name__=='__main__':main()
