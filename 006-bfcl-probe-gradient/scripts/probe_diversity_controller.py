"""One rollout job plus one gradient job; independent of current attribution study."""
import fcntl,time
from common import *
from q1_selection_controller import ENV,NS,call,archive
RUN=ROOT/'runs/probe-diversity-20261105';P=json.loads((RUN/'protocol.json').read_text());CONTAINER='/scratch/cvlab/home/shuli/data-rsi/006-bfcl-probe-gradient'
def tasks():
 return [{'key':q['key'],'stage':s,'job':f"pd-{q['key']}-{s}-1105",'directory':f"pd-{q['key']}-{s}-20261105"} for q in P['questions'] for s in ['rollouts','gradient']]
def submit(t):
 run=ROOT/'runs'/t['directory'];stage=t['stage'];image=P['eval_image'] if stage=='rollouts' else P['train_image'];memory=P['execution']['memory_'+('rollouts' if stage=='rollouts' else 'gradients')];script='probe_diversity_rollouts.py' if stage=='rollouts' else 'probe_diversity_gradients.py'
 cmd=['runai','submit','--name',t['job'],'--image',image,'--gpu','1','--node-pool','h200','--cpu','12','--memory',memory,'--backoff-limit','0','--large-shm','--existing-pvc','claimname=cvlab-scratch,path=/scratch','--command','--','/opt/datasense/bin/python',CONTAINER+'/scripts/'+script,t['key']]
 write(run/'submission.json',{'at':now(),'command':cmd});print(call(cmd),flush=True);state(run,'submitted','queued')
def main():
 with (RUN/'controller.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);todo=tasks()
  while True:
   try:
    pods=json.loads(call(['kubectl','get','pods','-n',NS,'-o','json']))['items'];jobs={x['metadata']['name'] for x in json.loads(call(['kubectl','get','runaijobs','-n',NS,'-o','json']))['items']}
   except RuntimeError as e:write(RUN/'controller-state.json',{'stage':'connection_retry','at':now(),'error':str(e)});time.sleep(20);continue
   for t in todo:
    items=[p for p in pods if p['metadata']['name'].startswith(t['job']+'-')]
    if not items:continue
    p=max(items,key=lambda p:p['metadata']['creationTimestamp']);phase=p['status']['phase']
    if phase in ['Succeeded','Failed']:
     archive(t,items)
     if phase=='Failed':raise RuntimeError(t['job']+' failed; no silent replacement')
     assert (ROOT/'runs'/t['directory']/'complete.json').exists()
     print(call(['runai','delete','job',t['job']]),flush=True);jobs.discard(t['job'])
   active={s:sum(t['job'] in jobs for t in todo if t['stage']==s) for s in ['rollouts','gradient']}
   def done(t):return (ROOT/'runs'/t['directory']/'complete.json').exists()
   for t in todo:
    if done(t) or t['job'] in jobs or active[t['stage']]>=1:continue
    if t['stage']=='gradient' and not (ROOT/f"runs/pd-{t['key']}-rollouts-20261105/complete.json").exists():continue
    if t['stage']=='gradient' and not (RUN/'gradient-implementation-ready.json').exists():continue
    submit(t);jobs.add(t['job']);active[t['stage']]+=1
   count=sum(done(t) for t in todo if t['stage']=='gradient');write(RUN/'progress.json',{'stage':'questions_analyzed','done':count,'total':8,'updated_at':now()})
   write(RUN/'controller-state.json',{'stage':'monitoring','at':now(),'active_jobs':[t['job'] for t in todo if t['job'] in jobs],'complete':[t['directory'] for t in todo if done(t)]})
   if all(done(t) and t['job'] not in jobs for t in todo):
    assert json.loads((RUN/'result.json').read_text())['complete'];write(RUN/'complete.json',{'finished_at':now()});write(RUN/'pipeline-complete.json',{'finished_at':now()});state(RUN,'complete','succeeded');return
   state(RUN,'rollouts_and_gradient_analysis','running');time.sleep(20)
if __name__=='__main__':
 try:main()
 except BaseException as e:state(RUN,'failed','failed',error=str(e),finished_at=now());write(RUN/'controller-failure.json',{'at':now(),'error':str(e)});raise
