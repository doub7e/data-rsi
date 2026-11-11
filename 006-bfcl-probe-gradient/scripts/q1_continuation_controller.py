"""Prepare the continuation study and train only after explicit storage resolution."""
import fcntl,os,subprocess,time
from common import *
import q1_selection_controller as cluster
RUN=ROOT/'runs/q1-continuation-20261104';PLAN=json.loads((RUN/'protocol.json').read_text());NS=cluster.NS;ENV=cluster.ENV;CONTAINER=cluster.CONTAINER

def tasks():
 out=[{'job':'q1c-probe-1104','directory':'q1c-probe-20261104','stage':'probe','gpus':1},{'job':'q1c-loader-audit-1104','directory':'q1c-loader-audit-20261104','stage':'loader','gpus':0},{'job':'q1c-scores-1104','directory':'q1c-scores-20261104','stage':'scores','gpus':2}]
 for stage in ['train','eval']:
  for v in PLAN['variants']:out.append({'job':f'q1c-{v}-{stage}-1104','directory':f'q1c-{v}-{stage}-20261104','stage':stage,'variant':v,'gpus':2 if stage=='train' else 1})
 out.append({'job':'q1c-compare-1104','directory':RUN.name,'stage':'compare','gpus':0});return out

def done(t):
 if (ROOT/'runs'/t['directory']/'complete.json').exists():return True
 return 'variant' in t and (ROOT/f"runs/q1c-{t['variant']}-train-20261104/undefined.json").exists()

def submit(t):
 d=ROOT/'runs'/t['directory'];stage=t['stage'];assert stage not in ['probe','loader']
 if stage=='train':assert (RUN/'storage-ready.json').exists(),'Storage has not been resolved'
 image=PLAN['eval_image'] if stage=='eval' else PLAN['train_image'];cmd=['runai','submit','--name',t['job'],'--image',image,'--gpu',str(t['gpus']),'--cpu','16' if t['gpus'] else '2','--memory','192G' if stage=='scores' else '128G' if t['gpus'] else '8G','--backoff-limit','0','--large-shm','--existing-pvc','claimname=cvlab-scratch,path=/scratch']
 if t['gpus']:cmd+=['--node-pool','h200']
 cmd+=['--command','--','/opt/datasense/bin/python',CONTAINER+f'/scripts/q1_continuation_{stage}.py']
 if 'variant' in t:cmd.append(t['variant'])
 write(d/'submission.json',{'command':cmd,'at':now()});print(cluster.call(cmd),flush=True);state(d,'submitted','queued')
 if stage=='compare':
  m=json.loads((d/'run.json').read_text());m['job']={'provider':'runai','name':t['job'],'namespace':NS};write(d/'run.json',m)

def selection_summary():
 p=ROOT/'runs/q1c-scores-20261104';data={m:{x['meta_index']:x for x in rows(p/f'{m}-scores.jsonl')} for m in ['F','K']};assert len(data['F'])==len(data['K'])==100 and set(data['F'])==set(data['K']);assert all(data['F'][i]['gradient_norm']==data['K'][i]['gradient_norm'] for i in data['F']);summary={}
 for v in PLAN['variants']:
  r=ROOT/f'runs/q1c-{v}-train-20261104'
  if (r/'undefined.json').exists():summary[v]=json.loads((r/'undefined.json').read_text());continue
  a=json.loads((r/'data/audit.json').read_text());assert len(a['meta_indices'])==len(set(a['meta_indices']))==50 and sha(r/'data/train-data.json')==a['data_sha256'];summary[v]=a
 write(p/'result.json',{'candidate_count':100,'six_scores_complete':True,'shared_candidate_gradient_norms_identical':True,'selections':summary,'at':now()});write(RUN/'selection-summary.json',summary)

def main():
 with (RUN/'controller.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);todo=tasks();state(RUN,'continuation_training_and_evaluation' if (RUN/'storage-ready.json').exists() else 'probe_and_selection',started_at=now())
  while True:
   try:
    pods=json.loads(cluster.call(['kubectl','get','pods','-n',NS,'-o','json']))['items'];jobs={x['metadata']['name'] for x in json.loads(cluster.call(['kubectl','get','runaijobs','-n',NS,'-o','json']))['items']}
   except RuntimeError as e:write(RUN/'controller-state.json',{'stage':'cluster_retry','error':str(e),'at':now()});time.sleep(20);continue
   for t in todo:
    items=[p for p in pods if p['metadata']['name'].startswith(t['job']+'-')]
    if not items:continue
    phase=max(items,key=lambda p:p['metadata']['creationTimestamp'])['status']['phase']
    if phase in ['Failed','Succeeded']:
     cluster.archive(t,items)
     if phase=='Failed':raise RuntimeError(t['job']+' failed; existing successful work retained')
     assert done(t),t['job']+' exited without completion marker';print(cluster.call(['runai','delete','job',t['job']]),flush=True);jobs.discard(t['job'])
   active={s:sum(t['job'] in jobs for t in todo if t['stage']==s) for s in ['train','eval']}
   policy=json.loads((RUN/'execution-policy.json').read_text()) if (RUN/'execution-policy.json').exists() else {}
   order=policy.get('training_order',PLAN['variants'])
   ordered=sorted(todo,key=lambda t:(0,order.index(t['variant'])) if t['stage']=='train' else (1,0))
   for t in ordered:
    if done(t) or t['job'] in jobs:continue
    s=t['stage'];ready=False
    if s=='scores':ready=(ROOT/'runs/q1c-probe-20261104/prepared-complete.json').exists() and (ROOT/'runs/q1c-probe-20261104/complete.json').exists() and (ROOT/'runs/q1c-loader-audit-20261104/complete.json').exists() and (RUN/'loader-policy.json').exists()
    if s=='train':ready=(RUN/'storage-ready.json').exists() and (ROOT/'runs/q1c-scores-20261104/complete.json').exists() and (ROOT/'runs'/t['directory']/'preparation-complete.json').exists() and active['train']<2 and not any(j.startswith('q1-') and '-train-' in j for j in jobs)
    if s=='eval':ready=(ROOT/f"runs/q1c-{t['variant']}-train-20261104/complete.json").exists() and active['eval']<2
    if s=='compare':ready=all(done(x) for x in todo if x['stage']=='eval')
    if ready:
     submit(t);jobs.add(t['job'])
     if s in active:active[s]+=1
   write(RUN/'controller-state.json',{'stage':'monitoring','at':now(),'active_jobs':[t['job'] for t in todo if t['job'] in jobs],'completed':[t['directory'] for t in todo if done(t)]})
   prepared=all(done(t) and t['job'] not in jobs for t in todo if t['stage'] in ['probe','loader','scores'])
   if prepared:
    if not (RUN/'selection-summary.json').exists():selection_summary()
    if not (RUN/'storage-ready.json').exists():
     state(RUN,'ready_for_continuation_waiting_storage','blocked',message='Fresh probe gradients and top50 selections complete; need storage capacity before SFT. No old checkpoints deleted.');write(RUN/'controller-state.json',{'stage':'waiting_storage','at':now(),'compute_jobs_active':False});print('PREPARATION COMPLETE; TRAINING REQUIRES STORAGE RESOLUTION',flush=True);return
   if all(done(t) for t in todo if t['stage']=='train') and not (RUN/'all-training-complete.json').exists():write(RUN/'all-training-complete.json',{'finished_at':now()})
   if all(done(t) and t['job'] not in jobs for t in todo):write(RUN/'pipeline-complete.json',{'finished_at':now()});state(RUN,'complete','succeeded');print('CONTINUATION STUDY COMPLETE',flush=True);return
   time.sleep(20)
if __name__=='__main__':
 try:main()
 except BaseException as e:write(RUN/'controller-state.json',{'stage':'failed','error':str(e),'at':now()});raise
