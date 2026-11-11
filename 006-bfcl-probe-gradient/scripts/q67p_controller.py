"""Wait for Q1, freeze its two winners, then run exactly two new SFTs and 3 evals."""
import fcntl,subprocess,time
from common import *
from q1_selection_controller import ENV,NS,call,archive,CONTAINER
RUN=ROOT/'runs/q67-pif-20261106'
def choose(results):
 return {mode:min([v for v in results if v.startswith(mode)],key=lambda v:(-results[v]['successes'],int(v[1:]))) for mode in ['f','k']}
def freeze(cfg):
 if 'selected' in cfg:return cfg
 results={v:json.loads((ROOT/f'runs/q1p-{v}-eval-20261105/result.json').read_text()) for v in ['f008','f032','f128','k008','k032','k128']}
 assert all(x['trials']==256 for x in results.values())
 cfg['selected']=choose(results);cfg['selection_frozen_at']=now();write(RUN/'protocol.json',cfg)
 write(RUN/'q1-winner-selection.json',{'at':now(),'selected':cfg['selected'],'all_q1_successes':{v:x['successes'] for v,x in results.items()},'rule':cfg['winner_rule'],'source_results_sha256':{v:sha(ROOT/f'runs/q1p-{v}-eval-20261105/result.json') for v in results}})
 for v in ['f','k']:
  for stage in ['train','eval']:
   d=ROOT/f'runs/q67p-{v}-{stage}-20261106';m=json.loads((d/'run.json').read_text());m['title']=m['title'].replace('最佳'+v.upper(),cfg['selected'][v].upper()).replace('· '+v.upper()+' ·','· '+cfg['selected'][v].upper()+' ·');m['parameters'].update(q1_selected=cfg['selected'][v],probe_rollouts=int(cfg['selected'][v][1:]));write(d/'run.json',m)
 print('FROZEN',cfg['selected'],flush=True);return cfg

def tasks():
 items=[('probe',None,'q67p-probe-20261106',1),('prepare',None,'q67p-prepare-20261106',1)]
 items += [('train',v,f'q67p-{v}-train-20261106',2) for v in ['f','k']]
 items += [('eval',v,f'q67p-{v}-eval-20261106',1) for v in ['random100','f','k']]
 items += [('compare',None,RUN.name,0)]
 return [{'stage':s,'variant':v,'directory':d,'gpus':g,'job':('q67p-compare-1106' if s=='compare' else d.replace('-20261106','-1106'))} for s,v,d,g in items]

def submit(t,cfg):
 s=t['stage'];d=ROOT/'runs'/t['directory'];g=t['gpus'];script='q67p_rollouts.py' if s in ['probe','eval'] else 'q67p_'+s+'.py';args=['probe'] if s=='probe' else [t['variant']] if s in ['train','eval'] else []
 image=cfg['eval_image'] if s in ['probe','eval'] else cfg['train_image']
 cmd=['runai','submit','--name',t['job'],'--image',image,'--gpu',str(g),'--cpu','16' if g else '4','--memory','256G' if s=='prepare' else '128G' if g else '16G','--backoff-limit','0','--large-shm','--existing-pvc','claimname=cvlab-scratch,path=/scratch']
 if g:cmd+=['--node-pool','h200']
 cmd+=['--command','--','/opt/datasense/bin/python',CONTAINER+'/scripts/'+script]+args
 write(d/'submission.json',{'at':now(),'command':cmd});print(call(cmd),flush=True)
 m=json.loads((d/'run.json').read_text());m['job']={'provider':'runai','name':t['job'],'namespace':NS};write(d/'run.json',m);state(d,'submitted','queued')

def main():
 with (RUN/'controller.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);todo=tasks()
  while True:
   cfg=json.loads((RUN/'protocol.json').read_text())
   if not (ROOT/cfg['gate']).exists():
    state(RUN,'waiting_for_q1_completion','planned');write(RUN/'controller-state.json',{'at':now(),'stage':'waiting_for_q1_completion','gate':cfg['gate']});time.sleep(30);continue
   cfg=freeze(cfg)
   ready=json.loads((RUN/'implementation-ready.json').read_text())
   for path,digest in ready['source_sha256'].items():assert sha(ROOT/path)==digest,'Source changed after preflight: '+path
   try:
    pods=json.loads(call(['kubectl','get','pods','-n',NS,'-o','json']))['items'];jobs={x['metadata']['name'] for x in json.loads(call(['kubectl','get','runaijobs','-n',NS,'-o','json']))['items']}
   except (RuntimeError,subprocess.TimeoutExpired) as e:
    write(RUN/'controller-state.json',{'at':now(),'stage':'cluster_connection_retry','error':str(e)});time.sleep(30);continue
   for t in todo:
    items=[x for x in pods if x['metadata']['name'].startswith(t['job']+'-')]
    if not items:continue
    p=max(items,key=lambda x:x['metadata']['creationTimestamp']);phase=p['status']['phase']
    if phase in ['Failed','Succeeded']:
     archive(t,items)
     if phase=='Failed':raise RuntimeError(t['job']+' failed; artifacts preserved; no blind retry')
     assert (ROOT/'runs'/t['directory']/'complete.json').exists(),t['job']+' missing completion'
     print(call(['runai','delete','job',t['job']]),flush=True);jobs.discard(t['job'])
   def done(stage,v=None):return any(t['stage']==stage and t['variant']==v and (ROOT/'runs'/t['directory']/'complete.json').exists() for t in todo)
   active={s:sum(t['job'] in jobs for t in todo if t['stage']==s) for s in ['train','eval']}
   for t in todo:
    s=t['stage'];v=t['variant']
    if done(s,v) or t['job'] in jobs:continue
    ok=s=='probe' or s=='prepare' and done('probe') or s=='train' and done('prepare') and active['train']<1 or s=='eval' and done('prepare') and (v=='random100' or done('train',v)) and active['eval']<1 or s=='compare' and all(done('eval',v) for v in ['random100','f','k'])
    if ok:
     submit(t,cfg);jobs.add(t['job'])
     if s in active:active[s]+=1
   write(RUN/'controller-state.json',{'at':now(),'stage':'monitoring','selected':cfg['selected'],'active_jobs':[t['job'] for t in todo if t['job'] in jobs],'complete':[t['directory'] for t in todo if done(t['stage'],t['variant'])]})
   if all(done(t['stage'],t['variant']) and t['job'] not in jobs for t in todo):
    write(RUN/'pipeline-complete.json',{'finished_at':now()});state(RUN,'complete','succeeded');return
   if not done('compare'):state(RUN,'probe_selection_training_evaluation','running')
   time.sleep(30)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  write(RUN/'controller-failure.json',{'at':now(),'error':str(e)});state(RUN,'controller_failed','failed',error=str(e));raise
