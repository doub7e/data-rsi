"""Persistent DAG control, model-server lifecycle, bounded infra recovery, scoped cleanup."""
import fcntl,time,sys,urllib.request
from common import *
from register import register,prepare
P=json.loads((ROOT/'configs/protocol.json').read_text());STUDY=ROOT/'runs/study'
NAMES={'assets':'ota-resample-assets-1107'}
def jobname(rid):return NAMES.get(rid,'ota-rs-'+rid+'-1107')
def complete(rid):return (ROOT/'runs'/rid/'complete.json').exists()
def archive(job,run,pods):
 for p in pods:
  uid=p['metadata']['uid'];folder=run/'cluster'/uid;folder.mkdir(parents=True,exist_ok=True);write(folder/'pod.json',p);log=call(['kubectl','logs','-n',NS,p['metadata']['name']],check=False,timeout=120);(folder/'pod.log').write_text(log.stdout+log.stderr)
 r=call(['runai','delete','job',job],check=False)
 if r.returncode and not any(s in (r.stdout+r.stderr).lower() for s in ['not found','could not find']):raise RuntimeError(r.stderr or r.stdout)
 write(run/'last-cleanup.json',{'at':now(),'job':job});print('CLEANUP',job,flush=True)
def submit(rid,script,args=(),gpu=0,cpu=8,memory='48G',image=None,env=None,service=False):
 job=jobname(rid);run=register(rid,rid,'validation' if service else 'analysis',job=job);image=image or (P['worker_image'] if script=='run_worker.sh' and complete('worker-image-check') else P['train_image'])
 cmd=['runai','submit','--name',job,'--image',image,'--gpu',str(gpu),'--cpu',str(cpu),'--memory',memory,'--backoff-limit','0','--run-as-uid','223974','--run-as-gid','11166','--environment','USER=shuli','--environment','LOGNAME=shuli','--large-shm','--existing-pvc','claimname=cvlab-scratch,path=/scratch']
 if gpu:cmd+=['--node-pool','h200']
 for k,v in (env or {}).items():cmd+=['--environment',k+'='+v]
 command=['/bin/bash',remote(ROOT/'scripts'/script)] if script.endswith('.sh') else ['python3' if service else '/opt/harbor/bin/python' if image==P.get('worker_image') else '/opt/sft/bin/python',remote(ROOT/'scripts'/script)]
 cmd+=['--command','--']+command+list(args);write(run/'submission.json',{'at':now(),'command':cmd});print(call(cmd).stdout,flush=True);state(run,'submitted','queued');return job

def ensure_service(rid,mode,jobs,pods,budget):
 job=jobname(rid);image=P['teacher_serve_image'] if mode=='teacher' else P['student_serve_image'];run=ROOT/'runs'/rid
 if any(j.startswith('ota-rs-serve-') and j!=job for j in jobs):return None
 if job not in jobs:
  if budget[0]+4>8:return None
  submit(rid,'serve.py',[mode],gpu=4,cpu=16,memory='128G',image=image,service=True);jobs.add(job);budget[0]+=4;return None
 matches=[p for p in pods if p['metadata'].get('labels',{}).get('workloadName')==job]
 if not matches:return None
 pod=max(matches,key=lambda p:p['metadata']['creationTimestamp']);container=pod['spec']['containers'][0];assert container['image']==image and remote(ROOT/'scripts/serve.py') in container.get('command',[])+container.get('args',[]),'Unexpected existing serving workload'
 if pod['status']['phase']!='Running':return None
 name='ota-resample-'+mode;code="import json,urllib.request;d=json.load(urllib.request.urlopen('http://127.0.0.1:8000/v1/models',timeout=3));assert any(m['id']=="+repr(name)+" for m in d['data']);print('ready')"
 if call(['kubectl','exec','-n',NS,pod['metadata']['name'],'--','python3','-c',code],check=False,timeout=20).returncode:return None
 endpoint='http://'+pod['status']['podIP']+':8000/v1';write(run/'endpoint.json',{'at':now(),'endpoint':endpoint,'pod':pod['metadata']['name'],'model':name,'transport':'PodIP; no ClusterIP Service permissions required'});state(run,'serving_ready','running')
 if mode=='teacher' and not complete('teacher-smoke'):
  code="""import json,urllib.request,importlib.metadata;from pathlib import Path
url='http://127.0.0.1:8000/v1/chat/completions';body={'model':'ota-resample-teacher','messages':[{'role':'user','content':'Compute 17+25 and return a JSON object with key answer.'}],'max_tokens':1024,'temperature':0.6,'seed':20261107};request=urllib.request.Request(url,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'});d=json.load(urllib.request.urlopen(request,timeout=180));m=d['choices'][0]['message'];assert m.get('content');out=Path('"""+remote(ROOT/'runs/teacher-smoke')+"""');(out/'response.json').write_text(json.dumps(d));(out/'result.json').write_text(json.dumps({'metrics':{'completion_tokens':d.get('usage',{}).get('completion_tokens',0)},'versions':{n:importlib.metadata.version(n) for n in ['vllm','torch','transformers']},'message_fields':list(m),'reasoning_chars':len(m.get('reasoning') or m.get('reasoning_content') or ''),'content':m['content']}));print('teacher_smoke_ok')"""
  response=call(['kubectl','exec','-n',NS,pod['metadata']['name'],'--','python3','-c',code],check=False,timeout=210)
  if response.returncode:write(ROOT/'runs/teacher-smoke/error.json',{'at':now(),'message':response.stderr[-3000:]});raise RuntimeError('Teacher HTTP smoke failed; inspect artifact')
  write(ROOT/'runs/teacher-smoke/complete.json',{'finished_at':now()});state(ROOT/'runs/teacher-smoke','complete','succeeded',finished_at=now())
 if mode!='teacher':
  marker=run/'generation-smoke-complete.json';previous=json.loads(marker.read_text()) if marker.exists() else {}
  if previous.get('pod_uid')!=pod['metadata']['uid']:
   state(run,'concurrent_generation_smoke','starting')
   smoke=call(['kubectl','exec','-n',NS,pod['metadata']['name'],'--','python3',remote(ROOT/'scripts/server_smoke.py'),mode,rid],check=False,timeout=600)
   if smoke.returncode:
    write(run/'generation-smoke-error.json',{'at':now(),'stderr':smoke.stderr[-2000:],'stdout':smoke.stdout[-1000:]});raise RuntimeError('Student concurrent generation smoke failed: '+rid)
   write(marker,{'finished_at':now(),'pod_uid':pod['metadata']['uid']});state(run,'serving_ready','running')
 return endpoint

def release_service(rid,jobs,pods,budget):
 job=jobname(rid)
 if job not in jobs:return
 run=ROOT/'runs'/rid;archive(job,run,[p for p in pods if p['metadata'].get('labels',{}).get('workloadName')==job]);jobs.remove(job);budget[0]-=4;write(run/'result.json',{'finished_at':now(),'metrics':{'released_gpus':4}});write(run/'complete.json',{'finished_at':now()});state(run,'complete','succeeded',finished_at=now())

def cleanup_sandboxes(rid):
 label='ota-resample-'+rid;pods=json.loads(call(['kubectl','get','pods','-n',NS,'-l','data-rsi-native='+label,'-o','json']).stdout)['items']
 for p in pods:
  job=p['metadata'].get('labels',{}).get('workloadName','');assert job.startswith('ota-native-')
  assert not any('persistentVolumeClaim' in v for v in p['spec'].get('volumes',[]))
  archive(job,ROOT/'runs'/rid/'sandbox-cleanup',[p])

def recover_terminal(jobs,pods):
 for manifest in sorted((ROOT/'runs').glob('*/run.json'),key=lambda p:(not p.parent.name.startswith('serve-'),p.parent.name)):
  run=manifest.parent;rid=run.name;job=(json.loads(manifest.read_text()).get('job') or {}).get('name')
  if not job or job not in jobs:continue
  matches=[p for p in pods if p['metadata'].get('labels',{}).get('workloadName')==job]
  if not matches or any(p['status']['phase'] not in ['Succeeded','Failed'] for p in matches):continue
  if rid.startswith('serve-'):
   consumers={'serve-teacher-a':['teacher-batch1','teacher-uniform'],'serve-teacher-b':['teacher-guided'],'serve-epoch1':['probe'],'serve-uniform':['eval-uniform'],'serve-guided':['eval-guided']}[rid]
   for consumer in consumers:
    client=jobname(consumer)
    if client in jobs and not complete(consumer):
     archive(client,ROOT/'runs'/consumer,[p for p in pods if p['metadata'].get('labels',{}).get('workloadName')==client]);jobs.remove(client);cleanup_sandboxes(consumer);state(ROOT/'runs'/consumer,'waiting_server_recovery','restarting',message='Inference server stopped; interrupted outputs retained for same-seed resume')
   archive(job,run,matches);jobs.remove(job)
   if all(complete(c) for c in consumers):write(run/'complete.json',{'finished_at':now()});state(run,'complete','succeeded',finished_at=now());continue
   path=run/'server-retries.json';history=json.loads(path.read_text()) if path.exists() else {'events':[]};history['events'].append({'at':now(),'reason':'Unexpected serving process termination; stopped dependent clients before retry','pod_uids':[p['metadata']['uid'] for p in matches]});write(path,history)
   if len(history['events'])<=2:state(run,'bounded_server_recovery','restarting');continue
   state(run,'server_retry_limit','failed',finished_at=now());raise RuntimeError(rid+' server recovery limit reached')
  if rid in ['teacher-batch1','teacher-uniform','teacher-guided','probe','eval-uniform','eval-guided']:cleanup_sandboxes(rid)
  archive(job,run,matches);jobs.remove(job)
  if (run/'complete.json').exists():continue
  st=json.loads((run/'state.json').read_text()) if (run/'state.json').exists() else {};error=st.get('error','')
  if error.startswith('UnresolvedRolloutErrors:') and (run/'plan.json').exists():
   items=json.loads((run/'plan.json').read_text())['items']
   if items and all((run/'trials'/x['id']/'complete.json').exists() for x in items):
    write(run/'metadata-finalization-required.json',{'at':now(),'reason':'All originally unresolved outcomes were separately postprocessed and all planned results now exist. Aggregate only; no additional rollout.','records':len(items)});state(run,'metadata_finalization','queued');continue
  infra=any(x in error for x in ['AuthenticationUnavailable','APIConnectionError','LLMRequestTimeoutError','ServiceUnavailableError','ConnectionError','NativeSandboxInterrupted'])
  old=run/'infrastructure-retries.json';history=json.loads(old.read_text()) if old.exists() else {'events':[]}
  if infra and len(history['events'])<4:
   history['events'].append({'at':now(),'error':error,'policy':'Resume only interrupted/auth trials; preserve finished outputs and seeds.'});write(old,history);state(run,'infra_recovery','queued');continue
  raise RuntimeError(rid+' failed; inspect preserved logs before retrying: '+error)

def main():
 with (STUDY/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  while True:
   try:
    pods=json.loads(call(['kubectl','get','pods','-n',NS,'-o','json']).stdout)['items'];objects=json.loads(call(['kubectl','get','runaijobs','-n',NS,'-o','json']).stdout)['items'];jobs={x['metadata']['name'] for x in objects}
   except Exception as e:state(STUDY,'waiting_cluster_auth','running',message=str(e));time.sleep(20);continue
   recover_terminal(jobs,pods)
   budget=[sum(float(c.get('resources',{}).get('limits',{}).get('nvidia.com/gpu',0)) for p in pods if p['metadata'].get('labels',{}).get('workloadName','').startswith(('ota-rs-','ota-resample-')) and p['status']['phase'] not in ['Succeeded','Failed'] for c in p['spec']['containers'])]
   if not (ROOT/'configs/task-inventory.json').exists():state(STUDY,'asset_preparation','running');time.sleep(20);continue
   prepare()
   for rid in ['teacher-batch1','teacher-uniform','teacher-guided','probe','eval-uniform','eval-guided']:
    if (ROOT/'runs'/rid/'metadata-finalization-required.json').exists() and not complete(rid) and jobname(rid) not in jobs:
     jobs.add(submit(rid,'finalize_rollouts.py',[rid],gpu=0,cpu=2,memory='16G',image=P['worker_image']))
   rl_ready=len(list((ROOT/'tasks/native/rl').glob('*/native-ready.json')))==728;dev_ready=len(list((ROOT/'tasks/native/dev').glob('*/native-ready.json')))==70
   # Release servers as soon as their consumers are done, before scheduling dependent compute.
   if complete('teacher-uniform'):release_service('serve-teacher-a',jobs,pods,budget)
   if complete('teacher-guided'):release_service('serve-teacher-b',jobs,pods,budget)
   for worker,server in [('probe','serve-epoch1'),('eval-uniform','serve-uniform'),('eval-guided','serve-guided')]:
    if complete(worker):release_service(server,jobs,pods,budget)
   if rl_ready and not complete('harness-smoke') and jobname('harness-smoke') not in jobs:
    jobs.add(submit('harness-smoke','run_worker.sh',['--preflight'],gpu=0,cpu=4,memory='16G'))
   if complete('assets') and rl_ready and not complete('teacher-uniform'):
    endpoint=ensure_service('serve-teacher-a','teacher',jobs,pods,budget)
    rid='teacher-batch1' if not complete('teacher-batch1') else 'teacher-uniform'
    if endpoint and complete('harness-smoke') and jobname(rid) not in jobs:jobs.add(submit(rid,'run_worker.sh',[rid],env={'OTA_RESAMPLE_API_BASE':endpoint}))
   for rid,data in [('sft-epoch1','teacher-batch1'),('sft-uniform','teacher-uniform'),('sft-guided','teacher-guided')]:
    if complete(data) and (rid=='sft-epoch1' or complete('sft-epoch1')) and not complete(rid) and jobname(rid) not in jobs and budget[0]+4<=8:
     jobs.add(submit(rid,'train.py',[rid],gpu=4,cpu=16,memory='256G'));budget[0]+=4
   if complete('sft-epoch1') and dev_ready and not complete('probe'):
    endpoint=ensure_service('serve-epoch1','sft-epoch1',jobs,pods,budget)
    if endpoint and jobname('probe') not in jobs:jobs.add(submit('probe','run_worker.sh',['probe'],env={'OTA_RESAMPLE_API_BASE':endpoint}))
   if complete('probe') and complete('gradient-bf16-contract-check') and not complete('probe-gradient-plan') and jobname('probe-gradient-plan') not in jobs:
    jobs.add(submit('probe-gradient-plan','prepare_probe_gradient.py',gpu=0,cpu=2,memory='16G',image=P['worker_image']))
   if complete('sft-epoch1') and complete('gradient-bf16-contract-check') and not complete('probe-gradient'):
    from question_gradient import run_id,records_for
    ready=[];probe_plan=json.loads((ROOT/'runs/probe/plan.json').read_text())['items']
    for task in json.loads((ROOT/'configs/dev-split.json').read_text())['probe_tasks']:
     items=[x for x in probe_plan if x['task']==task]
     if not all((ROOT/'runs/probe/trials'/x['id']/'complete.json').exists() for x in items):continue
     rows=records_for(task);scores=[x['score'] for x in rows]
     if not all(isinstance(x,(int,float)) for x in scores) or len(set(scores))<=1:continue
     ready.append(task)
    active=sum(j.startswith('ota-rs-probe-question-') for j in jobs)
    for task in ready:
     rid=run_id(task);register(rid,'Checkpoint1 · 单题reward梯度 · '+task,sources={'training_data':f'runs/{rid}/data/train.jsonl'})
     if not complete(rid) and jobname(rid) not in jobs and active<4 and budget[0]+1<=8:
      jobs.add(submit(rid,'question_gradient.py',[task],gpu=1,cpu=8,memory='128G'));budget[0]+=1;active+=1
    if ready and jobname('probe-gradient') not in jobs:
     done_rows=0
     for task in ready:
      f=ROOT/'runs'/run_id(task)/'progress.json'
      if f.exists():done_rows+=json.loads(f.read_text()).get('done',0)
     write(ROOT/'runs/probe-gradient/progress.json',{'stage':'known_eligible_question_gradients','done':done_rows,'total':32*len(ready),'scope_note':'Only independently finalized eligible questions; global mean waits for full probe eligibility.','updated_at':now()});state(ROOT/'runs/probe-gradient','independent_question_gradients','running')
    if complete('probe-gradient-plan'):
     plan=json.loads((ROOT/'runs/probe-gradient/plan.json').read_text());eligible=[q['task'] for q in plan['eligible']]
     if eligible and all(complete(run_id(task)) for task in eligible) and jobname('probe-gradient') not in jobs:
      jobs.add(submit('probe-gradient','merge_question_gradients.py',gpu=0,cpu=4,memory='128G'))
   if complete('probe-gradient'):
    for i in [0,1]:
     rid=f'attribution-candidates-{i}'
     no_signal=json.loads((ROOT/'runs/probe-gradient/probe-gradient-complete.json').read_text())['no_signal'];gpu=0 if no_signal else 1
     if not complete(rid) and jobname(rid) not in jobs and budget[0]+gpu<=8:jobs.add(submit(rid,'candidate_dots.py',[str(i)],gpu=gpu,cpu=2 if no_signal else 8,memory='16G' if no_signal else '192G'));budget[0]+=gpu
   if all(complete(f'attribution-candidates-{i}') for i in [0,1]) and not complete('attribution'):
    from select_environments import main as select
    select()
   if complete('attribution') and not complete('teacher-guided'):
    endpoint=ensure_service('serve-teacher-b','teacher',jobs,pods,budget)
    if endpoint and jobname('teacher-guided') not in jobs:jobs.add(submit('teacher-guided','run_worker.sh',['teacher-guided'],env={'OTA_RESAMPLE_API_BASE':endpoint}))
   for branch in ['uniform','guided']:
    rid='eval-'+branch
    if complete('sft-'+branch) and dev_ready and not complete(rid):
     endpoint=ensure_service('serve-'+branch,'sft-'+branch,jobs,pods,budget)
     if endpoint and complete('harness-smoke') and jobname(rid) not in jobs:jobs.add(submit(rid,'run_worker.sh',[rid],env={'OTA_RESAMPLE_API_BASE':endpoint}))
   stages=['assets','environments','teacher-batch1','sft-epoch1','teacher-uniform','probe','probe-gradient','attribution-candidates-0','attribution-candidates-1','attribution','teacher-guided','sft-uniform','sft-guided','eval-uniform','eval-guided'];done=sum(complete(x) for x in stages);write(STUDY/'progress.json',{'stage':'experiment_stages','done':done,'total':len(stages),'updated_at':now()});write(STUDY/'controller-state.json',{'at':now(),'active_jobs':sorted(j for j in jobs if j.startswith(('ota-rs-','ota-resample-'))),'requested_gpus':budget[0],'rl_ready':rl_ready,'dev_ready':dev_ready,'completed_stages':[s for s in stages if complete(s)]});state(STUDY,'pipeline_running','running')
   if complete('eval-uniform') and complete('eval-guided'):
    if not complete('comparison') and jobname('comparison') not in jobs:jobs.add(submit('comparison','aggregate.py',gpu=0,cpu=2,memory='8G'))
    if complete('comparison'):
     write(STUDY/'result.json',json.loads((ROOT/'runs/comparison/result.json').read_text()));write(STUDY/'complete.json',{'finished_at':now()});state(STUDY,'complete','succeeded',finished_at=now())
     from verify_final import main as verify
     verify();return
   time.sleep(20)
if __name__=='__main__':
 try:main()
 except BaseException as e:state(STUDY,'failed','failed',error=str(e),finished_at=now());raise
