"""Persistent DAG control, model-server lifecycle, bounded infra recovery, scoped cleanup."""
import fcntl,time,sys,urllib.request
from common import *
from register import register,prepare
P=json.loads((ROOT/'configs/protocol.json').read_text());STUDY=ROOT/'runs/study'
NAMES={'sft-guided':'ota-r4-sft-guided-r1-1109'}
def jobname(rid):return NAMES.get(rid,'ota-r4-'+rid+'-1108')
def complete(rid):return (ROOT/'runs'/rid/'complete.json').exists()
def archive(job,run,pods):
 for p in pods:
  uid=p['metadata']['uid'];folder=run/'cluster'/uid;folder.mkdir(parents=True,exist_ok=True);write(folder/'pod.json',p);log=call(['kubectl','logs','-n',NS,p['metadata']['name']],check=False,timeout=120);(folder/'pod.log').write_text(log.stdout+log.stderr)
 r=call(['runai','delete','job','-p','cvlab-shuli',job],check=False)
 if r.returncode and not any(s in (r.stdout+r.stderr).lower() for s in ['not found','could not find']):raise RuntimeError(r.stderr or r.stdout)
 write(run/'last-cleanup.json',{'at':now(),'job':job});print('CLEANUP',job,flush=True)
def submit(rid,script,args=(),gpu=0,cpu=8,memory='48G',image=None,env=None,service=False):
 job=jobname(rid);run=register(rid,rid,'validation' if service else 'analysis',job=job);image=image or (P['worker_image'] if script=='run_worker.sh' and complete('worker-image-check') else P['train_image'])
 cmd=['runai','submit','-p','cvlab-shuli','--name',job,'--image',image,'--gpu',str(gpu),'--cpu',str(cpu),'--memory',memory,'--backoff-limit','0','--run-as-uid','223974','--run-as-gid','11166','--environment','USER=shuli','--environment','LOGNAME=shuli','--large-shm','--existing-pvc','claimname=cvlab-scratch,path=/scratch']
 if gpu:cmd+=['--node-pool','h200']
 for k,v in (env or {}).items():cmd+=['--environment',k+'='+v]
 command=['/bin/bash',remote(ROOT/'scripts'/script)] if script.endswith('.sh') else ['python3' if service else '/opt/harbor/bin/python' if image==P.get('worker_image') else '/opt/sft/bin/python',remote(ROOT/'scripts'/script)]
 cmd+=['--command','--']+command+list(args);write(run/'submission.json',{'at':now(),'command':cmd});print(call(cmd).stdout,flush=True);state(run,'submitted','queued');return job

def server_identity_changed(previous,pod):
 if not previous:return False
 return previous.get('pod')!=pod['metadata']['name'] or bool(previous.get('pod_uid') and previous['pod_uid']!=pod['metadata']['uid'])

def pause_dependents(rid,reason,jobs,pods):
 consumers={'serve-epoch1':['probe'],'serve-uniform':['eval-uniform'],'serve-teacher-b':['teacher-guided'],'serve-guided':['eval-guided']}.get(rid,[])
 for consumer in consumers:
  client=jobname(consumer)
  if client not in jobs or complete(consumer):continue
  run=ROOT/'runs'/consumer;write(run/('inference-interruption-'+str(time.time_ns())+'.json'),{'at':now(),'server':rid,'reason':reason,'policy':'Preserve finished outcomes; stop stale-endpoint clients, then resume only interrupted attempts at new endpoint with same seeds.'});state(run,'stopping_stale_inference_client','restarting',message=reason)
  archive(client,run,[p for p in pods if p['metadata'].get('labels',{}).get('workloadName')==client]);jobs.remove(client)
  for _ in range(45):
   remaining=json.loads(call(['kubectl','get','pods','-n',NS,'-l','workloadName='+client,'-o','json']).stdout)['items']
   if not remaining:break
   time.sleep(2)
  else:raise RuntimeError('Client pod did not exit before sandbox cleanup: '+client)
  cleanup_sandboxes(consumer);state(run,'waiting_inference_recovery','restarting',message=reason)

def ensure_service(rid,mode,jobs,pods,budget):
 job=jobname(rid);image=P['teacher_serve_image'] if mode=='teacher' else P['student_serve_image'];run=ROOT/'runs'/rid
 if any(j.startswith('ota-r4-serve-') and j!=job for j in jobs):return None
 if job not in jobs:
  if budget[0]+4>8:return None
  submit(rid,'serve.py',[mode],gpu=4,cpu=16,memory='128G',image=image,service=True);jobs.add(job);budget[0]+=4;return None
 matches=[p for p in pods if p['metadata'].get('labels',{}).get('workloadName')==job]
 if not matches:
  pause_dependents(rid,'Inference pod missing; keep its RunAI job queued',jobs,pods);state(run,'waiting_inference_pod','queued');return None
 pod=max(matches,key=lambda p:p['metadata']['creationTimestamp']);container=pod['spec']['containers'][0];assert container['image']==image and remote(ROOT/'scripts/serve.py') in container.get('command',[])+container.get('args',[]),'Unexpected existing serving workload'
 previous=json.loads((run/'endpoint.json').read_text()) if (run/'endpoint.json').exists() else {}
 marker=run/'generation-smoke-complete.json'
 if previous and not previous.get('pod_uid') and marker.exists():previous['pod_uid']=json.loads(marker.read_text()).get('pod_uid')
 if server_identity_changed(previous,pod):pause_dependents(rid,'Inference pod identity changed; previous endpoint is no longer valid',jobs,pods)
 if pod['status']['phase']!='Running':
  pause_dependents(rid,'Inference pod is '+pod['status']['phase']+'; retain queued replacement',jobs,pods);state(run,'waiting_inference_pod','queued');return None
 name='ota-resample-'+mode;code="import json,urllib.request;d=json.load(urllib.request.urlopen('http://127.0.0.1:8000/v1/models',timeout=3));assert any(m['id']=="+repr(name)+" for m in d['data']);assert urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=3).status==200;print('ready')"
 readiness=call(['kubectl','exec','-n',NS,pod['metadata']['name'],'--','python3','-c',code],check=False,timeout=20)
 if readiness.returncode:
  failures=run/'readiness-failures.json';old=json.loads(failures.read_text()) if failures.exists() else {};count=(old.get('consecutive',0) if old.get('pod_uid')==pod['metadata']['uid'] else 0)+1;write(failures,{'at':now(),'pod_uid':pod['metadata']['uid'],'consecutive':count,'stderr':readiness.stderr[-1000:]})
  if count>=3:pause_dependents(rid,'Inference health/model check failed repeatedly; clients wait for recovery',jobs,pods)
  if count>=6 and previous.get('pod_uid')==pod['metadata']['uid']:
   history_path=run/'server-retries.json';history=json.loads(history_path.read_text()) if history_path.exists() else {'events':[]};history['events'].append({'at':now(),'reason':'Repeated health failure on previously-ready pod','pod_uid':pod['metadata']['uid']});write(history_path,history)
   archive(job,run,matches);jobs.remove(job);budget[0]-=4
   if len(history['events'])>2:state(run,'server_retry_limit','failed',finished_at=now());raise RuntimeError(rid+' health recovery limit reached')
   state(run,'bounded_health_recovery','restarting');return None
  state(run,'loading_or_recovering_model','starting');return None
 write(run/'readiness-failures.json',{'at':now(),'pod_uid':pod['metadata']['uid'],'consecutive':0})
 endpoint='http://'+pod['status']['podIP']+':8000/v1';write(run/'endpoint.json',{'at':now(),'endpoint':endpoint,'pod':pod['metadata']['name'],'pod_uid':pod['metadata']['uid'],'model':name,'transport':'PodIP; no ClusterIP Service permissions required'});state(run,'serving_ready','running')
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
 label='ota-r4-'+rid;pods=json.loads(call(['kubectl','get','pods','-n',NS,'-l','data-rsi-native='+label,'-o','json']).stdout)['items']
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
   pause_dependents(rid,'Inference server stopped; preserve finished outcomes and await client exit before cleanup',jobs,pods)
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
   budget=[sum(float(c.get('resources',{}).get('limits',{}).get('nvidia.com/gpu',0)) for p in pods if p['metadata'].get('labels',{}).get('workloadName','').startswith('ota-r4-') and p['status']['phase'] not in ['Succeeded','Failed'] for c in p['spec']['containers'])]
   if not (STUDY/'reuse-manifest.json').exists():state(STUDY,'importing_immutable_sources','running');time.sleep(20);continue
   for worker,server in [('probe','serve-epoch1'),('teacher-guided','serve-teacher-b'),('eval-uniform','serve-uniform'),('eval-guided','serve-guided')]:
    if complete(worker):release_service(server,jobs,pods,budget)
   for rid in ['probe','teacher-guided','eval-uniform','eval-guided']:
    if (ROOT/'runs'/rid/'metadata-finalization-required.json').exists() and not complete(rid) and jobname(rid) not in jobs:jobs.add(submit(rid,'finalize_rollouts.py',[rid],gpu=0,cpu=2,memory='16G',image=P['worker_image']))
   if not complete('probe'):
    endpoint=ensure_service('serve-epoch1','sft-epoch1',jobs,pods,budget)
    if endpoint and jobname('probe') not in jobs:jobs.add(submit('probe','run_worker.sh',['probe'],env={'OTA_RESAMPLE_API_BASE':endpoint}))
   if complete('probe') and not complete('probe-gradient-plan') and jobname('probe-gradient-plan') not in jobs:
    jobs.add(submit('probe-gradient-plan','prepare_probe_gradient.py',gpu=0,cpu=2,memory='16G',image=P['worker_image']))
   if complete('probe-gradient-plan') and not complete('probe-gradient'):
    plan=json.loads((ROOT/'runs/probe-gradient/plan.json').read_text());n=plan['num_workers']
    for i in range(n):
     rid=f'probe-gradient-shard-{i}';register(rid,f'全70×4 probe梯度 · 分片{i}',sources={'training_data':f'runs/{rid}/data/train.jsonl'})
     if not complete(rid) and jobname(rid) not in jobs and budget[0]+1<=8:
      jobs.add(submit(rid,'probe_gradient_shard.py',[str(i)],gpu=1,cpu=8,memory='128G'));budget[0]+=1
    if n and all(complete(f'probe-gradient-shard-{i}') for i in range(n)) and jobname('probe-gradient') not in jobs:
     jobs.add(submit('probe-gradient','merge_probe_gradient.py',gpu=0,cpu=4,memory='128G'))
    elif n and jobname('probe-gradient') not in jobs:
     done=0
     for i in range(n):
      f=ROOT/f'runs/probe-gradient-shard-{i}/progress.json'
      if f.exists():done+=json.loads(f.read_text()).get('done',0)
     write(ROOT/'runs/probe-gradient/progress.json',{'stage':'probe_gradient_rollouts','done':done,'total':4*plan['global_valid_questions'],'updated_at':now()});state(ROOT/'runs/probe-gradient','parallel_full_parameter_gradients','running')
   if complete('probe-gradient'):
    meta=json.loads((ROOT/'runs/probe-gradient/probe-gradient-complete.json').read_text());gpu=0 if meta['no_signal'] else 1
    for i in [0,1]:
     rid=f'attribution-candidates-{i}'
     if not complete(rid) and jobname(rid) not in jobs and budget[0]+gpu<=8:
      jobs.add(submit(rid,'candidate_dots.py',[str(i)],gpu=gpu,cpu=2 if not gpu else 8,memory='16G' if not gpu else '192G'));budget[0]+=gpu
   if all(complete(f'attribution-candidates-{i}') for i in [0,1]) and not complete('attribution'):
    from select_environments import main as select
    select()
   if complete('attribution') and not complete('teacher-guided'):
    endpoint=ensure_service('serve-teacher-b','teacher',jobs,pods,budget)
    if endpoint and jobname('teacher-guided') not in jobs:jobs.add(submit('teacher-guided','run_worker.sh',['teacher-guided'],env={'OTA_RESAMPLE_API_BASE':endpoint}))
   if complete('teacher-guided') and not complete('sft-guided') and jobname('sft-guided') not in jobs and budget[0]+4<=8:
    jobs.add(submit('sft-guided','train.py',['sft-guided'],gpu=4,cpu=16,memory='256G'));budget[0]+=4
   # Uniform repeats2–4 can overlap with gradient computation after epoch1 eval.
   for branch in ['uniform','guided']:
    rid='eval-'+branch
    if complete('sft-'+branch) and not complete(rid):
     endpoint=ensure_service('serve-'+branch,'sft-'+branch,jobs,pods,budget)
     if endpoint and jobname(rid) not in jobs:jobs.add(submit(rid,'run_worker.sh',[rid],env={'OTA_RESAMPLE_API_BASE':endpoint}))
   stages=['probe','probe-gradient-plan','probe-gradient','attribution-candidates-0','attribution-candidates-1','attribution','teacher-guided','sft-guided','eval-uniform','eval-guided','comparison'];done=sum(complete(x) for x in stages)
   write(STUDY/'progress.json',{'stage':'new_experiment_stages','done':done,'total':len(stages),'updated_at':now()});write(STUDY/'controller-state.json',{'at':now(),'active_jobs':sorted(j for j in jobs if j.startswith('ota-r4-')),'requested_gpus':budget[0],'completed_stages':[s for s in stages if complete(s)]});state(STUDY,'pipeline_running','running')
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
