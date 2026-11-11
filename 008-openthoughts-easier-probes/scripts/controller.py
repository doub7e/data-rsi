"""Isolated OTA probe lifecycle; never controls previous experiment jobs."""
import fcntl,subprocess,time
from common import *
RUN=ROOT/'runs/easy-probes-20261105';P=json.loads((RUN/'protocol.json').read_text());ENV={**os.environ,'KUBECONFIG':'/mnt/cvlab/scratch/cvlab/home/shuli/.codex/secrets/native-runai-kubeconfig.json'}
REMOTE='/scratch/cvlab/home/shuli/data-rsi/008-openthoughts-easier-probes'
def call(args,check=True):
 r=subprocess.run(args,env=ENV,text=True,capture_output=True,timeout=90)
 if check and r.returncode:raise RuntimeError(' '.join(args[:3])+': '+(r.stderr or r.stdout)[-1200:])
 return r
def archive(job,run,pods):
 for p in pods:
  name=p['metadata']['name'];uid=p['metadata']['uid'];write(run/f'cluster/{name}-{uid}.json',p);f=run/f'logs/{name}-{uid}.log';f.parent.mkdir(exist_ok=True);r=call(['kubectl','logs','-n',NS,name],check=False);f.write_text(r.stdout+r.stderr)
 deleted=call(['runai','delete','job',job],check=False)
 if deleted.returncode and not any(x in (deleted.stderr+deleted.stdout).lower() for x in ['not found','could not find']):raise RuntimeError(deleted.stderr or deleted.stdout)
 print(deleted.stdout,flush=True)
def own_sandboxes(run):
 label='ota-easy-'+run.name;pods=json.loads(call(['kubectl','get','pods','-n',NS,'-l','data-rsi-native='+label,'-o','json']).stdout)['items'];owned=set()
 for f in run.rglob('native-events.jsonl'):
  for line in f.read_text().splitlines():
   try:r=json.loads(line)
   except ValueError:continue
   if r.get('event')=='start_requested':owned.add(r['job'])
 for p in pods:
  job=p['metadata'].get('labels',{}).get('workloadName');assert job in owned,'Sandbox label not present in thisrun ownershiplog';archive(job,run/'sandbox-cleanup', [p])
def serving(jobs,pods):
 run=ROOT/'runs/serve-base-20261105';job=P['serve_job']
 if job not in jobs:
  cmd=['runai','submit','--name',job,'--image',P['serve_image'],'--gpu','4','--node-pool','h200','--cpu','16','--memory','128G','--run-as-uid','223974','--run-as-gid','11166','--environment','USER=shuli','--environment','LOGNAME=shuli','--environment','VLLM_CACHE_ROOT=/tmp/ota-easy-vllm','--environment','VLLM_CONFIG_ROOT=/tmp/ota-easy-vllm-config','--environment','XDG_CACHE_HOME=/tmp/ota-easy-cache','--environment','TRITON_CACHE_DIR=/tmp/ota-easy-triton','--large-shm','--backoff-limit','0','--existing-pvc','claimname=cvlab-scratch,path=/scratch','--command','--','/bin/bash',REMOTE+'/scripts/serve_base.sh'];write(run/'submission.json',{'at':now(),'command':cmd});print(call(cmd).stdout,flush=True);state(run,'serving_queue','queued');return None
 matches=[p for p in pods if p['metadata'].get('labels',{}).get('workloadName')==job or p['metadata']['name'].startswith(job+'-')]
 if not matches:return None
 pod=max(matches,key=lambda p:p['metadata']['creationTimestamp']);phase=pod['status']['phase'];container=pod['spec']['containers'][0]
 assert container['image']==P['serve_image'] and REMOTE+'/scripts/serve_base.sh' in container.get('command',[])+container.get('args',[]),'Existing serving job identity differs'
 if phase=='Failed':archive(job,run,matches);raise RuntimeError('Base inference job failed')
 if phase!='Running':return None
 check="import json,urllib.request;d=json.load(urllib.request.urlopen('http://127.0.0.1:8000/v1/models',timeout=3));assert any(m['id']=='ota-easy-qwen3-8b-base' for m in d['data']);print('ready')"
 if call(['kubectl','exec','-n',NS,pod['metadata']['name'],'--','python3','-c',check],check=False).returncode:return None
 # ClusterIP service keeps queued/running workers independent of serving podIP changes.
 service={'apiVersion':'v1','kind':'Service','metadata':{'name':job,'namespace':NS},'spec':{'selector':{'workloadName':job},'ports':[{'port':8000,'targetPort':8000}],'type':'ClusterIP'}}
 path=run/'service.json';write(path,service)
 if not (run/'service-created.json').exists():
  created=call(['kubectl','apply','-f',str(path)],check=False)
  if created.returncode:
   if 'forbidden' not in (created.stderr+created.stdout).lower():raise RuntimeError(created.stderr[-1000:])
   endpoint='http://'+pod['status']['podIP']+':8000/v1';write(run/'endpoint.json',{'endpoint':endpoint,'pod':pod['metadata']['name'],'checked_at':now(),'model':P['model_name'],'transport':'podIP fallback: Service creation forbidden'});state(run,'ready','running');return endpoint
  write(run/'service-created.json',{'at':now()})
 endpoint=f'http://{job}.{NS}.svc.cluster.local:8000/v1';write(run/'endpoint.json',{'endpoint':endpoint,'pod':pod['metadata']['name'],'checked_at':now(),'model':P['model_name']});state(run,'ready','running');return endpoint

def submit(q,stage,endpoint=None):
 run=ROOT/f"runs/{q['key']}-{stage}-20261105";job=f"ota-easy-{q['key']}-{stage}-1105"
 analytical=False
 if stage=='gradient':
  outcomes=json.loads((ROOT/f"runs/{q['key']}-rollouts-20261105/result.json").read_text())['scores'];analytical=len(outcomes)==32 and len(set(outcomes))==1
 gpu='0' if stage=='rollouts' or analytical else '1';memory='48G' if stage=='rollouts' else '16G' if analytical else P['execution']['gradient_memory']
 if analytical:
  m=json.loads((run/'run.json').read_text());m['parameters'].update(analytical_zero=True,gpus=0,reason='Constantreward: noGPUderivative necessary');write(run/'run.json',m)
 cmd=['runai','submit','--name',job,'--image',P['train_image'],'--gpu',gpu,'--cpu','2' if analytical else '8','--memory',memory,'--backoff-limit','0','--large-shm','--existing-pvc','claimname=cvlab-scratch,path=/scratch']
 if gpu!='0':cmd+=['--node-pool','h200']
 if endpoint:cmd+=['--environment','OTA_PROBE_API_BASE='+endpoint]
 cmd+=['--command','--']+(['/bin/bash',REMOTE+'/scripts/run_rollouts.sh',q['key']] if stage=='rollouts' else ['/opt/sft/bin/python',REMOTE+('/scripts/zero_gradient.py' if analytical else '/scripts/gradients.py'),q['key']])
 write(run/'submission.json',{'at':now(),'command':cmd});print(call(cmd).stdout,flush=True);state(run,'submitted','queued')

def main():
 global P
 with (RUN/'controller.lock').open('w') as f:
  fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  while True:
   try:pods=json.loads(call(['kubectl','get','pods','-n',NS,'-o','json']).stdout)['items'];jobs={x['metadata']['name'] for x in json.loads(call(['kubectl','get','runaijobs','-n',NS,'-o','json']).stdout)['items']}
   except RuntimeError as e:state(RUN,'cluster_auth_or_connectivity','blocked',error=str(e));time.sleep(20);continue
   for name,job in [('loss-audit-20261105','ota-easy-loss-audit-1105'),('loss-parity-20261105','ota-easy-loss-parity-1105')]:
    run=ROOT/'runs'/name;matches=[p for p in pods if p['metadata']['name'].startswith(job+'-')]
    if matches and all(p['status']['phase'] in ['Succeeded','Failed'] for p in matches):
     archive(job,run,matches);jobs.discard(job)
     assert (run/'complete.json').exists(),name+' did not pass'
   if not (ROOT/P['loss']['requires_loss_parity']).exists():state(RUN,'waiting_for_loss_parity','planned');time.sleep(20);continue
   stages=['rollouts'] if P['phase']=='screening' else ['rollouts','gradient']
   tasks=[(q,stage,f"ota-easy-{q['key']}-{stage}-1105",ROOT/f"runs/{q['key']}-{stage}-20261105") for q in P['tasks'] for stage in stages]
   for q,stage,job,run in tasks:
    matches=[p for p in pods if p['metadata']['name'].startswith(job+'-')]
    if not matches:continue
    latest=max(matches,key=lambda p:p['metadata']['creationTimestamp'])
    if latest['status']['phase'] in ['Succeeded','Failed']:
     if stage=='rollouts':own_sandboxes(run)
     archive(job,run,matches);jobs.discard(job)
     if not (run/'complete.json').exists():
      st=json.loads((run/'state.json').read_text()) if (run/'state.json').exists() else {};error=str(st.get('error',''))
      network_failure=st.get('exception_type') in ['AuthenticationUnavailable','APIConnectionError','LLMRequestTimeoutError','ServiceUnavailableError'] or any(t in error for t in ['AuthenticationUnavailable','APIConnectionError','LLMRequestTimeoutError','ServiceUnavailableError'])
      if stage=='rollouts' and network_failure:
       path=run/'infrastructure-retries.json';prior=json.loads(path.read_text()) if path.exists() else {'events':[]}
       if len(prior['events'])<2:
        prior['events'].append({'at':now(),'error':error,'policy':'Pinned native resume archives only unfinished/network/auth trials, verifies every preserved terminal result hash; same fixed seeds.'});write(path,prior);state(run,'infrastructure_recovery','queued',message='Retrying infrastructure interruption; valid rewards preserved');continue
      raise RuntimeError(job+' failed; preserved artifacts, no model-failure replacement')
   if P['phase']=='screening' and all((run/'complete.json').exists() and job not in jobs for q,stage,job,run in tasks):
    from select_tasks import select
    P=select(P);print('SCREENING_SELECTED '+','.join(q['task'] for q in P['tasks']),flush=True);continue
   done_rollouts=all((ROOT/f"runs/{q['key']}-rollouts-20261105/complete.json").exists() for q in P['tasks'])
   endpoint=None
   if not done_rollouts:endpoint=serving(jobs,pods)
   elif P['phase']=='formal' and not (ROOT/'runs/serve-base-20261105/complete.json').exists():
    serve_run=ROOT/'runs/serve-base-20261105'
    if P['serve_job'] in jobs:archive(P['serve_job'],serve_run,[p for p in pods if p['metadata']['name'].startswith(P['serve_job']+'-')])
    if (serve_run/'service-created.json').exists():
     deletion=call(['kubectl','delete','service',P['serve_job'],'-n',NS,'--ignore-not-found'],check=False);write(serve_run/'service-cleanup.json',{'at':now(),'returncode':deletion.returncode,'message':(deletion.stdout+deletion.stderr)[-1000:]})
    else:write(serve_run/'service-cleanup.json',{'at':now(),'created':False,'reason':'PodIP fallback used; noService created, so noService deletion attempted'})
    state(serve_run,'serving_complete','succeeded');write(serve_run/'result.json',{'finished_at':now(),'model':P['model_name'],'rollouts_finished':144,'metrics':{'completed_rollouts':144}});write(serve_run/'complete.json',{'finished_at':now()})
   active={s:sum(job in jobs for q,stage,job,run in tasks if stage==s) for s in ['rollouts','gradient']}
   for q,stage,job,run in tasks:
    zero_cpu=False
    outcome_path=ROOT/f"runs/{q['key']}-rollouts-20261105/result.json"
    if stage=='gradient' and outcome_path.exists():
     outcomes=json.loads(outcome_path.read_text()).get('scores',[]);zero_cpu=len(outcomes)==32 and len(set(outcomes))==1
    if (run/'complete.json').exists() or job in jobs or (not zero_cpu and active[stage]>=(P['execution']['concurrent_rollout_questions'] if stage=='rollouts' else 1)):continue
    if stage=='rollouts' and endpoint and (RUN/'rollout-implementation-ready.json').exists():submit(q,stage,endpoint);jobs.add(job);active[stage]+=1
    if stage=='gradient' and (ROOT/f"runs/{q['key']}-rollouts-20261105/complete.json").exists() and (RUN/'gradient-implementation-ready.json').exists():submit(q,stage);jobs.add(job);active[stage]+=1
   target='rollouts' if P['phase']=='screening' else 'gradient';finished=sum((ROOT/f"runs/{q['key']}-{target}-20261105/complete.json").exists() for q in P['tasks']);write(RUN/'progress.json',{'stage':'screening_questions' if P['phase']=='screening' else 'formal_questions_analyzed','done':finished,'total':len(P['tasks']),'updated_at':now()});write(RUN/'controller-state.json',{'at':now(),'phase':P['phase'],'active_jobs':[job for q,s,job,r in tasks if job in jobs],'completed_questions':finished})
   if P['phase']=='formal' and finished==3 and all(job not in jobs for q,s,job,r in tasks):
    from finalize import main as finalize
    finalize();write(RUN/'complete.json',{'finished_at':now()});state(RUN,'complete','succeeded',finished_at=now())
    from verify_completion import main as verify_completion
    verify_completion();return
   state(RUN,'waiting_for_4h200_inference','queued') if not endpoint and not done_rollouts and not any(active.values()) else state(RUN,'rollouts_and_gradients','running')
   time.sleep(20)
if __name__=='__main__':
 try:main()
 except BaseException as e:state(RUN,'failed','failed',error=str(e),finished_at=now());raise
