"""Seeded single-occurrence Harbor runs with immutable terminal-result recovery."""
import concurrent.futures,threading,subprocess,signal,time,math,sys,os,urllib.request
from common import *
from verifier_scores import valid_score,score_metrics
from transcripts import convert_agent
P=json.loads((ROOT/'configs/protocol.json').read_text());ACTIVE=set();LOCK=threading.Lock()
MODEL_FAILURES={'AgentTimeoutError','AgentEnvironmentTimeoutError','ContextLengthExceededError','ContextBudgetExceededError','OutputLengthExceededError','SummarizationTimeoutError'}
def config(item,run,plan):
 cfg=json.loads((ROOT/'configs/harbor-native-template.json').read_text());cfg.update(job_name=item['id'],jobs_dir=str(run/'jobs'),n_attempts=1,n_concurrent_trials=1,timeout_multiplier=P['timeout_multiplier'],datasets=[],tasks=[{'path':str(TASKS/item['kind']/item['task'])}]);cfg['retry']['max_retries']=0;cfg['verifier']['max_timeout_sec']=P['verifier_timeout_sec'];cfg['environment']['import_path']='seeded_env:SeededRunAIEnvironment'
 if item['kind']=='rl':cfg['environment']['kwargs']['seed_source']=str(TASKS/'rl'/item['task']/'environment/seeds')
 a=cfg['agents'][0];a.update(name=None,import_path='probe_agent:ProbeTerminus2',model_name='hosted_vllm/'+plan['served_model'],max_timeout_sec=P['agent_timeout_sec']);sampling=P['teacher_sampling'] if plan['teacher'] else P['student_sampling']
 a['kwargs'].update(api_base=os.environ['OTA_RESAMPLE_API_BASE'],temperature=sampling['temperature'],probe_seed=item['seed'],collect_rollout_details=True,enable_episode_logging=True,trajectory_dump_cadence='per_turn',store_all_messages=True,interleaved_thinking=P['interleaved_thinking'])
 a['kwargs']['llm_kwargs'].update(top_p=sampling['top_p'],top_k=sampling['top_k'])
 for key in ['presence_penalty','repetition_penalty']:
  if key in sampling:a['kwargs']['llm_kwargs'][key]=sampling[key]
 a['kwargs']['model_info'].update(max_input_tokens=P['context'],max_output_tokens=P['max_output_tokens']);return cfg

def wait_auth(run,seconds=300):
 from auth_guard import require_auth,AuthenticationUnavailable
 deadline=time.monotonic()+seconds;reported=False
 while True:
  try:require_auth();return
  except AuthenticationUnavailable:
   if not reported:
    with LOCK:write(run/'auth-pause.json',{'at':now(),'reason':'Waiting for existing credential publisher; no new refresh consumer','maximum_wait_seconds':seconds})
    reported=True
   if time.monotonic()>=deadline:raise
   time.sleep(2)

def cleanup_interrupted(item,run):
 from auth_guard import AuthenticationUnavailable,auth_error
 jobs=set()
 for path in (run/'jobs'/item['id']).rglob('native-events.jsonl'):
  for line in path.read_text().splitlines():
   try:event=json.loads(line)
   except ValueError:continue
   if event.get('event')=='start_requested':jobs.add(event['job'])
 for job in jobs:
  assert job.startswith('ota-native-');wait_auth(run)
  result=subprocess.run(['runai','delete','job','-p','cvlab-shuli',job],capture_output=True,text=True,timeout=90)
  if result.returncode and not any(x in (result.stdout+result.stderr).lower() for x in ['not found','could not find','does not exist']):
   if auth_error(result.stderr+result.stdout):raise AuthenticationUnavailable('Authentication rejected interrupted sandbox cleanup')
   raise RuntimeError('Interrupted sandbox cleanup failed: '+(result.stderr or result.stdout)[-500:])

def recoverable_trial(item,run,plan):
 for attempt in range(5):
  try:return trial(item,run,plan)
  except Exception as e:
   infrastructure=type(e).__name__=='AuthenticationUnavailable' or any(x in str(e) for x in ['AuthenticationUnavailable','APIConnectionError','LLMRequestTimeoutError','ServiceUnavailableError','NativeSandboxInterrupted',':CancelledError'])
   if not infrastructure:raise
   if attempt==4:
    write(run/'trials'/item['id']/'infra-retry-exhausted.json',{'at':now(),'error':str(e),'attempts':5,'policy':'Final terminal outcome lacking verifier reward scores0 per user; original attempts preserved'})
    return accept_terminal_result(item,run,allow_infrastructure_final=True)
   path=run/'trials'/item['id']/'infrastructure-retries.json';history=json.loads(path.read_text()) if path.exists() else {'events':[]};history['events'].append({'at':now(),'error':str(e),'policy':'Retry only this interrupted trial with same seed; preserve valid terminal results; do not cancel healthy peer trials.'});write(path,history);wait_auth(run);cleanup_interrupted(item,run);time.sleep(2)

def trial(item,run,plan):
 from auth_guard import require_auth
 wait_auth(run);folder=run/'trials'/item['id'];marker=folder/'complete.json'
 if marker.exists():return json.loads(marker.read_text())
 ready=TASKS/item['kind']/item['task']/'native-ready.json';assert ready.exists(),'Task native image not ready'
 directory=run/'jobs'/item['id']
 # Preserve an already-finished verifier result even if the parent died before
 # writing its derived complete marker. Never reroll such an outcome.
 if (directory/'config.json').exists():
  saved=json.loads((directory/'config.json').read_text());assert saved['agents'][0]['model_name']=='hosted_vllm/'+plan['served_model'];assert saved['agents'][0]['kwargs']['probe_seed']==item['seed']
  for result_file in directory.glob('*/result.json'):
   try:existing_result=json.loads(result_file.read_text())
   except (OSError,ValueError):continue
   reward=((existing_result.get('verifier_result') or {}).get('rewards') or {}).get('reward')
   if existing_result.get('finished_at') and valid_score(reward):return accept_terminal_result(item,run)
 if directory.exists() and not (directory/'config.json').exists():
  backup=run/'recovery'/('uninitialized-'+item['id']+'-'+str(time.time_ns()));backup.parent.mkdir(parents=True,exist_ok=True);directory.rename(backup)
 cfg=config(item,run,plan)
 if directory.exists():
  from native_recovery import archive_confirmed
  archive_confirmed(directory,run,item,cleanup_interrupted)
 existing=directory/'config.json'
 if existing.exists():
  saved=json.loads(existing.read_text());assert saved['agents'][0]['model_name']==cfg['agents'][0]['model_name']
  old=saved['agents'][0]['kwargs']['api_base'];new=cfg['agents'][0]['kwargs']['api_base']
  if old!=new:
   write(run/'recovery'/(item['id']+'-endpoint-'+str(time.time_ns())+'.json'),{'at':now(),'old_config':saved,'new_endpoint':new,'reason':'Same pinned model restarted on a different pod IP; preserve all terminal results and seeds'})
   saved['agents'][0]['kwargs']['api_base']=new;write(existing,saved)
 path=run/'configs'/(item['id']+'.json');write(path,cfg);log=run/'logs'/(item['id']+'.log');log.parent.mkdir(exist_ok=True)
 with log.open('a') as f:
  command=[sys.executable,str(NATIVE/'scripts/resume_native_job.py'),str(directory)] if (directory/'config.json').exists() else ['harbor','run','--config',str(path)]
  process=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  with LOCK:ACTIVE.add(process)
  try:
   while process.poll() is None:time.sleep(5);wait_auth(run)
  finally:
   if process.poll() is None:
    os.killpg(process.pid,signal.SIGINT)
    try:process.wait(timeout=45)
    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
   with LOCK:ACTIVE.discard(process)
 return accept_terminal_result(item,run)

def accept_terminal_result(item,run,allow_infrastructure_final=False):
 """Only parse a finished trial and export data; never submit, resume or generate."""
 folder=run/'trials'/item['id'];directory=run/'jobs'/item['id'];marker=folder/'complete.json'
 if marker.exists():return json.loads(marker.read_text())
 paths=list(directory.glob('*/result.json'));assert len(paths)==1,(item['id'],'missing terminal result')
 raw=json.loads(paths[0].read_text());assert raw.get('finished_at'),'Postprocessing requires a finished raw trial';error=raw.get('exception_info') or {};etype=error.get('exception_type');value=((raw.get('verifier_result') or {}).get('rewards') or {}).get('reward');valid=isinstance(value,(float,int)) and math.isfinite(value) and 0<=value<=1
 from native_recovery import confirmed,model_resource_limit
 resource_failure=model_resource_limit(paths[0].parent) if etype and etype not in MODEL_FAILURES else None
 assert value is None or valid,(item['id'],'Invalid nonmissing verifier reward',value)
 if not valid and not allow_infrastructure_final:
  if etype in {'AuthenticationUnavailable','APIConnectionError','LLMRequestTimeoutError','ServiceUnavailableError','CancelledError'}:raise RuntimeError(f"{item['id']} infrastructure/unclassified error:{etype}")
  if confirmed(paths[0].parent):raise RuntimeError(f"{item['id']} infrastructure/unclassified error:NativeSandboxInterrupted")
 score=float(value) if valid else 0.0
 agent=paths[0].parent/'agent'
 if not agent.exists():
  agents=sorted(paths[0].parent.glob('attempts/*/agent'));agent=agents[-1] if agents else paths[0].parent/'attempts/000/agent'
 index=agent/'probe-calls/index.jsonl';calls=[json.loads(x) for x in index.read_text().splitlines() if x.strip()] if index.exists() else []
 committed_agent_steps=sum(1 for path in agent.glob('trajectory*.json') for step in json.loads(path.read_text()).get('steps',[]) if step.get('source')=='agent' and not step.get('is_copied_context'))
 uncommitted_terminal_response=bool(error) and committed_agent_steps==0
 allow_empty=not calls or uncommitted_terminal_response
 if not list(agent.glob('trajectory*.json')) and allow_empty:rows=[];audit=[{'skip':'No recorded assistant generation; terminal raw outcome retained','exception_type':etype}]
 else:rows,audit=convert_agent(agent,item['task'],item.get('repeat',item['index']),allow_empty=allow_empty)
 folder.mkdir(parents=True,exist_ok=True)
 if not rows:write(folder/'empty-training-export.json',{'at':now(),'reason':'No accepted assistant turn is available for original SFT export; raw API attempts and terminal result are retained','exception_type':etype,'raw_api_responses':[str(p.relative_to(ROOT)) for p in (agent/'probe-calls').glob('call-*.json')],'gradient_contribution':'zero; retain official score in within-question mean and retain this rollout in denominator count; no fabricated tokens or replacement rollout'})
 from reasoning_audit import audit as audit_reasoning
 write(folder/'reasoning-preservation.json',audit_reasoning(agent,rows,allow_uncommitted_terminal_response=uncommitted_terminal_response))
 (folder/'train.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));write(folder/'conversion-audit.json',audit)
 assert (calls or (allow_empty and not rows)) and len({x['seed'] for x in calls})==len(calls)
 record={**item,'score':score,'full_pass':int(score==1),'raw_score':value,'reward_source':'verifier' if valid else 'missing_verifier_zero','imputed_zero':not valid,'score_policy':'missing_verifier_as_zero','resource_limit_classification':resource_failure,'exception_type':etype,'training_rows':len(rows),'calls':len(calls),'trial_result':str(paths[0].relative_to(ROOT)),'agent_dir':str(agent.relative_to(ROOT)),'finished_at':now()};write(marker,record);return record

def main(rid):
 run=ROOT/'runs'/rid;snapshot_sources(run,['rollouts.py','common.py','probe_agent.py','transcripts.py','reasoning_audit.py','seeded_env.py','loss_contract.py','native_recovery.py','verifier_scores.py']);plan=json.loads((run/'plan.json').read_text());os.environ['NATIVE_RUN_LABEL']='ota-c3-'+rid
 from auth_guard import require_auth
 wait_auth(run);state(run,'rollouts',started_at=now());results={};unresolved={};items=plan['items'];total=len(items)
 pending_file=run/'unresolved-trials.json'
 if pending_file.exists():write(run/'recovery'/('unresolved-'+str(time.time_ns())+'.json'),json.loads(pending_file.read_text()))
 write(pending_file,{'at':now(),'count':0,'records':[]})
 with urllib.request.urlopen(os.environ['OTA_RESAMPLE_API_BASE'].rstrip('/')+'/models',timeout=20) as f:models=json.load(f)
 assert any(x['id']==plan['served_model'] for x in models['data'])
 def refresh(row):
  results[row['id']]=row;write(run/'progress.json',{'stage':'native_rollouts','done':len(results),'total':total,'unresolved_trials':len(unresolved),**score_metrics(results.values()),'updated_at':now()});print('ROLLOUT',rid,len(results),total,flush=True)
 remaining=[]
 for item in items:
  marker=run/'trials'/item['id']/'complete.json'
  if marker.exists():
   cached=json.loads(marker.read_text());assert all(cached[k]==item[k] for k in ['id','task','seed']);assert 'repeat' not in item or cached['repeat']==item['repeat'];results[item['id']]=cached
  else:remaining.append(item)
 if results:refresh(next(iter(results.values())))
 if rid=='teacher-batch1' and items[0] in remaining:refresh(recoverable_trial(items[0],run,plan));remaining.remove(items[0])
 pool=concurrent.futures.ThreadPoolExecutor(max_workers=P['concurrent_sandboxes'])
 try:
  futures={pool.submit(recoverable_trial,x,run,plan):x for x in remaining}
  for future in concurrent.futures.as_completed(futures):
   try:row=future.result()
   except Exception as failure:
    item=futures[future];unresolved[item['id']]={'at':now(),'item':item,'error':str(failure),'exception_type':type(failure).__name__}
    write(pending_file,{'at':now(),'count':len(unresolved),'records':list(unresolved.values()),'policy':'Preserve unresolved result without assigning score; continue healthy independent trials. Stage cannot complete until all errors are resolved.'})
    state(run,'rollouts_with_unresolved_errors',message=f'{len(unresolved)} unresolved trial(s); healthy peers continue')
    print('UNRESOLVED_TRIAL',item['id'],repr(failure),flush=True);continue
   refresh(row)
 except BaseException as failure:
  state(run,'draining_after_failure','failed',error=str(failure),exception_type=type(failure).__name__,finished_at=now());print('PRIMARY_WORKER_FAILURE',repr(failure),flush=True)
  with LOCK:
   for process in list(ACTIVE):
    if process.poll() is None:
     try:os.killpg(process.pid,signal.SIGINT)
     except ProcessLookupError:pass
  raise
 finally:pool.shutdown(wait=True,cancel_futures=True)
 if unresolved:raise RuntimeError('UnresolvedRolloutErrors: '+','.join(sorted(unresolved)))
 assert len(results)==total
 finalize_rollouts(run,items,results)

def finalize_rollouts(run,items,results):
 """Merge already accepted outcomes only, without any agent/cluster calls."""
 total=len(items);assert total and set(results)=={x['id'] for x in items}
 data=run/'data';data.mkdir(exist_ok=True)
 with (data/'train.jsonl').open('w') as f:
  for x in items:f.write((run/'trials'/x['id']/'train.jsonl').read_text())
 records=[results[x['id']] for x in items];write(run/'result.json',{'finished_at':now(),'records':records,'metrics':{**score_metrics(records),'sft_rows':sum(x['training_rows'] for x in records),'empty_sft_exports':sum(x['training_rows']==0 for x in records),'resource_budget_failures':sum(x.get('resource_limit_classification') is not None for x in records)}});m=json.loads((run/'run.json').read_text());m['sources']['training_data']=str((data/'train.jsonl').relative_to(ROOT));write(run/'run.json',m);write(run/'complete.json',{'finished_at':now()});state(run,'complete','succeeded',finished_at=now())
if __name__=='__main__':
 try:main(sys.argv[1])
 except BaseException as e:state(ROOT/'runs'/sys.argv[1],'failed','failed',error=str(e),exception_type=type(e).__name__,finished_at=now());raise
