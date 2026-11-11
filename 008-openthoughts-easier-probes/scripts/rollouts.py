"""32 seeded native sandbox rollouts for one OTA probe task, on allocatedCPU."""
import concurrent.futures,os,shutil,signal,subprocess,sys,time,urllib.request,math,threading
from common import *
from transcripts import convert_agent
from score_rewards import score_of
P=json.loads((ROOT/'runs/easy-probes-20261105/protocol.json').read_text())
ACTIVE=set();ACTIVE_LOCK=threading.Lock()
MODEL_FAILURES={'AgentTimeoutError','AgentEnvironmentTimeoutError','ContextLengthExceededError','OutputLengthExceededError','SummarizationTimeoutError'}

def binary_success(value):
 # Validate official scores; returned binary flag is used only for full-pass display.
 if isinstance(value,(int,float)) and math.isfinite(value) and 0 <= value <= 1:
  return int(value == 1)
 return None

def job_config(q,run,rep,endpoint):
 cfg=json.loads((NATIVE/'production/harbor-native-template.json').read_text());cfg.update(job_name=f'r{rep:03d}',jobs_dir=str(run/'jobs'),n_attempts=1,n_concurrent_trials=1,timeout_multiplier=P['timeout_multiplier'],datasets=[],tasks=[{'path':str(ROOT/q['path'])}]);cfg['retry']['max_retries']=0;cfg['verifier']['max_timeout_sec']=P['verifier_timeout_sec']
 agent=cfg['agents'][0];agent.update(name=None,import_path='probe_agent:ProbeTerminus2',model_name='hosted_vllm/'+P['model_name'],max_timeout_sec=P['agent_timeout_sec'])
 agent['kwargs'].update(api_base=endpoint,temperature=P['temperature'],probe_seed=q['seed_offset']+rep*100000,collect_rollout_details=True,enable_episode_logging=True,trajectory_dump_cadence='per_turn',store_all_messages=True)
 agent['kwargs']['model_info'].update(max_input_tokens=P['context'],max_output_tokens=P['max_output_tokens'])
 return cfg

def run_trial(q,run,rep):
 from auth_guard import require_auth
 require_auth();name=f'r{rep:03d}';directory=run/'jobs'/name;marker=run/'trials'/name/'complete.json'
 if marker.exists():return json.loads(marker.read_text())
 if directory.exists() and not (directory/'config.json').exists():
  backup=run/'recovery'/now().replace(':','')/name;backup.parent.mkdir(parents=True,exist_ok=True);directory.rename(backup)
 cfg=job_config(q,run,rep,os.environ['OTA_PROBE_API_BASE'])
 path=run/'configs'/f'{name}.json';write(path,cfg);log=run/'logs'/f'{name}.log';log.parent.mkdir(exist_ok=True)
 with log.open('a') as f:
  command=[sys.executable,str(NATIVE/'scripts/resume_native_job.py'),str(directory)] if (directory/'config.json').exists() else ['harbor','run','--config',str(path)]
  process=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  with ACTIVE_LOCK:ACTIVE.add(process)
  try:
   while process.poll() is None:time.sleep(5);require_auth()
  finally:
   if process.poll() is None:
    os.killpg(process.pid,signal.SIGINT)
    try:process.wait(timeout=45)
    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
   with ACTIVE_LOCK:ACTIVE.discard(process)
 results=list(directory.glob('*/result.json'));assert len(results)==1,(rep,'missing/ambiguous actual trial result',results)
 result=json.loads(results[0].read_text());exception=result.get('exception_info') or {};etype=exception.get('exception_type');value=((result.get('verifier_result') or {}).get('rewards') or {}).get('reward')
 trial=results[0].parent;agent_dir=trial/'agent'
 if not agent_dir.exists():
  attempts=sorted(trial.glob('attempts/*/agent'));assert attempts;agent_dir=attempts[-1]
 from reward_audit import deleted_sympy_source
 cause=deleted_sympy_source(result,agent_dir)
 if etype and etype not in MODEL_FAILURES and cause is None:raise RuntimeError(f'Trial{rep} infrastructure/unclassified error:{etype}; not relabeled as model failure')
 reward=float(value) if binary_success(value) is not None else None
 if reward is None and (etype in MODEL_FAILURES or cause):reward=0
 assert reward is not None and 0<=reward<=1,(rep,'no valid score',value,etype)
 if cause:write(run/f'trials/r{rep:03d}/model-failure-evidence.json',cause)
 rows,audit=convert_agent(agent_dir,q['task'],rep)
 target=run/'trials'/name/'train.jsonl';target.parent.mkdir(parents=True,exist_ok=True);target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));write(target.with_name('conversion-audit.json'),audit)
 calls=[]
 index=agent_dir/'probe-calls/index.jsonl'
 if index.exists():calls=[json.loads(l) for l in index.read_text().splitlines() if l.strip()]
 assert calls and len({c['seed'] for c in calls})==len(calls),'Missing unique seeded call records'
 record={'repeat':rep,'reward':reward,'exception_type':etype,'raw_reward':value,'reward_definition':'official_score_v1','full_pass':int(reward==1),'reward_source':'verifier' if binary_success(value) is not None else 'model_caused_verifier_failure' if cause else 'model_budget_timeout','finished_at':now(),'training_rows':len(rows),'trial_result':str(results[0].relative_to(ROOT)),'agent_dir':str(agent_dir.relative_to(ROOT)),'calls':len(calls),'seed_offset':q['seed_offset']+rep*100000,'copied_assistant_context_steps':sum(a.get('copied_assistant_steps',0) for a in audit)}
 write(marker,record);return record

def main(key):
 from auth_guard import require_auth
 q=next(q for q in P['tasks'] if q['key']==key);run=ROOT/f'runs/{key}-rollouts-20261105';n=q.get('rollouts',32);os.environ['NATIVE_RUN_LABEL']='ota-easy-'+run.name;require_auth()
 assert (ROOT/P['loss']['requires_loss_parity']).exists()
 ready=json.loads((ROOT/'runs/easy-probes-20261105/rollout-implementation-ready.json').read_text())
 for path,digest in ready['source_sha256'].items():assert sha(ROOT/path)==digest,('source_changed',path)
 state(run,'rollouts',started_at=now())
 with urllib.request.urlopen(os.environ['OTA_PROBE_API_BASE'].rstrip('/')+'/models',timeout=20) as f:models=json.load(f)
 assert any(m['id']==P['model_name'] for m in models['data'])
 results={}
 def refresh(row):
  row=dict(row);row['reward']=score_of(row);row['full_pass']=int(row['reward']==1);row['reward_definition']='official_score_v1'
  results[row['repeat']]=row;write(run/'progress.json',{'stage':'native_rollouts','done':len(results),'total':n,'successes':sum(x['reward']==1 for x in results.values()),'updated_at':now()});print('ROLLOUT',key,len(results),sum(x['reward']==1 for x in results.values()),flush=True)
 refresh(run_trial(q,run,1))
 pool=concurrent.futures.ThreadPoolExecutor(max_workers=P['n_concurrent_trials'])
 try:
  futures=[pool.submit(run_trial,q,run,i) for i in range(2,n+1)]
  for future in concurrent.futures.as_completed(futures):refresh(future.result())
 except BaseException:
  with ACTIVE_LOCK:
   for process in list(ACTIVE):
    if process.poll() is None:
     try:os.killpg(process.pid,signal.SIGINT)
     except ProcessLookupError:pass
  raise
 finally:pool.shutdown(wait=True,cancel_futures=True)
 assert len(results)==n
 data=run/'data';data.mkdir(exist_ok=True)
 with (data/'train.jsonl').open('w') as f:
  for rep in range(1,n+1):f.write((run/f'trials/r{rep:03d}/train.jsonl').read_text())
 success=sum(x['reward']==1 for x in results.values());record={'finished_at':now(),'task':q['task'],'benchmark':q['benchmark'],'successes':success,'trials':n,'probe_reward_definition':'official_score_v1','scores':[results[i]['reward'] for i in range(1,n+1)],'mean_score':sum(x['reward'] for x in results.values())/n,'outcomes':[results[i]['reward'] for i in range(1,n+1)],'records':[results[i] for i in range(1,n+1)],'metrics':{'success_percent':100*success/n,'evaluated_trials':n,'mean_score':sum(x['reward'] for x in results.values())/n,'partial_credit_trials':sum(0<x['reward']<1 for x in results.values())}};write(run/'result.json',record);write(run/'complete.json',{'finished_at':now()});state(run,'complete','succeeded')
 m=json.loads((run/'run.json').read_text());m['sources']['training_data']=str((data/'train.jsonl').relative_to(ROOT));write(run/'run.json',m)
if __name__=='__main__':
 try:main(sys.argv[1])
 except BaseException as e:state(ROOT/f'runs/{sys.argv[1]}-rollouts-20261105','failed','failed',error=str(e),exception_type=type(e).__name__,finished_at=now());raise
