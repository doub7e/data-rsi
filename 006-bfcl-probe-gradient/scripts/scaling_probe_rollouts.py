"""Generate128 fresh trajectories for the frozen single-probe scaling study."""
import concurrent.futures,importlib.util,json,os,re,shutil,signal,subprocess,sys,time,urllib.request
from common import *
RUN=ROOT/'runs/probe194-scaling-20261103';PROTOCOL=json.loads((RUN/'protocol.json').read_text());QID=PROTOCOL['probe_id']
sys.path.insert(0,str(BASE/'scripts'))
from bfcl_registry import register
from history_protocol import install,extract_formatter,assert_clean_prompt,BLOCK,THINK

def preflight():
 register();record=install();assert record['version']=='strict-v2'
 from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler
 source=BASE/'docker/eval/bfcl/patches/bfcl_eval/model_handler/local_inference/qwen_fc.py';old=extract_formatter(source.read_text(),'legacy-v1');new=QwenFCHandler._format_prompt;count=0
 with (ROOT/'runs/probe-rollouts/rollouts.jsonl').open() as f:
  for line in f:
   row=json.loads(line)
   for step in row['steps']:
    parsed=BLOCK.findall(step['formatted_prompt'])
    if not any(role=='assistant' and '<think>' in THINK.sub('',body) for role,body in parsed):continue
    messages=[{'role':role,'content':body} for role,body in parsed]
    if messages[-1]['role']=='assistant' and not messages[-1]['content']:messages.pop()
    assert old(None,messages,[])==step['formatted_prompt'];clean=new(None,messages,[]);assert_clean_prompt(clean)
    assert [(a,b) for a,b in BLOCK.findall(clean) if a!='assistant']==[(a,b) for a,b in parsed if a!='assistant'];count+=1
 assert count==148;record['affected_prompt_replays_passed']=count;write(RUN/'harness-preflight.json',record)
 source=Path('/opt/bfcl-data');dest=RUN/'data'
 for sub in ('','possible_answer/'):
  selected=[r for r in rows(source/sub/'BFCL_v4_multi_turn_miss_func.json') if r['id']==QID];assert len(selected)==1
  target=dest/sub/'BFCL_v4_multi_turn_miss_func.json';target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(selected[0])+'\n')
 shutil.copytree(source/'multi_turn_func_doc',dest/'multi_turn_func_doc',dirs_exist_ok=True)

def extract(rep,out):
 generated=rows(out/'Qwen_Qwen3-4B-FC/multi_turn/BFCL_v4_multi_turn_miss_func_result.json');assert len(generated)==1 and generated[0]['id']==QID;raw=generated[0]
 if raw.get('traceback'):raise RuntimeError('Recorded rollout exception; retain for inspection: '+raw['traceback'][-1200:])
 scores=rows(out/'score/Qwen_Qwen3-4B-FC/multi_turn/BFCL_v4_multi_turn_miss_func_score.json');assert scores[0]['total_count']==1
 reward=int(scores[0]['correct_count']);assert reward in (0,1)
 steps=[]
 for turn in raw['inference_log']:
  if not isinstance(turn,dict):continue
  for k,v in turn.items():
   if k.startswith('step_'):
    entries=[x['content'] for x in v if x.get('role')=='inference_input'];assert len(entries)==1;s=entries[0];assert_clean_prompt(s['formatted_prompt']);assert s['prompt_token_replay_verified'];steps.append(s)
 assert steps and len({s['seed'] for s in steps})==len(steps)
 return {'id':QID,'repeat':rep,'reward':reward,'steps':steps,'history_protocol':'strict-v2'}

def main():
 os.environ.update(BFCL_PROJECT_ROOT=str(RUN/'preflight-runtime'),BFCL_DATA_DIR='/opt/bfcl-data',BFCL_HISTORY_PROTOCOL='strict-v2',BFCL_DROP_HISTORY_THINK='1',NO_PROXY='localhost,127.0.0.1',no_proxy='localhost,127.0.0.1')
 state(RUN,'harness_preflight',started_at=now());preflight()
 model=ROOT/'runs/probe-rollouts/model-64k';port=25194
 env={**os.environ,'MODEL':'Qwen/Qwen3-4B-FC','MODEL_PATH':str(model),'BFCL_DATA_DIR':str(RUN/'data'),'PATH':str(BASE/'scripts/bin')+os.pathsep+os.environ['PATH'],'LOCAL_SERVER_PORT':str(port),'HF_HOME':str(ROOT/'cache/huggingface')}
 (RUN/'logs').mkdir(exist_ok=True)
 with (RUN/'server.log').open('a') as log:
  server=subprocess.Popen(['vllm','serve',str(model),'--port',str(port),'--dtype','bfloat16','--max-model-len','65536','--tensor-parallel-size','1','--gpu-memory-utilization','0.85','--trust-remote-code','--max-num-seqs','32'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  clients=[]
  try:
   ready=False;opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
   for _ in range(600):
    if server.poll() is not None:raise RuntimeError('vLLM startup failed')
    try:
     with opener.open(f'http://127.0.0.1:{port}/v1/models',timeout=3) as response:ready=any(d['id']==str(model) for d in json.load(response)['data'])
    except Exception:pass
    if ready:break
    time.sleep(2)
   if not ready:raise TimeoutError('vLLM startup timed out')
   state(RUN,'generating128');order=[v for pair in zip(PROTOCOL['halves']['a'],PROTOCOL['halves']['b']) for v in pair]
   def rollout(rep):
    out=RUN/f'bfcl/r{rep:03d}';marker=out/'complete.json'
    if marker.exists():return rep
    out.mkdir(parents=True,exist_ok=True);local={**env,'PROBE_REPEAT':str(rep),'BFCL_PROJECT_ROOT':str(RUN/f'harness/r{rep:03d}')}
    commands=[[sys.executable,ROOT/'scripts/scaling_probe_cli.py','generate','--model','Qwen/Qwen3-4B-FC','--test-category','multi_turn_miss_func','--temperature','1.0','--backend','vllm','--num-gpus','1','--num-threads','1','--local-model-path',str(model),'--result-dir',out,'--skip-server-setup','--include-input-log'],['bfcl','evaluate','--model','Qwen/Qwen3-4B-FC','--test-category','multi_turn_miss_func','--result-dir',out,'--score-dir',out/'score']]
    with (RUN/f'logs/r{rep:03d}.log').open('a') as f:
     for cmd in commands:
      p=subprocess.Popen(list(map(str,cmd)),env=local,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);clients.append(p)
      if p.wait()!=0:raise RuntimeError(f'Rollout{rep} command failed; see logs/r{rep:03d}.log')
    trace=extract(rep,out);write(RUN/f'traces/r{rep:03d}.json',trace);write(marker,{'finished_at':now(),'reward':trace['reward'],'calls':len(trace['steps'])});return rep
   rollout(order[0]);print('FIRST_ROLLOUT_VERIFIED',order[0],flush=True)
   with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
    futures=[pool.submit(rollout,rep) for rep in order[1:]]
    for f in concurrent.futures.as_completed(futures):
     rep=f.result();done=len(list((RUN/'traces').glob('r*.json')));write(RUN/'rollout-progress.json',{'done':done,'total':128,'last_repeat':rep,'updated_at':now()});print('ROLLOUT_COMPLETE',done,rep,flush=True)
   traces=[json.loads((RUN/f'traces/r{i:03d}.json').read_text()) for i in range(1,129)]
   seeds=[s['seed'] for t in traces for s in t['steps']];assert len(set(seeds))==len(seeds)
   summary={'finished_at':now(),'rollouts':128,'successes':sum(t['reward'] for t in traces),'captured_calls':len(seeds),'all_history_prompts_clean':True,'all_prompt_token_ids_verified':True,'all_call_seeds_unique':True,'half_prefixes':{h:[{'n':n,'successes':sum(traces[i-1]['reward'] for i in indices[:n])} for n in PROTOCOL['prefixes_per_half']] for h,indices in PROTOCOL['halves'].items()},'length_capped_calls':sum(s['finish_reason']=='length' for t in traces for s in t['steps']),'total_output_tokens':sum(len(s['completion_token_ids']) for t in traces for s in t['steps'])}
   write(RUN/'rollout-summary.json',summary);write(RUN/'rollouts-complete.json',{'finished_at':now()});state(RUN,'rollouts_complete_waiting_gradients');print(json.dumps(summary),flush=True)
  finally:
   for p in clients+[server]:
    if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
   for p in clients+[server]:
    try:p.wait(timeout=20)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
if __name__=='__main__':
 try:main()
 except BaseException as e:state(RUN,'rollout_failed','failed',error=str(e));raise
