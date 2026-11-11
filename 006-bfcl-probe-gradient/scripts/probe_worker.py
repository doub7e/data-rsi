import concurrent.futures,importlib.util,json,os,random,shutil,signal,socket,subprocess,sys,time,urllib.request
from pathlib import Path
from common import *
RUN=ROOT/'runs/probe-rollouts'
CATS=CFG['probe_counts']

def prepare():
 import pyarrow.parquet as pq
 source=Path('/opt/bfcl-data');dest=RUN/'data'
 if not (RUN/'selection.json').exists():
  rng=random.Random(CFG['probe_seed']);ids={c:sorted(rng.sample([x['id'] for x in rows(source/f'BFCL_v4_multi_turn_{c}.json')],n)) for c,n in CATS.items()}
  write(RUN/'selection.json',{'seed':CFG['probe_seed'],'ids':ids,'total':32,'heldout':768,'frozen_at':now()})
 ids=json.loads((RUN/'selection.json').read_text())['ids']
 dest.mkdir(exist_ok=True)
 for c in CATS:
  for sub in ('','possible_answer/'):
   path=dest/sub/f'BFCL_v4_multi_turn_{c}.json';path.parent.mkdir(parents=True,exist_ok=True)
   selected=[r for r in rows(source/sub/f'BFCL_v4_multi_turn_{c}.json') if r['id'] in ids[c]];assert len(selected)==CATS[c]
   path.write_text(''.join(json.dumps(r)+'\n' for r in selected))
 shutil.copytree(source/'multi_turn_func_doc',dest/'multi_turn_func_doc',dirs_exist_ok=True)
 table=pq.read_table(BASE/'cache/pool/pool.parquet').to_pylist()
 orig=json.loads((BASE/'runs/random100-s42-20261102/data/audit.json').read_text())['indices']
 remaining=[i for i in range(len(table)) if i not in set(orig)]
 extended=sorted(orig+random.Random(CFG['candidate_extension_seed']).sample(remaining,100))
 assert len(set(extended))==200 and set(orig)<=set(extended)
 out=ROOT/'data';out.mkdir(exist_ok=True)
 payload=''.join(json.dumps(dict(table[i],pool_index=i,original100=i in set(orig)),ensure_ascii=False)+'\n' for i in extended)
 target=out/'candidates.jsonl'
 if target.exists():assert target.read_text()==payload
 else:target.write_text(payload)
 write(out/'candidate-manifest.json',{'seed_extension':43,'original100':orig,'indices':extended,'meta_indices':[table[i]['meta_index'] for i in extended],'validation_overlap':0,'sha256':sha(target),'frozen_at':now()})
 return ids

def summarize(repeats,ids):
 tasks={t:[] for group in ids.values() for t in group};records=[]
 for rep in range(1,repeats+1):
  out=RUN/f'bfcl/p{rep:02d}'
  for cat in CATS:
   scores=rows(out/f'score/Qwen_Qwen3-4B-FC/multi_turn/BFCL_v4_multi_turn_{cat}_score.json')
   assert scores[0]['total_count']==CATS[cat]
   bad={r['id'] for r in scores[1:] if 'id' in r};assert len(bad)==CATS[cat]-scores[0]['correct_count']
   generated=rows(out/f'Qwen_Qwen3-4B-FC/multi_turn/BFCL_v4_multi_turn_{cat}_result.json')
   assert len(generated)==CATS[cat] and {r['id'] for r in generated}==set(ids[cat])
   for row in generated:
    if row.get('traceback'):raise RuntimeError(f'Infrastructure/inference error in probe {rep}/{row["id"]}: '+row['traceback'][-500:])
    steps=[]
    for turn in row.get('inference_log',[]):
     if not isinstance(turn,dict):continue
     for k,v in turn.items():
      if k.startswith('step_'):
       capture=[x['content'] for x in v if x.get('role')=='inference_input']
       assert len(capture)==1
       step=capture[0];assert step['prompt_token_ids'] and step['completion_token_ids'];steps.append(step)
    assert steps
    r={'id':row['id'],'repeat':rep,'reward':int(row['id'] not in bad),'steps':steps}
    records.append(r);tasks[row['id']].append(r)
 mixed=[]
 for task,rs in tasks.items():
  assert len(rs)==repeats
  avg=sum(x['reward'] for x in rs)/repeats
  if 0<avg<1:mixed.append(task)
  for r in rs:r['advantage']=r['reward']-avg
 dest=RUN/'rollouts.jsonl';dest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
 report={'questions':32,'repeats':repeats,'rollouts':len(records),'mixed_questions':len(mixed),'mixed_ids':mixed,
  'successes':sum(r['reward'] for r in records),'task_success_rate':{k:sum(r['reward'] for r in v)/repeats for k,v in tasks.items()},
  'gradient_eligible_rollouts':sum(r['advantage']!=0 for r in records),'maximum_context_tokens':max(len(s['prompt_token_ids'])+len(s['completion_token_ids']) for r in records for s in r['steps']),
  'captured_steps':sum(len(r['steps']) for r in records),'sha256':sha(dest)}
 write(RUN/'result.json',report);return report

def main():
 state(RUN,'prepare');ids=prepare()
 spec=importlib.util.spec_from_file_location('up_eval','/opt/bfcl/eval.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 model=module.evaluation_model(MODEL,RUN/'model-64k');port=25031
 env={**os.environ,'MODEL':'Qwen/Qwen3-4B-FC','MODEL_PATH':model,'BFCL_DATA_DIR':str(RUN/'data'),
      'PATH':str(BASE/'scripts/bin')+os.pathsep+os.environ['PATH'],
      'BFCL_PROJECT_ROOT':str(RUN/'harness'),'BFCL_DROP_HISTORY_THINK':'1','LOCAL_SERVER_PORT':str(port),
      'NO_PROXY':'localhost,127.0.0.1','no_proxy':'localhost,127.0.0.1','HF_HOME':str(ROOT/'cache/huggingface')}
 log=RUN/'server.log'
 with log.open('a') as f:
  server=subprocess.Popen(['vllm','serve',model,'--port',str(port),'--dtype','bfloat16','--max-model-len','65536','--tensor-parallel-size','1','--gpu-memory-utilization','0.85','--trust-remote-code'],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  try:
   ready=False;opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
   for _ in range(600):
    if server.poll() is not None:raise RuntimeError('Probe vLLM startup failed')
    try:
     with opener.open(f'http://127.0.0.1:{port}/v1/models',timeout=3) as response:ready=any(d['id']==model for d in json.load(response)['data'])
    except Exception:pass
    if ready:break
    time.sleep(2)
   if not ready:raise TimeoutError('Probe vLLM startup timeout')
   state(RUN,'rollout',started_at=now())
   def repeat(rep):
    out=RUN/f'bfcl/p{rep:02d}';out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return
    local={**env,'PROBE_SEED':str(2026110300+rep),'BFCL_PROJECT_ROOT':str(RUN/f'harness/p{rep:02d}')}
    cats=','.join('multi_turn_'+c for c in CATS)
    call([sys.executable,ROOT/'scripts/probe_cli.py','generate','--model','Qwen/Qwen3-4B-FC','--test-category',cats,'--temperature','1.0','--backend','vllm','--num-gpus','1','--num-threads','8','--local-model-path',model,'--result-dir',out,'--skip-server-setup','--include-input-log'],RUN/f'logs/p{rep:02d}.log',env=local)
    call(['bfcl','evaluate','--model','Qwen/Qwen3-4B-FC','--test-category',cats,'--result-dir',out,'--score-dir',out/'score'],RUN/f'logs/p{rep:02d}.log',env=local)
    write(out/'complete.json',{'finished_at':now()});print('Probe repeat complete:',rep,flush=True)
   repeat(1)
   with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:list(pool.map(repeat,range(2,9)))
   report=summarize(8,ids)
   if report['mixed_questions']<4:
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(repeat,range(9,17)))
    report=summarize(16,ids)
   if report['mixed_questions']==0:raise RuntimeError('All probe advantages zero; no data-ranking signal')
   write(RUN/'complete.json',{'finished_at':now(),**report});state(RUN,'complete','succeeded')
  finally:
   if server.poll() is None:os.killpg(server.pid,signal.SIGTERM)
   try:server.wait(timeout=30)
   except subprocess.TimeoutExpired:os.killpg(server.pid,signal.SIGKILL);server.wait()
if __name__=='__main__':
 try:main()
 except BaseException as e:state(RUN,'probe','failed',error=str(e));raise
