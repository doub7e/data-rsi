"""Storage gate, exact historical prompt recovery, and report for priority FP3220.

Never generates substitute screening outcomes. Recovery requires the original
ordered manifest or its complete source curriculum manifest and frozen shuffle.
"""
import argparse,hashlib,json,os,random,shutil,time
from pathlib import Path
from fd_reliability_tracking import atomic_json,update_progress

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()

def ordered_records(manifest):
 records=manifest['prompts']
 if len(records)!=1600:raise ValueError('original1600 manifest required; no substitute small pool')
 if len({r['prompt_hash'] for r in records})!=1600:raise ValueError('duplicate prompt hashes')
 if manifest.get('seed',manifest.get('config',{}).get('selection_seed'))!=27115:raise ValueError('historical seed mismatch')
 if all('random_rank'in r for r in records):
  if [r['random_rank'] for r in records]!=list(range(1600)):raise ValueError('ordered random ranks changed')
  if manifest.get('difficulty_sorting') is not False or manifest.get('teacher_screening') is not True:raise ValueError('random-screened manifest contract mismatch')
  return records,'original ordered random1600 manifest'
 if all('curriculum_rank'in r for r in records):
  if [r['curriculum_rank'] for r in records]!=list(range(1600)):raise ValueError('source curriculum ranks changed')
  order=list(range(1600));random.Random(27115).shuffle(order)
  return [records[i] for i in order],'original source curriculum1600 + exact historical Python seed27115 shuffle'
 raise ValueError('manifest lacks original random/curriculum ranks')

def storage(out):
 start=time.monotonic();probe=out/'owned-storage-write-probe.bin';size=256*(1<<20)
 # The probe is newly created by this gate and removed afterwards. No other file cleanup.
 try:
  with probe.open('xb') as f:
   block=b'\xa5'*(1<<20)
   for _ in range(size//len(block)):f.write(block)
   f.flush();os.fsync(f.fileno())
  if probe.stat().st_size!=size:raise ValueError('incomplete storage write')
 finally:
  if probe.exists():probe.unlink()
 import importlib,importlib.metadata
 for name in ['verl.trainer.main_ppo','verl.workers.actor.dp_actor','verl.workers.fsdp_workers']:importlib.import_module(name)
 from fd_training_reward import compute_score,evaluate_answer,grader_metadata
 assert compute_score('DeepMath-103K',r'\boxed{2}','2')==1
 assert evaluate_answer(r'\boxed{2}','2')==1 and evaluate_answer(r'\boxed{3}','2')==0
 fs=shutil.disk_usage(out)
 if fs.free<120*(1<<30):raise ValueError('filesystem available space below120GiB final-checkpoint budget')
 atomic_json(out/'result.json',{'metrics':{'written_and_fsynced_bytes':size,'filesystem_free_gib':fs.free/(1<<30),'runtime_imports_passed':3,'grader_contract_passed':1},'uid':os.getuid(),'gid':os.getgid(),'scope':'worker-mounted-path write test; filesystem free is not a user quota guarantee','grader':grader_metadata(),'elapsed_seconds':time.monotonic()-start})
 update_progress(out.parent.parent,out.name,done=1,total=1,stage='mounted storage actual write/fsync and trainer/scorer imports passed')

def restore(a):
 import pyarrow as pa
 import pyarrow.compute as pc
 import pyarrow.parquet as pq
 from transformers import AutoTokenizer
 from fd_reliability_banks import render_prompt_ids
 out=a.output;manifest=json.loads(a.manifest.read_text());records,method=ordered_records(manifest);selected=records[:320]
 # Read scalar ability first; only materialize selected full rows from the cached source.
 table=pq.read_table(a.dataset)
 physical=[i for i,x in enumerate(table.column('ability').to_pylist()) if x=='math']
 indices=[physical[int(r['domain_index'])] for r in selected]
 rows=[table.slice(i,1).to_pylist()[0] for i in indices]
 tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
 metadata=[]
 for rank,(r,row) in enumerate(zip(selected,rows,strict=True)):
  ids=render_prompt_ids(tok,row['prompt']);actual=hashlib.sha256(str(ids).encode('ascii')).hexdigest()
  if actual!=r['prompt_hash']:raise ValueError(f'original prompt hash mismatch at training rank{rank}; dataset/tokenizer may differ')
  if len(ids)!=int(r['prompt_tokens']):raise ValueError('original prompt token count differs')
  if row['data_source']!=r['data_source']:raise ValueError('original source differs')
  count=int(r['correct_count']);draws=int(r['num_responses'])
  if draws not in [8,16] or not .125<=count/draws<=.875 or count/draws!=r['accuracy']:raise ValueError('original screening contract differs')
  info=dict(row.get('extra_info') or {});source_rank=int(r.get('source_curriculum_rank',r.get('curriculum_rank',-1)))
  info.update(screen_teacher_accuracy=r['accuracy'],screen_teacher_correct_count=count,screen_teacher_num_responses=draws,screen_candidate_position=r['candidate_position'],curriculum_rank=source_rank,curriculum_step=source_rank//32+1)
  row['extra_info']=info
  metadata.append({'training_rank':rank,'training_step':rank//16+1,'position_in_step':rank%16,'source_physical_row':indices[rank],'original_record':r,'verified_prompt_hash':actual,'messages':row['prompt'],'ground_truth':row['reward_model']['ground_truth']})
 # Full data roundtrip must preserve actual prompts/labels. Byte hash of rebuilt
 # parquet is new and is explicitly not presented as original parquet equality.
 pq.write_table(pa.Table.from_pylist(rows),out/'train.parquet')
 reloaded=pq.read_table(out/'train.parquet').to_pylist()
 assert len(reloaded)==320
 for left,right in zip(rows,reloaded,strict=True):
  assert left['prompt']==right['prompt'] and left['reward_model']==right['reward_model']
 with (out/'train.jsonl').open('w') as f:
  for x in metadata:f.write(json.dumps(x,ensure_ascii=False)+'\n')
 (out/'benchmarks').mkdir()
 hashes={}
 for name in ['aime24','aime25','hmmt25_feb','hmmt25_nov']:
  src=a.benchmarks/name/'test.jsonl';dest=out/'benchmarks'/name/'test.jsonl';dest.parent.mkdir();shutil.copy2(src,dest)
  assert len([x for x in dest.read_text().splitlines() if x.strip()])==30
  hashes[name]=digest(dest)
 atomic_json(out/'result.json',{'metrics':{'training_prompts':320,'original_prompt_hashes_verified':320,'distinct_training_prompts':len({r['prompt_hash'] for r in selected}),'training_steps':20,'benchmark_questions':120},'recovery_method':method,'original_manifest_sha256':digest(a.manifest),'original_parquet_sha256_from_manifest':manifest.get('sha256'),'rebuilt_parquet_sha256':digest(out/'train.parquet'),'claim':'same original first320 prompt identities/order verified by historical token hashes; rebuilt parquet serialization not claimed byte-identical','original_records':selected,'benchmark_sha256':hashes})
 update_progress(out.parent.parent,out.name,done=320,total=320,stage='original320 identities/order and cached source prompts verified; no teacher screening')

def report(root,out):
 protocol=json.loads((root/'protocol.json').read_text());selection_run=protocol.get('selection_run','select320');prepare_run=protocol.get('prepare_run','prepare-data');train=json.loads((root/'runs/train-fp32-richardson20/result.json').read_text());eval20=json.loads((root/'runs/eval20/result.json').read_text());base=json.loads((root/'runs/eval-base/result.json').read_text());data=json.loads((root/'runs'/selection_run/'result.json').read_text());benchmarks=json.loads((root/'runs'/prepare_run/'selection.json').read_text())['benchmarks']
 assert train['metrics']['optimizer_steps']==20 and train['metrics']['actual_training_trajectories']==20480
 assert eval20['benchmark_sha256']==base['benchmark_sha256']=={k:v['sha256'] for k,v in benchmarks.items()}
 for d in [eval20,base]:assert d['metrics']['evaluated_trials']==3840
 metrics={'training_steps':20,'actual_training_trajectories':20480,'math_avg_percent':eval20['metrics']['math_avg_percent'],'base_math_avg_percent':base['metrics']['math_avg_percent'],'delta_vs_base_pp':eval20['metrics']['math_avg_percent']-base['metrics']['math_avg_percent']}
 atomic_json(out/'result.json',{'metrics':metrics,'protocol':'protocol.json','limitations':['one training seed','historical47.27 uses another question pool/runtime/evaluation realization; no matched BF16 retraining yet'],'per_model':{'base':base['metrics'],'fp32_richardson20':eval20['metrics']}})
 lines=['**最高优先级FP32 Richardson20-step结果**','','原1600题清单不可访问；使用历史及已完成筛选记录构造的新320题固定池，非原题集复现；sigmoid(3s)，fallback0，score/loss clip1，N64，h1e-5，8H200。','','| 模型 | AIME24 | AIME25 | HMMT Feb | HMMT Nov | Math Avg |','|---|---:|---:|---:|---:|---:|']
 for name,d in [('Base',base),('FP32 Richardson20',eval20)]:lines.append('| '+name+' | '+' | '.join(f"{d['metrics'][b+'_accuracy_percent']:.2f}" for b in ['aime24','aime25','hmmt25_feb','hmmt25_nov'])+f" | {d['metrics']['math_avg_percent']:.2f} |")
 lines+=['','单训练seed；与历史47.27的差异不能只归因于FD精度，尚缺同运行环境的BF16对照。原始训练曲线和评测生成均保留。']
 (out/'report.md').write_text('\n'.join(lines)+'\n');shutil.copy2(out/'report.md',root/'REPORT.md');update_progress(root,out.name,done=1,total=1,stage='20-step and both complete four-benchmark evaluations verified')

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['storage','restore','report']);p.add_argument('--output',type=Path,required=True);p.add_argument('--manifest',type=Path);p.add_argument('--dataset',type=Path);p.add_argument('--model');p.add_argument('--benchmarks',type=Path);a=p.parse_args()
 if a.mode=='storage':storage(a.output)
 elif a.mode=='restore':restore(a)
 else:report(a.output.parent.parent,a.output)
if __name__=='__main__':main()
