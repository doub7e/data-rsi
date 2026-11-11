"""Exact mean of per-question score-centered training-loss gradients at checkpoint1."""
import sys,gc,math,collections
from common import *
from loss_contract import runtime,forward_loss
from gradient_ops import allocate,add_row
P=json.loads((ROOT/'configs/protocol.json').read_text());N=P['probe_rollouts_per_question'];SHARD=int(sys.argv[1]);RUN=ROOT/f'runs/probe-gradient-shard-{SHARD}'
def main():
 import torch
 snapshot_sources(RUN,['probe_gradient_shard.py','common.py','loss_contract.py','gradient_ops.py'])
 source=ROOT/'runs/probe';result=json.loads((source/'result.json').read_text());group=collections.defaultdict(list)
 for r in result['records']:group[r['task']].append(r)
 split=json.loads((ROOT/'configs/dev-split.json').read_text());assert set(group)==set(split['probe_tasks'])
 eligible=[];excluded=[]
 for task,rows in sorted(group.items()):
  assert len(rows)==N;scores=[r['score'] for r in rows]
  (eligible if len(set(scores))>1 else excluded).append({'task':task,'rows':rows,'mean_score':sum(scores)/N,'scores':scores})
 plan=json.loads((ROOT/'runs/probe-gradient/plan.json').read_text());total_valid=plan['global_valid_questions'];assert total_valid==len(eligible) and sha(source/'data/train.jsonl')==plan['data_sha256'];assigned=set(plan['assignments'][SHARD]);eligible=[q for q in eligible if q['task'] in assigned];assert eligible and len(eligible)==len(assigned)
 write(RUN/'assigned-questions.json',{'shard':SHARD,'questions':sorted(assigned),'global_valid_questions':total_valid,'coefficient_normalizer':total_valid})
 data=RUN/'data';data.mkdir(exist_ok=True)
 with (data/'train.jsonl').open('w') as f:
  for q in eligible:
   for rec in q['rows']:f.write((source/'trials'/rec['id']/'train.jsonl').read_text())

 if not eligible:
  write(RUN/'progress.json',{'stage':'constant_score_questions','done':70,'total':70,'updated_at':now()})
  write(RUN/'probe-gradient-complete.json',{'finished_at':now(),'no_signal':True,'valid_questions':0,'probe_norm':0.});write(RUN/'result.json',{'metrics':{'valid_probe_questions':0,'probe_norm':0.}});write(RUN/'complete.json',{'finished_at':now()});state(RUN,'complete','succeeded',finished_at=now());return
 model,trainer,tok,encode=runtime(RUN,precision='training_bf16');model.zero_grad(set_to_none=True);params=dict(model.named_parameters());accumulator=allocate(model);resume=RUN/'probe-accumulator.pt';completed=[];audit=[];data_hash=sha(source/'data/train.jsonl')
 if resume.exists():
  snapshot=torch.load(resume,map_location='cpu',weights_only=True);assert snapshot['data_sha256']==data_hash,'Probe data changed during resume';completed=snapshot['completed_questions'];audit=snapshot['encoding_audit'];assert set(completed)<={q['task'] for q in eligible}
  for n,p in params.items():accumulator[n].copy_(snapshot['gradient'][n]);del snapshot['gradient'][n]
  del snapshot;gc.collect()
 state(RUN,'probe_reward_gradient',started_at=now())
 for q in eligible:
  if q['task'] in completed:continue
  encoded=[]
  for rec in q['rows']:
   rows=[json.loads(l) for l in (source/'trials'/rec['id']/'train.jsonl').read_text().splitlines() if l.strip()];tokens=[encode(r['conversations']) for r in rows];encoded.append((rec,tokens))
  total=sum(row['target_tokens'] for rec,rows in encoded for row in rows)
  for index,(rec,rows) in enumerate(encoded,1):
   coefficient=-(rec['score']-q['mean_score'])/total_valid
   for row in rows:
    if not row['target_tokens'] or coefficient==0:continue
    add_row(model,trainer,row,total,accumulator,coefficient)
   write(RUN/'progress.json',{'stage':'probe_gradient','done':len(completed)*N+index,'total':len(eligible)*N,'question':q['task'],'updated_at':now()})
  for n,value in accumulator.items():assert torch.isfinite(value).all(),n
  completed.append(q['task']);audit.append({'task':q['task'],'target_tokens':total,'original_target_tokens':sum(row['original_targets'] for _,rows in encoded for row in rows),'rows':sum(len(rows) for _,rows in encoded),'rollouts':N,'zero_supervision_rollouts':sum(not any(row['target_tokens'] for row in rows) for _,rows in encoded),'zero_supervision_question':total==0});data={'completed_questions':list(completed),'data_sha256':data_hash,'encoding_audit':list(audit),'gradient':{n:value.detach().cpu() for n,value in accumulator.items()}};tmp=resume.with_suffix('.tmp');torch.save(data,tmp);tmp.replace(resume);del data;gc.collect()
  write(RUN/'probe-encoding-audit.json',audit)
  print('PROBE_GRADIENT',len(completed),len(eligible),q['task'],flush=True)
 norm2=0.
 for value in accumulator.values():
  v=value.reshape(-1)
  for off in range(0,len(v),1<<20):x=v[off:off+(1<<20)].double();norm2+=float(x@x)
 assert math.isfinite(norm2)
 write(RUN/'probe-gradient-complete.json',{'finished_at':now(),'valid_questions':len(eligible),'global_valid_questions':total_valid,'excluded_questions':len(excluded),'probe_norm':math.sqrt(norm2),'gradient_file':str(resume.relative_to(ROOT)),'forward_precision':'training_bf16','gradient_accumulation':'FP32 weighted sum of BF16 per-row gradients','loss_reference_sha256':sha(ROOT/'runs/sft-epoch1/train.yaml'),'no_signal':norm2==0.});write(RUN/'result.json',{'metrics':{'valid_probe_questions':len(eligible),'excluded_questions':len(excluded),'probe_norm':math.sqrt(norm2)}});write(RUN/'complete.json',{'finished_at':now()});state(RUN,'complete','succeeded',finished_at=now())
if __name__=='__main__':
 try:main()
 except BaseException as e:state(RUN,'probe_gradient_failed','failed',error=str(e),finished_at=now());raise
