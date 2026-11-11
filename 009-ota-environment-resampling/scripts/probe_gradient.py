"""Exact mean of per-question score-centered training-loss gradients at checkpoint1."""
import sys,gc,math,collections
from common import *
from loss_contract import runtime,forward_loss
P=json.loads((ROOT/'configs/protocol.json').read_text());RUN=ROOT/'runs/probe-gradient'
def main():
 import torch
 snapshot_sources(RUN,['probe_gradient.py','common.py','loss_contract.py'])
 source=ROOT/'runs/probe';result=json.loads((source/'result.json').read_text());group=collections.defaultdict(list)
 for r in result['records']:group[r['task']].append(r)
 split=json.loads((ROOT/'configs/dev-split.json').read_text());assert set(group)==set(split['probe_tasks'])
 eligible=[];excluded=[]
 for task,rows in sorted(group.items()):
  assert len(rows)==32;scores=[r['score'] for r in rows]
  (eligible if len(set(scores))>1 else excluded).append({'task':task,'rows':rows,'mean_score':sum(scores)/32,'scores':scores})
 write(RUN/'probe-questions.json',{'eligible':eligible,'excluded':excluded,'original_questions':14,'replacement_questions':0,'weight':'Equal average across eligible question gradients'})
 if not eligible:
  write(RUN/'progress.json',{'stage':'constant_score_questions','done':14,'total':14,'updated_at':now()})
  write(RUN/'probe-gradient-complete.json',{'finished_at':now(),'no_signal':True,'valid_questions':0,'probe_norm':0.});write(RUN/'result.json',{'metrics':{'valid_probe_questions':0,'probe_norm':0.}});write(RUN/'complete.json',{'finished_at':now()});state(RUN,'complete','succeeded',finished_at=now());return
 model,trainer,tok,encode=runtime(RUN);model.zero_grad(set_to_none=True);params=dict(model.named_parameters());resume=RUN/'probe-accumulator.pt';completed=[];audit=[];data_hash=sha(source/'data/train.jsonl')
 if resume.exists():
  snapshot=torch.load(resume,map_location='cpu',weights_only=True);assert snapshot['data_sha256']==data_hash,'Probe data changed during resume';completed=snapshot['completed_questions'];audit=snapshot['encoding_audit'];assert set(completed)<={q['task'] for q in eligible}
  for n,p in params.items():p.grad=snapshot['gradient'][n].to('cuda');del snapshot['gradient'][n]
  del snapshot;gc.collect()
 state(RUN,'probe_reward_gradient',started_at=now())
 for q in eligible:
  if q['task'] in completed:continue
  encoded=[]
  for rec in q['rows']:
   rows=[json.loads(l) for l in (source/'trials'/rec['id']/'train.jsonl').read_text().splitlines() if l.strip()];tokens=[encode(r['conversations']) for r in rows];encoded.append((rec,tokens))
  total=sum(row['target_tokens'] for rec,rows in encoded for row in rows);assert total>0
  for index,(rec,rows) in enumerate(encoded,1):
   coefficient=-(rec['score']-q['mean_score'])/len(eligible)
   for row in rows:
    if not row['target_tokens'] or coefficient==0:continue
    loss=forward_loss(model,trainer,row,total)*coefficient;loss.backward();del loss
   write(RUN/'progress.json',{'stage':'probe_gradient','done':len(completed)*32+index,'total':len(eligible)*32,'question':q['task'],'updated_at':now()})
  for n,p in params.items():assert p.grad is not None and torch.isfinite(p.grad).all(),n
  completed.append(q['task']);audit.append({'task':q['task'],'target_tokens':total,'original_target_tokens':sum(row['original_targets'] for _,rows in encoded for row in rows),'rows':sum(len(rows) for _,rows in encoded),'rollouts':32});data={'completed_questions':list(completed),'data_sha256':data_hash,'encoding_audit':list(audit),'gradient':{n:p.grad.detach().cpu() for n,p in params.items()}};tmp=resume.with_suffix('.tmp');torch.save(data,tmp);tmp.replace(resume);del data;gc.collect()
  write(RUN/'probe-encoding-audit.json',audit)
  print('PROBE_GRADIENT',len(completed),len(eligible),q['task'],flush=True)
 norm2=0.
 for p in params.values():
  v=p.grad.reshape(-1)
  for off in range(0,len(v),1<<20):x=v[off:off+(1<<20)].double();norm2+=float(x@x)
 assert math.isfinite(norm2)
 write(RUN/'probe-gradient-complete.json',{'finished_at':now(),'valid_questions':len(eligible),'excluded_questions':len(excluded),'probe_norm':math.sqrt(norm2),'gradient_file':str(resume.relative_to(ROOT)),'loss_reference_sha256':sha(ROOT/'runs/sft-epoch1/train.yaml'),'no_signal':norm2==0.});write(RUN/'result.json',{'metrics':{'valid_probe_questions':len(eligible),'excluded_questions':len(excluded),'probe_norm':math.sqrt(norm2)}});write(RUN/'complete.json',{'finished_at':now()});state(RUN,'complete','succeeded',finished_at=now())
if __name__=='__main__':
 raise RuntimeError('Legacy prototype disabled; use prepare_probe_gradient.py, BF16 probe_gradient_shard.py and merge_probe_gradient.py')
 try:main()
 except BaseException as e:state(RUN,'probe_gradient_failed','failed',error=str(e),finished_at=now());raise
