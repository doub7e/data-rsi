"""Exact full-parameter probe dot each original teacher trajectory update direction."""
import sys,math,gc
from common import *
from loss_contract import runtime,forward_loss
from gradient_ops import allocate,add_row
RUN=ROOT/'runs/attribution'
def main(shard):
 import torch
 torch.set_num_threads(4);meta=json.loads((ROOT/'runs/probe-gradient/probe-gradient-complete.json').read_text());source=ROOT/'runs/teacher-batch1';records=json.loads((source/'result.json').read_text())['records'];assigned=records[shard::2];out=ROOT/f'runs/attribution-candidates-{shard}';out.mkdir(exist_ok=True);snapshot_sources(out,['candidate_dots.py','common.py','loss_contract.py','gradient_ops.py']);state(out,'candidate_gradient_dots',started_at=now())
 data=out/'data';data.mkdir(exist_ok=True)
 with (data/'train.jsonl').open('w') as f:
  for rec in assigned:f.write((source/'trials'/rec['id']/'train.jsonl').read_text())
 m=json.loads((out/'run.json').read_text());m['sources']['training_data']=str((data/'train.jsonl').relative_to(ROOT));write(out/'run.json',m)
 if meta['no_signal']:
  write(out/'progress.json',{'stage':'no_probe_signal_zero_dots','done':len(assigned),'total':len(assigned),'updated_at':now()})
  write(out/'result.json',{'records':[{'id':r['id'],'task':r['task'],'dot':0.,'reason':'No nonzero mean probe gradient; uniform sampling fallback'} for r in assigned],'metrics':{'scored_trajectories':len(assigned),'all_dots_zero':1}});write(out/'complete.json',{'finished_at':now()});state(out,'complete','succeeded',finished_at=now());return
 checkpoint=torch.load(ROOT/meta['gradient_file'],map_location='cpu',weights_only=True);probe=checkpoint['gradient'];del checkpoint;model,trainer,tok,encode=runtime(out,precision='training_bf16');params=dict(model.named_parameters());accumulator=allocate(model);assert set(params)==set(probe);results=[]
 for idx,rec in enumerate(assigned,1):
  path=out/'records'/(rec['id']+'.json')
  if path.exists():results.append(json.loads(path.read_text()));continue
  rows=[encode(json.loads(l)['conversations']) for l in (source/'trials'/rec['id']/'train.jsonl').read_text().splitlines() if l.strip()];targets=sum(x['target_tokens'] for x in rows);model.zero_grad(set_to_none=True);nll=0.
  for value in accumulator.values():value.zero_()
  if targets:
   for row in rows:
    if row['target_tokens']:
     nll+=add_row(model,trainer,row,targets,accumulator)
  dot=0.;norm2=0.
  if targets:
   for n,value in accumulator.items():
    assert torch.isfinite(value).all();g=value.reshape(-1);v=probe[n].reshape(-1)
    for off in range(0,len(g),1<<20):
     a=g[off:off+(1<<20)].double();b=v[off:off+(1<<20)].to('cuda',dtype=torch.float64);dot-=float(a@b);norm2+=float(a@a)
  assert math.isfinite(dot);r={'id':rec['id'],'task':rec['task'],'dot':dot,'target_tokens':targets,'original_targets':sum(x['original_targets'] for x in rows),'sft_rows':len(rows),'mean_token_nll':nll if targets else None,'candidate_update_norm':math.sqrt(norm2),'score_definition':'dot(mean_probe_reward_gradient, -grad trajectory token-mean SFTloss)','checkpoint':'sft-epoch1','forward_precision':'training_bf16','gradient_accumulation':'FP32'};write(path,r);results.append(r);write(out/'progress.json',{'stage':'candidate_full_parameter_dots','done':idx,'total':len(assigned),'updated_at':now()});print('CANDIDATE_DOT',shard,idx,len(assigned),dot,flush=True)
 write(out/'result.json',{'finished_at':now(),'records':results,'metrics':{'scored_trajectories':len(results)}});write(out/'complete.json',{'finished_at':now()});state(out,'complete','succeeded',finished_at=now())
if __name__=='__main__':
 try:main(int(sys.argv[1]))
 except BaseException as e:state(ROOT/f'runs/attribution-candidates-{sys.argv[1]}','failed','failed',error=str(e),finished_at=now());raise
