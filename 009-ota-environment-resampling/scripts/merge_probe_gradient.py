"""Sum full-parameter FP32 shard accumulators; no projection or factorization."""
import math,gc
from common import *
RUN=ROOT/'runs/probe-gradient'
def main():
 import torch
 torch.set_num_threads(4);state(RUN,'reduce_probe_gradient_shards',started_at=now());plan=json.loads((RUN/'plan.json').read_text());assert plan['num_workers']>0;gradient=None;completed=[];audits=[]
 for i in range(plan['num_workers']):
  source=ROOT/f'runs/probe-gradient-shard-{i}';assert (source/'complete.json').exists();data=torch.load(source/'probe-accumulator.pt',map_location='cpu',weights_only=True);assert data['data_sha256']==plan['data_sha256'];assert set(data['completed_questions'])==set(plan['assignments'][i]);completed+=data['completed_questions'];audits+=data['encoding_audit'];partial=data['gradient'];del data
  if gradient is None:gradient=partial
  else:
   assert set(partial)==set(gradient)
   for name in list(partial):gradient[name].add_(partial.pop(name))
  del partial;gc.collect();write(RUN/'progress.json',{'stage':'reduce_probe_gradient_shards','done':i+1,'total':plan['num_workers'],'updated_at':now()})
 assert len(completed)==len(set(completed))==plan['global_valid_questions'];norm2=0.
 for value in gradient.values():
  flat=value.reshape(-1)
  for off in range(0,len(flat),1<<20):chunk=flat[off:off+(1<<20)].double();assert torch.isfinite(chunk).all();norm2+=float(chunk@chunk)
 path=RUN/'probe-accumulator.pt';tmp=path.with_suffix('.tmp');torch.save({'gradient':gradient,'completed_questions':completed,'encoding_audit':audits,'data_sha256':plan['data_sha256']},tmp);tmp.replace(path);write(RUN/'probe-encoding-audit.json',audits)
 meta={'finished_at':now(),'valid_questions':plan['global_valid_questions'],'excluded_questions':len(plan['excluded']),'probe_norm':math.sqrt(norm2),'gradient_file':str(path.relative_to(ROOT)),'loss_reference_sha256':sha(ROOT/'runs/sft-epoch1/train.yaml'),'no_signal':norm2==0.,'shards':plan['num_workers'],'forward_precision':'training_bf16','gradient_accumulation':'FP32','reduction':'SUM of per-shard vectors already divided by GLOBAL valid-question count; FP32 accumulation, full parameter dimension.'};write(RUN/'probe-gradient-complete.json',meta);write(RUN/'result.json',{'finished_at':now(),'metrics':{'valid_probe_questions':meta['valid_questions'],'excluded_questions':meta['excluded_questions'],'probe_norm':meta['probe_norm'],'gradient_workers':meta['shards']}});write(RUN/'complete.json',{'finished_at':now()});state(RUN,'complete','succeeded',finished_at=now())
if __name__=='__main__':
 try:main()
 except BaseException as e:state(RUN,'failed','failed',error=str(e),finished_at=now());raise
