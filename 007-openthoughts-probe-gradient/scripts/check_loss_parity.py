"""GPU parity against actual training loss; no model update."""
import math
from common import *
from loss_contract import runtime,forward_loss

def main():
 import torch
 run=ROOT/'runs/loss-parity-20261105';state(run,'loading_original_training_stack',started_at=now());assert torch.cuda.device_count()==1 and 'H200' in torch.cuda.get_device_name(0);torch.set_num_threads(4)
 model,trainer,tok,encode=runtime(run)
 fixtures=[{'conversations':[{'role':'user','content':'Inspect the working directory.'},{'role':'assistant','content':'<think>I should inspect the files before deciding.</think>\n{"commands":[{"keystrokes":"ls -la\\n","duration":0.1}]}'},{'role':'user','content':'Output: data.csv'},{'role':'assistant','content':'<think>The file exists; inspect its header.</think>\n{"commands":[{"keystrokes":"head -n 2 data.csv\\n","duration":0.1}]}'}]},json.loads((TRAIN/'runs/repro-10k-4h200-20261030/data/train.jsonl').open().readline())]
 # Use two different lengths, each with multiple supervised assistant turns.
 fixtures[1]['conversations']=fixtures[1]['conversations'][:4]
 rr=[encode(x['conversations']) for x in fixtures];counts=[x['target_tokens'] for x in rr];assert counts[0]!=counts[1];total=sum(counts)
 model.zero_grad(set_to_none=True);values=[]
 for r in rr:
  loss=forward_loss(model,trainer,r,total);values.append(float(loss.detach()));loss.backward()
 reference={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters()};model.zero_grad(set_to_none=True)
 # Equivalent implementation used for reward weighting: per-row summed CE,
 # then one shared logical-batch denominator; no per-row length reweighting.
 sums=[]
 for r in rr:
  loss=forward_loss(model,trainer,r,1);sums.append(float(loss.detach()));(loss/total).backward()
 na=nb=dot=delta=0.
 for n,p in model.named_parameters():
  a=reference[n].to('cuda');b=p.grad;d=a-b;na+=float((a*a).sum(dtype=torch.float64));nb+=float((b*b).sum(dtype=torch.float64));dot+=float((a*b).sum(dtype=torch.float64));delta+=float((d*d).sum(dtype=torch.float64))
 cosine=dot/math.sqrt(na*nb);relative=math.sqrt(delta/na);lossrel=abs(sum(values)-sum(sums)/total)/abs(sum(values))
 assert cosine>.9999 and relative<.02 and lossrel<.005,(cosine,relative,lossrel)
 # Original full training row exercises native cutoff and all-assistant supervision.
 real=json.loads((TRAIN/'runs/repro-10k-4h200-20261030/data/train.jsonl').open().readline());full=encode(real['conversations']);model.zero_grad(set_to_none=True);loss=forward_loss(model,trainer,full,full['target_tokens']);loss.backward()
 assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
 result={'finished_at':now(),'target_counts':counts,'shared_denominator':total,'trainer_batch_loss':sum(values),'sum_ce_div_shared_denominator':sum(sums)/total,'relative_loss_error':lossrel,'gradient_cosine':cosine,'gradient_relative_l2':relative,'full_training_row_length':len(full['input_ids']),'full_training_row_targets':full['target_tokens'],'full_training_row_loss':float(loss.detach()),'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'metrics':{'loss_parity_relative_error':lossrel,'gradient_parity_cosine':cosine},'interpretation':'The original Trainer/Liger loss is called directly. Test compares its accumulated-token denominator with algebraically equivalent token-sum accumulation, on unchanged original base weights; not a full optimizer or ZeRO training equivalence claim.'}
 write(run/'result.json',result);write(run/'complete.json',{'finished_at':now()});state(run,'complete','succeeded');print(json.dumps(result),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:state(ROOT/'runs/loss-parity-20261105','failed','failed',error=str(e),finished_at=now());raise
