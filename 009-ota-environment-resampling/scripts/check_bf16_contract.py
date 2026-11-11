"""Compare new BF16-forward/FP32-buffer gradient to direct BF16 Trainer reference."""
import math,gc
from common import *
from loss_contract import runtime
from gradient_ops import allocate,add_row
RUN=ROOT/'runs/gradient-bf16-contract-check'
def main():
 import torch
 state(RUN,'actual_training_precision_check',started_at=now());rows=[json.loads(x) for x in (ROOT/'runs/teacher-batch1/data/train.jsonl').read_text().splitlines() if x.strip()];ordered=sorted(enumerate(rows),key=lambda x:sum(len(m['content']) for m in x[1]['conversations']));index,raw=ordered[len(ordered)//2];model,trainer,tok,encode=runtime(RUN,precision='training_bf16');row=encode(raw['conversations']);acc=allocate(model);torch.manual_seed(42);adapted=add_row(model,trainer,row,row['target_tokens'],acc)
 old={n:v.cpu() for n,v in acc.items()};del acc;gc.collect();torch.cuda.empty_cache();model.zero_grad(set_to_none=True);torch.manual_seed(42)
 inputs={'input_ids':torch.tensor([row['input_ids']],device='cuda'),'labels':torch.tensor([row['labels']],device='cuda'),'attention_mask':torch.ones((1,len(row['input_ids'])),device='cuda',dtype=torch.long)}
 with torch.autocast('cuda',enabled=False):loss=trainer.compute_loss(model,inputs,num_items_in_batch=torch.tensor(row['target_tokens'],device='cuda'))
 direct=float(loss.detach());loss.backward();a2=b2=dot=diff2=0.
 for name,p in model.named_parameters():
  aa=old.pop(name).reshape(-1);bb=p.grad.reshape(-1)
  for off in range(0,len(aa),1<<20):
   a=aa[off:off+(1<<20)].to('cuda',dtype=torch.float64);b=bb[off:off+(1<<20)].double();a2+=float(a@a);b2+=float(b@b);dot+=float(a@b);d=a-b;diff2+=float(d@d)
 cosine=dot/math.sqrt(a2*b2);relative_loss=abs(adapted-direct)/max(abs(direct),1e-30)
 x=torch.tensor([.123,-.456,.789],device='cuda',dtype=torch.bfloat16);buffer=torch.zeros_like(x,dtype=torch.float32);buffer.add_(x,alpha=.3);buffer.add_(x,alpha=-.2);assert torch.allclose(buffer,x.float()*.1,rtol=1e-6,atol=1e-7)
 result={'finished_at':now(),'sample_index':index,'input_tokens':len(row['input_ids']),'target_tokens':row['target_tokens'],'adapted_loss':adapted,'direct_bf16_trainer_loss':direct,'metrics':{'relative_loss_difference':relative_loss,'gradient_cosine':cosine,'gradient_relative_l2_difference':math.sqrt(diff2/b2),'fp32_weighted_accumulation_verified':1},'reference':'Direct BF16 model + actualTrainer.compute_loss, autocast disabled, matching DeepSpeed module.bfloat16() forward. New helper uses BF16 model and FP32 accumulation.'};write(RUN/'result.json',result);assert relative_loss<1e-6 and cosine>.999,'Actual-training BF16 parity failed';write(RUN/'complete.json',{'finished_at':now()});state(RUN,'complete','succeeded',finished_at=now());print(json.dumps(result),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:state(RUN,'failed','failed',error=str(e),finished_at=now());raise
