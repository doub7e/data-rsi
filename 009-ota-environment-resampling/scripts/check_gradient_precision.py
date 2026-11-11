"""Quantify legacy FP32-autocast vs actual DeepSpeed BF16-forward at checkpoint1."""
import math,gc,inspect
from common import *
from loss_contract import runtime,forward_loss
RUN=ROOT/'runs/gradient-precision-check'
def main():
 import torch
 from deepspeed.runtime.engine import DeepSpeedEngine
 state(RUN,'precision_parity',started_at=now());source=inspect.getsource(DeepSpeedEngine._configure_distributed_model);(RUN/'deepspeed-module-dtype.py').write_text(source);assert 'bfloat16' in source
 rows=[json.loads(line) for line in (ROOT/'runs/teacher-batch1/data/train.jsonl').read_text().splitlines() if line.strip()];ordered=sorted(enumerate(rows),key=lambda x:sum(len(m['content']) for m in x[1]['conversations']));index,raw=ordered[len(ordered)//2];model,trainer,tok,encode=runtime(RUN,precision='legacy_fp32_autocast');row=encode(raw['conversations']);assert row['target_tokens']>0
 torch.manual_seed(42);model.zero_grad(set_to_none=True);loss=forward_loss(model,trainer,row,row['target_tokens']);legacy=float(loss.detach());loss.backward();old={n:p.grad.detach().cpu() for n,p in model.named_parameters()};del loss;model.zero_grad(set_to_none=True);gc.collect();model.bfloat16();assert all(p.dtype==torch.bfloat16 for p in model.parameters());torch.cuda.empty_cache()
 torch.manual_seed(42);loss=forward_loss(model,trainer,row,row['target_tokens']);actual=float(loss.detach());loss.backward();a2=b2=dot=diff2=0.
 for name,p in model.named_parameters():
  a=old.pop(name).reshape(-1);b=p.grad.reshape(-1)
  for off in range(0,len(a),1<<20):
   aa=a[off:off+(1<<20)].to('cuda',dtype=torch.float64);bb=b[off:off+(1<<20)].double();a2+=float(aa@aa);b2+=float(bb@bb);dot+=float(aa@bb);difference=aa-bb;diff2+=float(difference@difference)
 result={'finished_at':now(),'training_dtype_evidence':'zero3 bf16.enabled=true; actualDeepSpeed _configure_distributed_model source and savedcheckpointdtypeBF16','sample_index':index,'task':raw['task'],'input_tokens':len(row['input_ids']),'target_tokens':row['target_tokens'],'legacy_fp32_autocast_loss':legacy,'bf16_model_loss':actual,'metrics':{'relative_loss_difference':abs(legacy-actual)/max(abs(actual),1e-30),'gradient_cosine':dot/math.sqrt(a2*b2),'gradient_relative_l2_difference':math.sqrt(diff2/max(b2,1e-30)),'legacy_gradient_norm':math.sqrt(a2),'bf16_gradient_norm':math.sqrt(b2)},'note':'Checks numericalforward/gradientprecision only. Dataset templates, labels and token normalization are identical.'};write(RUN/'result.json',result);write(RUN/'complete.json',{'finished_at':now()});state(RUN,'complete','succeeded',finished_at=now());print(json.dumps(result),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:state(RUN,'failed','failed',error=str(e),finished_at=now());raise
