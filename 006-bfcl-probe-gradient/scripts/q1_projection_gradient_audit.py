"""Diagnose the preflight norm discrepancy without changing acceptance thresholds."""
from common import *
import math,time

def main():
 import torch
 from gradients import load_model,candidate_encoder,backward,test_gradient
 run=ROOT/'runs/q1p-gradient-audit-20261105';state(run,'gradient_replay',started_at=now());assert torch.cuda.device_count()==1
 initial={'matmul_precision':torch.get_float32_matmul_precision(),'allow_tf32':torch.backends.cuda.matmul.allow_tf32,'allow_bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0)}
 torch.set_num_threads(4);test_gradient();model=load_model(MODEL);encode=candidate_encoder();data={x['meta_index']:x for x in rows(ROOT/'data/candidates.jsonl')};old={x['meta_index']:x for x in rows(ROOT/'runs/q1-scores-20261104/F-scores.jsonl')};results=[]
 for idx in [1281,106]:
  samples=encode(data[idx]);denom=sum(sum(x!=-100 for x in y[1:]) for x,y in samples);reference=None
  for mode in ['original_defaults','repeat_original','projection_settings']:
   if mode=='projection_settings':torch.set_num_threads(12);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
   model.zero_grad(set_to_none=True)
   for x,y in samples:backward(model,x,y,1/denom,'sum')
   norm2=0.;refnorm2=0.;dot=0.;diff2=0.
   for name,p in model.named_parameters():
    g=p.grad;norm2+=float((g*g).sum(dtype=torch.float64))
    if reference is not None:
     r=reference[name].to('cuda');refnorm2+=float((r*r).sum(dtype=torch.float64));dot+=float((r*g).sum(dtype=torch.float64));delta=r-g;diff2+=float((delta*delta).sum(dtype=torch.float64));del r,delta
   if reference is None:reference={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters()}
   norm=math.sqrt(norm2);result={'meta_index':idx,'mode':mode,'target_tokens':denom,'input_sha256':__import__('hashlib').sha256(json.dumps(samples).encode()).hexdigest(),'norm':norm,'old_norm':old[idx]['gradient_norm'],'relative_norm_difference':norm/old[idx]['gradient_norm']-1,'cosine_vs_first':dot/math.sqrt(norm2*refnorm2) if refnorm2 else None,'relative_vector_error_vs_first':math.sqrt(diff2/refnorm2) if refnorm2 else None};results.append(result);print(json.dumps(result),flush=True);write(run/'result.json',{'initial_settings':initial,'comparisons':results})
  del reference
 write(run/'complete.json',{'finished_at':now()});state(run,'complete','succeeded')
if __name__=='__main__':
 try:main()
 except BaseException as e:state(ROOT/'runs/q1p-gradient-audit-20261105','failed','failed',error=str(e));raise
