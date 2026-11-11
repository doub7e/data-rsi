"""BF16 model forward/backward matching DeepSpeed; FP32 weighted accumulation."""
from loss_contract import forward_loss

def allocate(model):
 import torch
 return {name:torch.zeros_like(parameter,dtype=torch.float32,device='cuda') for name,parameter in model.named_parameters()}

def add_row(model,trainer,row,denominator,accumulator,coefficient=1.):
 if not row['target_tokens'] or coefficient==0:return 0.
 model.zero_grad(set_to_none=True);loss=forward_loss(model,trainer,row,denominator);value=float(loss.detach());loss.backward();del loss
 for name,parameter in model.named_parameters():
  assert parameter.grad is not None,name
  accumulator[name].add_(parameter.grad,alpha=coefficient)
 model.zero_grad(set_to_none=True)
 return value
