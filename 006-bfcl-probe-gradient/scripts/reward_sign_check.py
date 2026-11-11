"""Analytic two-action check of the actual ascent/descent sign convention."""
import torch
from common import *
x=torch.zeros(2,requires_grad=True)
ce0=torch.nn.functional.cross_entropy(x[None],torch.tensor([0]))
ce1=torch.nn.functional.cross_entropy(x[None],torch.tensor([1]))
# Rewards [1,0], baseline .5, M2: implementation backprops -adv/M * NLL.
g=torch.autograd.grad(-.25*ce0+.25*ce1,x,retain_graph=True)[0]
u0=-torch.autograd.grad(ce0,x,retain_graph=True)[0]
u1=-torch.autograd.grad(ce1,x)[0]
cos=lambda a,b:float(torch.dot(a,b)/(a.norm()*b.norm()))
assert torch.allclose(g,torch.tensor([.25,-.25]))
assert cos(g,u0)>.99999 and cos(g,u1)<-.99999
result={'probe_gradient':g.tolist(),'successful_sft_update':u0.tolist(),'failed_sft_update':u1.tolist(),
        'success_cosine':cos(g,u0),'failure_cosine':cos(g,u1),'checked_at':now(),
        'note':'Analytic sign convention check only; does not change frozen ranking.'}
write(ROOT/'runs/gradient-smoke/reward-sign-check.json',result);print(result)
