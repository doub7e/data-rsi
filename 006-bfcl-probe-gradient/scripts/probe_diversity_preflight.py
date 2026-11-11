"""Tiny CPU integration tests inside the existing training image."""
from common import *
def main():
 import torch,numpy as np
 from probe_diversity_math import unit_check,coefficients,pair_from_gram
 from gradients import test_gradient
 run=ROOT/'runs/pd-preflight-20261105';state(run,'numerical_preflight');torch.set_num_threads(2)
 result={'algebra':unit_check(),'autograd':test_gradient()}
 # The implemented CPU block-accumulation and GPU-style chunked Gram must match
 # direct centered gradients, including no successes in an entire32-sample block.
 torch.manual_seed(8);scores=torch.randn(128,97,dtype=torch.float32);rewards=torch.tensor([0]*32+[i%2 for i in range(96)]);sums=[];wins=[]
 for b in range(4):
  u=torch.zeros(97);w=torch.zeros(97)
  for i in range(32*b,32*(b+1)):
   u.add_(scores[i])
   if rewards[i]:w.add_(scores[i])
  sums.extend([u,w]);wins.append(int(rewards[b*32:(b+1)*32].sum()))
 matrix=torch.stack(sums).double();gram=torch.zeros(8,8,dtype=torch.float64)
 for offset in range(0,97,13):x=matrix[:,offset:offset+13];gram+=x@x.T
 for a,b in [([0,1],[2,3]),([0,2],[1,3]),([0,3],[1,2])]:
  aa=coefficients(a,wins,[32]*4);bb=coefficients(b,wins,[32]*4);actual=pair_from_gram(gram.numpy(),aa,bb);g=[]
  for subset in [a,b]:
   ids=[i for block in subset for i in range(block*32,(block+1)*32)];rr=rewards[ids].double();g.append(((rr-rr.mean())[:,None]*scores[ids].double()).mean(0))
  expected=float(torch.nn.functional.cosine_similarity(g[0],g[1],dim=0));assert abs(expected-actual['cosine'])<2e-7
 result['fp32_block_sum_matches_direct_centering']=True
 write(run/'result.json',result);write(run/'complete.json',{'finished_at':now()});state(run,'complete','succeeded');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
