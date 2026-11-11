"""Dense two-sided JL projection and exact dual ridge influence, without rank-c factors."""
import hashlib,math
import numpy as np
import torch

def seed_for(seed,name,side):return int.from_bytes(hashlib.sha256(f'{seed}:{name}:{side}'.encode()).digest()[:8],'little')%(2**63-1)
class DenseProjection:
 def __init__(self,specs,width,seed,stream_width=1024,device='cuda'):
  self.width=width;self.seed=seed;self.specs=specs;self.device=device;self.cache={};self.entries=[];offset=0
  for name,shape in specs:
   shape=tuple(shape)
   target=tuple(min(x,width) for x in shape) if len(shape)==2 else shape
   size=math.prod(target);self.entries.append({'name':name,'shape':shape,'target':target,'start':offset,'stop':offset+size});offset+=size
  self.dimension=offset;self.stream_width=stream_width
 def matrix(self,name,side,source,target):
  key=(name,side)
  if key not in self.cache:
   generator=torch.Generator(device='cpu');generator.manual_seed(seed_for(self.seed,name,side))
   # Fixed 1024-wide RNG stream makes every smaller projection an exact nested prefix.
   signs=torch.randint(0,2,(source,self.stream_width),generator=generator,dtype=torch.int8)[:,:target].to(torch.float32)
   signs.mul_(2).sub_(1).div_(math.sqrt(target));self.cache[key]=signs.to(self.device)
  return self.cache[key]
 @torch.no_grad()
 def project(self,grads):
  out=np.empty(self.dimension,dtype=np.float32)
  for e in self.entries:
   g=grads[e['name']]
   if len(e['shape'])==2:
    a,b=e['shape'];pa,pb=e['target'];left=self.matrix(e['name'],'out',a,pa);right=self.matrix(e['name'],'in',b,pb)
    z=(left.T@g)@right
   else:z=g
   out[e['start']:e['stop']]=z.detach().reshape(-1).cpu().numpy()
  return out
 def close(self):self.cache.clear()

def extract_block(features,entry,width,max_width):
 n=len(features);target=entry['target'];block=features[:,entry['start']:entry['stop']]
 if len(entry['shape'])==2:
  a,b=target;pa,pb=[min(x,width) for x in entry['shape']]
  block=block.reshape(n,a,b)[:,:pa,:pb]
  # Max-width projections were normalized by their own output dimensions.
  scale=math.sqrt(a*b/(pa*pb));return np.ascontiguousarray(block.reshape(n,-1))*scale
 return np.ascontiguousarray(block)

@torch.no_grad()
def dual_scores(z,q,damping):
 """Rows z are candidate ascent updates; q are reward ascent probes.
 Returns s=(ZZ^T+lambda I)^-1 ZQ^T, equal to Z(Z^TZ+lambda I)^-1 Q^T.
 """
 assert z.dtype==q.dtype==torch.float64 and damping>0
 gram=z@z.T;cross=z@q.T;matrix=gram+torch.eye(len(z),dtype=z.dtype,device=z.device)*damping
 chol=torch.linalg.cholesky(matrix);score=torch.cholesky_solve(cross,chol)
 residual=float(torch.linalg.vector_norm(matrix@score-cross)/torch.linalg.vector_norm(cross).clamp_min(1e-30))
 return score,gram,cross,residual

def unit_checks():
 from pathlib import Path
 import sys
 sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reference'))
 from lorif_reference.curvature import CurvatureModel
 from lorif_reference.factors import Factors
 from lorif_reference.influence import influence_scores
 torch.manual_seed(20261105);z=torch.randn(7,12,dtype=torch.float64);q=torch.randn(3,12,dtype=torch.float64);damping=.2
 score,gram,cross,res=dual_scores(z,q,damping)
 direct=z@torch.linalg.solve(z.T@z+torch.eye(12,dtype=z.dtype)*damping,q.T)
 torch.testing.assert_close(score,direct,rtol=1e-10,atol=1e-10)
 _,s,vh=torch.linalg.svd(z,full_matrices=False)
 curvature=CurvatureModel((3,4),vh.T,s,damping)
 def exact_factors(x):return Factors(x.reshape(len(x),3,4),torch.eye(4,dtype=x.dtype).expand(len(x),4,4))
 reference=influence_scores(exact_factors(q),exact_factors(z),curvature).T
 torch.testing.assert_close(score,reference,rtol=1e-10,atol=1e-10)
 # No preconditioner: correct sign is ascent dot ascent, and strong ridge tends to it.
 large,_,_,_=dual_scores(z,q,1e8);torch.testing.assert_close(large*1e8,cross,rtol=1e-5,atol=1e-5)
 grads={'w':torch.randn(12,9),'norm':torch.randn(9)};specs=[(k,tuple(v.shape)) for k,v in grads.items()]
 big=DenseProjection(specs,8,123,stream_width=16,device='cpu');small=DenseProjection(specs,4,123,stream_width=16,device='cpu');a=big.project(grads)[None];b=small.project(grads)
 recovered=np.concatenate([extract_block(a,e,4,8)[0] for e in big.entries]);np.testing.assert_allclose(recovered,b,rtol=2e-5,atol=2e-5)
 # Linearity is necessary for reward-weighted accumulation in projected space.
 g2={k:torch.randn_like(v) for k,v in grads.items()};lhs=big.project({k:.3*grads[k]-.7*g2[k] for k in grads});rhs=.3*big.project(grads)-.7*big.project(g2);np.testing.assert_allclose(lhs,rhs,rtol=2e-5,atol=2e-5)
 return {'dual_vs_primal_max_error':float((score-direct).abs().max()),'dual_vs_lorif_max_error':float((score-reference).abs().max()),'solve_residual':res,'nested_projection_max_error':float(np.max(np.abs(recovered-b))),'linearity_max_error':float(np.max(np.abs(lhs-rhs))),'rank_c':'No approximation; identity factors only in tiny upstream-equivalence test'}
