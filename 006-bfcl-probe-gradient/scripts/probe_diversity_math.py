"""Small exact algebra used by full-parameter split-half diagnostics."""
import math
import numpy as np

def coefficients(blocks,successes,counts):
 n=sum(counts[b] for b in blocks);wins=sum(successes[b] for b in blocks);out=np.zeros(2*len(counts),dtype=np.float64)
 if wins not in (0,n):
  for b in blocks:out[2*b]=-wins/(n*n);out[2*b+1]=1/n
 return out

def pair_from_gram(gram,a,b):
 na=float(a@gram@a);nb=float(b@gram@b);dot=float(a@gram@b);scale=max(float(np.abs(gram).max()),1.)
 assert na>=-1e-10*scale and nb>=-1e-10*scale
 na=max(na,0.);nb=max(nb,0.);cos=dot/math.sqrt(na*nb) if na>0 and nb>0 else None
 assert cos is None or abs(cos)<1.000001
 return {'norm_a':math.sqrt(na),'norm_b':math.sqrt(nb),'dot':dot,'cosine':min(1.,max(-1.,cos)) if cos is not None else None}

def unit_check():
 rng=np.random.default_rng(5);s=rng.normal(size=(128,31));rewards=rng.integers(0,2,128);vectors=[];wins=[];counts=[32]*4
 for b in range(4):
  x=s[b*32:(b+1)*32];y=rewards[b*32:(b+1)*32];vectors.extend([x.sum(0),x[y==1].sum(0)]);wins.append(int(y.sum()))
 v=np.array(vectors);gram=v@v.T
 for aa,bb in [([0,1],[2,3]),([0,2],[1,3]),([0,3],[1,2])]:
  a=coefficients(aa,wins,counts);b=coefficients(bb,wins,counts);actual=pair_from_gram(gram,a,b)
  gs=[]
  for blocks in [aa,bb]:
   ids=[i for block in blocks for i in range(32*block,32*(block+1))];y=rewards[ids];gs.append(((y-y.mean())[:,None]*s[ids]).mean(0))
  np.testing.assert_allclose(a@v,gs[0],atol=1e-14);np.testing.assert_allclose(b@v,gs[1],atol=1e-14);expected=float(gs[0]@gs[1]/np.linalg.norm(gs[0])/np.linalg.norm(gs[1]));assert abs(actual['cosine']-expected)<1e-12
 zero=coefficients([0,1],[32,32,0,0],counts);assert not zero.any() and pair_from_gram(gram,zero,np.ones(8))['cosine'] is None
 return {'three_partitions_match_direct_centered_gradients':True,'constant_reward_cosine_is_undefined':True}
