"""CPU-only RunAI integration tests; no real influence experiment before dot gate."""
import copy
from common import *
import numpy as np
import torch
import q1_projected_influence as worker
from q1_projection_math import DenseProjection,extract_block,unit_checks

def main():
 run=ROOT/'runs/q1p-preflight-20261105';state(run,'integration_tests');torch.set_num_threads(2)
 result={'algebra':unit_checks()}
 from gradients import test_gradient
 result['gradient']=test_gradient()
 gen=torch.Generator().manual_seed(104);specs=[('w',(12,9)),('n',(9,))];projectors=[DenseProjection(specs,8,s,stream_width=1024,device='cpu') for s in worker.P['seeds']]
 raw=[{'w':torch.randn(12,9,generator=gen),'n':torch.randn(9,generator=gen)} for _ in range(206)]
 features=[np.stack([p.project(x) for x in raw]) for p in projectors];norms=np.array([[float((x[n]*x[n]).sum()) for n,_ in specs] for x in raw[:200]])
 ids=list(range(200));out=worker.score_features(features,projectors[0].entries,norms,ids,8,run/'synthetic',device='cpu')
 for (seed,width),res in out.items():
  direct=np.zeros((200,6));seed_index=worker.P['seeds'].index(seed)
  for k,e in enumerate(projectors[0].entries):
   block=torch.from_numpy(extract_block(features[seed_index],e,width,8)).double();z,q=block[:200],block[200:];refdim=math.prod(min(x,512) for x in e['shape']) if len(e['shape'])==2 else math.prod(e['shape']);lam=.5*norms[:,k].sum()/refdim
   direct+=(z@torch.linalg.solve(z.T@z+lam*torch.eye(z.shape[1],dtype=torch.float64),q.T)).numpy()
  np.testing.assert_allclose(res['influence'],direct,rtol=1e-8,atol=1e-9)
 result['integration']='Projected block kernels, damping and summed dual scores match direct primal solves for200 candidates and6 queries, two widths/two seeds.'
 # Use real archived exact dot columns as fixtures to test acceptance and rejection logic.
 cfg=worker.CFG2;src={m:{x['meta_index']:x for x in rows(ROOT/f'runs/q1-scores-20261104/{m}-scores.jsonl')} for m in ['F','K']};ids=sorted(src['F'])
 exact=[[src[v[0].upper()][i]['scores'][str(int(v[1:]))]['dot'] for v in cfg['variants']] for i in ids]
 fixture={key:{'projected_dot':exact,'influence':exact,'projected_norm2':[1.]*200,'raw_norm2':[1.]*200} for key in out}
 good=worker.check_quality(fixture,ids,8);assert good['passed'];bad=copy.deepcopy(fixture);bad[(worker.P['seeds'][1],8)]['influence']=(-np.array(exact)).tolist();rejected=worker.check_quality(bad,ids,8);assert not rejected['passed']
 result['quality_gate']={'identical_scores_pass':good['passed'],'reversed_scores_rejected':not rejected['passed']}
 write(run/'result.json',result);write(run/'complete.json',{'finished_at':now()});state(run,'complete','succeeded');print(json.dumps(result),flush=True)
if __name__=='__main__':
 import math
 try:main()
 except BaseException as e:state(ROOT/'runs/q1p-preflight-20261105','failed','failed',error=str(e));raise
