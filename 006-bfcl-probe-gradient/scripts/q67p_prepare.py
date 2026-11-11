"""Resumable single-width attribution for one new question; no convergence gates."""
import gc,math,time
from common import *
import numpy as np
import torch
from gradients import load_model,backward,candidate_encoder,test_gradient
from q1_projection_math import DenseProjection,dual_scores
STUDY=ROOT/'runs/q67-pif-20261106';RUN=ROOT/'runs/q67p-prepare-20261106'

def save_array(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
 with tmp.open('wb') as f:np.save(f,value,allow_pickle=False)
 tmp.replace(path)

def materialize(cfg,ids,scores):
 sys.path.insert(0,str(BASE/'upstream/code/bfcl/selection'));from recipe.expand import expand_trajectory
 candidates={x['meta_index']:x for x in rows(ROOT/'data/candidates.jsonl')};audits={}
 for j,v in enumerate(cfg['variants']):
  ranking=sorted([{'meta_index':idx,'influence':float(scores[i,j])} for i,idx in enumerate(ids)],key=lambda x:(-x['influence'],x['meta_index']))
  selected=sorted([candidates[x['meta_index']] for x in ranking[:100]],key=lambda x:x['pool_index']);samples=[]
  for row in selected:
   for s in expand_trajectory(row['messages']):samples.append(dict(instruction=s['prompt'],input='',output=s['target'],system=s['system'],history=[list(x) for x in s['history']]))
  run=ROOT/f'runs/q67p-{v}-train-20261106';d=run/'data';d.mkdir(exist_ok=True)
  text=json.dumps(samples,ensure_ascii=False);dest=d/'train-data.json'
  if dest.exists():assert dest.read_text()==text,'Frozen training data changed'
  else:dest.write_text(text)
  (d/'train.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in selected));write(d/'ranking.json',ranking)
  audit={'variant':v,'source_q1_variant':cfg['selected'][v],'meta_indices':[x['meta_index'] for x in selected],'trajectories':100,'expanded_rows':len(samples),'candidate_sha256':cfg['candidate_sha256'],'data_sha256':sha(dest)}
  write(d/'audit.json',audit);audits[v]=audit
  write(run/'preparation-complete.json',{'completed_at':now(),'model_path':str(MODEL),'data_path':str(d),'expanded_rows':len(samples)})
  m=json.loads((run/'run.json').read_text());m['parameters'].update(audit);m['artifacts'].append({'label':'Selection ranking','path':str((d/'ranking.json').relative_to(ROOT))});write(run/'run.json',m)
 return audits

def main():
 cfg=json.loads((STUDY/'protocol.json').read_text());assert (ROOT/cfg['gate']).exists() and cfg['selected'];p=cfg['projection'];variants=cfg['variants'];sizes={v:int(cfg['selected'][v][1:]) for v in variants}
 assert torch.cuda.device_count()==1 and 'H200' in torch.cuda.get_device_name(0)
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
 state(RUN,'probe_gradients',started_at=now());write(RUN/'gradient-unit-check.json',test_gradient())
 source=ROOT/'runs/q67p-probe-20261106';assert (source/'complete.json').exists()
 traces={i:json.loads((source/f'traces/r{i:03d}.json').read_text()) for i in cfg['rollout_order'][:max(sizes.values())]}
 assert all(t['id']==cfg['probe_id'] and t['history_protocol']=='strict-v2' for t in traces.values())
 prefixes={v:cfg['rollout_order'][:n] for v,n in sizes.items()};rates={v:sum(traces[i]['reward'] for i in subset)/len(subset) for v,subset in prefixes.items()}
 write(RUN/'probe-prefixes.json',{'selected':cfg['selected'],'prefixes':prefixes,'reward_rates':rates,'trace_sha256':{str(i):sha(source/f'traces/r{i:03d}.json') for i in traces}})
 if any(x in (0,1) for x in rates.values()):
  write(RUN/'degenerate-probe.json',{'at':now(),'rates':rates,'policy':'Frozen prefix has no reward variance. No automatic resampling, replacement question or arbitrary tie-ranked training.'})
  raise RuntimeError('Degenerate frozen probe prefix; preserved diagnostics, no silent resampling')
 model=load_model(ROOT/cfg['model_view']);params=dict(model.named_parameters());specs=[(n,tuple(x.shape)) for n,x in params.items()];assert sum(math.prod(s) for n,s in specs)==4022468096
 projector=DenseProjection(specs,p['width'],p['seed'],stream_width=p['stream_width']);entries=projector.entries;dimension=projector.dimension
 write(RUN/'layout.json',{'dimension':dimension,'entries':entries,'width':p['width'],'seed':p['seed'],'stream_width':p['stream_width']})
 query_path=RUN/'probe-state.pt'
 if query_path.exists():
  saved=torch.load(query_path,map_location='cpu',weights_only=False);queries=saved['queries'];done=saved['done'];assert saved['selected']==cfg['selected']
 else:queries=np.zeros((2,dimension),dtype=np.float32);done=0
 order=cfg['rollout_order'][:max(sizes.values())]
 for position,rep in enumerate(order,1):
  if position<=done:continue
  trace=traces[rep];accum={v:{n:torch.zeros_like(x) for n,x in params.items()} for v in variants if rep in prefixes[v]}
  for step in trace['steps']:
   modes=[v for v in accum if v=='f' or step['turn_index']==cfg['key_turn']]
   if not modes:continue
   assert step['prompt_token_replay_verified'];prompt=step['prompt_token_ids'];completion=step['completion_token_ids'];model.zero_grad(set_to_none=True)
   backward(model,prompt+completion,[-100]*len(prompt)+completion,-1.,'sum')
   with torch.no_grad():
    for n,x in params.items():
     assert x.grad is not None
     for v in modes:accum[v][n].add_(x.grad)
  for v in accum:
   value=projector.project(accum[v]);assert np.isfinite(value).all();queries[variants.index(v)]+=((trace['reward']-rates[v])/sizes[v])*value
  del accum,value
  # Each checkpoint atomically includes both sums and the exact processed prefix.
  if position%8==0 or position==len(order):
   tmp=query_path.with_suffix('.tmp');torch.save({'queries':queries,'done':position,'selected':cfg['selected']},tmp);tmp.replace(query_path)
  write(RUN/'progress.json',{'stage':'probe_gradients','done':position,'total':len(order),'updated_at':now()});print('PROBE',position,flush=True)
 del params,model;gc.collect();torch.cuda.empty_cache();assert np.isfinite(queries).all() and all(np.linalg.norm(x)>0 for x in queries)
 model=load_model(MODEL);params=dict(model.named_parameters());assert specs==[(n,tuple(x.shape)) for n,x in params.items()]
 candidates=rows(ROOT/'data/candidates.jsonl');ids=[x['meta_index'] for x in candidates];assert len(set(ids))==len(ids)==200 and sha(ROOT/'data/candidates.jsonl')==cfg['candidate_sha256']
 encode=candidate_encoder();state(RUN,'candidate_gradients')
 for i,row in enumerate(candidates):
  directory=RUN/f'features/c{i:03d}';marker=directory/'complete.json'
  if marker.exists():
   record=json.loads(marker.read_text());assert record['meta_index']==row['meta_index'] and (directory/'projection.npy').stat().st_size==record['bytes'];continue
  samples=encode(row);denom=sum(sum(x!=-100 for x in labels[1:]) for _,labels in samples);assert denom>0
  model.zero_grad(set_to_none=True)
  for tokens,labels in samples:backward(model,tokens,labels,1/denom,'sum')
  grads={n:x.grad for n,x in params.items()};norms=[]
  for n,g in grads.items():
   assert g is not None and torch.isfinite(g).all();norms.append(float((g*g).sum(dtype=torch.float64)))
  value=-projector.project(grads);assert np.isfinite(value).all();save_array(directory/'projection.npy',value)
  write(marker,{'meta_index':row['meta_index'],'raw_norm2':norms,'target_tokens':denom,'bytes':(directory/'projection.npy').stat().st_size,'finished_at':now()})
  write(RUN/'progress.json',{'stage':'candidate_gradients','done':i+1,'total':200,'updated_at':now()});print('CANDIDATE',i+1,flush=True)
 del model,params;projector.close();gc.collect();torch.cuda.empty_cache()
 # Dense features remain persisted per candidate for restart; no rank-c truncation.
 features=np.empty((200,dimension),dtype=np.float32);raw_norms=[]
 for i in range(200):
  d=RUN/f'features/c{i:03d}';features[i]=np.load(d/'projection.npy',allow_pickle=False);raw_norms.append(json.loads((d/'complete.json').read_text())['raw_norm2'])
 raw_norms=np.array(raw_norms);assert np.isfinite(features).all();scores=np.zeros((200,2),dtype=np.float64);diagnostics=[];state(RUN,'preconditioning')
 for k,e in enumerate(entries):
  trace=float(raw_norms[:,k].sum())
  if trace==0:continue
  dim=math.prod(e['target']);damping=p['damping_multiplier']*trace/dim
  z=torch.from_numpy(features[:,e['start']:e['stop']]).to('cuda',dtype=torch.float64);q=torch.from_numpy(queries[:,e['start']:e['stop']]).to('cuda',dtype=torch.float64)
  score,gram,cross,residual=dual_scores(z,q,damping);assert residual<1e-7 and torch.isfinite(score).all();scores+=score.cpu().numpy()
  path=RUN/f'blocks/{k:03d}.pt';path.parent.mkdir(exist_ok=True);tmp=path.with_suffix('.tmp');torch.save({'parameter':e['name'],'gram':gram.cpu(),'cross':cross.cpu(),'damping':damping,'candidate_ids':ids,'variants':variants},tmp);tmp.replace(path)
  diagnostics.append({'parameter':e['name'],'damping':damping,'residual':residual});write(RUN/'progress.json',{'stage':'preconditioning','done':k+1,'total':len(entries),'updated_at':now()})
  del z,q,score,gram,cross
 assert np.isfinite(scores).all();write(RUN/'scores.json',{'candidate_ids':ids,'variants':variants,'scores':scores.tolist(),'blocks':diagnostics})
 audits=materialize(cfg,ids,scores);write(RUN/'result.json',{'finished_at':now(),'selected':cfg['selected'],'audits':audits,'metrics':{'candidates':200,'selected_per_model':100,'projection_dimension':dimension,'maximum_solve_residual':max(x['residual'] for x in diagnostics)}})
 write(RUN/'complete.json',{'finished_at':now()});state(RUN,'complete','succeeded')
if __name__=='__main__':
 try:main()
 except BaseException as e:state(RUN,'failed','failed',error=str(e));raise
