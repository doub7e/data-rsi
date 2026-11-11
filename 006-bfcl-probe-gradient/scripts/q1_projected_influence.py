"""Base-model dense projected empirical-Fisher influence; RunAI-only, after dot study.

All candidate parameters included. No rank-c approximation. Exact 200x200 dual solve
per named parameter; projected features live in worker RAM, kernels and scores persist.
"""
import gc,math,time
from common import *
import numpy as np
import torch
from q1_projection_math import DenseProjection,extract_block,dual_scores,unit_checks
from q1_attribution_worker import materialize
RUN=ROOT/'runs/q1p-prepare-20261105';STUDY=ROOT/'runs/q1-pif-20261105'
CFG2=json.loads((STUDY/'protocol.json').read_text());P=CFG2['projection']

def persist_tensor(path,payload):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(path)

def freeze_sources():
 ready=json.loads((STUDY/'implementation-ready.json').read_text())
 for path,digest in ready['source_sha256'].items():assert sha(ROOT/path)==digest,'Implementation changed after ready marker: '+path
 assert (ROOT/CFG2['gate']).exists(),'Dot study must finish before influence computation'
 assert sha(ROOT/'data/candidates.jsonl')==CFG2['candidate_sha256']

def collect(width,round_dir,expected_params=4022468096,device="cuda"):
 from gradients import load_model,backward,candidate_encoder,test_gradient
 variants=CFG2['variants'];candidates=rows(ROOT/'data/candidates.jsonl');assert len(candidates)==200
 model=load_model(ROOT/'runs/probe-rollouts/model-64k');params=dict(model.named_parameters());specs=[(name,tuple(p.shape)) for name,p in params.items()]
 assert sum(math.prod(s) for n,s in specs)==expected_params
 projectors=[DenseProjection(specs,width,seed,stream_width=max(P['widths']),device=device) for seed in P['seeds']]
 features=[np.empty((206,p.dimension),dtype=np.float32) for p in projectors]
 write(round_dir/'layout.json',{'width':width,'dimension':projectors[0].dimension,'entries':projectors[0].entries,'seeds':P['seeds'],'dense_feature_bytes':sum(x.nbytes for x in features),'format':'dense FP32 in worker RAM, no rank-c; persisted Gram matrices and cross-products suffice to reproduce all scores'})
 sums=[{mode:{kind:np.zeros(pr.dimension,dtype=np.float32) for kind in ['all','success']} for mode in ['F','K']} for pr in projectors]
 accumulated={mode:{name:torch.zeros_like(param) for name,param in params.items()} for mode in ['F','K']}
 rewards=0;start=time.monotonic()
 for index,rep in enumerate(CFG2['rollout_order'],1):
  trace=json.loads((ROOT/CFG2['prepared_run']/f'prepared/r{rep:03d}.json').read_text());assert trace['id']==CFG2['probe_id']
  rewards+=trace['reward']
  for mode in accumulated:
   for value in accumulated[mode].values():value.zero_()
  for step in trace['steps']:
   assert step['prompt_token_replay_verified'];prompt=step['prompt_token_ids'];completion=step['completion_token_ids'];model.zero_grad(set_to_none=True)
   backward(model,prompt+completion,[-100]*len(prompt)+completion,-1.,'sum')
   with torch.no_grad():
    for name,param in params.items():
     assert param.grad is not None
     accumulated['F'][name].add_(param.grad)
     if step['turn_index']==CFG2['key_turn']:accumulated['K'][name].add_(param.grad)
  for j,pr in enumerate(projectors):
   for mode in ['F','K']:
    value=pr.project(accumulated[mode]);assert np.isfinite(value).all();sums[j][mode]['all']+=value
    if trace['reward']:sums[j][mode]['success']+=value
    if index in CFG2['sizes']:
     v=f'{mode.lower()}{index:03d}';features[j][200+variants.index(v)]=(sums[j][mode]['success']-(rewards/index)*sums[j][mode]['all'])/index
    del value
  write(round_dir/'progress.json',{'stage':'probe_gradients','done':index,'total':128,'elapsed_seconds':time.monotonic()-start,'updated_at':now()});print('PROBE',width,index,flush=True)
 del sums,accumulated,params,model;gc.collect();torch.cuda.empty_cache()
 model=load_model(MODEL);params=dict(model.named_parameters());assert specs==[(n,tuple(p.shape)) for n,p in params.items()];encode=candidate_encoder();raw_norms=np.zeros((200,len(specs)),dtype=np.float64)
 old={r['meta_index']:r for r in rows(ROOT/'runs/q1-scores-20261104/F-scores.jsonl')};norm_errors=[]
 for index,row in enumerate(candidates):
  samples=encode(row);denom=sum(sum(x!=-100 for x in labels[1:]) for ids,labels in samples);assert denom==old[row['meta_index']]['target_tokens']
  model.zero_grad(set_to_none=True)
  for ids,labels in samples:backward(model,ids,labels,1/denom,'sum')
  grads={name:param.grad for name,param in params.items()}
  for k,(name,_) in enumerate(specs):
   g=grads[name];assert g is not None and torch.isfinite(g).all();raw_norms[index,k]=float((g*g).sum(dtype=torch.float64))
  norm=math.sqrt(raw_norms[index].sum());relative=abs(norm/old[row['meta_index']]['gradient_norm']-1)
  repeat_audit={'meta_index':row['meta_index'],'first_norm':norm,'archived_norm':old[row['meta_index']]['gradient_norm'],'first_relative_norm_difference':relative,'repeated':False}
  if relative>=2e-4:
   # Empirically observed BF16 backward repeat variability. Do not confuse norm
   # equality with vector agreement; require a full-vector repeat audit here.
   reference={name:g.detach().cpu().clone() for name,g in grads.items()};grads.clear();model.zero_grad(set_to_none=True)
   for ids2,labels2 in samples:backward(model,ids2,labels2,1/denom,'sum')
   dot=refnorm=diffnorm=0.;grads={name:param.grad for name,param in params.items()}
   for k,(name,_) in enumerate(specs):
    g=grads[name];r=reference[name].to(device);raw_norms[index,k]=float((g*g).sum(dtype=torch.float64));refnorm+=float((r*r).sum(dtype=torch.float64));dot+=float((g*r).sum(dtype=torch.float64));delta=g-r;diffnorm+=float((delta*delta).sum(dtype=torch.float64));del r,delta
   repeated_norm=math.sqrt(raw_norms[index].sum());repeated_relative=abs(repeated_norm/old[row['meta_index']]['gradient_norm']-1);cosine=dot/math.sqrt(refnorm*repeated_norm**2);vector_error=math.sqrt(diffnorm/refnorm)
   repeat_audit.update(repeated=True,repeated_norm=repeated_norm,repeated_relative_norm_difference=repeated_relative,repeat_cosine=cosine,repeat_relative_vector_error=vector_error)
   write(round_dir/f'numerical-audits/{row["meta_index"]}.json',repeat_audit)
   assert max(relative,repeated_relative)<P['numerical_audit']['maximum_archived_norm_difference'] and cosine>=P['numerical_audit']['minimum_repeat_cosine'] and vector_error<=P['numerical_audit']['maximum_repeat_vector_error'],('candidate-gradient-repeat-mismatch',repeat_audit)
   relative=repeated_relative;del reference
  write(round_dir/f"numerical-audits/{row['meta_index']}.json",repeat_audit);norm_errors.append(relative)
  for j,pr in enumerate(projectors):features[j][index]=-pr.project(grads)
  write(round_dir/'progress.json',{'stage':'candidate_gradients','done':index+1,'total':200,'elapsed_seconds':time.monotonic()-start,'updated_at':now()});print('CANDIDATE',width,index+1,flush=True)
 write(round_dir/'candidate-gradient-check.json',{'max_relative_norm_difference':max(norm_errors),'model':'original base','candidate_ids':[x['meta_index'] for x in candidates]})
 del model,params,grads,g;gc.collect()
 for pr in projectors:pr.close()
 torch.cuda.empty_cache()
 return features,projectors[0].entries,raw_norms,[x['meta_index'] for x in candidates]

def score_features(features,entries,raw_norms,ids,width,round_dir,device="cuda"):
 results={};widths=[width//2,width]
 for seed_index,seed in enumerate(P['seeds']):
  for w in widths:
   directory=round_dir/f'seed-{seed}/width-{w}';directory.mkdir(parents=True,exist_ok=True)
   total=np.zeros((200,6),dtype=np.float64);direct=np.zeros_like(total);norms=np.zeros(200,dtype=np.float64);block_metrics=[]
   for k,e in enumerate(entries):
    block=extract_block(features[seed_index],e,w,width);tensor=torch.from_numpy(block).to(device,dtype=torch.float64);z=tensor[:200];q=tensor[200:]
    trace=float(raw_norms[:,k].sum());refdim=math.prod(min(x,P['damping_reference_width']) for x in e['shape']) if len(e['shape'])==2 else math.prod(e['shape'])
    if trace==0:
     assert float(z.abs().max())==0;continue
    damping=P['damping_multiplier']*trace/refdim
    s,gram,cross,res=dual_scores(z,q,damping);assert res<1e-7,(e['name'],res)
    total+=s.cpu().numpy();direct+=cross.cpu().numpy();norms+=gram.diag().cpu().numpy()
    persist_tensor(directory/f'block-{k:03d}.pt',{'parameter':e['name'],'gram':gram.cpu(),'cross':cross.cpu(),'damping':damping,'raw_trace':trace,'dimension':z.shape[1],'reference_dimension':refdim,'candidate_ids':ids,'variants':CFG2['variants']})
    block_metrics.append({'parameter':e['name'],'damping':damping,'residual':res,'dimension':z.shape[1]})
    del block,tensor,z,q,s,gram,cross
   result={'seed':seed,'width':w,'candidate_ids':ids,'variants':CFG2['variants'],'influence':total.tolist(),'projected_dot':direct.tolist(),'projected_norm2':norms.tolist(),'raw_norm2':raw_norms.sum(axis=1).tolist(),'blocks':block_metrics};write(directory/'result.json',result);results[(seed,w)]=result
 return results

def check_quality(results,ids,width):
 from scipy.stats import spearmanr
 source={m:{r['meta_index']:r for r in rows(ROOT/f'runs/q1-scores-20261104/{m}-scores.jsonl')} for m in ['F','K']}
 exact=np.array([[source[v[0].upper()][i]['scores'][str(int(v[1:]))]['dot'] for v in CFG2['variants']] for i in ids]);checks=[]
 def comparison(a,b,label,relative_limit):
  for j,v in enumerate(CFG2['variants']):
   x=np.array(a)[:,j];y=np.array(b)[:,j];xc=x-x.mean();yc=y-y.mean();relative=float(np.linalg.norm(xc-yc)/max(np.linalg.norm(yc),1e-30));rho=float(spearmanr(x,y).statistic)
   topx=set(sorted(range(200),key=lambda k:(-x[k],ids[k]))[:100]);topy=set(sorted(range(200),key=lambda k:(-y[k],ids[k]))[:100]);overlap=len(topx&topy)
   passed=math.isfinite(rho) and rho>=P['quality']['minimum_spearman'] and relative<=relative_limit and overlap>=P['quality']['minimum_top100_overlap']
   checks.append({'comparison':label,'variant':v,'spearman':rho,'centered_relative_l2':relative,'top100_overlap':overlap,'passed':passed})
 for seed in P['seeds']:
  big=results[(seed,width)];small=results[(seed,width//2)]
  comparison(big['projected_dot'],exact,f'seed{seed}: projected dot vs exact full-parameter dot',P['quality']['maximum_relative_l2'])
  comparison(small['influence'],big['influence'],f'seed{seed}: influence width{width//2} vs {width}',P['quality']['maximum_relative_l2'])
  normrel=np.abs(np.sqrt(np.array(big['projected_norm2'])/np.array(big['raw_norm2']))-1)
  checks.append({'comparison':f'seed{seed}: candidate norm preservation','median':float(np.median(normrel)),'p95':float(np.quantile(normrel,.95)),'passed':bool(np.median(normrel)<=.02 and np.quantile(normrel,.95)<=.05)})
 comparison(results[(P['seeds'][0],width)]['influence'],results[(P['seeds'][1],width)]['influence'],'independent projection seeds',P['quality']['maximum_relative_l2'])
 return {'width':width,'passed':all(c['passed'] for c in checks),'checks':checks,'interpretation':'Numerical projection fidelity and stability only; not proof of downstream attribution validity. No evaluation outcomes used to choose dimensions.'}

def main():
 freeze_sources();assert torch.cuda.device_count()==1 and 'H200' in torch.cuda.get_device_name(0)
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
 state(RUN,'numerical_preflight',started_at=now());write(RUN/'unit-checks.json',unit_checks())
 from gradients import test_gradient
 write(RUN/'gradient-unit-checks.json',test_gradient())
 accepted=None
 for width in P['widths']:
  round_dir=RUN/f'width-{width}';round_dir.mkdir(exist_ok=True);state(RUN,'collect_projected_gradients',width=width)
  features,entries,norms,ids=collect(width,round_dir);state(RUN,'exact_dual_curvature_and_quality',width=width)
  results=score_features(features,entries,norms,ids,width,round_dir);del features;gc.collect();torch.cuda.empty_cache()
  quality=check_quality(results,ids,width);write(round_dir/'quality.json',quality);print('QUALITY',width,quality['passed'],flush=True)
  if quality['passed']:
   accepted=(width,results[(P['seeds'][0],width)],quality);break
 if accepted is None:
  write(RUN/'quality-failure.json',{'at':now(),'reason':'Projection failed predeclared fidelity/stability checks through width1024. No training datasets materialized; do not silently lower thresholds.'});raise RuntimeError('Projected influence quality gate failed; preserve diagnostics and stop before training')
 width,result,quality=accepted;matrix=np.array(result['influence']);rankings={v:[{'meta_index':i,'influence':float(matrix[k,j]),'projection_seed':P['seeds'][0],'projection_width':width} for k,i in enumerate(ids)] for j,v in enumerate(CFG2['variants'])}
 audits=materialize(CFG2,rankings,'influence');write(RUN/'result.json',{'finished_at':now(),'accepted_width':width,'selected_seed':P['seeds'][0],'quality':quality,'audits':audits,'definition':P['score'],'limitations':P['limitations']});write(RUN/'complete.json',{'finished_at':now()});state(RUN,'complete','succeeded')
if __name__=='__main__':
 try:main()
 except BaseException as e:state(RUN,'failed','failed',error=str(e));raise
