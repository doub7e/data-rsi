"""Exact full-parameter reward-weighted original OTA training-loss gradients."""
import sys,gc,math,collections,time
from common import *
from loss_contract import runtime,forward_loss
P=json.loads((ROOT/'runs/easy-probes-20261105/protocol.json').read_text())
def layer(name):
 if name.startswith('model.layers.'):return 'layer-'+name.split('.')[2].zfill(2)
 if name.startswith('model.embed_tokens.'):return 'embedding'
 if name.startswith('lm_head.'):return 'lm_head'
 return 'final_norm_other'
def coefficients(blocks,score_sums,targets,counts,constant=False):
 import numpy as np
 v=np.zeros(8);n=sum(counts[i] for i in blocks);r=sum(score_sums[i] for i in blocks);denom=sum(targets[i] for i in blocks)
 if constant or not denom:return v
 for i in blocks:v[2*i]=r/n/denom;v[2*i+1]=-1/denom
 return v

def pair(gram,a,b):
 import numpy as np
 na=float(a@gram@a);nb=float(b@gram@b);dot=float(a@gram@b)
 assert na>=-1e-8 and nb>=-1e-8
 na=max(na,0.);nb=max(nb,0.);c=dot/math.sqrt(na*nb) if na*nb else None
 assert c is None or abs(c)<1.00001
 return {'norm_a':math.sqrt(na),'norm_b':math.sqrt(nb),'dot':dot,'cosine':None if c is None else max(-1.,min(1.,c))}

def unit_check():
 import numpy as np
 rng=np.random.default_rng(21);raw=rng.normal(size=(32,11));rewards=rng.choice([0.,.1,.5,.75,1.],32);tokens=rng.integers(10,3000,32);vectors=[];sums=[];totals=[]
 for b in range(4):
  x=raw[b*8:(b+1)*8];y=rewards[b*8:(b+1)*8];vectors.extend([x.sum(0),(x*y[:,None]).sum(0)]);sums.append(float(y.sum()));totals.append(int(tokens[b*8:(b+1)*8].sum()))
 matrix=np.array(vectors);gram=matrix@matrix.T
 for aa,bb in P['partitions']:
  a=coefficients(aa,sums,totals,[8]*4);b=coefficients(bb,sums,totals,[8]*4);out=pair(gram,a,b);direct=[]
  for blocks in [aa,bb]:
   ids=[i for block in blocks for i in range(block*8,(block+1)*8)];r=rewards[ids];direct.append(-((r-r.mean())[:,None]*raw[ids]).sum(0)/tokens[ids].sum())
  np.testing.assert_allclose(a@matrix,direct[0],atol=1e-14);np.testing.assert_allclose(b@matrix,direct[1],atol=1e-14);expected=direct[0]@direct[1]/np.linalg.norm(direct[0])/np.linalg.norm(direct[1]);assert abs(out['cosine']-expected)<1e-10
 assert not coefficients([0,1],[4,4,0,0],[100]*4,[8]*4,constant=True).any()
 # Reward offsets cancel in every half; positive scaling leaves cosine unchanged.
 offset=.3;scale=2.7
 shifted=np.array([v for b in range(4) for v in [matrix[2*b],matrix[2*b+1]+offset*matrix[2*b]]])
 for blocks in [[0,1],[0,2],[0,3]]:
  a=coefficients(blocks,sums,totals,[8]*4);z=coefficients(blocks,[x+8*offset for x in sums],totals,[8]*4);np.testing.assert_allclose(a@matrix,z@shifted,atol=1e-14)
 return {'fractional_score_gram_equals_direct_gradient':True,'constant_half_score_0_5_exact_zero':True,'reward_shift_invariance':True,'full_training_token_normalization':True}

def main(key):
 import torch,numpy as np
 q=next(q for q in P['tasks'] if q['key']==key);run=ROOT/f'runs/{key}-gradient-20261105';source=ROOT/f'runs/{key}-rollouts-20261105';assert (source/'complete.json').exists();assert (ROOT/P['loss']['requires_loss_parity']).exists();assert torch.cuda.device_count()==1 and 'H200' in torch.cuda.get_device_name(0);torch.set_num_threads(4)
 ready=json.loads((ROOT/'runs/easy-probes-20261105/gradient-implementation-ready.json').read_text())
 for path,digest in ready['source_sha256'].items():assert sha(ROOT/path)==digest,('source_changed',path)
 state(run,'load_training_loss_stack',started_at=now());write(run/'estimator-unit-check.json',unit_check());model,trainer,tok,encode=runtime(run);params=dict(model.named_parameters());layers=model.config.num_hidden_layers;results=json.loads((source/'result.json').read_text());rewards={i:float(r) for i,r in enumerate(results['scores'],1)}
 encoded={};audit=[];targets={};original_targets={}
 for rep in range(1,33):
  records=[json.loads(x) for x in (source/f'trials/r{rep:03d}/train.jsonl').read_text().splitlines() if x.strip()];encoded[rep]=[encode(row['conversations']) for row in records];targets[rep]=sum(x['target_tokens'] for x in encoded[rep]);original_targets[rep]=sum(x['original_targets'] for x in encoded[rep])
  write(run/f'encoded/r{rep:03d}.json',{'repeat':rep,'reward':rewards[rep],'rows':encoded[rep]});audit.append({'repeat':rep,'rows':len(records),'target_tokens':targets[rep],'original_targets':original_targets[rep],'zero_target_rows':sum(r['zero_targets_after_cutoff'] for r in encoded[rep]),'truncated_rows':sum(x['original_tokens']>len(x['input_ids']) for x in encoded[rep])})
 write(run/'encoding-audit.json',{'rollouts':audit,'loss_contract':json.loads((run/'loss-contract.json').read_text()),'no_silent_rollout_drops':len(encoded)==32});m=json.loads((run/'run.json').read_text());m['sources'].update(training_data=str((source/'data/train.jsonl').relative_to(ROOT)),trajectories=str((source/'jobs').relative_to(ROOT)));write(run/'run.json',m)
 score_sums=[sum(rewards[i] for i in b) for b in P['blocks']];block_targets=[sum(targets[i] for i in b) for b in P['blocks']];curve=[];prefix_a=None;individual=[];start=time.monotonic();constant=len(set(rewards.values()))==1;by_layer={}
 if not constant:
  accum=[];done=0;state(run,'training_loss_gradients')
  for block,assigned in enumerate(P['blocks']):
   sums=[{n:torch.zeros_like(p,device='cpu') for n,p in params.items()} for _ in range(2)];accum.append(sums)
   for pos,rep in enumerate(assigned,1):
    model.zero_grad(set_to_none=True)
    valid=[r for r in encoded[rep] if r['target_tokens']]
    ce_sum=0.;norm2=0.
    for row in valid:
     loss=forward_loss(model,trainer,row,1);ce_sum+=float(loss.detach());loss.backward();del loss
    if valid:
     for name,param in params.items():
      assert param.grad is not None and torch.isfinite(param.grad).all();norm2+=float((param.grad*param.grad).sum(dtype=torch.float64));g=param.grad.detach().cpu();sums[0][name].add_(g)
      if rewards[rep]:sums[1][name].add_(g,alpha=rewards[rep])
    individual.append({'repeat':rep,'reward':rewards[rep],'target_tokens':targets[rep],'ce_sum':ce_sum,'mean_token_nll':ce_sum/targets[rep] if targets[rep] else None,'gradient_norm_sum':math.sqrt(norm2),'gradient_norm_mean_tokens':math.sqrt(norm2)/targets[rep] if targets[rep] else None});write(run/'individual-gradient-stats.json',individual)
    if block in (0,2) and pos==4:
     selected=assigned[:4];rate=sum(rewards[i] for i in selected)/4;denom=sum(targets[i] for i in selected);current={n:torch.zeros_like(u) if len({rewards[i] for i in selected})==1 or not denom else (rate*u-sums[1][n])/denom for n,u in sums[0].items()}
     if block==0:prefix_a=current
     else:
      na=nb=dot=0.
      for n,a in prefix_a.items():
       av=a.reshape(-1);bv=current[n].reshape(-1)
       for off in range(0,len(av),1<<20):
        x=av[off:off+(1<<20)].to('cuda',dtype=torch.float64);y=bv[off:off+(1<<20)].to('cuda',dtype=torch.float64);na+=float(x@x);nb+=float(y@y);dot+=float(x@y)
      curve.append({'rollouts':8,'successes_a':sum(rewards[i]==1 for i in P['blocks'][0][:4]),'successes_b':sum(rewards[i]==1 for i in selected),'norm_a':math.sqrt(na),'norm_b':math.sqrt(nb),'dot':dot,'cosine':dot/math.sqrt(na*nb) if na*nb else None});prefix_a=None;del current
    done+=1;write(run/'progress.json',{'stage':'training_loss_gradients','done':done,'total':32,'updated_at':now()});print('GRADIENT',key,done,flush=True)
  del model,params,trainer;gc.collect();torch.cuda.empty_cache();state(run,'full_parameter_gram')
  by_layer=collections.defaultdict(lambda:np.zeros((8,8),dtype=np.float64));names=list(accum[0][0])
  for k,name in enumerate(names):
   vectors=[accum[b][j][name].reshape(-1) for b in range(4) for j in range(2)];gram=torch.zeros((8,8),device='cuda',dtype=torch.float64)
   for off in range(0,len(vectors[0]),1<<20):x=torch.stack([v[off:off+(1<<20)] for v in vectors]).to('cuda',dtype=torch.float64);gram+=x@x.T
   array=gram.cpu().numpy();by_layer['all']+=array;by_layer[layer(name)]+=array
   for b in range(4):
    for j in range(2):del accum[b][j][name]
   write(run/'progress.json',{'stage':'parameter_blocks','done':k+1,'total':len(names),'updated_at':now()})
 else:
  for n in ['all','embedding','lm_head','final_norm_other']+[f'layer-{i:02d}' for i in range(layers)]:by_layer[n]=np.zeros((8,8))
  curve.append({'rollouts':8,'successes_a':sum(rewards[i]==1 for i in P['blocks'][0][:4]),'successes_b':sum(rewards[i]==1 for i in P['blocks'][2][:4]),'norm_a':0.,'norm_b':0.,'dot':0.,'cosine':None});write(run/'progress.json',{'stage':'constant_reward_analytic_zero','done':32,'total':32,'updated_at':now()})
 splits=[]
 for a,b in P['partitions']:
  ca=coefficients(a,score_sums,block_targets,[8]*4,constant=len({rewards[i] for j in a for i in P['blocks'][j]})==1);cb=coefficients(b,score_sums,block_targets,[8]*4,constant=len({rewards[i] for j in b for i in P['blocks'][j]})==1);splits.append({'a':a,'b':b,'successes_a':sum(rewards[i]==1 for j in a for i in P['blocks'][j]),'successes_b':sum(rewards[i]==1 for j in b for i in P['blocks'][j]),'mean_score_a':sum(score_sums[j] for j in a)/(8*len(a)),'mean_score_b':sum(score_sums[j] for j in b)/(8*len(b)),'targets_a':sum(block_targets[i] for i in a),'targets_b':sum(block_targets[i] for i in b),'statistics':{n:pair(g,ca,cb) for n,g in by_layer.items()}})
 curve.append({'rollouts':32,**splits[0]['statistics']['all'],'successes_a':splits[0]['successes_a'],'successes_b':splits[0]['successes_b']});metrics={'mean_score':sum(rewards.values())/32,'score_variance':sum((x-sum(rewards.values())/32)**2 for x in rewards.values())/32,'partial_credit_trials':sum(0<x<1 for x in rewards.values()),'success_percent':100*sum(x==1 for x in rewards.values())/32,'rollouts':32,'sft_rows':sum(x['rows'] for x in audit),'retained_target_fraction':sum(targets.values())/max(sum(original_targets.values()),1)}
 for c in curve:
  if c['cosine'] is not None:metrics[f'split_half_cosine_{c["rollouts"]}']=c['cosine']
 result={'finished_at':now(),'key':key,'task':q['task'],'benchmark':q['benchmark'],'model':P['model'],'successes':sum(x==1 for x in rewards.values()),'score_sum':sum(rewards.values()),'mean_score':sum(rewards.values())/32,'probe_reward_definition':'official_score_v1','trials':32,'curve':curve,'partitions':splits,'layers':layers,'encoding':audit,'individual_gradients':individual,'all_rewards_constant':constant,'metrics':metrics,'loss_definition':P['loss'],'limitations':P['limitations'],'elapsed_seconds':time.monotonic()-start};write(run/'result.json',result);write(run/'block-gram.json',{'parameter_block_grams':None if constant else {n:g.tolist() for n,g in by_layer.items()},'raw_block_grams_computed':not constant,'analytic_centered_gradient_zero':constant,'score_sums':score_sums,'scores':[rewards[i] for i in range(1,33)],'vector_basis':'For each8-rollout block: U=sum gradCE_sum; W=sum score_i*gradCE_sum; not binary success sums.','valid_target_tokens':block_targets,'rollout_blocks':P['blocks']})
 from report import question_report,summary_report
 question_report(run,result);write(run/'complete.json',{'finished_at':now()});state(run,'complete','succeeded');summary_report()
if __name__=='__main__':
 try:main(sys.argv[1])
 except BaseException as e:state(ROOT/f'runs/{sys.argv[1]}-gradient-20261105','failed','failed',error=str(e),finished_at=now());raise
