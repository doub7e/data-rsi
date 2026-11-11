"""One H200, exact full-parameter block sums; persist only sufficient small Grams."""
import argparse,collections,gc,math,re,time
from common import *
from probe_diversity_math import coefficients,pair_from_gram,unit_check
STUDY=ROOT/'runs/probe-diversity-20261105';P=json.loads((STUDY/'protocol.json').read_text())
def block_name(name):
 if name.startswith('model.layers.'):return 'layer-'+name.split('.')[2].zfill(2)
 return 'embedding_lm_head' if name.startswith('model.embed_tokens.') else 'final_norm_other'

def main(key):
 import torch
 import numpy as np
 from gradients import load_model,backward,test_gradient
 ready=json.loads((STUDY/'gradient-implementation-ready.json').read_text())
 for path,digest in ready['source_sha256'].items():assert sha(ROOT/path)==digest,('source_changed',path)
 q=next(x for x in P['questions'] if x['key']==key);run=ROOT/f'runs/pd-{key}-gradient-20261105';source=ROOT/f'runs/pd-{key}-rollouts-20261105'
 assert (source/'complete.json').exists();assert torch.cuda.device_count()==1 and 'H200' in torch.cuda.get_device_name(0);torch.set_num_threads(4)
 state(run,'gradient_preflight',started_at=now());write(run/'unit-checks.json',{'estimator':unit_check(),'autograd':test_gradient()})
 traces={i:json.loads((source/f'traces/r{i:03d}.json').read_text()) for i in range(1,129)};assert all(x['id']==q['id'] and x['history_protocol']=='strict-v2' for x in traces.values())
 reward={i:t['reward'] for i,t in traces.items()};wins=[sum(reward[i] for i in b) for b in P['blocks']];curve={};prefix_a={};layers={};tokens={i:sum(len(s['completion_token_ids']) for s in t['steps']) for i,t in traces.items()};start=time.monotonic()
 counts=[32]*4;all_constant=sum(wins) in (0,128)
 if not all_constant:
  model=load_model(ROOT/P['model_view']);params=dict(model.named_parameters());assert sum(p.numel() for p in params.values())==4022468096
  # CPU accumulators keep full parameter vectors, not sketches or rank-c factors.
  accum=[];completed=0
  state(run,'full_parameter_gradients')
  for block,assigned in enumerate(P['blocks']):
   sums=[{name:torch.zeros_like(p,device='cpu') for name,p in params.items()} for _ in range(2)];accum.append(sums)
   for position,rep in enumerate(assigned,1):
    model.zero_grad(set_to_none=True);trace=traces[rep]
    for step in trace['steps']:
     assert step['prompt_token_replay_verified'];p,c=step['prompt_token_ids'],step['completion_token_ids'];backward(model,p+c,[-100]*len(p)+c,-1.,'sum')
    for name,param in params.items():
     assert param.grad is not None and torch.isfinite(param.grad).all();g=param.grad.detach().cpu();sums[0][name].add_(g)
     if reward[rep]:sums[1][name].add_(g)
    if block in (0,2) and position in [4,16]:
     rate=sum(reward[i] for i in assigned[:position])/position
     current={name:(torch.zeros_like(u) if rate in (0,1) else (sums[1][name]-rate*u)/position) for name,u in sums[0].items()}
     if block==0:prefix_a[position]=current
     else:
      na=nb=dot=0.
      for name,a in prefix_a[position].items():
       av=a.reshape(-1);bv=current[name].reshape(-1)
       for offset in range(0,len(av),1<<20):
        x=av[offset:offset+(1<<20)].to('cuda',dtype=torch.float64);y=bv[offset:offset+(1<<20)].to('cuda',dtype=torch.float64);na+=float(x@x);nb+=float(y@y);dot+=float(x@y)
      curve[position]={'total_rollouts':2*position,'per_half':position,'successes_a':sum(reward[i] for i in P['blocks'][0][:position]),'successes_b':sum(reward[i] for i in assigned[:position]),'norm_a':math.sqrt(na),'norm_b':math.sqrt(nb),'dot':dot,'cosine':dot/math.sqrt(na*nb) if na*nb else None};del prefix_a[position],current
      write(run/'partial-curve.json',[curve[n] for n in sorted(curve)])
    completed+=1;write(run/'progress.json',{'stage':'full_parameter_gradients','done':completed,'total':128,'updated_at':now(),'elapsed_seconds':time.monotonic()-start});print('GRADIENT',key,completed,flush=True)
  del model,params,param,g;gc.collect();torch.cuda.empty_cache();state(run,'exact_scalar_gram_comparison')
  layers=collections.defaultdict(lambda:np.zeros((8,8),dtype=np.float64))
  names=list(accum[0][0]);write(run/'progress.json',{'stage':'parameter_block_gram','done':0,'total':len(names),'updated_at':now()})
  for k,name in enumerate(names):
   vectors=[accum[b][kind][name].reshape(-1) for b in range(4) for kind in range(2)];gram=torch.zeros((8,8),device='cuda',dtype=torch.float64)
   for offset in range(0,len(vectors[0]),1<<20):
    matrix=torch.stack([v[offset:offset+(1<<20)] for v in vectors]).to('cuda',dtype=torch.float64);gram+=matrix@matrix.T
   value=gram.cpu().numpy();layers['all']+=value;layers[block_name(name)]+=value
   for b in range(4):
    for kind in range(2):del accum[b][kind][name]
   write(run/'progress.json',{'stage':'parameter_block_gram','done':k+1,'total':len(names),'updated_at':now()})
  del accum;gc.collect()
 else:
  for name in ['all','embedding_lm_head','final_norm_other']+[f'layer-{i:02d}' for i in range(36)]:layers[name]=np.zeros((8,8))
  for n in [4,16]:curve[n]={'total_rollouts':2*n,'per_half':n,'successes_a':sum(reward[i] for i in P['blocks'][0][:n]),'successes_b':sum(reward[i] for i in P['blocks'][2][:n]),'norm_a':0.,'norm_b':0.,'dot':0.,'cosine':None}
 partitions=[]
 for a,b in P['partitions']:
  ca=coefficients(a,wins,counts);cb=coefficients(b,wins,counts);stats={name:pair_from_gram(gram,ca,cb) for name,gram in layers.items()};partitions.append({'blocks_a':a,'blocks_b':b,'successes_a':sum(wins[i] for i in a),'successes_b':sum(wins[i] for i in b),'statistics':stats})
 mainpair=partitions[0];curve[64]={'total_rollouts':128,'per_half':64,'successes_a':mainpair['successes_a'],'successes_b':mainpair['successes_b'],**mainpair['statistics']['all']}
 failures=collections.Counter();failure_turns=collections.Counter()
 for rep in range(1,129):
  score=rows(source/f'bfcl/r{rep:03d}/score/Qwen_Qwen3-4B-FC/multi_turn/BFCL_v4_multi_turn_{q["category"]}_score.json')
  assert int(score[0]['correct_count'])==reward[rep]
  if not reward[rep]:
   error=score[1].get('error',{}) if len(score)>1 else {};failures[str(error.get('error_type','unknown'))]+=1;match=re.search(r'\bturn\s*(\d+)\b',str(error.get('error_message','')),re.I);failure_turns[match.group(1) if match else 'unspecified']+=1
 question=json.loads((source/'question.json').read_text());examples={str(y):next((i for i in range(1,129) if reward[i]==y),None) for y in [0,1]}
 metrics={'probe_success_percent':100*sum(wins)/128,'rollouts':128}
 for n,x in curve.items():
  if x['cosine'] is not None:metrics[f'split_half_cosine_{2*n}']=x['cosine']
 result={'finished_at':now(),'probe_id':q['id'],'key':key,'category':q['category'],'successes':sum(wins),'trials':128,'curve':[curve[n] for n in sorted(curve)],'partitions':partitions,'all_rewards_constant':all_constant,'first_failure_types':dict(failures),'first_failure_turns':dict(failure_turns),'representative_repeat_by_reward':examples,'question_user_turns':question.get('question'),'question_classes':question.get('involved_classes'),'output_tokens':{'mean':float(np.mean(list(tokens.values()))),'median':float(np.median(list(tokens.values()))),'min':min(tokens.values()),'max':max(tokens.values())},'length_capped_calls':sum(s['finish_reason']=='length' for t in traces.values() for s in t['steps']),'elapsed_seconds':time.monotonic()-start,'metrics':metrics,'limitations':P['limitations']}
 write(run/'block-gram.json',{'block_rollouts':P['blocks'],'successes':wins,'counts':counts,'gram_by_layer':{name:gram.tolist() for name,gram in layers.items()},'precision':'Full FP32 parameter-gradient block sums; FP64 chunked Gram, no projection'})
 write(run/'result.json',result)
 from probe_diversity_report import question_report,summary_report
 question_report(run,result);write(run/'complete.json',{'finished_at':now()});state(run,'complete','succeeded');summary_report()
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('key');args=parser.parse_args()
 try:main(args.key)
 except BaseException as e:state(ROOT/f'runs/pd-{args.key}-gradient-20261105','failed','failed',error=str(e),finished_at=now());raise
