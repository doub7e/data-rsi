"""Single clean probe: full candidate200 scores plus balanced-half stability."""
import argparse,collections,fcntl,gc,hashlib,json,math,os,re,statistics,subprocess,sys,time
from pathlib import Path
from common import *
RUN=ROOT/'runs/single-probe194-20261103';QID='multi_turn_miss_func_194'
BLOCK=re.compile(r'<\|im_start\|>([^\n]+)\n(.*?)(?:<\|im_end\|>|$)',re.S)
THINK=re.compile(r'<think>(.*?)</think>',re.S)
def group(name):
 if name.startswith('model.embed_tokens'):return 'embedding_lm_head'
 if '.self_attn.' in name:return 'attention'
 if '.mlp.' in name:return 'mlp'
 return 'norm_other'
def prepare():
 selected=[]
 with (ROOT/'runs/probe-rollouts/rollouts.jsonl').open() as f:
  for line in f:
   r=json.loads(line)
   if r['id']==QID:selected.append(r)
 assert len(selected)==8 and sum(r['reward'] for r in selected)==4
 for r in selected:
  assert r['advantage']==r['reward']-.5
  for s in r['steps']:
   assert not any(role=='assistant' and (any(m.group(1).strip() for m in THINK.finditer(body)) or '<think>' in THINK.sub('',body)) for role,body in BLOCK.findall(s['formatted_prompt']))
 good=sorted(r['repeat'] for r in selected if r['reward']);bad=sorted(r['repeat'] for r in selected if not r['reward'])
 halves={'a':sorted(good[:2]+bad[:2]),'b':sorted(good[2:]+bad[2:])}
 frozen={'probe_id':QID,'selection_rule':'Unique clean mixed-outcome probe with success_rate0.5; selected before candidate scoring, not a random representative of all probes.',
  'halves':halves,'primary':'Mean of the two balanced4-rollout probe gradients, equivalent to all8 estimator.','candidate_count':200,
  'notes':['Reuse the existing clean recorded rollouts; no new sampling.','Half split is deliberately outcome-balanced; stability diagnostic, not independent training replications.'],
  'created_at':now()}
 if (RUN/'selection.json').exists():assert json.loads((RUN/'selection.json').read_text())['halves']==halves
 else:write(RUN/'selection.json',frozen)
 (RUN/'probe-rollouts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in selected))
 for r in rows(ROOT/'runs/probe-rollouts/data/BFCL_v4_multi_turn_miss_func.json'):
  if r['id']==QID:write(RUN/'probe-task.json',r)
 candidates=rows(ROOT/'data/candidates.jsonl');assert len(candidates)==200
 wanted={r['meta_index'] for r in candidates}
 # Full task descriptions live in metadata, not in the reduced selection pool.
 source=json.loads((BASE/'cache/envscaler/envscaler_sft_traj_9k_metadata.json').read_text())
 metadata={str(i):{'task_info':m['task_info'],'traj_type':m['traj_type']} for i,m in enumerate(source) if i in wanted};del source
 write(RUN/'candidate-metadata.json',metadata)
 probe_summary=[]
 for r in selected:
  rep=r['repeat'];base=ROOT/f'runs/probe-rollouts/bfcl/p{rep:02d}';failure=None;raw=None
  for item in rows(base/'score/Qwen_Qwen3-4B-FC/multi_turn/BFCL_v4_multi_turn_miss_func_score.json'):
   if item.get('id')==QID:failure={k:v for k,v in item.items() if k not in ['inference_log','prompt','model_result_raw','possible_answer']}
  for item in rows(base/'Qwen_Qwen3-4B-FC/multi_turn/BFCL_v4_multi_turn_miss_func_result.json'):
   if item['id']==QID:raw=item;break
  write(RUN/f'probe-raw/rep{rep}.json',raw)
  probe_summary.append({'repeat':rep,'reward':r['reward'],'calls':len(r['steps']),'output_tokens':sum(len(s['completion_token_ids']) for s in r['steps']),
   'failure':failure})
 write(RUN/'probe-outcomes.json',probe_summary)
 return halves

def ensure_prepared():
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/"preparation.lock").open("a") as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if not (RUN/"preparation-complete.json").exists():
   prepare()
   write(RUN/"preparation-complete.json",{"finished_at":now()})

def worker(rank):
 import torch
 from safetensors import safe_open
 from gradients import load_model,candidate_encoder,backward,save_probe,test_gradient
 torch.set_num_threads(4);assert torch.cuda.device_count()==2;torch.cuda.set_device(rank)
 write(RUN/f'worker{rank}-unit-check.json',test_gradient())
 if rank==0 and not (RUN/'probe-gradient/complete.json').exists():
  model=load_model(ROOT/'runs/probe-rollouts/model-64k')
  traces=rows(RUN/'probe-rollouts.jsonl');halves=json.loads((RUN/'selection.json').read_text())['halves']
  for key,reps in halves.items():
   model.zero_grad(set_to_none=True)
   for r in traces:
    if r['repeat'] not in reps:continue
    for s in r['steps']:
     p,c=s['prompt_token_ids'],s['completion_token_ids']
     backward(model,p+c,[-100]*len(p)+c,-r['advantage']/4,'sum')
   save_probe(model,RUN/f'probe-gradient/{key}')
  write(RUN/'probe-gradient/complete.json',{'finished_at':now()})
  del model;gc.collect();torch.cuda.empty_cache()
 while not (RUN/'probe-gradient/complete.json').exists():time.sleep(3)
 model=load_model(MODEL);encode=candidate_encoder()
 probes=[]
 for key in ('a','b'):
  tensors={};directory=RUN/f'probe-gradient/{key}';manifest=json.loads((directory/'manifest.json').read_text())
  assert manifest['parameters']==4022468096
  for item in manifest['files']:
   with safe_open(str(directory/item['file']),framework='pt',device='cpu') as f:
    for name in f.keys():tensors[name]=f.get_tensor(name).to('cuda')
  probes.append(tensors)
 a,b=probes;pnorms={key:0.0 for key in ('a','b','full')};parts={k:0.0 for k in ('embedding_lm_head','attention','mlp','norm_other')};cross=0.
 for name,p in model.named_parameters():
  aa,bb=a[name],b[name];full=(aa+bb)*.5
  pnorms['a']+=float((aa*aa).sum(dtype=torch.float64));pnorms['b']+=float((bb*bb).sum(dtype=torch.float64));pnorms['full']+=float((full*full).sum(dtype=torch.float64));cross+=float((aa*bb).sum(dtype=torch.float64));parts[group(name)]+=float((full*full).sum(dtype=torch.float64))
 pnorms={k:math.sqrt(v) for k,v in pnorms.items()};assert all(v>0 for v in pnorms.values())
 write(RUN/f'worker{rank}-probe-stats.json',{'norms':pnorms,'half_cosine':cross/(pnorms['a']*pnorms['b']),'squared_norm_by_group':parts})
 candidates=rows(ROOT/'data/candidates.jsonl')[rank::2];out=RUN/f'worker{rank}-scores.jsonl';seen={x['meta_index'] for x in rows(out)} if out.exists() else set();start=time.monotonic()
 for i,row in enumerate(candidates):
  if row['meta_index'] in seen:continue
  samples=encode(row);denom=sum(sum(v!=-100 for v in y[1:]) for _,y in samples);model.zero_grad(set_to_none=True)
  for ids,labels in samples:backward(model,ids,labels,1/denom,'sum')
  norm2=da=db=0.;gnorm={k:0. for k in parts};gdot={k:0. for k in parts}
  for name,p in model.named_parameters():
   assert p.grad is not None;v=p.grad;assert torch.isfinite(v).all()
   n=float((v*v).sum(dtype=torch.float64));d1=-float((v*a[name]).sum(dtype=torch.float64));d2=-float((v*b[name]).sum(dtype=torch.float64))
   norm2+=n;da+=d1;db+=d2;gnorm[group(name)]+=n;gdot[group(name)]+=(d1+d2)*.5
  norm=math.sqrt(norm2);dot=(da+db)*.5
  entry={'meta_index':row['meta_index'],'pool_index':row['pool_index'],'env_id':row['env_id'],'traj_type':row['traj_type'],'steps':row['steps'],
   'cosine':dot/(norm*pnorms['full']),'cosine_half_a':da/(norm*pnorms['a']),'cosine_half_b':db/(norm*pnorms['b']),
   'dot':dot,'gradient_norm':norm,'probe_norm':pnorms['full'],'target_tokens':denom,'input_tokens':sum(len(x) for x,_ in samples),
   'parameter_groups':{k:{'dot':gdot[k],'candidate_norm':math.sqrt(gnorm[k]),'probe_norm':math.sqrt(parts[k]),
     'cosine':gdot[k]/math.sqrt(gnorm[k]*parts[k]) if gnorm[k]*parts[k] else None} for k in parts}}
  assert all(math.isfinite(entry[k]) for k in ['cosine','cosine_half_a','cosine_half_b'])
  with out.open('a') as f:f.write(json.dumps(entry)+'\n')
  write(RUN/f'worker{rank}-progress.json',{'done':i+1,'total':len(candidates),'elapsed_seconds':time.monotonic()-start});print(json.dumps({k:entry[k] for k in ['meta_index','cosine','cosine_half_a','cosine_half_b']}),flush=True)
 write(RUN/f'worker{rank}-complete.json',{'finished_at':now()})

def analyze():
 import numpy as np
 from scipy.stats import spearmanr
 scores=[x for rank in (0,1) for x in rows(RUN/f'worker{rank}-scores.jsonl')];assert len(scores)==200 and len({x['meta_index'] for x in scores})==200
 scores.sort(key=lambda x:(-x['cosine'],x['meta_index']));old={r['meta_index']:r for r in rows(ROOT/'runs/gradient-candidates-0/scores.jsonl')+rows(ROOT/'runs/gradient-candidates-1/scores.jsonl')}
 candidates={r['meta_index']:r for r in rows(ROOT/'data/candidates.jsonl')};metadata=json.loads((RUN/'candidate-metadata.json').read_text());catalog=[]
 for rank,s in enumerate(scores,1):
  r=candidates[s['meta_index']];assistant=[m['content'] for m in r['messages'] if m['role']=='assistant'];visible=[re.sub(r'<think>.*?</think>','',t,flags=re.S).strip() for t in assistant];s['tool_call_reply_fraction']=sum('<tool_call>' in t for t in visible)/len(visible);s['reasoning_character_fraction']=sum(len(m.group(0)) for t in assistant for m in re.finditer(r'<think>.*?</think>',t,re.S))/sum(map(len,assistant));s['rank']=rank;s['aggregate_original_cosine']=old[s['meta_index']]['cosine'];s['mean_turn_nll']=old[s['meta_index']]['mean_turn_nll']
  system=r['messages'][0]['content'];intro=system.split('## Brief Introduction:',1)[-1].split('## Environment Rules',1)[0].split('# Tools',1)[0].strip()
  task=metadata[str(s['meta_index'])]['task_info'].get('task','')
  catalog.append({'rank':rank,'meta_index':s['meta_index'],'env_id':s['env_id'],'cosine':s['cosine'],'half_a':s['cosine_half_a'],'half_b':s['cosine_half_b'],'steps':s['steps'],'intro':intro[:900],'task':task})
 write(RUN/'scores.json',scores);write(RUN/'catalog.json',catalog)
 write(RUN/'selected-detail.json',[dict(catalog[i],messages=candidates[catalog[i]['meta_index']]['messages'],parameter_groups=scores[i]['parameter_groups']) for i in [0,1,2,3,4,95,99,100,104,195,196,197,198,199]])
 (RUN/'ranked-trajectories.jsonl').write_text(''.join(json.dumps(dict(candidates[s['meta_index']],single_probe_rank=s['rank'],single_probe_cosine=s['cosine']),ensure_ascii=False)+'\n' for s in scores))
 a=sorted(scores,key=lambda r:-r['cosine_half_a']);b=sorted(scores,key=lambda r:-r['cosine_half_b']);stats=json.loads((RUN/'worker0-probe-stats.json').read_text())
 s1=json.loads((RUN/'worker1-probe-stats.json').read_text());assert math.isclose(stats['norms']['full'],s1['norms']['full'],rel_tol=1e-6)
 result={'finished_at':now(),'probe_id':QID,'candidate_count':200,'score_min':min(s['cosine'] for s in scores),'score_max':max(s['cosine'] for s in scores),'positive_scores':sum(s['cosine']>0 for s in scores),
  'probe_gradient':stats,'half_rank_spearman':float(spearmanr([s['cosine_half_a'] for s in scores],[s['cosine_half_b'] for s in scores]).statistic),
  'half_top20_overlap':len({s['meta_index'] for s in a[:20]}&{s['meta_index'] for s in b[:20]}),
  'half_top100_overlap':len({s['meta_index'] for s in a[:100]}&{s['meta_index'] for s in b[:100]}),
  'correlations':{k:float(spearmanr([s['cosine'] for s in scores],[s[k] for s in scores]).statistic) for k in ['steps','input_tokens','target_tokens','gradient_norm','mean_turn_nll','tool_call_reply_fraction','reasoning_character_fraction','aggregate_original_cosine']}}
 result['metrics']={k:result[k] for k in ['candidate_count','positive_scores','half_rank_spearman','half_top20_overlap','half_top100_overlap']};write(RUN/'result.json',result)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,ax=plt.subplots(1,3,figsize=(14,4));ax[0].hist([s['cosine'] for s in scores],bins=25);ax[0].set_xlabel('Single-probe cosine');ax[0].set_ylabel('Trajectories')
 ax[1].scatter([s['cosine_half_a'] for s in scores],[s['cosine_half_b'] for s in scores],s=12,alpha=.65);ax[1].set_xlabel('Balanced half A cosine');ax[1].set_ylabel('Balanced half B cosine');ax[1].set_title(f"Rank correlation {result['half_rank_spearman']:.3f}")
 ax[2].scatter([s['target_tokens'] for s in scores],[s['cosine'] for s in scores],s=12,alpha=.65);ax[2].set_xlabel('Supervised target tokens');ax[2].set_ylabel('Single-probe cosine');fig.tight_layout();fig.savefig(RUN/'analysis.png',dpi=160);fig.savefig(RUN/'analysis.pdf');plt.close(fig)
 state(RUN,'complete','succeeded');write(RUN/'complete.json',{'finished_at':now()});print(json.dumps(result),flush=True)

def main():
 state(RUN,'prepare',started_at=now());ensure_prepared();children=[];handles=[]
 try:
  for rank in (0,1):
   if (RUN/f'worker{rank}-complete.json').exists():continue
   f=(RUN/f'worker{rank}.log').open('a');handles.append(f)
   children.append(subprocess.Popen([sys.executable,__file__,'--worker',str(rank)],stdout=f,stderr=subprocess.STDOUT))
  state(RUN,'single_probe_gradient_and_candidate_scores')
  while any(p.poll() is None for p in children):
   if any(p.poll() not in (None,0) for p in children):raise RuntimeError('A worker failed; inspect worker logs')
   time.sleep(10)
  assert all(p.returncode==0 for p in children)
  state(RUN,'analysis');analyze()
 finally:
  for p in children:
   if p.poll() is None:p.terminate()
  for p in children:
   try:p.wait(timeout=20)
   except subprocess.TimeoutExpired:p.kill();p.wait()
  for f in handles:f.close()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--worker',type=int,choices=[0,1]);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
 try:
  if a.prepare_only:ensure_prepared()
  elif a.worker is None:main()
  else:worker(a.worker)
 except BaseException as e:
  if a.worker is None:state(RUN,'failed','failed',error=str(e))
  raise
