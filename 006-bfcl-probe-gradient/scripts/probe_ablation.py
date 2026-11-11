"""Paired full-parameter token/turn ablations, compared in GPU memory."""
import argparse,collections,gc,json,math,os,sys,time
from pathlib import Path
from common import *
MODES=['all_tokens','no_reasoning','key_turn_all','key_turn_no_reasoning']
def visible_mask(ids):
 mask=[];first_open=ids.index(151667) if 151667 in ids else len(ids);first_close=ids.index(151668) if 151668 in ids else len(ids)
 inside=first_close<first_open
 for token in ids:
  if token==151667:inside=True;mask.append(False)
  elif token==151668:mask.append(False);inside=False
  else:mask.append(not inside)
 return mask

def verify_masked_loss():
 import torch
 from transformers import Qwen3Config,Qwen3ForCausalLM
 from gradients import target_loss
 torch.manual_seed(2194)
 cfg=Qwen3Config(vocab_size=47,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8,tie_word_embeddings=True,attention_dropout=0.)
 model=Qwen3ForCausalLM(cfg).float().eval();ids=torch.randint(0,47,(1,20));labels=torch.full_like(ids,-100)
 for pos in [8,10,11,14,16,19]:labels[:,pos]=ids[:,pos]
 expected=model(input_ids=ids,labels=labels,use_cache=False).loss*6;expected.backward();reference={n:p.grad.clone() for n,p in model.named_parameters()};model.zero_grad(set_to_none=True)
 actual,count=target_loss(model,ids,labels,'sum');actual.backward();error=max(float((p.grad-reference[n]).abs().max()) for n,p in model.named_parameters())
 assert count==6 and abs(float(actual-expected))<2e-5 and error<2e-6
 return {'noncontiguous_targets':count,'loss_abs_difference':abs(float(actual-expected)),'gradient_max_abs_difference':error}

def prepare_inputs(run,cfg):
 source=ROOT/cfg['source_run'];records=[];audit={'rollouts':0,'calls':0,'zero_target_calls':collections.Counter(),'target_tokens':collections.Counter(),'key_turn':cfg['key_turn']}
 for rep in range(1,129):
  trace=json.loads((source/f'traces/r{rep:03d}.json').read_text());raw=rows(source/f'bfcl/r{rep:03d}/Qwen_Qwen3-4B-FC/multi_turn/BFCL_v4_multi_turn_{cfg["category"]}_result.json')[0]
  turn=-1;mapping=[]
  for item in raw['inference_log']:
   if not isinstance(item,dict):continue
   if 'begin_of_turn_query' in item:turn+=1
   for name,events in item.items():
    if not name.startswith('step_'):continue
    capture=[e['content'] for e in events if e.get('role')=='inference_input'];assert len(capture)==1;mapping.append((turn,capture[0]))
  assert len(mapping)==len(trace['steps']) and trace['id']==cfg['probe_id']
  for s,(turn,logged) in zip(trace['steps'],mapping):
   assert s['prompt_token_ids']==logged['prompt_token_ids'] and s['completion_token_ids']==logged['completion_token_ids'];s['turn_index']=turn
   vis=visible_mask(s['completion_token_ids']);key=turn==cfg['key_turn'];counts={'all_tokens':len(vis),'no_reasoning':sum(vis),'key_turn_all':len(vis) if key else 0,'key_turn_no_reasoning':sum(vis) if key else 0}
   for mode,n in counts.items():audit['target_tokens'][mode]+=n;audit['zero_target_calls'][mode]+=n==0
   audit['calls']+=1
  write(run/f'prepared/r{rep:03d}.json',trace);audit['rollouts']+=1
  records.append({'repeat':rep,'reward':trace['reward'],'calls':len(trace['steps']),'key_turn_calls':sum(s['turn_index']==cfg['key_turn'] for s in trace['steps'])})
 audit['records']=records;audit['target_tokens']=dict(audit['target_tokens']);audit['zero_target_calls']=dict(audit['zero_target_calls']);write(run/'mask-audit.json',audit)
 from probe_followup_artifacts import export_rollouts
 export_rollouts(source,cfg['category'],cfg['probe_id'])
 m=json.loads((run/'run.json').read_text());m['sources'].update(trajectories=str((source/'jobs').relative_to(ROOT)),evaluation_plan=str((source/'evaluation-plan.json').relative_to(ROOT)));write(run/'run.json',m)
 write(run/'prepared-complete.json',{'finished_at':now()})

def labels_for(s,mode,key_turn):
 c=s['completion_token_ids'];keep=[True]*len(c) if mode in ('all_tokens','key_turn_all') else visible_mask(c)
 if mode.startswith('key_turn') and s['turn_index']!=key_turn:keep=[False]*len(c)
 return [-100]*len(s['prompt_token_ids'])+[t if k else -100 for t,k in zip(c,keep)],sum(keep)

def block(name):
 if name.startswith('model.layers.'):return 'layer-'+name.split('.')[2].zfill(2)
 if name.startswith('model.embed_tokens'):return 'embedding_lm_head'
 return 'final_norm_other'
def add_stats(sums,key,a,b):
 import torch
 q=sums.setdefault(key,[0.,0.,0.]);q[0]+=float((a*a).sum(dtype=torch.float64));q[1]+=float((b*b).sum(dtype=torch.float64));q[2]+=float((a*b).sum(dtype=torch.float64))
def finalize(q):
 a,b,d=q;return {'norm_a':math.sqrt(a),'norm_b':math.sqrt(b),'dot':d,'cosine':d/math.sqrt(a*b) if a*b else None}

def main(run_name):
 import torch,torch.distributed as dist
 from datetime import timedelta
 from safetensors import safe_open
 from gradients import load_model,backward,test_gradient
 rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.set_num_threads(4)
 dist.init_process_group('nccl',timeout=timedelta(hours=2));run=ROOT/'runs'/run_name;cfg=json.loads((run/'protocol.json').read_text());half='ab'[rank];source=ROOT/cfg['source_run'];partition=json.loads((source/'protocol.json').read_text())['halves'];assigned=partition[half];assert len(assigned)==64
 if rank==0:
  state(run,'preparing_masks',started_at=now());prepare_inputs(run,cfg)
 dist.barrier()
 trials=[json.loads((run/f'prepared/r{rep:03d}.json').read_text()) for rep in assigned];mean=sum(t['reward'] for t in trials)/64
 write(run/f'rank-{rank}-unit-check.json',{'standard':test_gradient(),'masked_loss':verify_masked_loss()})
 assert visible_mask([151667,7,151668,8,9])==[False,False,False,True,True]
 assert visible_mask([151667,7,8])==[False,False,False]
 assert visible_mask([7,151668,8])==[False,False,True]
 assert visible_mask([7,8])==[True,True]
 model=load_model(ROOT/'runs/probe-rollouts/model-64k');names=[n for n,p in model.named_parameters()]
 modes=MODES[1:] if cfg['reuse_full_gradient'] else MODES
 accum={mode:{n:torch.zeros_like(p,device='cpu' if mode=='all_tokens' else 'cuda') for n,p in model.named_parameters()} for mode in modes}
 progress=[];start=time.monotonic()
 if rank==0:state(run,'gradient_ablations')
 for index,t in enumerate(trials,1):
  weight=(t['reward']-mean)/64;counts={};zero={};likelihood={}
  for mode in modes:
   model.zero_grad(set_to_none=True);target_count=0;calls=0;compared_tokens=0;hf_nll=vllm_nll=0.
   for step in t['steps']:
    labels,n=labels_for(step,mode,cfg['key_turn'])
    if not n:continue
    target_count+=n;calls+=1
    if weight:
     loss,_=backward(model,step['prompt_token_ids']+step['completion_token_ids'],labels,-1.,'sum')
     if step.get('generation_logprobs_aligned'):
      c=step['completion_token_ids'];selected=labels[-len(c):];hf_nll+=loss;vllm_nll-=sum(lp for lp,label in zip(step['generation_logprobs'],selected) if label!=-100);compared_tokens+=n
   if weight and target_count:
    for name,p in model.named_parameters():
     assert p.grad is not None and torch.isfinite(p.grad).all();g=p.grad.detach()
     if mode=='all_tokens':g=g.cpu()
     accum[mode][name].add_(g,alpha=weight)
   counts[mode]=target_count;zero[mode]=target_count==0;likelihood[mode]={'compared_tokens':compared_tokens,'hf_nll':hf_nll,'vllm_actual_decode_nll':vllm_nll}
  progress.append({'repeat':t['repeat'],'reward':t['reward'],'target_tokens':counts,'zero_target_trajectory':zero,'likelihood_comparison':likelihood})
  write(run/f'rank-{rank}-progress.json',{'done':index,'total':64,'reward_mean':mean,'elapsed_seconds':time.monotonic()-start,'updated_at':now()})
  if index%8==0:print('PROGRESS',half,index,flush=True)
 write(run/f'rank-{rank}-trajectory-counts.json',progress)
 del model;gc.collect();torch.cuda.empty_cache();dist.barrier()
 if rank==0:state(run,'comparing_in_memory')
 results={};own_cross=collections.defaultdict(float)
 for mi,mode in enumerate(MODES):
  sums={}
  if mode=='all_tokens' and cfg['reuse_full_gradient']:
   directory=source/f'gradient-{half}/n064';manifest=json.loads((directory/'manifest.json').read_text());assert manifest['parameters']==4022468096
   def tensors():
    for item in manifest['files']:
     with safe_open(str(directory/item['file']),framework='pt',device='cpu') as f:
      for name in f.keys():yield name,f.get_tensor(name).to('cuda')
  else:
   def tensors():
    for name in names:yield name,accum[mode][name].to('cuda')
  for name,a in tensors():
   for other in MODES[mi+1:]:
    b_local=accum[other][name].to('cuda');own_cross[mode+'__'+other]+=float((a*b_local).sum(dtype=torch.float64))
   b=torch.empty_like(a) if rank==0 else a
   dist.broadcast(b,src=1)
   if rank==0:
    add_stats(sums,'all',a,b);add_stats(sums,block(name),a,b)
   del a,b
  if rank==0:
   results[mode]={k:finalize(v) for k,v in sums.items()};write(run/'partial-results.json',results);print('MODE_RESULT',mode,json.dumps(results[mode]['all']),flush=True)
  dist.barrier()
 cross=[None,None];dist.all_gather_object(cross,dict(own_cross))
 if rank==0:
  for rank_i,values in enumerate(cross):
   norm_key='norm_a' if rank_i==0 else 'norm_b'
   for pair,dot in list(values.items()):
    a,b=pair.split('__');den=results[a]['all'][norm_key]*results[b]['all'][norm_key];values[pair]={'dot':dot,'cosine':dot/den if den else None}
  result={'finished_at':now(),'probe_id':cfg['probe_id'],'rollouts':128,'key_turn':cfg['key_turn'],'results':results,'within_half_mode_alignment':{'a':cross[0],'b':cross[1]},'metrics':{mode:results[mode]['all']['cosine'] for mode in MODES},'interpretation':'All modes use the same recorded inputs, half assignment and trajectory-level advantages. Reasoning is masked from loss, not erased from current context. Key turns are fixed by protocol; no outcome-dependent turn selection.'}
  if cfg['reuse_full_gradient']:
   original=json.loads((source/'result.json').read_text())['curve'][-1]['cosine'];result['full_gradient_reuse_difference']=abs(results['all_tokens']['all']['cosine']-original);assert result['full_gradient_reuse_difference']<1e-10
  write(run/'result.json',result);plot(run,result)
  if cfg['probe_id']=='multi_turn_base_22':
   from probe_followup_artifacts import comparison_artifacts
   comparison_artifacts(run)
  state(run,'complete','succeeded');write(run/'complete.json',{'finished_at':now()})
 dist.barrier();dist.destroy_process_group()

def plot(run,result):
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(13,4));labels=['All tokens','No reasoning','Key turn: all','Key turn: visible'];v=[result['results'][m]['all']['cosine'] for m in MODES];axes[0].bar(labels,[x if x is not None else float('nan') for x in v]);axes[0].axhline(0,color='gray',lw=1);axes[0].tick_params(axis='x',rotation=25);axes[0].set_ylabel('Split-half cosine');axes[0].set_title(result['probe_id']+' (64+64)')
 for mode,label in zip(MODES,labels):axes[1].plot(range(36),[result['results'][mode][f'layer-{i:02d}']['cosine'] for i in range(36)],label=label)
 axes[1].axhline(0,color='gray',lw=1);axes[1].set_xlabel('Transformer block');axes[1].set_ylabel('Layer split-half cosine');axes[1].legend(fontsize=8);fig.tight_layout();fig.savefig(run/'ablation.png',dpi=180);fig.savefig(run/'ablation.pdf');plt.close(fig)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('run_name');args=parser.parse_args()
 try:main(args.run_name)
 except BaseException as e:
  if int(os.environ.get('LOCAL_RANK','0'))==0:state(ROOT/'runs'/args.run_name,'failed','failed',error=str(e))
  raise
