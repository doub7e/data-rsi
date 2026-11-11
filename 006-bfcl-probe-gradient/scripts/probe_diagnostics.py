"""Real-model gradient replay and HF/vLLM per-token likelihood diagnostics."""
import argparse,gc,importlib.metadata,json,math,os,subprocess,sys,time
from common import *
RUN=ROOT/'runs/probe-diagnostics-20261103';SOURCE=ROOT/'runs/probe194-scaling-20261103';VIEW=ROOT/'runs/probe-rollouts/model-64k'
def prepare():
 selected=[]
 for rep in (1,2):
  row=json.loads((SOURCE/f'traces/r{rep:03d}.json').read_text())
  for i,s in enumerate(row['steps']):selected.append({'id':f'r{rep:03d}-s{i:02d}','repeat':rep,'call_index':i,'prompt_token_ids':s['prompt_token_ids'],'completion_token_ids':s['completion_token_ids']})
 row=json.loads((SOURCE/'traces/r023.json').read_text())
 for i in (0,len(row['steps'])-1):
  s=row['steps'][i];selected.append({'id':f'r023-s{i:02d}','repeat':23,'call_index':i,'prompt_token_ids':s['prompt_token_ids'],'completion_token_ids':s['completion_token_ids']})
 write(RUN/'likelihood-cases.json',selected)
 return selected

def vllm_phase():
 import torch,transformers,vllm
 from vllm import LLM,SamplingParams
 cases=prepare();llm=LLM(model=str(VIEW),dtype='bfloat16',max_model_len=65536,tensor_parallel_size=1,gpu_memory_utilization=.7,enable_prefix_caching=False,max_num_seqs=8,trust_remote_code=True)
 write(RUN/'vllm-runtime.json',{'torch':torch.__version__,'transformers':transformers.__version__,'vllm':vllm.__version__,'config_sha256':sha(VIEW/'config.json'),'dtype':'bfloat16','prefix_cache':False,'reason':'Forced prompt_logprobs replay uses no prefix cache; fresh decode logprobs separately check decode vs prefill.'})
 params=SamplingParams(temperature=0,max_tokens=1,prompt_logprobs=1,skip_special_tokens=False)
 outputs=llm.generate([{'prompt_token_ids':c['prompt_token_ids']+c['completion_token_ids']} for c in cases],params,use_tqdm=False)
 scored=[]
 for c,o in zip(cases,outputs):
  p,t=c['prompt_token_ids'],c['completion_token_ids'];assert list(o.prompt_token_ids)==p+t and len(o.prompt_logprobs)==len(p)+len(t)
  lp=[o.prompt_logprobs[len(p)+i][tok].logprob for i,tok in enumerate(t)]
  scored.append(dict(c,source='recorded_forced_prefill',vllm_logprobs=lp))
 selected=[cases[0],next(c for c in reversed(cases) if c['repeat']==1),next(c for c in cases if c['repeat']==2),next(c for c in reversed(cases) if c['repeat']==2),cases[-2],cases[-1]]
 fresh=[]
 for i,c in enumerate(selected):
  for where,offset in [('start',0),('near_end',max(0,len(c['completion_token_ids'])-64))]:
   prefix=c['prompt_token_ids']+c['completion_token_ids'][:offset]
   fresh.append({'id':c['id']+'-'+where,'prompt_token_ids':prefix,'seed':194000+i*2+(where=='near_end')})
 outputs=llm.generate([{'prompt_token_ids':c['prompt_token_ids']} for c in fresh],[SamplingParams(temperature=1,top_p=1,top_k=-1,max_tokens=64,logprobs=1,seed=c['seed'],skip_special_tokens=False) for c in fresh],use_tqdm=False)
 for c,o in zip(fresh,outputs):
  gen=o.outputs[0];ids=list(gen.token_ids);assert ids and gen.logprobs
  c.update(completion_token_ids=ids,vllm_logprobs=[lp[t].logprob for t,lp in zip(ids,gen.logprobs)],source='fresh_actual_decode',finish_reason=gen.finish_reason)
 replay=llm.generate([{'prompt_token_ids':c['prompt_token_ids']+c['completion_token_ids']} for c in fresh],params,use_tqdm=False)
 for c,o in zip(fresh,replay):c['vllm_prefill_logprobs']=[o.prompt_logprobs[len(c['prompt_token_ids'])+i][t].logprob for i,t in enumerate(c['completion_token_ids'])]
 write(RUN/'vllm-reference.json',scored+fresh);write(RUN/'vllm-complete.json',{'finished_at':now(),'recorded_calls':len(scored),'fresh_decode_calls':len(fresh)})
 core=getattr(llm.llm_engine,'engine_core',None)
 if core is not None and callable(getattr(core,'shutdown',None)):core.shutdown()
 del llm;gc.collect()

def bucket(name):
 if name.startswith('model.layers.'):return 'layer-'+name.split('.')[2].zfill(2)
 if name.startswith('model.embed_tokens'):return 'embedding_lm_head'
 return 'final_norm_other'
def repeat_gradients(model,rep):
 import torch
 from gradients import backward
 trace=json.loads((SOURCE/f'traces/r{rep:03d}.json').read_text());reference=None;checks=[];losses=[]
 for repeat in range(2):
  model.zero_grad(set_to_none=True);loss=0.
  for s in trace['steps']:
   p,c=s['prompt_token_ids'],s['completion_token_ids'];val,_=backward(model,p+c,[-100]*len(p)+c,-1.,'sum');loss+=val
  losses.append(loss)
  if repeat==0:reference={k:p.grad.detach().clone() for k,p in model.named_parameters()};continue
  sums={}
  for name,p in model.named_parameters():
   a,b=reference[name],p.grad;assert torch.isfinite(a).all() and torch.isfinite(b).all()
   vals=[float((a*a).sum(dtype=torch.float64)),float((b*b).sum(dtype=torch.float64)),float((a*b).sum(dtype=torch.float64)),float(((a-b)**2).sum(dtype=torch.float64))]
   for k in ('all',bucket(name)):
    q=sums.setdefault(k,[0.,0.,0.,0.])
    for j,v in enumerate(vals):q[j]+=v
  checks={k:{'norm_first':math.sqrt(a),'norm_second':math.sqrt(b),'cosine':dot/math.sqrt(a*b) if a*b else None,'relative_l2':math.sqrt(err/a) if a else None} for k,(a,b,dot,err) in sums.items()}
 result={'repeat_id':rep,'reward':trace['reward'],'calls':len(trace['steps']),'nll_sums':losses,'gradient_checks':checks,'passes':checks['all']['cosine']>=.9999 and checks['all']['relative_l2']<=.01}
 write(RUN/f'gradient-repeat-r{rep:03d}.json',result);del reference;model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();return result

def hf_logprobs(model,case,autocast):
 import torch
 import torch.nn.functional as F
 from contextlib import nullcontext
 p,c=case['prompt_token_ids'],case['completion_token_ids'];ids=torch.tensor([p+c],device='cuda')
 values=[]
 with torch.no_grad(),(torch.autocast('cuda',dtype=torch.bfloat16) if autocast else nullcontext()):
  h=model.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).last_hidden_state
  for offset in range(0,len(c),128):
   selected=h[0,len(p)-1+offset:len(p)-1+min(offset+128,len(c))];logits=model.lm_head(selected)
   target=torch.tensor(c[offset:offset+128],device='cuda');values.extend(F.log_softmax(logits.float(),dim=-1).gather(-1,target[:,None]).squeeze(-1).cpu().tolist())
 assert len(values)==len(c);return values

def hf_phase():
 import torch,transformers
 from gradients import load_model,test_gradient
 torch.set_num_threads(4);assert torch.cuda.device_count()==1
 model=load_model(VIEW);write(RUN/'hf-runtime.json',{'torch':torch.__version__,'transformers':transformers.__version__,'config_sha256':sha(VIEW/'config.json'),'gradient_mode':'fp32 parameters, bf16 autocast, sdpa, train(), attention_dropout0, checkpointing nonreentrant','unit_check':test_gradient()})
 repeats=[repeat_gradients(model,rep) for rep in (1,2,23)];write(RUN/'gradient-repeat-summary.json',{'cases':repeats,'all_pass':all(x['passes'] for x in repeats)})
 cases=json.loads((RUN/'vllm-reference.json').read_text());result=[]
 for c in cases:result.append({'id':c['id'],'source':c['source'],'mixed_train_logprobs':hf_logprobs(model,c,True)})
 model.eval()
 for c,r in zip(cases,result):r['mixed_eval_logprobs']=hf_logprobs(model,c,True)
 del model;gc.collect();torch.cuda.empty_cache()
 from transformers import AutoModelForCausalLM
 model=AutoModelForCausalLM.from_pretrained(VIEW,torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
 write(RUN/'hf-bf16-buffers.json',{name:str(b.dtype) for name,b in model.named_buffers()})
 for c,r in zip(cases,result):r['bf16_eval_logprobs']=hf_logprobs(model,c,False)
 write(RUN/'hf-logprobs.json',result);del model;gc.collect();torch.cuda.empty_cache()
 analyze()

def analyze():
 import numpy as np
 refs=json.loads((RUN/'vllm-reference.json').read_text());hfs=json.loads((RUN/'hf-logprobs.json').read_text());assert len(refs)==len(hfs)
 records=[]
 def metrics(a,b):
  a=np.asarray(a);b=np.asarray(b);d=a-b
  return {'tokens':len(d),'mean_signed_difference':float(d.mean()),'mean_absolute_difference':float(np.abs(d).mean()),'p99_absolute_difference':float(np.quantile(np.abs(d),.99)),'max_absolute_difference':float(np.abs(d).max()),'rmse':float(np.sqrt((d*d).mean())),'nll_reference':float(-b.sum()),'nll_hf':float(-a.sum())}
 groups={}
 for ref,hf in zip(refs,hfs):
  assert ref['id']==hf['id']
  item={'id':ref['id'],'source':ref['source'],'comparisons':{}}
  for key in ['mixed_train_logprobs','mixed_eval_logprobs','bf16_eval_logprobs']:
   item['comparisons'][key]=metrics(hf[key],ref['vllm_logprobs'])
   a,b=groups.setdefault((ref['source'],key),([],[]));a.extend(hf[key]);b.extend(ref['vllm_logprobs'])
  item['mixed_train_vs_eval']=metrics(hf['mixed_train_logprobs'],hf['mixed_eval_logprobs'])
  if 'vllm_prefill_logprobs' in ref:item['vllm_decode_vs_prefill']=metrics(ref['vllm_logprobs'],ref['vllm_prefill_logprobs'])
  records.append(item)
 aggregate={source:{key:metrics(*groups[(source,key)]) for (s,key) in groups if s==source} for source in sorted({s for s,k in groups})}
 result={'finished_at':now(),'repeat_gradient_all_pass':json.loads((RUN/'gradient-repeat-summary.json').read_text())['all_pass'],'likelihood_aggregate':aggregate,'likelihood_per_case':records,'interpretation_thresholds':{'repeat_cosine_min':.9999,'repeat_relative_l2_max':.01,'logprob_investigation_mean_abs':.03,'logprob_investigation_p99_abs':.2,'note':'Logprob thresholds are investigation triggers, not proof of identical policies.'}}
 write(RUN/'result.json',result);state(RUN,'complete','succeeded');write(RUN/'complete.json',{'finished_at':now()});print(json.dumps({'repeat_gradient_all_pass':result['repeat_gradient_all_pass'],'likelihood_aggregate':aggregate}),flush=True)

def main():
 state(RUN,'vllm_logprob_reference',started_at=now())
 call([sys.executable,__file__,'vllm'],RUN/'vllm.log')
 state(RUN,'hf_replay_and_logprob_checks');call([sys.executable,__file__,'hf'],RUN/'hf.log')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','vllm','hf']);a=p.parse_args()
 try:
  if a.mode=='run':main()
  elif a.mode=='vllm':vllm_phase()
  else:hf_phase()
 except BaseException as e:state(RUN,'failed','failed',error=str(e));raise
