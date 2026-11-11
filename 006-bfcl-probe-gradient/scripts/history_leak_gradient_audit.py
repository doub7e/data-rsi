"""Decompose contributions from recorded leaking contexts; diagnostic, not corrected-policy evaluation."""
import gc,json,math,re,time
import torch
from common import *
from gradients import load_model,load_probe,candidate_encoder,backward
RUN=ROOT/'runs/history-leak-gradient-audit-20261103'
BLOCK=re.compile(r'<\|im_start\|>([^\n]+)\n(.*?)(?:<\|im_end\|>|$)',re.S)
THINK=re.compile(r'<think>(.*?)</think>',re.S)
def leaked(step):
 for role,body in BLOCK.findall(step['formatted_prompt']):
  if role=='assistant' and (any(m.group(1).strip() for m in THINK.finditer(body)) or '<think>' in THINK.sub('',body)):return True
 return False

def main():
 torch.set_num_threads(4);assert torch.cuda.device_count()==1 and 'H200' in torch.cuda.get_device_name(0)
 state(RUN,'isolate_leak_gradient',started_at=now());started=time.monotonic()
 traces=rows(ROOT/'runs/probe-rollouts/rollouts.jsonl');report=json.loads((ROOT/'runs/probe-rollouts/result.json').read_text())
 affected=[(r,j,s) for r in traces if r['advantage']!=0 for j,s in enumerate(r['steps']) if leaked(s)]
 assert len(affected)==21,len(affected)
 model=load_model(ROOT/'runs/probe-rollouts/model-64k');model.zero_grad(set_to_none=True)
 locations=[]
 for i,(r,j,s) in enumerate(affected):
  p,c=s['prompt_token_ids'],s['completion_token_ids']
  backward(model,p+c,[-100]*len(p)+c,-r['advantage']/(32*report['repeats']),'sum')
  locations.append({'id':r['id'],'repeat':r['repeat'],'step':j,'advantage':r['advantage'],'prompt_tokens':len(p),'target_tokens':len(c)})
  write(RUN/'progress.json',{'stage':'leak_backward','done':i+1,'total':21})
 leak={n:(p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)) for n,p in model.named_parameters()}
 original=load_probe(model);model.zero_grad(set_to_none=True)
 a2=l2=b2=al=ab=0.0
 for name,a in original.items():
  l=leak[name];b=a-l
  a2+=float((a*a).sum(dtype=torch.float64));l2+=float((l*l).sum(dtype=torch.float64));b2+=float((b*b).sum(dtype=torch.float64))
  al+=float((a*l).sum(dtype=torch.float64));ab+=float((a*b).sum(dtype=torch.float64))
 an,ln,bn=math.sqrt(a2),math.sqrt(l2),math.sqrt(b2);assert an>0 and bn>0
 delta2=0.0
 for name,a in original.items():
  change=a/an-(a-leak[name])/bn;delta2+=float((change*change).sum(dtype=torch.float64))
 delta=math.sqrt(delta2)
 ranking=json.loads((ROOT/'data/ranking.json').read_text());assert math.isclose(an,ranking[0]['probe_norm'],rel_tol=1e-6)
 result={'original_probe_norm':an,'leak_component_norm':ln,'remaining_probe_norm':bn,'leak_norm_over_original':ln/an,
  'cos_original_remaining':ab/(an*bn),'cos_original_leak':al/(an*ln) if ln else None,'unit_direction_distance':delta,
  'cosine_change_bound':delta,'definitely_negative_after_omission':sum(r['cosine']+delta+1e-6<0 for r in ranking),
  'total_candidates':200,'affected_calls':locations,'affected_rollouts':len({(r['id'],r['repeat']) for r,_,_ in affected}),
  'scope':'Remove exactly21 leaking-context call gradients from the original estimator while retaining original rewards/advantages. This is a decomposition diagnostic, not a corrected-harness policy estimate.'}
 write(RUN/'decomposition.json',result);print(json.dumps({k:v for k,v in result.items() if k!='affected_calls'}),flush=True)
 del model;gc.collect();torch.cuda.empty_cache()
 state(RUN,'representative_candidate_checks');model=load_model(MODEL);encode=candidate_encoder()
 candidates={r['meta_index']:r for r in rows(ROOT/'data/candidates.jsonl')}
 positions=[0,1,9,24,49,74,99,100,124,149,174,199];comparisons=[]
 for pos in positions:
  prior=ranking[pos];row=candidates[prior['meta_index']];examples=encode(row);total=sum(sum(x!=-100 for x in y[1:]) for _,y in examples)
  model.zero_grad(set_to_none=True)
  for ids,labels in examples:backward(model,ids,labels,1/total,'sum')
  n2=da=dl=db=0.0
  for name,p in model.named_parameters():
   if p.grad is None:continue
   g=p.grad.detach();a=original[name];l=leak[name]
   n2+=float((g*g).sum(dtype=torch.float64));da-=float((g*a).sum(dtype=torch.float64));dl-=float((g*l).sum(dtype=torch.float64));db-=float((g*(a-l)).sum(dtype=torch.float64))
  n=math.sqrt(n2);ca=da/(n*an);cb=db/(n*bn);cl=dl/(n*ln) if ln else 0.0
  error=abs(ca-prior['cosine']);assert error<5e-5,(row['meta_index'],error)
  entry={'original_rank':pos+1,'meta_index':row['meta_index'],'saved_cosine':prior['cosine'],'recomputed_original_cosine':ca,
    'cosine_without_leaking_calls':cb,'cosine_with_leak_component':cl,'change':cb-ca,'recompute_error':error}
  comparisons.append(entry);write(RUN/'candidate-checks.json',comparisons);write(RUN/'progress.json',{'stage':'candidate_checks','done':len(comparisons),'total':len(positions)})
  print(json.dumps(entry),flush=True)
 result.update(candidate_checks=comparisons,finished_at=now(),elapsed_seconds=time.monotonic()-started,
  metrics={'affected_gradient_calls':21,'direction_distance':delta,'provably_remaining_negative':result['definitely_negative_after_omission'],
   'sampled_candidates':len(comparisons),'sampled_remaining_negative':sum(x['cosine_without_leaking_calls']<0 for x in comparisons)})
 write(RUN/'result.json',result);state(RUN,'complete','succeeded')
if __name__=='__main__':
 try:main()
 except BaseException as e:state(RUN,'audit','failed',error=str(e));raise
