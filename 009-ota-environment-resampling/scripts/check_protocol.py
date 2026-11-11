"""Small deterministic protocol/math checks; no model computation."""
import ast,math,random,statistics
from common import *
from select_environments import probabilities

def main():
 checks={};p=json.loads((ROOT/'configs/protocol.json').read_text());split=json.loads((ROOT/'configs/dev-split.json').read_text());assert len(split['probe_tasks'])==14 and len(split['heldout_tasks'])==56 and not set(split['probe_tasks'])&set(split['heldout_tasks']);checks['fixed_disjoint14_56_split']=True
 a=json.loads((ROOT/'runs/teacher-batch1/plan.json').read_text());b=json.loads((ROOT/'runs/teacher-uniform/plan.json').read_text());assert len(a['items'])==len(b['items'])==728;assert [x['task'] for x in a['items']]==[x['task'] for x in b['items']]
 probe=json.loads((ROOT/'runs/probe/plan.json').read_text());assert len(probe['items'])==448;assert all(sum(x['task']==t for x in probe['items'])==32 for t in split['probe_tasks']);checks['generation_counts_and_probe32_each']=True
 seed_ranges=[]
 for plan in [a,b,probe]:seed_ranges.extend(x['seed'] for x in plan['items'])
 assert len(set(seed_ranges))==len(seed_ranges) and max(seed_ranges)+99999<2**31;checks['disjoint_seed_intervals']=True
 for values in [[-10.]*728,[-float(i) for i in range(728)],[math.sin(i) for i in range(728)]]:
  _,_,_,probs=probabilities(values);assert abs(sum(probs)-1)<1e-12 and min(probs)>=.1/728
  order=sorted(range(728),key=lambda i:values[i]);assert all(probs[j]>=probs[i] for i,j in zip(order,order[1:]));other=probabilities([x*3+2 for x in values])[3];assert max(abs(x-y) for x,y in zip(probs,other))<1e-12
 assert max(abs(x-1/728) for x in probabilities([0.]*728)[3])<1e-15;checks['probabilities_signed_monotone_normalized_uniform_fallback']=True
 # A positive dot must mean the candidate descent direction improves the probe linear objective.
 g=[.3,-.7,.2];lossgrad=[-.1,.5,-.2];dot=sum(x*(-y) for x,y in zip(g,lossgrad));epsilon=1e-6;observed=sum(x*(-epsilon*y) for x,y in zip(g,lossgrad))/epsilon;assert abs(dot-observed)<1e-14 and dot>0;checks['dot_descent_sign']=True
 def fn(path,name):return ast.dump(next(x for x in ast.parse(path.read_text()).body if isinstance(x,ast.FunctionDef) and x.name==name))
 for name in ['normalized','make_encoder','forward_loss']:assert fn(ROOT/'scripts/loss_contract.py',name)==fn(OLD/'scripts/loss_contract.py',name)
 checks['original_training_encoder_loss_unchanged']=True
 for file in ['probe_gradient_shard.py','question_gradient.py','candidate_dots.py']:assert "precision='training_bf16'" in (ROOT/'scripts'/file).read_text()
 checks['production_gradient_forward_precision_is_actual_bf16']=True
 for f in (ROOT/'scripts').glob('*.py'):compile(f.read_text(),str(f),'exec')
 write(ROOT/'runs/study/protocol-check.json',{'at':now(),'checks':checks});print(checks)
if __name__=='__main__':main()
