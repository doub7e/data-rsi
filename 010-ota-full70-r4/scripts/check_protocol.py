"""Small deterministic contract checks, without models or GPU work."""
import ast,sys,json,math
from unittest.mock import patch
from common import *
from verifier_scores import score_metrics
from select_environments import probabilities

def main():
 plans={r:json.loads((ROOT/'runs'/r/'plan.json').read_text()) for r in ['probe','eval-uniform','eval-guided']};checks={}
 for rid,p in plans.items():
  assert len(p['items'])==280 and len({x['task'] for x in p['items']})==70
  assert all(sorted(x['repeat'] for x in p['items'] if x['task']==task)==[1,2,3,4] for task in {x['task'] for x in p['items']})
  assert len({x['seed'] for x in p['items']})==280
 assert [x['seed'] for x in plans['eval-uniform']['items']]==[x['seed'] for x in plans['eval-guided']['items']]
 assert not {x['seed'] for x in plans['probe']['items']}&{x['seed'] for x in plans['eval-guided']['items']};checks['three_models70x4_and_seed_contract']=True
 m=score_metrics([{'score':1.,'raw_score':1.},{'score':0.,'raw_score':None}]);assert m['mean_score']==.5 and m['missing_verifier_scored_zero']==1;checks['missing_reward_zero_denominator']=True
 for values in [[0.]*728,[math.sin(i) for i in range(728)]]:
  _,_,_,p=probabilities(values);assert abs(sum(p)-1)<1e-12;assert min(p)>=.1/728
 checks['same_signed_dot_probability_rule']=True
 def fn(path,name):return ast.dump(next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name))
 for n in ['normalized','make_encoder','forward_loss']:assert fn(ROOT/'scripts/loss_contract.py',n)==fn(ORIGIN/'scripts/loss_contract.py',n)
 for file in ['probe_gradient_shard.py','candidate_dots.py']:assert "precision='training_bf16'" in (ROOT/'scripts'/file).read_text()
 checks['actual_training_loss_encoder_and_bf16_precision_unchanged']=True
 reuse=json.loads((ROOT/'runs/study/reuse-manifest.json').read_text());assert len(reuse['source_snapshots'])==70
 for x in reuse['source_snapshots']:assert sha(x['source_record'])==x['record_sha256']
 sys.path.insert(0,str(NATIVE/'native'));import rollouts
 with patch.object(rollouts,'wait_auth'),patch.object(rollouts.subprocess,'Popen',side_effect=AssertionError('Cached old repeat must not execute')):
  r=rollouts.trial(plans['eval-uniform']['items'][0],ROOT/'runs/eval-uniform',plans['eval-uniform']);assert r['repeat']==1 and r.get('reused_from')
 checks['original_uniform70_unchanged_and_never_rerolled']=True
 for p in (ROOT/'scripts').glob('*.py'):compile(p.read_text(),str(p),'exec')
 write(ROOT/'runs/study/protocol-check.json',{'at':now(),'checks':checks});print(checks)
if __name__=='__main__':main()
