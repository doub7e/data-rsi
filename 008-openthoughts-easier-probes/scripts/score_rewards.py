"""Official score rewards; binary full-pass remains a descriptive metric only."""
import math,statistics
from common import *
def score_of(record):
 value=record.get('raw_reward')
 if value is None and record.get('trial_result'):
  raw=json.loads((ROOT/record['trial_result']).read_text());value=((raw.get('verifier_result') or {}).get('rewards') or {}).get('reward')
 if value is None and record.get('reward_source') in ['model_budget_timeout','model_caused_verifier_failure']:value=0.
 if not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=1:raise ValueError(f'Invalid official score: {value!r}')
 return float(value)
def normalize(run):
 path=run/'result.json';data=json.loads(path.read_text())
 if data.get('probe_reward_definition')=='official_score_v1':return data
 scores=[score_of(r) for r in data['records']]
 assert len(scores)==data['trials']
 write(run/'revisions/binary-result-before-score-migration.json',data)
 for r,s in zip(data['records'],scores):r.update(reward=s,probe_reward=s,full_pass=int(s==1),reward_definition='official_score_v1')
 data.update(scores=scores,outcomes=scores,score_sum=sum(scores),mean_score=statistics.mean(scores),score_variance=statistics.pvariance(scores),successes=sum(s==1 for s in scores),probe_reward_definition='official_score_v1')
 data.setdefault('metrics',{}).update(mean_score=data['mean_score'],score_variance=data['score_variance'],partial_credit_trials=sum(0<s<1 for s in scores),success_percent=100*data['successes']/len(scores))
 write(path,data);return data
