"""Original-verifier-only scores; missing measurements never become numeric rewards."""
import math

def valid_score(value):
 return isinstance(value,(int,float)) and math.isfinite(value) and 0<=value<=1

def score_metrics(records):
 rows=list(records);scored=[r for r in rows if valid_score(r.get('score'))];n=len(rows);m=len(scored);k=sum(r['score']==1 for r in scored)
 return {'evaluated_trials':n,'verifier_scored_trials':m,'verifier_unscored_trials':n-m,'verifier_score_coverage_percent':100*m/n if n else None,'mean_score':sum(r['score'] for r in scored)/m if m else None,'full_passes':k,'full_pass_percent':100*k/n if n else None,'full_pass_percent_among_scored':100*k/m if m else None,'full_pass_percent_upper_bound':100*(k+n-m)/n if n else None}
