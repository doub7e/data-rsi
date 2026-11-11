"""This experiment explicitly scores missing terminal verifier rewards as zero."""
import math

def valid_score(value):
 return isinstance(value,(int,float)) and math.isfinite(value) and 0<=value<=1

def score_metrics(records):
 rows=list(records);n=len(rows);assert all(valid_score(r.get('score')) for r in rows)
 original=sum(valid_score(r.get('raw_score')) for r in rows);k=sum(r['score']==1 for r in rows)
 return {'evaluated_trials':n,'verifier_scored_trials':original,'missing_verifier_scored_zero':n-original,'verifier_score_coverage_percent':100*original/n if n else None,'mean_score':sum(r['score'] for r in rows)/n if n else None,'full_passes':k,'full_pass_percent':100*k/n if n else None}
