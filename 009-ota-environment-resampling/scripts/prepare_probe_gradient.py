"""Freeze score eligibility and balanced question shards before gradient computation."""
import collections,math
from common import *
from verifier_scores import valid_score
RUN=ROOT/'runs/probe-gradient-plan'
def main():
 state(RUN,'plan_probe_gradient',started_at=now());source=ROOT/'runs/probe';records=json.loads((source/'result.json').read_text())['records'];group=collections.defaultdict(list)
 for r in records:group[r['task']].append(r)
 split=json.loads((ROOT/'configs/dev-split.json').read_text());assert set(group)==set(split['probe_tasks']);eligible=[];excluded=[]
 for task,rows in sorted(group.items()):
  assert len(rows)==32;scores=[r['score'] for r in rows];missing=[r['id'] for r in rows if not valid_score(r.get('score'))];q={'task':task,'scores':scores,'mean_score':sum(scores)/32 if not missing else None,'records':rows}
  if missing:q.update(exclusion_reason='missing_original_verifier_scores',missing_score_trials=missing);excluded.append(q);continue
  if len(set(scores))==1:q['exclusion_reason']='constant_original_verifier_score';excluded.append(q);continue
  work=0.
  for rec in rows:
   for line in (source/'trials'/rec['id']/'train.jsonl').read_text().splitlines():
    if line.strip():work+=min(32768,sum(len(m['content']) for m in json.loads(line)['conversations'])/3.5)
  q['estimated_input_tokens_for_scheduling']=work;eligible.append(q)
 n=min(4,len(eligible));assignments=[[] for _ in range(n)];loads=[0.]*n
 for q in sorted(eligible,key=lambda q:(-q['estimated_input_tokens_for_scheduling'],q['task'])):
  i=min(range(n),key=lambda i:(loads[i],i));assignments[i].append(q['task']);loads[i]+=q['estimated_input_tokens_for_scheduling']
 plan={'at':now(),'eligible':eligible,'excluded':excluded,'global_valid_questions':len(eligible),'num_workers':n,'assignments':assignments,'estimated_shard_loads':loads,'data_sha256':sha(source/'data/train.jsonl'),'formula':'Compute each complete, nonconstant question gradient independently; sum and divide by final eligible-question count. Questions with any missing original verifier reward are excluded without replacement.','assignment_note':'Greedy balance by capped character-length estimate only; no gradient or evaluation result beyond predefined eligibility affects shard assignment.'};write(ROOT/'runs/probe-gradient/plan.json',plan);write(ROOT/'runs/probe-gradient/probe-questions.json',plan)
 if not eligible:
  parent=ROOT/'runs/probe-gradient';write(parent/'probe-gradient-complete.json',{'finished_at':now(),'no_signal':True,'valid_questions':0,'probe_norm':0.});write(parent/'result.json',{'metrics':{'valid_probe_questions':0,'probe_norm':0.}});write(parent/'progress.json',{'stage':'constant_score_questions','done':14,'total':14,'updated_at':now()});write(parent/'complete.json',{'finished_at':now()});state(parent,'complete','succeeded',finished_at=now())
 write(RUN/'result.json',{'finished_at':now(),'metrics':{'valid_questions':len(eligible),'excluded_questions':len(excluded),'gradient_workers':n}});write(RUN/'complete.json',{'finished_at':now()});state(RUN,'complete','succeeded',finished_at=now())
if __name__=='__main__':
 try:main()
 except BaseException as e:state(RUN,'failed','failed',error=str(e),finished_at=now());raise
