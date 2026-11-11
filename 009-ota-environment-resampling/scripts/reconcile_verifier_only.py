"""Reconcile derived records to immutable original verifier outputs, without rollouts."""
import json,sys,collections
from common import *
from verifier_scores import valid_score,score_metrics
from rollouts import accept_terminal_result,finalize_rollouts
RUN=ROOT/'runs/verifier-authority-audit'

def backup(path):
 if not path.exists():return
 dest=RUN/'previous-derived-records'/path.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True)
 assert not dest.exists(),'Migration is one-shot; inspect existing audit before resume'
 dest.write_bytes(path.read_bytes())

def main():
 state(RUN,'reconcile_original_verifier_outputs',started_at=now());changed=[];raw_hashes={}
 for rid in ['probe','eval-uniform']:
  run=ROOT/'runs'/rid
  for name in ['result.json','complete.json','state.json','run.json','progress.json','baseline-subsets.json','board-verification.json']:backup(run/name)
  for marker in sorted((run/'trials').glob('*/complete.json')):
   record=json.loads(marker.read_text());value=record.get('raw_score');expected=float(value) if valid_score(value) else None
   if record['score']==expected:continue
   rawpath=ROOT/record['trial_result'];raw=json.loads(rawpath.read_text());actual=((raw.get('verifier_result') or {}).get('rewards') or {}).get('reward');assert actual==value and not valid_score(actual)
   raw_hashes[str(rawpath.relative_to(ROOT))]=sha(rawpath);backup(marker);old_score=record['score'];record.update(score=None,full_pass=None,reward_source='verifier_unavailable',score_policy='original_verifier_only',reconciled_at=now());write(marker,record);changed.append({'run':rid,'id':record['id'],'old_score':old_score,'new_score':None,'original_verifier_reward':actual,'raw_result':record['trial_result']})
  if rid=='probe':
   item=next(x for x in json.loads((run/'plan.json').read_text())['items'] if x['id']=='p0446');rawpath=next((run/'jobs/p0446').glob('*/result.json'));raw_hashes[str(rawpath.relative_to(ROOT))]=sha(rawpath);rec=accept_terminal_result(item,run);assert rec['score'] is None and rec['raw_score'] is None
  items=json.loads((run/'plan.json').read_text())['items'];records={x['id']:json.loads((run/'trials'/x['id']/'complete.json').read_text()) for x in items}
  assert all(all(records[x['id']][k]==x[k] for k in ['id','task','seed']) for x in items)
  finalize_rollouts(run,items,records)
  progress=run/'verifier-only-progress.json';write(progress,{'stage':'original_verifier_terminal_records','done':len(records),'total':len(items),**score_metrics(records.values()),'updated_at':now()})
  manifest=json.loads((run/'run.json').read_text());manifest['sources']['progress']=str(progress.relative_to(ROOT));manifest.setdefault('parameters',{})['reward_policy']='original_verifier_only';manifest['artifacts'].append({'label':'原verifier分数核查与保留的旧统计','path':'runs/verifier-authority-audit/result.json'});write(run/'run.json',manifest)
  write(RUN/'progress.json',{'stage':'reconcile_runs','done':1 if rid=='probe' else 2,'total':2,'updated_at':now()})
 assert all(sha(ROOT/p)==h for p,h in raw_hashes.items()),'Original Harbor result changed'
 r=ROOT/'runs/eval-uniform';data=json.loads((r/'result.json').read_text());bytask={x['task']:x for x in data['records']};split=json.loads((ROOT/'configs/dev-split.json').read_text());subsets={name:score_metrics(bytask[t] for t in tasks) for name,tasks in [('all70',split['all_tasks']),('heldout56',split['heldout_tasks']),('probe14',split['probe_tasks'])]};write(r/'baseline-subsets.json',{'at':now(),'score_policy':'original_verifier_only','subsets':subsets,'reward_sources':dict(collections.Counter(x['reward_source'] for x in data['records'])),'note':'Means use available original verifier scores, with explicit coverage. full_pass_percent counts confirmed full passes over all attempts and is a lower bound when outcomes are unscored.'})
 probe=json.loads((ROOT/'runs/probe/result.json').read_text());group=collections.defaultdict(list)
 for x in probe['records']:group[x['task']].append(x)
 incomplete=[t for t,rows in group.items() if any(not valid_score(r['score']) for r in rows)]
 from question_gradient import run_id
 for task in incomplete:
  path=ROOT/'runs'/run_id(task)/'run.json'
  if path.exists():
   backup(path);m=json.loads(path.read_text());m.setdefault('parameters',{}).update(attribution_eligible_original_verifier_only=False,exclusion_reason='Incomplete original verifier score set; computed vector retains its original previous-label provenance');m['artifacts'].append({'label':'此梯度不再纳入归因的原因','path':'runs/verifier-authority-audit/result.json'});write(path,m)
 write(RUN/'result.json',{'finished_at':now(),'user_instruction':'按原verifier来','changed_derived_records':changed,'original_raw_results_sha256_unchanged':raw_hashes,'incomplete_probe_questions':incomplete,'metrics':{'removed_imputed_zeros':len(changed),'probe_raw_rollouts':448,'probe_unscored':probe['metrics']['verifier_unscored_trials'],'uniform_unscored':data['metrics']['verifier_unscored_trials'],'raw_results_modified':0},'no_rollout_or_verifier_reexecution':True});write(RUN/'complete.json',{'finished_at':now()});state(RUN,'complete','succeeded',finished_at=now())
if __name__=='__main__':
 try:main()
 except BaseException as e:state(RUN,'failed','failed',error=str(e),finished_at=now());raise
