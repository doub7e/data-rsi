"""Metadata-only recovery of a terminal trial with zero committed assistant steps."""
from common import *
from rollouts import accept_terminal_result,finalize_rollouts
RUN=ROOT/'runs/eval-finalization-repair'
def main():
 state(RUN,'recover_existing_terminal_outcome',started_at=now());snapshot_sources(RUN,['repair_terminal_export.py','rollouts.py','reasoning_audit.py','transcripts.py','common.py'])
 run=ROOT/'runs/eval-guided';proof=json.loads((RUN/'preserved-inputs.json').read_text());items=json.loads((run/'plan.json').read_text())['items'];plan=json.loads((run/'plan.json').read_text());item=next(x for x in items if x['id']=='e0167');rawpath=ROOT/proof['raw_trial_path'];assert sha(rawpath)==proof['raw_trial_sha256'];raw=json.loads(rawpath.read_text());assert raw['finished_at'] and raw.get('verifier_result') is None and raw['exception_info']['exception_type']=='RuntimeError'
 cfg=json.loads((run/'jobs/e0167/config.json').read_text());assert cfg['agents'][0]['kwargs']['probe_seed']==item['seed'] and cfg['agents'][0]['model_name']=='hosted_vllm/'+plan['served_model']
 for ident,digest in proof['completed_record_sha256'].items():assert sha(run/'trials'/ident/'complete.json')==digest
 row=accept_terminal_result(item,run);assert row['score']==0 and row['raw_score'] is None and row['imputed_zero'] and row['training_rows']==0
 assert sha(rawpath)==proof['raw_trial_sha256']
 for ident,digest in proof['completed_record_sha256'].items():assert sha(run/'trials'/ident/'complete.json')==digest
 records={}
 for x in items:
  d=json.loads((run/'trials'/x['id']/'complete.json').read_text());assert all(d[k]==x[k] for k in ['id','task','seed','repeat']);records[x['id']]=d
 assert len(records)==280;finalize_rollouts(run,items,records)
 write(run/'finalization-progress.json',{'stage':'verified_terminal_record_merge','done':280,'total':280,'unresolved_trials':0,'updated_at':now()});m=json.loads((run/'run.json').read_text());m['sources']['progress']='runs/eval-guided/finalization-progress.json';m['artifacts'].append({'label':'末条轨迹导出异常修复审计','path':'runs/eval-finalization-repair/result.json'});write(run/'run.json',m)
 result={'finished_at':now(),'metrics':{'recovered_terminal_records':1,'preserved_complete_records':279,'new_rollouts':0,'merged_records':280,'recovered_score':0,'empty_training_exports_added':1},'resolved_trial':'e0167','raw_result_unchanged':True,'completed279_unchanged':True,'source_exception':'RuntimeError: sandbox already Succeeded before first command','policy':'Existing missing-verifier0 rule; raw API response retained but no committed assistant steps, so empty SFT export. No reroll or fabricated conversation.','unresolved_history_preserved':'runs/eval-guided/unresolved-trials.json'}
 write(RUN/'result.json',result);write(RUN/'complete.json',{'finished_at':now()});state(RUN,'complete','succeeded',finished_at=now());print(json.dumps(result))
if __name__=='__main__':
 try:main()
 except BaseException as e:state(RUN,'failed','failed',error=str(e),finished_at=now());raise
