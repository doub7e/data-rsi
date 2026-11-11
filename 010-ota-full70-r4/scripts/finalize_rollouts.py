"""CPU-only merge of complete per-trial records; never generate or resume a trial."""
import sys,json
from common import ROOT,state,now,write
from rollouts import finalize_rollouts
rid=sys.argv[1];run=ROOT/'runs'/rid
try:
 items=json.loads((run/'plan.json').read_text())['items'];records={}
 for item in items:
  path=run/'trials'/item['id']/'complete.json';assert path.exists(),item['id'];record=json.loads(path.read_text())
  assert all(record[k]==item[k] for k in ['id','task','seed']);records[item['id']]=record
 assert len(records)==len(items) and len(items)>0
 finalize_rollouts(run,items,records)
 # User-approved board counter correction; preserve original worker progress
 # and all unresolved-error history. Count only validated terminal records.
 progress=run/'finalization-progress.json'
 write(progress,{'stage':'verified_terminal_record_merge','done':len(records),'total':len(items),'updated_at':now()})
 manifest=json.loads((run/'run.json').read_text());manifest['sources']['progress']=str(progress.relative_to(ROOT));write(run/'run.json',manifest)
 write(run/'metadata-only-finalization.json',{'at':now(),'merged_records':len(records),'no_new_rollouts':True})
except BaseException as error:state(run,'metadata_finalization_failed','failed',error=str(error),finished_at=now());raise
