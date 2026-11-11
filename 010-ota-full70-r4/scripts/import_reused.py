"""Import immutable source references/data and original Uniform repeat1, without inference."""
import os,shutil
from common import *
from verifier_scores import valid_score,score_metrics

def links(src,dst):
 if dst.exists():return
 shutil.copytree(src,dst,copy_function=os.link,symlinks=False)

def main():
 audit=ROOT/'runs/study/reuse-manifest.json';assert not audit.exists(),'Already imported; never silently redo reuse'
 source=ORIGIN/'runs/teacher-batch1';target=ROOT/'runs/teacher-batch1'
 (target/'data').mkdir(exist_ok=True);os.link(source/'data/train.jsonl',target/'data/train.jsonl')
 for p in sorted((source/'trials').iterdir()):
  if not p.is_dir():continue
  dst=target/'trials'/p.name;dst.mkdir(parents=True)
  os.link(p/'train.jsonl',dst/'train.jsonl')
 # Read-only links permit the existing viewer to read source traces inside this experiment.
 src=ORIGIN/'runs/eval-uniform';dst=ROOT/'runs/eval-uniform';items=json.loads((dst/'plan.json').read_text())['items'];rows=json.loads((src/'result.json').read_text())['records'];assert len(rows)==70
 snapshots=[]
 for item,record in zip(items[:70],rows):
  assert all(record[k]==item[k] for k in ['id','task','seed'])
  original_marker=src/'trials'/item['id']/'complete.json';source_digest=sha(original_marker)
  (dst/'jobs').mkdir(exist_ok=True);(dst/'trials').mkdir(exist_ok=True)
  links(src/'jobs'/item['id'],dst/'jobs'/item['id']);links(src/'trials'/item['id'],dst/'trials'/item['id'])
  value=record.get('raw_score');assert value is None or valid_score(value)
  imported={**record,'repeat':1,'score':float(value) if valid_score(value) else 0.,'full_pass':int(value==1),'reward_source':'verifier' if valid_score(value) else 'missing_verifier_zero','imputed_zero':not valid_score(value),'score_policy':'missing_verifier_as_zero','reused_from':f'{ORIGIN.name}/runs/eval-uniform/trials/{item["id"]}','source_record_sha256':source_digest,'imported_at':now()}
  # Atomic replace breaks only the new hardlink; the old record is unchanged.
  write(dst/'trials'/item['id']/'complete.json',imported);assert sha(original_marker)==source_digest
  raw=ORIGIN/record['trial_result'];st=raw.stat();snapshots.append({'id':item['id'],'source_record':str(original_marker),'record_sha256':source_digest,'source_raw_result':str(raw),'raw_size':st.st_size,'raw_mtime_ns':st.st_mtime_ns,'raw_inode':st.st_ino})
 write(dst/'progress.json',{'stage':'reused_first_repeat_new_repeats_pending','done':70,'total':280,'reused_trials':70,'new_trials_completed':0,'updated_at':now()})
 write(audit,{'at':now(),'source_experiment':ORIGIN.name,'uniform_reused_trials':70,'uniform_new_trials_planned':210,'raw_files':'Read-only use of same-filesystem hardlinks. Cached imported records bypass all Harbor execution; no source raw record may be edited.','source_snapshots':snapshots,'first_training_data_sha256':sha(source/'data/train.jsonl')})
 print('Imported70 original Uniform trials and first teacher training data; source record hashes unchanged.')
if __name__=='__main__':main()
