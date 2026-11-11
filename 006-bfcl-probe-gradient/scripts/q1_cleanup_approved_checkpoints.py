"""Execute only the exact old-checkpoint cleanup explicitly approved by the user."""
import shutil
from common import *
RUN=ROOT/'runs/q1-selection-20261104'
def main():
 proposal=RUN/'old-checkpoint-cleanup-proposal.json';plan=json.loads(proposal.read_text());approval=json.loads((RUN/'cleanup-approval.json').read_text())
 assert approval['approved'] is True and approval['proposal_sha256']==sha(proposal)
 allowed={BASE/'runs/random100-s42-20261102',ROOT/'runs/train-top100',ROOT/'runs/train-bottom100'}
 assert {Path(r['run']) for r in plan['runs']}==allowed
 removal_roots=[Path(d['path']).resolve() for r in plan['runs'] for d in r['remove_only_resume_directories']]
 # Protect any final-export symlink targets as well as regular final files.
 for root in allowed:
  for f in (root/'checkpoint').iterdir():
   if f.is_file() or f.is_symlink():
    target=f.resolve(strict=True)
    assert not any(target.is_relative_to(d) for d in removal_roots), 'Final export depends on a proposed removal: '+str(f)
 completed=[]
 for record in plan['runs']:
  root=Path(record['run']);checkpoint=root/'checkpoint';assert (root/'training-complete.json').exists()
  for f in record['final_weight_headers_and_lengths_verified']:
   s=(checkpoint/f['file']).stat();assert s.st_size==f['bytes'] and s.st_mtime_ns==f['mtime_ns']
  for rec in record['remove_only_resume_directories']:
   d=Path(rec['path']);assert d.parent==checkpoint and d.name.startswith('checkpoint-') and not d.is_symlink()
   if not d.exists():
    prior=RUN/'cleanup-progress.json';assert prior.exists() and str(d) in json.loads(prior.read_text())['removed'];completed.append(str(d));continue
   actual={str(p.relative_to(d)):p.stat().st_size for p in d.rglob('*') if p.is_file()};assert actual=={f['file']:f['bytes'] for f in rec['files']},'Checkpoint changed since review'
   dest=RUN/'old-checkpoint-metadata'/root.name/d.name;dest.mkdir(parents=True,exist_ok=True)
   for f in rec['files']:
    if f['bytes']<20_000_000:
     p=d/f['file'];target=dest/f['file'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
   shutil.rmtree(d);completed.append(str(d));write(RUN/'cleanup-progress.json',{'removed':completed,'updated_at':now()})
  for f in record['final_weight_headers_and_lengths_verified']:
   s=(checkpoint/f['file']).stat();assert s.st_size==f['bytes'] and s.st_mtime_ns==f['mtime_ns']
  note={'at':now(),'user_approved':True,'reason':'Release completed-run resume copies after new study hit actual NFS quota exhaustion','removed_resume_directories':record['remove_only_resume_directories'],'final_weights_retained':record['final_weight_headers_and_lengths_verified'],'small_checkpoint_metadata_preserved_in':str((RUN/'old-checkpoint-metadata'/root.name).relative_to(ROOT.parent))}
  write(root/'checkpoint-retention.json',note)
 write(RUN/'cleanup-complete.json',{'finished_at':now(),'removed':completed,'logical_bytes_reclaimed':plan['total_bytes'],'final_models_retained':True,'proposal_sha256':sha(proposal)})
 print('Approved cleanup complete:',len(completed),'directories;',plan['total_bytes']/2**30,'GiB')
if __name__=='__main__':main()
