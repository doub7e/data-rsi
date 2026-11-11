"""Read-only inventory and verification of already completed SFT checkpoints."""
from common import *
def main():
 from safetensors import safe_open
 run=ROOT/'runs/q1-storage-audit-20261104';state(run,'verify_completed_models',started_at=now());records=[]
 for source in [BASE/'runs/random100-s42-20261102',ROOT/'runs/train-top100',ROOT/'runs/train-bottom100']:
  checkpoint=source/'checkpoint';assert (source/'training-complete.json').exists() and (checkpoint/'training-complete.txt').exists()
  trainer=json.loads((checkpoint/'trainer_state.json').read_text());assert trainer['global_step']==trainer['max_steps'] and trainer['epoch']==2
  index=json.loads((checkpoint/'model.safetensors.index.json').read_text());names=sorted(set(index['weight_map'].values()));weights=[]
  for name in names:
   file=checkpoint/name
   with safe_open(str(file),framework='pt',device='cpu') as f:assert len(f.keys())>0
   weights.append({'file':name,'bytes':file.stat().st_size,'sha256':sha(file)})
  copies=[]
  for d in sorted(checkpoint.glob('checkpoint-*')):
   assert d.is_dir() and not d.is_symlink();items=[{'file':str(p.relative_to(d)),'bytes':p.stat().st_size} for p in d.rglob('*') if p.is_file()];ts=json.loads((d/'trainer_state.json').read_text())
   copies.append({'path':str(d),'step':ts['global_step'],'bytes':sum(f['bytes'] for f in items),'optimizer_bytes':sum(f['bytes'] for f in items if f['file']=='optimizer.pt'),'files':items})
  records.append({'run':str(source),'final_step':trainer['global_step'],'final_weights':weights,'resume_checkpoints':copies})
  write(run/'progress.json',{'verified_models':len(records),'total':3,'updated_at':now()})
 result={'at':now(),'read_only':True,'completed_models_verified':records,'resume_checkpoint_bytes':sum(c['bytes'] for r in records for c in r['resume_checkpoints']),'optimizer_bytes':sum(c['optimizer_bytes'] for r in records for c in r['resume_checkpoints'])};write(run/'result.json',result);write(run/'complete.json',{'finished_at':now()});state(run,'complete','succeeded');print(json.dumps({k:v for k,v in result.items() if k!='completed_models_verified'}))
if __name__=='__main__':main()
