"""Same validated SFT recipe, with bounded resume retention for this six-run study."""
import argparse,importlib.util,os,shutil,sys
from common import *
def main(v):
 run=ROOT/f'runs/q1p-{v}-train-20261105';cfg=json.loads((ROOT/'runs/q1-pif-20261105/protocol.json').read_text());assert v in cfg['variants']
 spec=importlib.util.spec_from_file_location('q1_sft',BASE/'scripts/random100_worker.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 m.ROOT=ROOT;m.RUN=run
 os.environ.update(SFT_SAVE_STEPS='20',SFT_SAVE_TOTAL_LIMIT='1',HF_DATASETS_CACHE=str(run/'cache/datasets'))
 if not (run/'training-complete.json').exists():m.train(now().replace(':','').replace('-',''))
 checkpoint=run/'checkpoint';idx=json.loads((checkpoint/'model.safetensors.index.json').read_text());weights=sorted(set(idx['weight_map'].values()));assert weights
 manifest=[{'file':f,'bytes':(checkpoint/f).stat().st_size,'sha256':sha(checkpoint/f)} for f in weights];assert all(x['bytes']>0 for x in manifest)
 write(run/'final-weights.json',{'files':manifest,'config_sha256':sha(checkpoint/'config.json'),'finished_at':now()})
 # These are only redundant resumable copies created by this new run; final exported model remains.
 removed=[]
 for d in checkpoint.glob('checkpoint-*'):
  assert d.is_dir() and not d.is_symlink() and (d/'trainer_state.json').exists()
  removed.append({'directory':d.name,'files':[{'path':str(p.relative_to(d)),'bytes':p.stat().st_size} for p in d.rglob('*') if p.is_file()]});shutil.rmtree(d)
 write(run/'resume-retention.json',{'reason':'Training complete and final exported weights SHA256 verified; retain final model and logs.','removed':removed,'at':now()})
 if (run/'cache/datasets').exists():shutil.rmtree(run/'cache/datasets')
 write(run/'complete.json',{'finished_at':now()});state(run,'complete','succeeded')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('variant');a=p.parse_args()
 try:main(a.variant)
 except BaseException as e:state(ROOT/f'runs/q1-{a.variant}-train-20261105','failed','failed',error=str(e));raise
