"""Materialize the frozen random50 continuation control inside allocated workers."""
from common import *
def random50():
 cfg=json.loads((ROOT/'runs/q1-continuation-20261104/protocol.json').read_text());run=ROOT/'runs/q1c-random50-train-20261104'
 if (run/'preparation-complete.json').exists():return
 candidates=rows(ROOT/cfg['candidate_path']);selected=sorted([r for r in candidates if r['meta_index'] in cfg['random50_meta_indices']],key=lambda r:r['pool_index']);assert len(selected)==50
 sys.path.insert(0,str(BASE/'upstream/code/bfcl/selection'));from recipe.expand import expand_trajectory
 samples=[]
 for row in selected:
  for s in expand_trajectory(row['messages']):samples.append(dict(instruction=s['prompt'],input='',output=s['target'],system=s['system'],history=[list(x) for x in s['history']]))
 dest=run/'data';dest.mkdir(exist_ok=True);(dest/'train-data.json').write_text(json.dumps(samples,ensure_ascii=False));(dest/'train.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
 audit={'variant':'random50','trajectories':50,'expanded_rows':len(samples),'meta_indices':[r['meta_index'] for r in selected],'sampling':'uniform without replacement, fixed seed44','data_sha256':sha(dest/'train-data.json'),'candidate_sha256':cfg['candidate_sha256'],'unique_environments':len({r['env_id'] for r in selected})};write(dest/'audit.json',audit)
 write(run/'preparation-complete.json',{'completed_at':now(),'model_path':str(BASE/'runs/random100-s42-20261102/checkpoint'),'data_path':str(dest),'expanded_rows':len(samples)})
 m=json.loads((run/'run.json').read_text());m['parameters'].update(data_rows=50,expanded_rows=len(samples),sampling='random50 seed44');m['artifacts'].append({'label':'Random50 selection audit','path':str((dest/'audit.json').relative_to(ROOT))});write(run/'run.json',m)
if __name__=='__main__':random50()
