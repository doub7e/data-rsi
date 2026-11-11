"""Freeze cosine ranking and materialize both groups with the original SFT recipe."""
import json,sys,statistics,math
from common import *

def main():
 candidates={r['meta_index']:r for r in rows(ROOT/'data/candidates.jsonl')}
 scores=[r for rank in (0,1) for r in rows(ROOT/f'runs/gradient-candidates-{rank}/scores.jsonl')]
 assert len(scores)==200 and len({s['meta_index'] for s in scores})==200 and {s['meta_index'] for s in scores}==set(candidates)
 assert all(math.isfinite(s['cosine']) and -1.000001<=s['cosine']<=1.000001 and s['gradient_norm']>0 for s in scores)
 assert all(s['probe_norm']>0 and math.isclose(s['probe_norm'],scores[0]['probe_norm'],rel_tol=1e-6) for s in scores),'Probe basis differs across workers'
 assert all(s['pool_index']==candidates[s['meta_index']]['pool_index'] and s['objective_weighting']=='mean NLL over all supervised trajectory tokens' for s in scores)
 scores.sort(key=lambda r:(-r['cosine'],r['meta_index']))
 sys.path.insert(0,str(BASE/'upstream/code/bfcl/selection'))
 from recipe.expand import expand_trajectory
 summaries={}
 for group,chosen in [('top100',scores[:100]),('bottom100',scores[100:])]:
  dest=ROOT/f'runs/train-{group}/data';dest.mkdir(parents=True,exist_ok=True)
  selected=[candidates[s['meta_index']] for s in chosen];selected.sort(key=lambda r:r['pool_index'])
  samples=[]
  for row in selected:
   for s in expand_trajectory(row['messages']):samples.append(dict(instruction=s['prompt'],input='',output=s['target'],system=s['system'],history=[list(x) for x in s['history']]))
  target=dest/'train-data.json';payload=json.dumps(samples,ensure_ascii=False)
  if target.exists():assert target.read_text()==payload
  else:target.write_text(payload)
  (dest/'train.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
  summary={'group':group,'trajectories':100,'expanded_rows':len(samples),'meta_indices':[r['meta_index'] for r in selected],
   'original100_overlap':sum(r['original100'] for r in selected),'unique_environments':len({r['env_id'] for r in selected}),
   'cosine_mean':statistics.mean(r['cosine'] for r in chosen),'cosine_min':min(r['cosine'] for r in chosen),'cosine_max':max(r['cosine'] for r in chosen),
   'train_data_sha256':sha(target),'input_tokens':sum(r['input_tokens'] for r in chosen),'target_tokens':sum(r['target_tokens'] for r in chosen)}
  summaries[group]=summary;write(dest/'audit.json',summary)
  write(dest.parent/'preparation-complete.json',{'completed_at':now(),'model_path':str(MODEL),'data_path':str(dest),'expanded_rows':len(samples)})
  meta=json.loads((dest.parent/'run.json').read_text());meta['parameters']={'model':'Qwen3-4B','data_rows':100,'expanded_rows':len(samples),'gpus':2,'gpu_type':'H200','epochs':2,'learning_rate':5e-6,'global_batch_size':32,'cutoff_len':16384,'sampling':'cosine-'+group};meta['artifacts'].append({'label':'Selection audit','path':str((dest/'audit.json').relative_to(ROOT))});write(dest.parent/'run.json',meta)
 write(ROOT/'data/ranking.json',scores);write(ROOT/'data/selection-summary.json',summaries)
 assert sum(x['original100_overlap'] for x in summaries.values())==100
 write(ROOT/'data/selection-complete.json',{'finished_at':now(),'scores_sha256':sha(ROOT/'data/ranking.json')})
 print(json.dumps(summaries),flush=True)
if __name__=='__main__':main()
