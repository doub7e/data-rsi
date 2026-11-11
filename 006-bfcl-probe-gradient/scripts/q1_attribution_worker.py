"""RunAI CPU preparation/comparison for sequential base-initialized attribution studies."""
import argparse,math
from common import *
DATE='20261105'
def config(method):return json.loads((ROOT/f'runs/q1-{method}-{DATE}/protocol.json').read_text())
def materialize(cfg,rankings,score_key):
 sys.path.insert(0,str(BASE/'upstream/code/bfcl/selection'));from recipe.expand import expand_trajectory
 candidates={r['meta_index']:r for r in rows(ROOT/'data/candidates.jsonl')};assert len(candidates)==200 and sha(ROOT/'data/candidates.jsonl')==cfg['candidate_sha256']
 audits={}
 for v in cfg['variants']:
  ranking=rankings[v];assert len(ranking)==200 and {x['meta_index'] for x in ranking}==set(candidates)
  assert all(math.isfinite(x[score_key]) for x in ranking)
  ordered=sorted(ranking,key=lambda x:(-x[score_key],x['meta_index']))
  selected=sorted([candidates[x['meta_index']] for x in ordered[:100]],key=lambda x:x['pool_index'])
  run=ROOT/f"runs/{cfg['prefix']}-{v}-train-{DATE}";d=run/'data';d.mkdir(exist_ok=True)
  samples=[]
  for row in selected:
   for s in expand_trajectory(row['messages']):samples.append(dict(instruction=s['prompt'],input='',output=s['target'],system=s['system'],history=[list(x) for x in s['history']]))
  payload=json.dumps(samples,ensure_ascii=False);dest=d/'train-data.json'
  if dest.exists():assert dest.read_text()==payload,'Frozen data changed'
  else:dest.write_text(payload)
  (d/'train.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
  write(d/'ranking.json',ordered)
  previous=json.loads((ROOT/f'runs/q1-{v}-train-20261104/data/audit.json').read_text())
  audit={'variant':v,'method':cfg['method'],'trajectories':100,'expanded_rows':len(samples),'meta_indices':[r['meta_index'] for r in selected],'unique_environments':len({r['env_id'] for r in selected}),'candidate_sha256':cfg['candidate_sha256'],'data_sha256':sha(dest),'cosine_top100_overlap':len(set(previous['meta_indices'])&{r['meta_index'] for r in selected}),'score_key':score_key,'score_min':ordered[99][score_key],'score_max':ordered[0][score_key]}
  write(d/'audit.json',audit);audits[v]=audit
  write(run/'preparation-complete.json',{'completed_at':now(),'model_path':str(MODEL),'data_path':str(d),'expanded_rows':len(samples)})
  manifest=json.loads((run/'run.json').read_text());manifest['parameters'].update(data_rows=100,expanded_rows=len(samples),sampling=cfg['method']+'_top100');manifest['artifacts'] += [{'label':'Selection audit','path':str((d/'audit.json').relative_to(ROOT))}];write(run/'run.json',manifest)
 return audits

def prepare_dot():
 cfg=config('dot');run=ROOT/f"runs/{cfg['prefix']}-prepare-{DATE}";state(run,'materializing_dot_selection')
 scores={m:rows(ROOT/cfg['score_source']/f'{m}-scores.jsonl') for m in ['F','K']}
 rankings={}
 for v in cfg['variants']:
  m=v[0].upper();n=str(int(v[1:]));rankings[v]=[{'meta_index':x['meta_index'],'dot':x['scores'][n]['dot'],'cosine':x['scores'][n]['cosine'],'gradient_norm':x['gradient_norm']} for x in scores[m]]
  for x in scores[m]:
   s=x['scores'][n];assert math.isclose(s['dot'],s['cosine']*x['gradient_norm']*s['probe_norm'],rel_tol=2e-6,abs_tol=1e-7)
 audits=materialize(cfg,rankings,'dot');write(run/'result.json',{'finished_at':now(),'audits':audits,'score_files':{m:sha(ROOT/cfg['score_source']/f'{m}-scores.jsonl') for m in scores}});write(run/'complete.json',{'finished_at':now()});state(run,'complete','succeeded');print(json.dumps({v:{k:a[k] for k in ['expanded_rows','cosine_top100_overlap']} for v,a in audits.items()}),flush=True)

def compare(method):
 import numpy as np
 from scipy.stats import binomtest
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 cfg=config(method);run=ROOT/f'runs/q1-{method}-{DATE}'
 data={v:json.loads((ROOT/f"runs/{cfg['prefix']}-{v}-eval-{DATE}/result.json").read_text()) for v in cfg['variants']}
 controls={k:json.loads((ROOT/cfg['controls'][k]/'result.json').read_text()) for k in ['base','random100']}
 cosine={v:json.loads((ROOT/cfg['controls']['cosine'][v]/'result.json').read_text()) for v in cfg['variants']}
 dot={v:json.loads((ROOT/f'runs/q1d-{v}-eval-{DATE}/result.json').read_text()) for v in cfg['variants']} if method=='pif' else {}
 rng=np.random.default_rng(20261105194);draws=rng.integers(0,256,size=(20000,256));contrasts=[]
 for v,x in data.items():
  refs={**controls,'matched_cosine':cosine[v]}
  if dot:refs['matched_dot']=dot[v]
  for label,y in refs.items():
   assert x['trials']==y['trials']==256 and len(x['outcomes'])==len(y['outcomes'])==256
   a=np.array(x['outcomes']);b=np.array(y['outcomes']);diff=a-b;boot=diff[draws].mean(axis=1);wins=int(((a==1)&(b==0)).sum());losses=int(((a==0)&(b==1)).sum())
   contrasts.append({'variant':v,'control':label,'delta_pp':100*float(diff.mean()),'bootstrap95_pp':(100*np.quantile(boot,[.025,.975])).tolist(),'wins':wins,'losses':losses,'p':float(binomtest(wins,wins+losses,.5).pvalue) if wins+losses else 1.})
 prev=0
 for rank,i in enumerate(sorted(range(len(contrasts)),key=lambda i:contrasts[i]['p'])):
  prev=max(prev,min(1.,contrasts[i]['p']*(len(contrasts)-rank)));contrasts[i]['holm_p']=prev
 result={'finished_at':now(),'evaluations':data,'controls':controls,'cosine_controls':cosine,'dot_controls':dot,'contrasts':contrasts,'limitations':cfg['limitations'],'metrics':{v+'_q1_success_percent':100*x['accuracy'] for v,x in data.items()}}
 write(run/'result.json',result)
 labels=list(data);xx=np.arange(len(labels));fig,ax=plt.subplots(figsize=(10,5))
 for offset,(name,ds) in enumerate([('Cosine',cosine),(method.upper(),data)]+([('Dot',dot)] if dot else [])):
  ax.bar(xx+(offset-.5)*.24,[ds[v]['accuracy']*100 for v in labels],width=.23,label=name)
 ax.axhline(100*controls['random100']['accuracy'],color='black',linestyle='--',label='random100');ax.set_xticks(xx,labels);ax.set_ylabel('Q1 success / 256 trials (%)');ax.legend();fig.tight_layout();fig.savefig(run/'comparison.png',dpi=180);fig.savefig(run/'comparison.pdf');plt.close(fig)
 lines=[f'# Q1 {method} selection from original base model','', '| Variant | Success /256 | Rate | Cosine rate |','|---|---:|---:|---:|']
 for v,x in data.items():lines.append(f"| {v} | {x['successes']} | {100*x['accuracy']:.2f}% | {100*cosine[v]['accuracy']:.2f}% |")
 lines+=['','| Variant | Control | Delta pp | Paired95% pp | Holm p |','|---|---|---:|---:|---:|']
 for x in contrasts:lines.append(f"| {x['variant']} | {x['control']} | {x['delta_pp']:+.2f} | {x['bootstrap95_pp'][0]:+.2f} to {x['bootstrap95_pp'][1]:+.2f} | {x['holm_p']:.4g} |")
 lines+=['','![Comparison](comparison.png)','']+['- '+x for x in cfg['limitations']];(run/'REPORT.md').write_text('\n'.join(lines)+'\n')
 m=json.loads((run/'run.json').read_text());m['artifacts'] += [{'label':'Results report','path':str((run/'REPORT.md').relative_to(ROOT))},{'label':'Comparison','path':str((run/'comparison.png').relative_to(ROOT))}];write(run/'run.json',m);write(run/'complete.json',{'finished_at':now()});state(run,'complete','succeeded')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare-dot','compare-dot','compare-pif']);a=p.parse_args();prepare_dot() if a.action=='prepare-dot' else compare(a.action.split('-')[1])
