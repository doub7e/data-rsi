"""Final single-question comparison; execute on a RunAI CPU worker."""
import math
from common import *
RUN=ROOT/'runs/q1-selection-20261104'
def main():
 import numpy as np
 from scipy.stats import binomtest,spearmanr
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 cfg=json.loads((RUN/'protocol.json').read_text());variants=['base','random100']+cfg['variants'];data={v:json.loads((ROOT/f'runs/q1-{v}-eval-20261104/result.json').read_text()) for v in variants}
 assert all(d['trials']==256 and len(d['outcomes'])==256 for d in data.values())
 rng=np.random.default_rng(20261104194);draws=rng.integers(0,256,size=(20000,256));contrasts=[]
 for control in ['random100','base']:
  for v in cfg['variants']:
   x=np.array(data[v]['outcomes']);y=np.array(data[control]['outcomes']);diff=x-y;boot=diff[draws].mean(axis=1);wins=int(((x==1)&(y==0)).sum());losses=int(((x==0)&(y==1)).sum());p=binomtest(wins,wins+losses,.5).pvalue if wins+losses else 1.
   contrasts.append({'variant':v,'control':control,'delta_percentage_points':100*float(diff.mean()),'paired_bootstrap95_pp':(100*np.quantile(boot,[.025,.975])).tolist(),'discordant_wins':wins,'discordant_losses':losses,'mcnemar_exact_p':float(p)})
 adjusted=0.
 for rank,i in enumerate(sorted(range(len(contrasts)),key=lambda i:contrasts[i]['mcnemar_exact_p'])):
  adjusted=max(adjusted,min(1.,contrasts[i]['mcnemar_exact_p']*(len(contrasts)-rank)));contrasts[i]['holm12_p']=adjusted
 selection={v:json.loads((ROOT/f'runs/q1-{v}-train-20261104/data/audit.json').read_text()) for v in cfg['variants']}
 scores={m:{r['meta_index']:r for r in rows(ROOT/f'runs/q1-scores-20261104/{m}-scores.jsonl')} for m in ['F','K']};ids=sorted(scores['F']);overlap=[]
 for a in cfg['variants']:
  for b in cfg['variants']:
   if a>=b:continue
   sa=[scores[a[0].upper()][i]['scores'][str(int(a[1:]))]['cosine'] for i in ids];sb=[scores[b[0].upper()][i]['scores'][str(int(b[1:]))]['cosine'] for i in ids]
   overlap.append({'a':a,'b':b,'top100_overlap':len(set(selection[a]['meta_indices'])&set(selection[b]['meta_indices'])),'score_spearman':float(spearmanr(sa,sb).statistic)})
 result={'finished_at':now(),'evaluations':data,'contrasts':contrasts,'selection':selection,'selection_comparisons':overlap,'limitations':cfg['limitations'],'metrics':{v+'_q1_success_percent':100*d['accuracy'] for v,d in data.items()}}
 write(RUN/'result.json',result)
 fig,axes=plt.subplots(1,2,figsize=(13,4.5));p=np.array([data[v]['accuracy'] for v in variants])*100;cis=np.array([data[v]['wilson95'] for v in variants])*100
 axes[0].bar(variants,p,color=['gray','gray']+['tab:blue']*3+['tab:orange']*3);axes[0].errorbar(variants,p,yerr=np.stack([p-cis[:,0],cis[:,1]-p]),fmt='none',ecolor='black',capsize=3);axes[0].set_ylabel('Q1 success (%) with Wilson95% CI');axes[0].tick_params(axis='x',rotation=35)
 primary=[c for c in contrasts if c['control']=='random100'];ds=np.array([c['delta_percentage_points'] for c in primary]);ci=np.array([c['paired_bootstrap95_pp'] for c in primary]);axes[1].errorbar(cfg['variants'],ds,yerr=np.stack([ds-ci[:,0],ci[:,1]-ds]),fmt='o',capsize=3);axes[1].axhline(0,color='gray');axes[1].set_ylabel('Difference vs random100 (percentage points)');axes[1].tick_params(axis='x',rotation=35);fig.tight_layout();fig.savefig(RUN/'comparison.png',dpi=180);fig.savefig(RUN/'comparison.pdf');plt.close(fig)
 lines=['# Q1 single-question selection: F/K × 8/32/128','', 'All current reasoning retained. Fresh evaluation uses256 trials/model and a disjoint seed namespace. This is targeted same-question adaptation; intervals cover evaluation sampling, not training-seed or task variability.','', '| Model | Successes /256 | Success rate | Wilson95% |','|---|---:|---:|---:|']
 for v,d in data.items():lines.append(f"| {v} | {d['successes']} | {100*d['accuracy']:.2f}% | {100*d['wilson95'][0]:.2f}–{100*d['wilson95'][1]:.2f}% |")
 lines += ['', '| Selected model | Control | Difference pp | Paired bootstrap95% pp | Holm12 p |','|---|---|---:|---:|---:|']
 for c in contrasts:lines.append(f"| {c['variant']} | {c['control']} | {c['delta_percentage_points']:+.2f} | {c['paired_bootstrap95_pp'][0]:+.2f}–{c['paired_bootstrap95_pp'][1]:+.2f} | {c['holm12_p']:.4g} |")
 lines += ['', '![Q1 comparison](comparison.png)','', 'Selection overlap, full score correlations, exact outcomes and provenance: [result.json](result.json).','', *['- '+s for s in cfg['limitations']]]
 (RUN/'REPORT.md').write_text('\n'.join(lines)+'\n');m=json.loads((RUN/'run.json').read_text());m['artifacts'] += [{'label':'Results report','path':str((RUN/'REPORT.md').relative_to(ROOT))},{'label':'Success-rate comparison','path':str((RUN/'comparison.png').relative_to(ROOT))}];write(RUN/'run.json',m);state(RUN,'complete','succeeded');write(RUN/'complete.json',{'finished_at':now()})
if __name__=='__main__':main()
