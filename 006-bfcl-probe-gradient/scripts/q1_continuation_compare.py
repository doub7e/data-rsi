"""Continuation comparison with independent seeds and the original frozen control."""
from common import *
RUN=ROOT/'runs/q1-continuation-20261104'
def main():
 import numpy as np
 from scipy.stats import binomtest,spearmanr
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 cfg=json.loads((RUN/'protocol.json').read_text());data={'starting_checkpoint':json.loads((ROOT/cfg['baseline_result']).read_text())};undefined={}
 for v in cfg['variants']:
  r=ROOT/f'runs/q1c-{v}-train-20261104'
  if (r/'undefined.json').exists():undefined[v]=json.loads((r/'undefined.json').read_text());continue
  data[v]=json.loads((ROOT/f'runs/q1c-{v}-eval-20261104/result.json').read_text())
 assert 'random50' in data and all(x['trials']==256 for x in data.values())
 rng=np.random.default_rng(2026110450);draws=rng.integers(0,256,size=(20000,256));contrasts=[]
 pairs=[(v,c) for v in cfg['variants'] if v!='random50' and v in data for c in ['random50','starting_checkpoint']]+[('random50','starting_checkpoint')]
 for v,c in pairs:
  x=np.array(data[v]['outcomes']);y=np.array(data[c]['outcomes']);d=x-y;wins=int(((x==1)&(y==0)).sum());losses=int(((x==0)&(y==1)).sum());p=binomtest(wins,wins+losses,.5).pvalue if wins+losses else 1.
  contrasts.append({'variant':v,'control':c,'delta_pp':float(d.mean()*100),'paired_bootstrap95_pp':(100*np.quantile(d[draws].mean(axis=1),[.025,.975])).tolist(),'discordant_wins':wins,'discordant_losses':losses,'mcnemar_p':float(p)})
 adjusted=0.
 for rank,i in enumerate(sorted(range(len(contrasts)),key=lambda i:contrasts[i]['mcnemar_p'])):
  # Keep the full planned13-comparison correction even if zero-direction settings were undefined.
  adjusted=max(adjusted,min(1.,contrasts[i]['mcnemar_p']*(13-rank)));contrasts[i]['holm13_p']=adjusted
 selections={v:json.loads((ROOT/f'runs/q1c-{v}-train-20261104/data/audit.json').read_text()) for v in cfg['variants'] if v not in undefined}
 overlap=[{'a':a,'b':b,'top50_overlap':len(set(selections[a]['meta_indices'])&set(selections[b]['meta_indices']))} for a in selections for b in selections if a<b]
 result={'finished_at':now(),'evaluations':data,'undefined_settings':undefined,'contrasts':contrasts,'selection':selections,'overlap':overlap,'limitations':cfg['limitations'],'metrics':{v+'_q1_success_percent':x['accuracy']*100 for v,x in data.items()}};write(RUN/'result.json',result)
 names=list(data);p=np.array([data[v]['accuracy'] for v in names])*100;ci=np.array([data[v]['wilson95'] for v in names])*100
 fig,ax=plt.subplots(figsize=(11,4.5));ax.bar(names,p);ax.errorbar(names,p,yerr=np.stack([p-ci[:,0],ci[:,1]-p]),fmt='none',ecolor='black',capsize=3);ax.set_ylabel('Q1 success (%) with Wilson95% CI');ax.tick_params(axis='x',rotation=30);fig.tight_layout();fig.savefig(RUN/'comparison.png',dpi=180);fig.savefig(RUN/'comparison.pdf');plt.close(fig)
 lines=['# Q1 continuation from random100 checkpoint','', 'Six gradient-selected50 settings and one frozen random50 control; all restart the optimizer and scheduler and train2 more epochs from the same random100 weights. Probe and evaluation seeds are disjoint.','', '| Model | Successes /256 | Accuracy | Wilson95% |','|---|---:|---:|---:|']
 for v,d in data.items():lines.append(f"| {v} | {d['successes']} | {100*d['accuracy']:.2f}% | {100*d['wilson95'][0]:.2f}–{100*d['wilson95'][1]:.2f}% |")
 lines+=['','| Model | Control | Delta pp | Paired95% pp | Holm13 p |','|---|---|---:|---:|---:|']
 for c in contrasts:lines.append(f"| {c['variant']} | {c['control']} | {c['delta_pp']:+.2f} | {c['paired_bootstrap95_pp'][0]:+.2f}–{c['paired_bootstrap95_pp'][1]:+.2f} | {c['holm13_p']:.4g} |")
 if undefined:lines+=['','Undefined zero-gradient settings, no resampling: '+json.dumps(undefined)]
 lines+=['','![Success rates](comparison.png)','',*['- '+s for s in cfg['limitations']]];(RUN/'REPORT.md').write_text('\n'.join(lines)+'\n');m=json.loads((RUN/'run.json').read_text());m['artifacts'] += [{'label':'Continuation comparison','path':str((RUN/'REPORT.md').relative_to(ROOT))},{'label':'Success-rate plot','path':str((RUN/'comparison.png').relative_to(ROOT))}];write(RUN/'run.json',m);write(RUN/'complete.json',{'finished_at':now()});state(RUN,'complete','succeeded')
if __name__=='__main__':main()
