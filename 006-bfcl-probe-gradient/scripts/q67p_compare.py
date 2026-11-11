"""Two predeclared paired comparisons, all executed in a RunAI CPU job."""
from common import *
def main():
 import numpy as np
 from scipy.stats import binomtest
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 run=ROOT/'runs/q67-pif-20261106';cfg=json.loads((run/'protocol.json').read_text());data={v:json.loads((ROOT/f'runs/q67p-{v}-eval-20261106/result.json').read_text()) for v in ['random100','f','k']}
 for v,x in data.items():assert x['probe_id']==cfg['probe_id'] and x['trials']==len(x['outcomes'])==256
 rng=np.random.default_rng(2026110667);draws=rng.integers(0,256,size=(20000,256));contrasts=[]
 for v in ['f','k']:
  a=np.array(data[v]['outcomes']);b=np.array(data['random100']['outcomes']);d=a-b;wins=int(((a==1)&(b==0)).sum());losses=int(((a==0)&(b==1)).sum())
  contrasts.append({'variant':v,'control':'random100','delta_pp':100*float(d.mean()),'bootstrap95_pp':(100*np.quantile(d[draws].mean(axis=1),[.025,.975])).tolist(),'wins':wins,'losses':losses,'p':float(binomtest(wins,wins+losses,.5).pvalue) if wins+losses else 1.})
 prev=0
 for rank,i in enumerate(sorted(range(2),key=lambda i:contrasts[i]['p'])):
  prev=max(prev,min(1.,contrasts[i]['p']*(2-rank)));contrasts[i]['holm_p']=prev
 result={'finished_at':now(),'probe_id':cfg['probe_id'],'selected':cfg['selected'],'evaluations':data,'contrasts':contrasts,'limitations':cfg['limitations'],'metrics':{v+'_success_percent':100*x['accuracy'] for v,x in data.items()}}
 write(run/'result.json',result)
 labels=['random100',cfg['selected']['f'].upper(),cfg['selected']['k'].upper()];fig,ax=plt.subplots(figsize=(7,4));ax.bar(labels,[100*x['accuracy'] for x in data.values()]);ax.set_ylabel('miss_func_67 success /256 (%)');fig.tight_layout();fig.savefig(run/'comparison.png',dpi=180);fig.savefig(run/'comparison.pdf');plt.close(fig)
 lines=['# 第二题：预条件投影与随机100','',f"Question: {cfg['probe_id']}",'','| Model | Success /256 | Rate |','|---|---:|---:|']
 for v,x in data.items():lines.append(f"| {v} | {x['successes']} | {100*x['accuracy']:.2f}% |")
 lines+=['','| Model | Delta vs random pp | Paired95% pp | Holm p |','|---|---:|---|---:|']
 for c in contrasts:lines.append(f"| {c['variant']} | {c['delta_pp']:+.2f} | {c['bootstrap95_pp']} | {c['holm_p']:.4g} |")
 lines+=['','![Comparison](comparison.png)','']+['- '+x for x in cfg['limitations']];(run/'REPORT.md').write_text('\n'.join(lines)+'\n')
 m=json.loads((run/'run.json').read_text());m['artifacts']+=[{'label':'Comparison report','path':str((run/'REPORT.md').relative_to(ROOT))},{'label':'Comparison chart','path':str((run/'comparison.png').relative_to(ROOT))}];write(run/'run.json',m)
 write(run/'complete.json',{'finished_at':now()});state(run,'complete','succeeded')
if __name__=='__main__':main()
