"""Paired held-out BFCL analysis; no model calls or new training."""
import json,statistics
import numpy as np
from common import *
RUN=ROOT/'runs/comparison';CATS=['base','miss_func','miss_param','long_context']

def scores(directory):
 result={}
 for rep in (1,2,3):
  out=directory/f'bfcl/random100-e{rep}';assert (out/'verified-summary.json').exists()
  for cat in CATS:
   generated=rows(out/f'Qwen_Qwen3-4B-FC/multi_turn/BFCL_v4_multi_turn_{cat}_result.json')
   scored=rows(out/f'score/Qwen_Qwen3-4B-FC/multi_turn/BFCL_v4_multi_turn_{cat}_score.json')
   assert len(generated)==200 and scored[0]['total_count']==200
   ids={r['id'] for r in generated};assert len(ids)==200
   bad={r['id'] for r in scored[1:] if 'id' in r};assert bad<=ids and len(bad)==200-scored[0]['correct_count']
   for task in ids:result[(rep,task)]=float(task not in bad)
 return result

def main():
 state(RUN,'comparison',started_at=now())
 probe={x for v in json.loads((ROOT/'runs/probe-rollouts/selection.json').read_text())['ids'].values() for x in v}
 allscores={'random100':scores(BASE/'runs/eval-random100-s42-20261102'),
            'top100':scores(ROOT/'runs/eval-top100'),'bottom100':scores(ROOT/'runs/eval-bottom100')}
 keys=set(allscores['random100']);assert all(set(s)==keys for s in allscores.values())
 allids=sorted({task for _,task in keys});heldout=[t for t in allids if t not in probe];assert len(heldout)==768
 table={};arrays={}
 for name,s in allscores.items():
  repeat=[statistics.mean(s[(rep,t)] for t in heldout) for rep in (1,2,3)]
  arrays[name]=np.array([statistics.mean(s[(rep,t)] for rep in (1,2,3)) for t in heldout])
  table[name]={'heldout_accuracy':statistics.mean(repeat),'heldout_repeat_accuracy':repeat,'heldout_repeat_std':statistics.stdev(repeat),
    'probe_accuracy':statistics.mean(s[(rep,t)] for rep in (1,2,3) for t in probe),
    'all800_accuracy':statistics.mean(s.values()),
    'category_accuracy':{c:statistics.mean(s[(rep,t)] for rep in (1,2,3) for t in heldout if t.startswith('multi_turn_'+c+'_')) for c in CATS}}
 rng=np.random.default_rng(20261103);resamples=rng.integers(0,len(heldout),size=(10000,len(heldout)))
 differences={}
 for a,b in [('top100','bottom100'),('top100','random100'),('bottom100','random100')]:
  delta=arrays[a]-arrays[b];boot=delta[resamples].mean(axis=1);lo,hi=np.quantile(boot,[.025,.975])
  differences[a+'-minus-'+b]={'difference_pp':float(delta.mean()*100),'paired_task_bootstrap_95ci_pp':[float(lo*100),float(hi*100)],'bootstrap_replicates':10000}
 selected=json.loads((ROOT/'data/selection-summary.json').read_text())
 signal=json.loads((ROOT/'runs/probe-rollouts/result.json').read_text())
 result={'finished_at':now(),'main_evaluation':'768 BFCL tasks excluding frozen32 probes; 3 repetitions/model','models':table,
  'paired_differences':differences,'selection':selected,
  'probe_signal':{k:signal[k] for k in ('questions','repeats','rollouts','mixed_questions','successes','gradient_eligible_rollouts','maximum_context_tokens')},
  'selection_cosines_all_negative':max(v['cosine_max'] for v in selected.values())<0,
  'bootstrap_note':'Resample tasks; average three repeats within task. These intervals do not cover training-seed variation.',
  'limitations':['Benchmark-feedback pilot, not blind DataSense selection.','Probe coverage excludes long_context.','Held-out tasks can share tool environments with probes; this is not an unseen-environment test.','One training seed; equal trajectories but unequal expanded sample/token budgets.','Cosine uses raw SGD direction; the fixed training optimizer is AdamW.']}
 result['metrics']={name+'_heldout_accuracy_percent':v['heldout_accuracy']*100 for name,v in table.items()}
 result['metrics']['top_minus_bottom_pp']=differences['top100-minus-bottom100']['difference_pp']
 def interpretation(pair):
  lo,hi=differences[pair]['paired_task_bootstrap_95ci_pp']
  return 'positive evidence in this fixed-pool/seed pilot' if lo>0 else 'negative evidence in this fixed-pool/seed pilot' if hi<0 else 'inconclusive at the paired-task 95% interval'
 result['interpretation']={'primary_top_vs_bottom':interpretation('top100-minus-bottom100'),
   'exploratory_top_vs_random':interpretation('top100-minus-random100')}
 write(RUN/'result.json',result)
 write(RUN/'paired-task-scores.json',[{'id':t,**{name:float(arrays[name][i]) for name in arrays}} for i,t in enumerate(heldout)])
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(11,4))
 names=['random100','top100','bottom100'];axes[0].bar(names,[table[n]['heldout_accuracy']*100 for n in names],color=['gray','seagreen','coral']);axes[0].set_ylabel('Held-out BFCL accuracy (%)');axes[0].set_title('768 tasks, mean of 3 repeats')
 ranking=json.loads((ROOT/'data/ranking.json').read_text());axes[1].hist([r['cosine'] for r in ranking],bins=25,color='steelblue');axes[1].axvline((ranking[99]['cosine']+ranking[100]['cosine'])/2,color='black',linestyle='--');axes[1].set_xlabel('Cosine with reward-centered probe gradient');axes[1].set_ylabel('Training trajectories');fig.tight_layout();fig.savefig(RUN/'comparison.png',dpi=160);fig.savefig(RUN/'comparison.pdf');plt.close(fig)
 lines=['# Probe-gradient pilot results','', 'Main metric excludes all32 probe tasks: 768 held-out tasks, 3 repeats per model.','', '| Model | Held-out accuracy | Repeat SD | Probe accuracy | All800 accuracy |','|---|---:|---:|---:|---:|']
 for n in names:
  v=table[n];lines.append(f"| {n} | {100*v['heldout_accuracy']:.2f}% | {100*v['heldout_repeat_std']:.2f} pp | {100*v['probe_accuracy']:.2f}% | {100*v['all800_accuracy']:.2f}% |")
 lines+=['','## Paired comparisons','']
 for name,d in differences.items():lines.append(f"- {name}: {d['difference_pp']:+.2f} pp; paired task bootstrap95% CI {d['paired_task_bootstrap_95ci_pp']} pp.")
 lines+=['',f"Primary top-vs-bottom: {result['interpretation']['primary_top_vs_bottom']}.",f"Exploratory top-vs-random: {result['interpretation']['exploratory_top_vs_random']}."]
 lines+=['','## Selection audit','','| Group | Trajectories | Expanded rows | Original100 overlap | Mean cosine | Supervised tokens |','|---|---:|---:|---:|---:|---:|']
 for name,v in selected.items():lines.append(f"| {name} | 100 | {v['expanded_rows']} | {v['original100_overlap']} | {v['cosine_mean']:.6f} | {v['target_tokens']} |")
 lines+=['',f"Probe: {signal['questions']} questions × {signal['repeats']} repeats; {signal['mixed_questions']} questions have nonzero empirical reward variance. Original random100 expands to1853 samples."]
 if result['selection_cosines_all_negative']:lines+=['','All200 cosine scores were negative. Top100 means relatively better alignment, not a positive predicted gain. The ranking/sign was not flipped after observing this.']
 lines+=['','## Limits','']+['- '+x for x in result['limitations']]
 (RUN/'REPORT.md').write_text('\n'.join(lines)+'\n')
 meta=json.loads((RUN/'run.json').read_text());meta['artifacts'] += [{'label':'Comparison chart','path':str((RUN/'comparison.png').relative_to(ROOT))},{'label':'Report','path':str((RUN/'REPORT.md').relative_to(ROOT))},{'label':'Full result','path':str((RUN/'result.json').relative_to(ROOT))}];write(RUN/'run.json',meta)
 write(RUN/'complete.json',{'finished_at':now()});state(RUN,'complete','succeeded')
if __name__=='__main__':
 try:main()
 except BaseException as e:state(RUN,'comparison','failed',error=str(e));raise
