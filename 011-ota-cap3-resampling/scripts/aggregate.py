"""Four conditions,70 tasks×4: cap3 resampling versus frozen baselines."""
import collections,csv,math,statistics
from common import *
from verifier_scores import valid_score,score_metrics
RUN=ROOT/'runs/comparison'

def summarize_tasks(records,tasks):
 group=collections.defaultdict(list)
 for r in records:group[r['task']].append(r)
 assert set(group)==set(tasks)
 result=[]
 for task in tasks:
  rows=sorted(group[task],key=lambda x:x['repeat']);assert [r['repeat'] for r in rows]==[1,2,3,4]
  assert len({r['seed'] for r in rows})==4 and all(valid_score(r['score']) for r in rows)
  result.append({'task':task,'scores':[r['score'] for r in rows],'seeds':[r['seed'] for r in rows],'mean_score':sum(r['score'] for r in rows)/4,'full_passes':sum(r['score']==1 for r in rows),'success_rate':sum(r['score']==1 for r in rows)/4,'missing_verifier_scored_zero':sum(not valid_score(r.get('raw_score')) for r in rows),'trial_ids':[r['id'] for r in rows]})
 return result

def paired_summary(a,b,rng,np):
 assert [x['task'] for x in a]==[x['task'] for x in b]
 da=np.array([y['mean_score']-x['mean_score'] for x,y in zip(a,b)]);ds=np.array([y['success_rate']-x['success_rate'] for x,y in zip(a,b)]);idx=rng.integers(0,len(a),size=(10000,len(a)))
 return {'tasks':len(a),'repeats_per_task':4,'score_difference':float(da.mean()),'score_difference_task_bootstrap95':np.quantile(da[idx].mean(1),[.025,.975]).tolist(),'success_rate_difference_pp':100*float(ds.mean()),'success_rate_difference_task_bootstrap95_pp':(100*np.quantile(ds[idx].mean(1),[.025,.975])).tolist(),'improved_task_means':int((da>0).sum()),'worsened_task_means':int((da<0).sum()),'tied_task_means':int((da==0).sum()),'bootstrap_unit':'Question: each model averaged over4 repeats before resampling the same70 task indices; never treat280 rollouts as280 independent questions.'}

def main():
 import numpy as np
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 snapshot_sources(RUN,['aggregate.py','verifier_scores.py','common.py']);state(RUN,'four_condition_comparison',started_at=now());tasks=json.loads((ROOT/'configs/dev-split.json').read_text())['all_tasks'];models={};raws={};summary={};rng=np.random.default_rng(20261113)
 mapping=[('epoch1','baseline-epoch1'),('uniform','baseline-uniform'),('uncapped','baseline-uncapped-guided'),('cap3','eval-guided')]
 for name,rid in mapping:
  run=ROOT/'runs'/rid;assert (run/'complete.json').exists();rows=json.loads((run/'result.json').read_text())['records'];assert len(rows)==280;raws[name]=rows;models[name]=summarize_tasks(rows,tasks);summary[name]=score_metrics(rows)
 for name in ['uniform','uncapped']:assert [r['seed'] for r in raws[name]]==[r['seed'] for r in raws['cap3']]
 pairs={b+'_minus_'+a:paired_summary(models[a],models[b],rng,np) for a,b in [('uniform','cap3'),('uncapped','cap3'),('epoch1','cap3')]}
 sampled=json.loads((ROOT/'runs/sampling/probabilities.json').read_text());teacher=json.loads((ROOT/'runs/teacher-guided/result.json').read_text());counts=collections.Counter(r['task'] for r in teacher['records']);assert len(teacher['records'])==728 and max(counts.values())<=3
 assert counts==collections.Counter(x['task'] for x in sampled['draws']);train=json.loads((ROOT/'runs/sft-guided/result.json').read_text());audit=json.loads((ROOT/'runs/sft-guided/data-encoding-audit.json').read_text());assert train['epoch']>=.999
 metrics={}
 for name,s in summary.items():
  for k in ['mean_score','full_pass_percent','full_passes','missing_verifier_scored_zero']:metrics[name+'_'+k]=s[k]
 for name,p in pairs.items():metrics[name+'_score_delta']=p['score_difference'];metrics[name+'_success_rate_delta_pp']=p['success_rate_difference_pp']
 limits=['All70tasks supplied probe feedback; not independent heldout generalization.','Single training seed. Task bootstrap describes variability across these70tasks, not training-seed variance.','Fresh teacher generation with new seeds; comparison changes both resampling cap and realized teacher trajectories, not an exact deletion-only ablation.','Probability cap plus dependent count rounding changes the sampling joint distribution; counts are not independent multinomial draws.','Failure trajectories retained, no reward-based filtering. Missing terminal verifier scores count0 with rawscore/errors preserved.','Original firstepoch/Uniform/uncapped Guided280outcomes reused; cap3 uses identical post-training task-repeat seeds.']
 result={'finished_at':now(),'summary':summary,'pairwise':pairs,'per_task':models,'sampling':sampled['metrics'],'count_histogram':sampled['count_histogram'],'teacher_metrics':teacher['metrics'],'training':{'epoch':train['epoch'],'steps':train['global_step'],'sft_rows':train['sft_rows'],'target_tokens':audit['target_tokens'],'original_target_tokens':audit['original_target_tokens'],'loss':train['metrics']['train_loss']},'original_training':json.loads((ROOT/'configs/original-comparison.json').read_text())['training'],'metrics':metrics,'limitations':limits};write(RUN/'result.json',result)
 with (RUN/'per-task.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['task']+[name+'_'+k for name in models for k in ['mean_score','full_passes','missing_verifier']])
  for i,t in enumerate(tasks):w.writerow([t]+[v for name in models for v in [models[name][i]['mean_score'],models[name][i]['full_passes'],models[name][i]['missing_verifier_scored_zero']]])
 lines=['# 上限3环境重采样：完整70题×4次结果','', '| 模型 | 次数 | 平均score | 满分次数 | 满分率 | 缺分记0 |','|---|---:|---:|---:|---:|---:|']
 for name,s in summary.items():lines.append(f"| {name} | 280 | {s['mean_score']:.5f} | {s['full_passes']} | {s['full_pass_percent']:.2f}% | {s['missing_verifier_scored_zero']} |")
 lines+=['','每题先平均4次，再对70题做配对bootstrap；不是280个独立问题。','', '| 比较 | 平均score差 | 95%题目bootstrap区间 | 满分率差(pp) |','|---|---:|---|---:|']
 for name,p in pairs.items():lines.append(f"| {name} | {p['score_difference']:+.5f} | {p['score_difference_task_bootstrap95']} | {p['success_rate_difference_pp']:+.3f} |")
 lines+=['','## 抽样与训练','',json.dumps(result['sampling'],ensure_ascii=False),json.dumps(result['training'],ensure_ascii=False),'','![四组对照](comparison.png)','','## 限制','']+['- '+s for s in limits];(RUN/'REPORT.md').write_text('\n'.join(lines)+'\n')
 names=list(models);fig,axes=plt.subplots(1,2,figsize=(11,4));axes[0].bar(names,[summary[n]['mean_score'] for n in names]);axes[0].set_ylabel('Mean verifier score');axes[1].bar(names,[summary[n]['full_pass_percent'] for n in names]);axes[1].set_ylabel('Full-pass rate (%)');fig.suptitle('70 feedback-used tasks x4; missing verifier=0');fig.tight_layout();fig.savefig(RUN/'comparison.png',dpi=160);fig.savefig(RUN/'comparison.pdf');plt.close(fig)
 fig,ax=plt.subplots(figsize=(8,4));old=sorted([r['old_probability'] for r in sampled['rows']],reverse=True);new=sorted([r['probability'] for r in sampled['rows']],reverse=True);ax.plot(range(1,729),old,label='Uncapped');ax.plot(range(1,729),new,label='Cap3');ax.axhline(3/728,ls='--',color='gray',label='3/728 cap');ax.set(xlabel='Environment rank',ylabel='Probability');ax.legend();fig.tight_layout();fig.savefig(RUN/'sampling.png',dpi=160);plt.close(fig)
 for rid in ['comparison','study']:
  p=ROOT/'runs'/rid/'run.json';m=json.loads(p.read_text());m['artifacts'] += [{'label':label,'path':str((RUN/file).relative_to(ROOT))} for label,file in [('完整比较报告','REPORT.md'),('四组结果对照','comparison.png'),('概率上限对比','sampling.png'),('逐题四次结果','per-task.csv'),('统计与置信区间','result.json')]];write(p,m)
 write(RUN/'complete.json',{'finished_at':now()});state(RUN,'complete','succeeded',finished_at=now())
if __name__=='__main__':
 try:main()
 except BaseException as e:state(RUN,'failed','failed',error=str(e),finished_at=now());raise
