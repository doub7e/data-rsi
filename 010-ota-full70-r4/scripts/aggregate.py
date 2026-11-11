"""Three checkpoints,70 tasks×4: fixed missing-as-zero scores and task-level inference."""
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
 snapshot_sources(RUN,['aggregate.py','verifier_scores.py','common.py']);state(RUN,'three_model_four_repeat_comparison',started_at=now());tasks=json.loads((ROOT/'configs/dev-split.json').read_text())['all_tasks'];models={};raws={};summary={};rng=np.random.default_rng(20261113)
 for model,rid in [('epoch1','probe'),('uniform','eval-uniform'),('guided','eval-guided')]:
  run=ROOT/'runs'/rid;assert (run/'complete.json').exists();rows=json.loads((run/'result.json').read_text())['records'];assert len(rows)==280;raws[model]=rows;models[model]=summarize_tasks(rows,tasks);s=score_metrics(rows);s['tasks_with_any_full_pass_in4']=sum(x['full_passes']>0 for x in models[model]);s['tasks_with_all4_full_passes']=sum(x['full_passes']==4 for x in models[model]);s['per_task_full_pass_histogram']={str(k):sum(x['full_passes']==k for x in models[model]) for k in range(5)};summary[model]=s
 assert [r['seed'] for r in raws['uniform']]==[r['seed'] for r in raws['guided']]
 pairs={}
 for a,b in [('epoch1','uniform'),('epoch1','guided'),('uniform','guided')]:pairs[b+'_minus_'+a]=paired_summary(models[a],models[b],rng,np)
 probe=json.loads((ROOT/'runs/probe-gradient/probe-gradient-complete.json').read_text());sampling=json.loads((ROOT/'runs/attribution/probabilities.json').read_text());teacher=json.loads((ROOT/'runs/teacher-guided/result.json').read_text());oldteacher=json.loads((ORIGIN/'runs/teacher-uniform/result.json').read_text())
 training={}
 for name,src in [('epoch1',ORIGIN/'runs/sft-epoch1'),('uniform',ORIGIN/'runs/sft-uniform'),('guided',ROOT/'runs/sft-guided')]:
  tr=json.loads((src/'result.json').read_text());audit=json.loads((src/'data-encoding-audit.json').read_text());training[name]={'source':str(src),'epoch':tr['epoch'],'cumulative_training_stages':1 if name=='epoch1' else 2,'steps':tr['global_step'],'target_tokens':audit['target_tokens'],'sft_rows':tr['sft_rows'],'loss':tr['metrics']['train_loss']}
 byenv=collections.defaultdict(list)
 for row in teacher['records']:byenv[row['task']].append(row)
 top=[{'task':r['task'],'probability':r['probability'],'draws':r['draw_count'],'fresh_teacher':score_metrics(byenv[r['task']])} for r in sorted(sampling['rows'],key=lambda r:r['probability'],reverse=True)[:10]]
 metrics={}
 for m,s in summary.items():
  for k in ['mean_score','full_pass_percent','full_passes','missing_verifier_scored_zero','tasks_with_any_full_pass_in4']:metrics[m+'_'+k]=s[k]
 for k,v in pairs.items():metrics[k+'_mean_score_delta']=v['score_difference'];metrics[k+'_success_rate_delta_pp']=v['success_rate_difference_pp']
 limitations=['All70 tasks provided probe feedback; this is feedback-used benchmark performance, not independent heldout generalization.','Only one training seed. Four inference repeats reduce single-rollout noise but do not measure training-seed variance.','Missing terminal verifier scores explicitly count as0 per user; original rewards/errors and missingness counts are retained.','Uniform repeats1 are reused from009, repeats2–4 are new; same pinned checkpoint, harness, task images, sampling and post-training seeds.','Epoch1/probe seeds differ from post-training seeds; Uniform and newGuided share task-repeat seeds.','Equal728 regenerated teacher trajectories, not necessarily equal unique environments or target tokens.','Raw full-parameter SFT-loss gradient dots; not exact AdamW-preconditioned influence.']
 result={'finished_at':now(),'score_policy':'missing_verifier_as_zero','feedback_tasks':70,'independent_heldout_tasks':0,'summary':summary,'pairwise':pairs,'per_task':models,'probe':probe,'sampling':sampling['metrics'],'training':training,'teacher_data':{'uniform':oldteacher['metrics'],'guided':teacher['metrics']},'top_environment_regeneration':top,'metrics':metrics,'limitations':limitations};write(RUN/'result.json',result)
 with (RUN/'per-task.csv').open('w',newline='') as f:
  writer=csv.writer(f);writer.writerow(['task']+[m+'_'+k for m in models for k in ['mean_score','full_passes_out_of4','missing_verifier']])
  for i,t in enumerate(tasks):writer.writerow([t]+[v for m in models for v in [models[m][i]['mean_score'],models[m][i]['full_passes'],models[m][i]['missing_verifier_scored_zero']]])
 lines=['# 全70题×4次：第一轮、Uniform与新Guided','', '所有70题用于probe反馈；本表不能解读为独立留出集结果。缺少原verifier分数的已结束尝试按用户要求计0，raw_score与异常保持原样。','', '| 模型 | 尝试数 | 平均score | 满分次数 | 满分率 | 缺分计0次数 | 4次中至少一次满分的题数 |','|---|---:|---:|---:|---:|---:|---:|']
 for name,s in summary.items():lines.append(f"| {name} | 280 | {s['mean_score']:.5f} | {s['full_passes']} | {s['full_pass_percent']:.2f}% | {s['missing_verifier_scored_zero']} | {s['tasks_with_any_full_pass_in4']}/70 |")
 lines+=['','先将每题4次结果取均值，再按相同70道题进行配对比较和bootstrap。','', '| 比较 | 平均score差 | 95%题目bootstrap区间 | 满分率差(pp) |','|---|---:|---|---:|']
 for name,s in pairs.items():lines.append(f"| {name} | {s['score_difference']:+.5f} | {s['score_difference_task_bootstrap95']} | {s['success_rate_difference_pp']:+.3f} |")
 lines+=['',f"有效probe题数：{probe['valid_questions']}；排除恒分题数：{probe['excluded_questions']}。",'',json.dumps(sampling['metrics'],ensure_ascii=False),'','## 训练量','', '| 模型 | 本阶段epochs | 本阶段optimizer steps | 监督目标tokens |','|---|---:|---:|---:|']
 for name,t in training.items():lines.append(f"| {name} | {t['epoch']} | {t['steps']} | {t['target_tokens']} |")
 lines+=['','![模型对照](comparison.png)','','![采样概率](sampling.png)','','![训练曲线](training-curves.png)','','## 限制','']+['- '+x for x in limitations];(RUN/'REPORT.md').write_text('\n'.join(lines)+'\n')
 names=['epoch1','uniform','guided'];fig,axes=plt.subplots(1,2,figsize=(10,4));axes[0].bar(names,[summary[m]['mean_score'] for m in names]);axes[0].set_ylabel('Mean score (missing verifier = 0)');axes[0].set_ylim(0,1);axes[1].bar(names,[summary[m]['full_pass_percent'] for m in names]);axes[1].set_ylabel('Full-pass rate over280 rollouts (%)');axes[1].set_ylim(0,100);fig.suptitle('All70 feedback-used tasks,4 repeats per model');fig.tight_layout();fig.savefig(RUN/'comparison.png',dpi=160);fig.savefig(RUN/'comparison.pdf');plt.close(fig)
 probabilities=sorted([r['probability'] for r in sampling['rows']],reverse=True);fig,ax=plt.subplots(figsize=(8,4));ax.plot(range(1,729),probabilities);ax.axhline(1/728,color='gray',linestyle='--',label='Uniform');ax.set_xlabel('Environment rank');ax.set_ylabel('Sampling probability');ax.legend();fig.tight_layout();fig.savefig(RUN/'sampling.png',dpi=160);plt.close(fig)
 fig,ax=plt.subplots(figsize=(8,4))
 for name,src in [('epoch1',ORIGIN/'runs/sft-epoch1'),('uniform',ORIGIN/'runs/sft-uniform'),('guided',ROOT/'runs/sft-guided')]:
  points=[json.loads(l) for l in (src/'checkpoint/trainer_log.jsonl').read_text().splitlines() if l.strip()];points=[x for x in points if 'loss' in x];ax.plot([x.get('current_steps',x.get('step',i)) for i,x in enumerate(points)],[x['loss'] for x in points],label=name)
 ax.set_xlabel('Optimizer step within stage');ax.set_ylabel('Training loss');ax.legend();fig.tight_layout();fig.savefig(RUN/'training-curves.png',dpi=160);plt.close(fig)
 for rid in ['comparison','study']:
  path=ROOT/'runs'/rid/'run.json';m=json.loads(path.read_text());m['artifacts'] += [{'label':label,'path':str((RUN/file).relative_to(ROOT))} for label,file in [('完整报告','REPORT.md'),('模型对照图','comparison.png'),('采样分布','sampling.png'),('训练曲线','training-curves.png'),('逐题四次结果','per-task.csv'),('详细结果与置信区间','result.json')]];write(path,m)
 write(RUN/'progress.json',{'stage':'three_model_comparison','done':3,'total':3,'updated_at':now()});write(RUN/'complete.json',{'finished_at':now()});state(RUN,'complete','succeeded',finished_at=now())
if __name__=='__main__':
 try:main()
 except BaseException as e:state(RUN,'failed','failed',error=str(e),finished_at=now());raise
