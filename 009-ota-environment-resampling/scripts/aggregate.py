"""Paired comparison using only original verifier measurements, with missingness explicit."""
import math
from common import *
from verifier_scores import valid_score,score_metrics
RUN=ROOT/'runs/comparison'

def compare_subset(models,tasks,rng,np):
 coverage={b:score_metrics(models[b][t] for t in tasks) for b in models}
 paired=[t for t in tasks if all(valid_score(models[b][t].get('score')) for b in models)]
 a=np.array([models['uniform'][t]['score'] for t in paired],dtype=float);b=np.array([models['guided'][t]['score'] for t in paired],dtype=float);delta=b-a
 if paired:
  indices=rng.integers(0,len(paired),size=(10000,len(paired)));boot=delta[indices].mean(1);binary=(b==1).astype(float)-(a==1).astype(float);bootbinary=binary[indices].mean(1)
  paired_values={'paired_uniform_mean_score':float(a.mean()),'paired_guided_mean_score':float(b.mean()),'score_difference':float(delta.mean()),'score_difference_bootstrap95':np.quantile(boot,[.025,.975]).tolist(),'full_pass_difference_percentage_points':100*float(binary.mean()),'full_pass_difference_bootstrap95_pp':(100*np.quantile(bootbinary,[.025,.975])).tolist()}
 else:paired_values={k:None for k in ['paired_uniform_mean_score','paired_guided_mean_score','score_difference','score_difference_bootstrap95','full_pass_difference_percentage_points','full_pass_difference_bootstrap95_pp']}
 lower=[];upper=[]
 for t in tasks:
  u=models['uniform'][t]['score'];g=models['guided'][t]['score'];ulo,uhi=(u,u) if valid_score(u) else (0.,1.);glo,ghi=(g,g) if valid_score(g) else (0.,1.);lower.append(glo-uhi);upper.append(ghi-ulo)
 return {'tasks':len(tasks),'paired_scored_tasks':len(paired),'unpaired_due_to_missing_verifier':[t for t in tasks if t not in paired],'coverage':coverage,'uniform_mean_score':coverage['uniform']['mean_score'],'guided_mean_score':coverage['guided']['mean_score'],**paired_values,'full_subset_score_difference_unknown_outcome_bounds':[sum(lower)/len(tasks),sum(upper)/len(tasks)],'improved_tasks':int((delta>0).sum()),'worsened_tasks':int((delta<0).sum()),'tied_tasks':int((delta==0).sum()),'uncertainty_note':'Bootstrap is conditional on jointly verifier-scored tasks. Separate full-subset bounds allow any0..1 outcome for unscored tasks; these bounds do not assign scores to missing observations.'}

def main():
 import numpy as np
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 state(RUN,'paired_original_verifier_analysis',started_at=now());split=json.loads((ROOT/'configs/dev-split.json').read_text());models={}
 for branch in ['uniform','guided']:
  raw=json.loads((ROOT/f'runs/eval-{branch}/result.json').read_text())['records'];assert len(raw)==70;models[branch]={r['task']:r for r in raw};assert set(models[branch])==set(split['all_tasks'])
 rng=np.random.default_rng(20261110);subsets={};metrics={}
 for name,tasks in [('all70',split['all_tasks']),('probe14',split['probe_tasks']),('heldout56',split['heldout_tasks'])]:
  v=compare_subset(models,tasks,rng,np);subsets[name]=v
  metrics.update({name+'_uniform_mean_score':v['uniform_mean_score'],name+'_guided_mean_score':v['guided_mean_score'],name+'_paired_score_difference':v['score_difference'],name+'_paired_scored_tasks':v['paired_scored_tasks'],name+'_uniform_unscored':v['coverage']['uniform']['verifier_unscored_trials'],name+'_guided_unscored':v['coverage']['guided']['verifier_unscored_trials']})
 data=[];training=[];teacher_records={}
 for rid in ['teacher-batch1','teacher-uniform','teacher-guided']:
  result=json.loads((ROOT/'runs'/rid/'result.json').read_text());data.append({'stage':rid,**result['metrics']});teacher_records[rid]=result['records']
 for rid in ['sft-epoch1','sft-uniform','sft-guided']:
  result=json.loads((ROOT/'runs'/rid/'result.json').read_text());audit=json.loads((ROOT/'runs'/rid/'data-encoding-audit.json').read_text());training.append({'stage':rid,'steps':result['global_step'],'epoch':result['epoch'],**audit})
 sampled=json.loads((ROOT/'runs/attribution/probabilities.json').read_text());probe=json.loads((ROOT/'runs/probe-gradient/probe-gradient-complete.json').read_text());records=[]
 for t in split['all_tasks']:
  u=models['uniform'][t]['score'];g=models['guided'][t]['score'];records.append({'task':t,'subset':'probe' if t in split['probe_tasks'] else 'heldout','uniform_score':u,'guided_score':g,'difference':g-u if valid_score(g) and valid_score(u) else None})
 # Descriptive regeneration audit; frozen sampling is never revised by this analysis.
 import collections
 by_environment=collections.defaultdict(list)
 for row in teacher_records['teacher-guided']:by_environment[row['task']].append(row)
 initial={r['id']:r for r in teacher_records['teacher-batch1']}
 regenerated=[{'task':task,**score_metrics(rows)} for task,rows in sorted(by_environment.items())]
 top_selected=[{'id':r['id'],'task':r['task'],'probability':r['probability'],'planned_draws':r['draw_count'],'original_teacher_score':initial[r['id']]['score'],'fresh_teacher':score_metrics(by_environment[r['task']])} for r in sorted(sampled['rows'],key=lambda x:x['probability'],reverse=True)[:10]]
 write(RUN/'regeneration-audit.json',{'frozen_sampling':True,'top_selected_environments':top_selected,'per_environment':regenerated,'note':'Descriptive actual teacher outcomes; no success-only filtering or probability changes.'})
 limitations=['Single training seed and one eval rollout/task/model; task bootstrap excludes training-seed uncertainty.','Heldout56 is primary. Probe14 was used for feedback and is not independent validation.','Missing original verifier scores remain null. No custom failure zeros. Paired estimates condition on joint score availability; report coverage and full-subset unknown-outcome bounds.','Probe questions require32 original verifier scores and nonconstant outcomes; incomplete questions are excluded without replacement.','Equal728 new trajectories per branch; supervised token counts and exported segment counts may differ.','Uniform visits each environment once; guided samples with replacement, so coverage/multiplicity differ.','Raw full-parameter SFT-loss gradient dots, not AdamW-preconditioned influence or exact on-policy gradients.','All728 RL training tasks are NL2Bash/terminal-automation; TB-dev covers broader task categories.','Budget failures without an accepted assistant turn have zero exported SFT rows; rejected raw generation remains preserved.']
 result={'finished_at':now(),'score_policy':'original_verifier_only','subsets':subsets,'per_task':records,'teacher_data':data,'training_data':training,'sampling':sampled['metrics'],'probe':probe,'top_selected_environment_regeneration':top_selected,'metrics':metrics,'limitations':limitations};write(RUN/'result.json',result)
 def fmt(x):return 'NA' if x is None else f'{x:.5f}'
 lines=['# 环境重采样：Uniform vs probe-dot guided','', '原verifier没有返回分数时保留null，不人为补0。未参与probe的56题为主要比较；置信区间基于两模型都有原verifier分数的同一批题目。','', '| 子集 | 计划题数 | 配对已评分题数 | Uniform配对均值 | Guided配对均值 | 配对差值 | bootstrap95%区间 |','|---|---:|---:|---:|---:|---:|---|']
 for name,v in subsets.items():lines.append(f"| {name} | {v['tasks']} | {v['paired_scored_tasks']} | {fmt(v['paired_uniform_mean_score'])} | {fmt(v['paired_guided_mean_score'])} | {fmt(v['score_difference'])} | {v['score_difference_bootstrap95']} |")
 lines+=['','## 原verifier覆盖率与缺失结果','', '| 子集 | Uniform已评分/计划 | Guided已评分/计划 | 完整子集差值的缺失结果上下界 |','|---|---:|---:|---|']
 for name,v in subsets.items():lines.append(f"| {name} | {v['coverage']['uniform']['verifier_scored_trials']}/{v['tasks']} | {v['coverage']['guided']['verifier_scored_trials']}/{v['tasks']} | {v['full_subset_score_difference_unknown_outcome_bounds']} |")
 lines+=['','上下界是允许未知结果在0至1范围变化得到的不确定性范围，不是给缺失结果填分。','', '## 数据与训练量','', '| 阶段 | 轨迹数 | SFT片段数 | 有分数样本的教师均值 |','|---|---:|---:|---:|']
 for x in data:lines.append(f"| {x['stage']} | {x['evaluated_trials']} | {x['sft_rows']} | {fmt(x['mean_score'])} |")
 lines+=['','| 训练阶段 | optimizer steps | 目标tokens | 原始目标tokens保留率 |','|---|---:|---:|---:|']
 for x in training:lines.append(f"| {x['stage']} | {x['steps']} | {x['target_tokens']} | {100*x['retained_target_fraction']:.2f}% |")
 lines+=['','## 再生成的实际表现','', '| 高概率环境 | 首批教师score | 抽样次数 | 新轨迹有分数/总数 | 新轨迹平均score |','|---|---:|---:|---:|---:|']
 for x in top_selected:lines.append(f"| {x['task']} | {fmt(x['original_teacher_score'])} | {x['planned_draws']} | {x['fresh_teacher']['verifier_scored_trials']}/{x['fresh_teacher']['evaluated_trials']} | {fmt(x['fresh_teacher']['mean_score'])} |")
 lines+=['','## 选样','',json.dumps(sampled['metrics'],ensure_ascii=False),f"\n有效probe题数：{probe['valid_questions']}；平均probe范数：{probe['probe_norm']}。",'', '![Score comparison](comparison.png)','', '![Sampling probabilities](sampling.png)','', '## 限制','']+['- '+x for x in limitations];(RUN/'REPORT.md').write_text('\n'.join(lines)+'\n')
 fig,ax=plt.subplots(figsize=(8,4));names=list(subsets);xx=np.arange(len(names));ax.bar(xx-.18,[subsets[n]['paired_uniform_mean_score'] if subsets[n]['paired_uniform_mean_score'] is not None else np.nan for n in names],width=.36,label='Uniform');ax.bar(xx+.18,[subsets[n]['paired_guided_mean_score'] if subsets[n]['paired_guided_mean_score'] is not None else np.nan for n in names],width=.36,label='Probe-dot guided');ax.set_xticks(xx,[n+'\npaired '+str(subsets[n]['paired_scored_tasks'])+'/'+str(subsets[n]['tasks']) for n in names]);ax.set_ylabel('Mean original verifier score (paired tasks)');ax.set_ylim(0,1);ax.legend();fig.tight_layout();fig.savefig(RUN/'comparison.png',dpi=160);fig.savefig(RUN/'comparison.pdf');plt.close(fig)
 probs=sorted([r['probability'] for r in sampled['rows']],reverse=True);fig,ax=plt.subplots(figsize=(8,4));ax.plot(range(1,len(probs)+1),probs);ax.axhline(1/len(probs),color='gray',linestyle='--',label='Uniform');ax.set_xlabel('Environment rank by probability');ax.set_ylabel('Sampling probability');ax.legend();fig.tight_layout();fig.savefig(RUN/'sampling.png',dpi=160);plt.close(fig)
 fig,ax=plt.subplots(figsize=(8,4))
 for rid in ['sft-epoch1','sft-uniform','sft-guided']:
  file=ROOT/'runs'/rid/'checkpoint/trainer_log.jsonl';points=[json.loads(x) for x in file.read_text().splitlines() if x.strip()];points=[x for x in points if 'loss' in x];ax.plot([x.get('current_steps',x.get('step',i)) for i,x in enumerate(points)],[x['loss'] for x in points],label=rid)
 ax.set_xlabel('Optimizer step within stage');ax.set_ylabel('Training loss');ax.legend();fig.tight_layout();fig.savefig(RUN/'training-curves.png',dpi=160);plt.close(fig)
 for rid in ['comparison','study']:
  path=ROOT/'runs'/rid/'run.json';m=json.loads(path.read_text());m['artifacts'] += [{'label':label,'path':str((RUN/name).relative_to(ROOT))} for label,name in [('最终配对报告','REPORT.md'),('评测比较图','comparison.png'),('采样概率','sampling.png'),('训练曲线','training-curves.png'),('逐题结果、评分覆盖率与置信区间','result.json'),('高概率环境的再生成结果','regeneration-audit.json')]];write(path,m)
 write(RUN/'complete.json',{'finished_at':now()});state(RUN,'complete','succeeded',finished_at=now())
if __name__=='__main__':
 try:main()
 except BaseException as e:state(RUN,'failed','failed',error=str(e),finished_at=now());raise
