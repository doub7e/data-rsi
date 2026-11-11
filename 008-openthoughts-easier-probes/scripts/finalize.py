"""Finish score-based report from saved metadata, with no model execution."""
import contextlib,io
from common import *
RUN=ROOT/'runs/easy-probes-20261105'
def main():
 from analyze_results import main as analyze
 with contextlib.redirect_stdout(io.StringIO()):analyze()
 p=json.loads((RUN/'protocol.json').read_text());screen=json.loads((RUN/'screening-results.json').read_text());rows=json.loads((RUN/'interpretation.json').read_text())['questions'];fmt=lambda x:'无定义' if x is None else f'{x:.5f}'
 lines=['# OTA 简单题：原始 Qwen3-8B score-based probe','', '使用官方原始 score；权重为 score−本 half 平均 score，不做标准差归一化，不二值化。六题各8条筛查，按冻结的分数变化规则选三题，再各采32条新轨迹。筛查不进入正式梯度。','', '| 筛查任务 | 原始scores | 平均score | 方差 | 满分/8（仅描述） | 选入 |','|---|---|---:|---:|---:|---|']
 chosen={q['task'] for q in p['tasks']}
 for x in screen['candidates']:lines.append(f"| {x['candidate']['task']} | {x['scores']} | {x['mean_score']:.4f} | {x['score_variance']:.4f} | {x['successes']}/8 | {'是' if x['candidate']['task'] in chosen else '否'} |")
 lines+=['','| 正式任务 | 平均score | score方差 | 满分/32（仅描述） | 主划分cosine | 划分2 | 划分3 |','|---|---:|---:|---:|---:|---:|---:|']
 metrics={'screened_tasks':6,'screening_rollouts':48,'formal_rollouts':96,'formal_questions':3}
 for x in rows:
  lines.append(f"| {x['task']} | {x['mean_score']:.5f} | {x['score_variance']:.5f} | {x['successes']}/32 | "+' | '.join(fmt(y['global']['cosine']) for y in x['partitions'])+' |');metrics[x['key']+'_mean_score']=x['mean_score'];metrics[x['key']+'_score_variance']=x['score_variance']
 lines+=['','每个 rollout 的所有 SFT 片段共用其 score。原训练的全 assistant（含 reasoning）监督、32K截断和有效目标token归一化保持不变：g_H=−Σ_i(score_i−mean_H score)∇CE_sum(i)/T_H。训练式上下文重放仍不等于严格 on-policy gradient。','', '![比较图](comparison.png)','']
 for x in rows:
  q=x['key'];lines += [f"## {x['task']}",'',f"目标tokens保留率 {100*x['retained_target_fraction']:.2f}%；pooled32梯度范数 {x['pooled_gradient_norm']:.6f}。",'', '| 划分 | A/B平均score | cosine | 正cosine层/有效层 | 层中位数 |','|---|---|---:|---|---:|']
  for i,z in enumerate(x['partitions'],1):
   lines.append(f"| {i} | {z['mean_score_a']:.4f}/{z['mean_score_b']:.4f} | {fmt(z['global']['cosine'])} | {z['layers']['positive']}/{z['layers']['defined']} | {fmt(z['layers']['median'])} |")
   if z['global']['cosine'] is not None:metrics[q+f'_partition{i}_cosine']=z['global']['cosine']
  if x.get('reward_baseline_decomposition'):
   lines+=['','分解为 reward加权项 A=−Σscore∇CE_sum/T 与 baseline项 B=mean(score)Σ∇CE_sum/T；g=A+B。它们不是成功项/失败项。','', '| 划分 | A跨半cosine | B跨半cosine | 半1：A/B范数 | 半2：A/B范数 |','|---|---:|---:|---|---|']
   for i,z in enumerate(x['reward_baseline_decomposition'],1):
    a,b=z['halves'];lines.append(f"| {i} | {fmt(z['across_half_reward_weighted_term_cosine'])} | {fmt(z['across_half_baseline_term_cosine'])} | {a['reward_weighted_term_norm']:.5f}/{a['baseline_term_norm']:.5f} | {b['reward_weighted_term_norm']:.5f}/{b['baseline_term_norm']:.5f} |")
  else:lines+=['','所有score相同，中心化梯度解析为零，cosine无定义。']
  lines+=['',f"[逐层表](../{q}-gradient-20261105/layers.csv) · [原始统计](../{q}-gradient-20261105/result.json)",'']
 lines+=['## 限制','', '筛选的是分数有变化的简单题，不代表整个benchmark；三种划分共享数据，并非独立重复。高cosine不保证更新后score提高；本轮没有执行参数更新。原始grader定义score的意义，部分得分也可能只衡量文件格式等子条件。','', '[score迁移记录](score-migration.json) · [正式协议](formal-protocol.json) · [完整分解](interpretation.json)']
 (RUN/'ANALYSIS.md').write_text('\n'.join(lines)+'\n');result=json.loads((RUN/'result.json').read_text());result['metrics'].update(metrics);result['probe_reward_definition']='official_score_v1';write(RUN/'result.json',result)
 m=json.loads((RUN/'run.json').read_text());known={a['path'] for a in m['artifacts']}
 for label,name in [('最终score-based分析','ANALYSIS.md'),('结构化梯度分解','interpretation.json')]:
  path=str((RUN/name).relative_to(ROOT))
  if path not in known:m['artifacts'].append({'label':label,'path':path})
 write(RUN/'run.json',m);write(RUN/'analysis-complete.json',{'finished_at':now(),'metrics':metrics});print('SCORE_ANALYSIS_COMPLETE',flush=True)
if __name__=='__main__':main()
