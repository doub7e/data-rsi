"""Assemble the completed scaling report from scalar artifacts (no GPU work)."""
import json,math,statistics
from common import *
RUN=ROOT/'runs/probe194-scaling-20261103'
def main():
 assert (RUN/'complete.json').exists()
 result=json.loads((RUN/'result.json').read_text());roll=json.loads((RUN/'rollout-summary.json').read_text());audit=json.loads((RUN/'fresh-trace-audit.json').read_text());p=json.loads((RUN/'protocol.json').read_text())
 assert roll['rollouts']==128 and audit['prompt_checks']==roll['captured_calls']
 partitions=result['final_three_partitions'];cs=[r['cosine'] for r in partitions];assert all(c is not None and math.isfinite(c) for c in cs)
 lines=['# Probe194：增加 rollout 数量后的 split-half 梯度稳定性','',f"128条全新rollout已完成，其中{roll['successes']}条成功。主划分在64+64条时的全参数梯度cosine为 **{result['curve'][-1]['cosine']:.6f}**；三种预先固定分法的范围为 **{min(cs):.6f}–{max(cs):.6f}**。",'', '## 样本量曲线','', '| 总rollout数 | 每半条数 | A成功数 | B成功数 | split-half cosine |','|---:|---:|---:|---:|---:|']
 for row in result['curve']:
  c='不可计算（零梯度）' if row['cosine'] is None else f"{row['cosine']:.6f}"
  lines.append(f"| {row['total_rollouts']} | {row['per_half']} | {row['successes_a']} | {row['successes_b']} | {c} |")
 lines += ['', '![样本量曲线与三种最终划分](scaling.png)','', '## 128条时的三种固定分法','', '将两个预先固定序列分为四块：A前32、A后32、B前32、B后32。以下每个半组都是64条，所有分法均在新rollout产生前固定。','', '| 分法 | 左半成功数 | 右半成功数 | cosine |','|---|---:|---:|---:|']
 names=['A全部 vs B全部','A前+B前 vs A后+B后','A前+B后 vs A后+B前']
 for name,row in zip(names,partitions):lines.append(f"| {name} | {row['successes_a']} | {row['successes_b']} | {row['cosine']:.6f} |")
 lines += ['', '这三种分法共享同一批128条轨迹，不是三次独立采样实验；各样本量也使用嵌套前缀，不应当作独立观测计算显著性。','', '## 计算方法与控制变量','', '- 使用同一道 `multi_turn_miss_func_194` 和原始 Qwen3-4B 初始化，不更新模型权重。','- 新生成128条轨迹，固定随机排列分为A/B各64条；取每半前4、8、16、32、64条。没有按reward配平、挑选或丢弃轨迹。','- 每个半组、每个前缀独立计算自己的平均reward。`g = mean_i [(R_i - mean_half(R)) * grad log P(y_i|x)]`，所有生成tokens计入，包括当前reasoning、工具调用、回答和EOS；prompt仅提供条件，不计入loss。','- 为高效累积，保存成功轨迹log-prob梯度和P、失败轨迹梯度和F，再构成 `g = ((1-r)P - rF)/n`。这与直接中心化autograd目标的微型模型对照最大误差为8.94e-8。','- 比较全部4,022,468,096个参数的cosine，没有做参数抽样、投影或embedding近似；参数/累计为FP32，前向使用BF16 autocast，内积和范数归约为FP64。','- T=1、top_p=1、top_k=-1、每次调用最多4096生成tokens、64K YaRN；每个rollout/调用都有不同的固定seed，实际prompt/completion token IDs均保存。','- 保存各规模的聚合梯度，以及32/64条时的成功/失败累计和用于重分组；没有保存128份逐轨迹完整梯度。','', '## 与此前8条分析的关系','', '此前8条的平衡半组cosine为−0.001427；它用了按结果配平的分组，并在同一rollout各调用复用seed。本次使用全新轨迹、不按结果配平、每次调用独立seed，并启用了strict-v2修复。因此不能把旧8与新128的差异全部归因于样本量。上表新8→新128才是同一协议下的扩量比较。','', '## Harness修复与新数据审计','', f"- 实际评测进程加载strict-v2运行时补丁；有效源码哈希与先前修复一致，148个原受影响prompt回放通过。共享BFCL入口已默认启用该版本；显式设置legacy-v1仍可复现旧formatter。镜像本身没有重建。",f"- 全部{roll['captured_calls']}个实际prompt的历史reasoning检查通过，prompt token IDs逐次核对一致，调用seed全部唯一。",f"- 生成{roll['total_output_tokens']:,}个tokens，最大实际context为{audit['maximum_context_tokens']}tokens。",f"- {len(audit['unclosed_think_outputs'])}条输出在未闭合think时达到长度限制；它们之后的{audit['unique_later_calls_after_unclosed_output']}次调用都没有携带这些历史reasoning。当前截断输出仍按实际生成token计算梯度，并未从数据中删除。",'- 官方BFCL scorer逐条评分，全部128条有原始日志、捕获token和reward；失败轨迹保留。','', '## 解释边界','', '本实验只检验这一题的Monte Carlo probe梯度方向重复性，没有重新给候选200条打分，也没有重训top/bottom模型。较高的split-half cosine不自动证明训练数据排名稳定或下游训练有效；旧的全32题probe及其旧评测也未被本实验替换。','', '## 存储恢复记录','', '梯度计算已完成，但最终失败轨迹累计和F64的写入触发共享存储Disk quota exceeded。全部10个主梯度快照、P64和32条时的累计和均已完整保存。保留失败日志后，仅删除约28.5GB未完成的辅助副本。主曲线直接读取完整g向量；额外分组所需F64通过 `F=((1-r)P64-64*g64)/r` 在内存中重构，没有重新生成或筛选数据。主划分的直接结果与重构结果进行数值核对，差值见verification.json。','', '## 产物','', '[完整统计](result.json) · [采样协议与分组](protocol.json) · [生成汇总](rollout-summary.json) · [实际轨迹审计](fresh-trace-audit.json) · [代码哈希](code-provenance.json) · [PDF图表](scaling.pdf)','', '原始逐次生成与官方评分在 `bfcl/r001` 至 `bfcl/r128`；token轨迹在 `traces/`；全部128条已投影到看板轨迹查看器。']
 (RUN/'REPORT.md').write_text('\n'.join(lines)+'\n')
 m=json.loads((RUN/'run.json').read_text());m['status']='succeeded';m['completed_at']=result['finished_at']
 existing={a['path'] for a in m['artifacts']}
 for label,name in [('Analysis report','REPORT.md'),('Three-split results','result.json'),('Fresh trace audit','fresh-trace-audit.json'),('Harness preflight','harness-preflight.json'),('PDF figure','scaling.pdf')]:
  path=f'runs/{RUN.name}/{name}'
  if path not in existing:m['artifacts'].append({'label':label,'path':path})
 write(RUN/'run.json',m)
 check={'checked_at':now(),'samples':128,'curve_points':len(result['curve']),'final_partitions':len(partitions),'direct_vs_reconstructed_128_cosine_difference':abs(result['curve'][-1]['cosine']-partitions[0]['cosine']),'all_new_prompt_tokens_verified':roll['all_prompt_token_ids_verified'],'all_new_histories_clean':audit['all_later_histories_clean'],'half_sample_counts':[len(p['halves'][h]) for h in 'ab'],'half_overlap':len(set(p['halves']['a'])&set(p['halves']['b']))}
 assert check['half_overlap']==0 and check['direct_vs_reconstructed_128_cosine_difference']<1e-6;write(RUN/'verification.json',check);print(json.dumps(check,indent=2))
if __name__=='__main__':main()
