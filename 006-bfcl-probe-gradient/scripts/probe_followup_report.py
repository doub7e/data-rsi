"""Assemble completed diagnostics and paired two-probe ablations."""
import csv,json,math,statistics
from pathlib import Path
from common import *
MODES=['all_tokens','no_reasoning','key_turn_all','key_turn_no_reasoning']
LABELS={'all_tokens':'全部tokens','no_reasoning':'去掉reasoning','key_turn_all':'仅关键turn（含reasoning）','key_turn_no_reasoning':'仅关键turn且去掉reasoning'}
def fmt(x):return '不可计算' if x is None else f'{x:.6f}'
def case_report(run):
 r=json.loads((run/'result.json').read_text());mask=json.loads((run/'mask-audit.json').read_text());cfg=json.loads((run/'protocol.json').read_text());counts=[json.loads((run/f'rank-{i}-trajectory-counts.json').read_text()) for i in (0,1)]
 likelihood={}
 for mode in MODES:
  entries=[x.get('likelihood_comparison',{}).get(mode) for rows in counts for x in rows];entries=[x for x in entries if x and x['compared_tokens']]
  if entries:
   n=sum(x['compared_tokens'] for x in entries);hf=sum(x['hf_nll'] for x in entries);v=sum(x['vllm_actual_decode_nll'] for x in entries)
   likelihood[mode]={'tokens':n,'hf_nll':hf,'vllm_decode_nll':v,'mean_signed_logprob_difference':(v-hf)/n,'absolute_trajectory_nll_gap_per_token':sum(abs(x['hf_nll']-x['vllm_actual_decode_nll']) for x in entries)/n,'note':'Trajectory sums allow cancellation. The latter is not a per-token absolute error.'}
 write(run/'likelihood-aggregate.json',likelihood)
 text=f"# {r['probe_id']}：token/turn梯度消融\n\n固定128条轨迹、原64+64分组、每半独立reward均值。关键用户回合为0-based turn {r['key_turn']}。所有模式保持实际输入上下文，只改变监督token位置；没有重新采样或按结果配平。\n\n| 模式 | 监督tokens | 全参数cosine | A范数 | B范数 |\n|---|---:|---:|---:|---:|\n"
 for mode in MODES:
  s=r['results'][mode]['all'];text+=f"| {LABELS[mode]} | {mask['target_tokens'][mode]} | {fmt(s['cosine'])} | {s['norm_a']:.4f} | {s['norm_b']:.4f} |\n"
 text+='\n![消融与逐层cosine](ablation.png)\n\n## 逐层结果\n\n| 模式 | Decoder层最小cosine | 最大cosine | 中位数 | 正值层数/36 |\n|---|---:|---:|---:|---:|\n'
 table=[]
 for mode in MODES:
  vals=[r['results'][mode][f'layer-{i:02d}']['cosine'] for i in range(36)];finite=[v for v in vals if v is not None]
  text+=f"| {LABELS[mode]} | {fmt(min(finite) if finite else None)} | {fmt(max(finite) if finite else None)} | {fmt(statistics.median(finite) if finite else None)} | {sum(v>0 for v in finite)}/36 |\n"
  tot=r['results'][mode]['all']
  for layer,s in r['results'][mode].items():
   if layer=='all':continue
   table.append({'mode':mode,'layer':layer,'cosine':s['cosine'],'norm_a':s['norm_a'],'norm_b':s['norm_b'],'squared_norm_fraction_a':s['norm_a']**2/tot['norm_a']**2 if tot['norm_a'] else None,'squared_norm_fraction_b':s['norm_b']**2/tot['norm_b']**2 if tot['norm_b'] else None})
 with (run/'layer-cosines.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
 text+='\n[完整逐层表](layer-cosines.csv)同时给出平方范数占比，避免把小范数层的高cosine误作全局强信号。全局cosine由完整向量计算，不是逐层cosine平均。\n\n## 解释边界\n\n- “去掉reasoning”保留工具调用、普通可见回答和EOS，不等于仅保留结构化工具调用。Reasoning仍留在当前调用上下文中，只不作为预测目标。\n- 关键turn中的所有模型调用均保留；其他turn贡献为零。缺失目标token的轨迹仍计入64条及原reward baseline，不会从分母删除。\n- 去掉reasoning或只选关键turn会改变梯度目标，是诊断消融，不能默认视为原完整policy-gradient估计的无偏降方差版本。\n- HF/vLLM存在已量化的log-prob数值差异；本比较固定原HF方法，不声称精确on-policy等价。\n- 第一题关键turn来自此前定性审计；第二题关键turn在新rollout生成前按任务语义固定。没有搜索哪一层/哪个turn的cosine最高后再只报告该处。\n- 只报告一个预先固定split；尚未据此重新选择训练数据或验证下游训练收益。\n'
 if likelihood:text+='\n## 实际decode likelihood对照\n\n第二题记录了实际decode log-prob。按相同mask汇总的HF/vLLM NLL差异见[likelihood-aggregate.json](likelihood-aggregate.json)；这些是轨迹求和的差异，不能当作逐token绝对误差。\n'
 (run/'REPORT.md').write_text(text)
 m=json.loads((run/'run.json').read_text());m['status']='succeeded';m['completed_at']=r['finished_at'];existing={a['path'] for a in m['artifacts']}
 for label,name in [('Case report','REPORT.md'),('Layer table','layer-cosines.csv'),('Actual decode likelihood','likelihood-aggregate.json')]:
  path=f'runs/{run.name}/{name}'
  if path not in existing:m['artifacts'].append({'label':label,'path':path})
 write(run/'run.json',m);return r

def main():
 first=ROOT/'runs/probe194-ablation-20261103';second=ROOT/'runs/probe22-ablation-20261103';data=[]
 for run in (first,second):
  if (run/'complete.json').exists():data.append(case_report(run))
 if len(data)!=2:return
 diagnostic=json.loads((ROOT/'runs/probe-diagnostics-20261103/result.json').read_text());reps=json.loads((ROOT/'runs/probe-diagnostics-20261103/gradient-repeat-summary.json').read_text())['cases']
 text='# 两道probe的梯度诊断与消融对比\n\n## 计算一致性\n\n三个真实固定轨迹的重复梯度cosine均约0.99996，相对L2差约0.84%–0.89%；最低逐层重复cosine仍大于0.9999。HF train/eval的likelihood相同。HF与vLLM log-prob平均绝对差约0.016nats、P99约0.24nats，尾部达到调查阈值，不能声称严格数值等价。详见[诊断报告](../probe-diagnostics-20261103/REPORT.md)。\n\n## 两题同规模对比\n\n每题使用128条轨迹、固定64+64划分；各模式共享轨迹、reward和分组，只改变loss mask。\n\n| 题目 | 全部tokens | 去掉reasoning | 仅关键turn（全部token） | 关键turn且去掉reasoning |\n|---|---:|---:|---:|---:|\n'
 for r in data:text+='| '+r['probe_id']+' | '+' | '.join(fmt(r['results'][m]['all']['cosine']) for m in MODES)+' |\n'
 text+='\n![两题消融和逐层cosine](comparison.png)\n\n第一题是旅行预订/缺失工具恢复，关键turn为客服工具恢复回合；第二题是模拟文件系统到模拟发帖工具的流程，关键turn为最终认证和发帖回合。它们不是完全同构的任务，因此跨题差异不能仅归因于某一个因素。\n\n[第一题详细报告](../probe194-ablation-20261103/REPORT.md) · [第二题详细报告](REPORT.md) · [两题逐层CSV](layer-comparison.csv) · [完整对比JSON](comparison.json)\n\n这些消融改变梯度目标；方向更稳定不自动证明选出的训练数据更有用。当前没有新训练模型，也没有重新评测原top/bottom模型。\n'
 (second/'COMPARISON.md').write_text(text);m=json.loads((second/'run.json').read_text());m['artifacts'] += [{'label':'Two-probe comparison','path':f'runs/{second.name}/COMPARISON.md'},{'label':'Comparison figure','path':f'runs/{second.name}/comparison.png'},{'label':'Combined layer CSV','path':f'runs/{second.name}/layer-comparison.csv'}];write(second/'run.json',m)
if __name__=='__main__':main()
