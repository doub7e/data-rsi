"""Render a small aggregate chart and report; no trajectory/model computation."""
from common import *
RUN=ROOT/'runs/pif-failure-top8-20261106'
def main():
 assert (RUN/'data-audit-complete.json').exists()
 r=json.loads((RUN/'result.json').read_text());a=r['models'];notes=json.loads((RUN/'manual-top8-notes.json').read_text())['notes']
 assert all(x['successes']+sum(x['first_failure_turns'].values())==256 and 'None' not in x['first_failure_turns'] for x in a.values())
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(12,4))
 for ax,p,n in zip(axes,['q1','q67'],[7,5]):
  for j,m in enumerate(['random','f','k']):ax.bar([i+(j-1)*.24 for i in range(n)],[a[p+'_'+m]['first_failure_turns'].get(str(i),0) for i in range(n)],width=.23,label=m)
  ax.set_title('Q1' if p=='q1' else 'Question 67');ax.set_xticks(range(n));ax.set_xlabel('Checker-reported failed turn (zero-based)');ax.set_ylabel('Trials /256');ax.legend()
 fig.tight_layout();fig.savefig(RUN/'failure-turns.png',dpi=180);fig.savefig(RUN/'failure-turns.pdf');plt.close(fig)
 lines=['# 预条件投影：失败模式与top训练轨迹','', '分析范围：Q1 和第二题67，各比较随机100、F128、K128，共1536条已完成评测；每组读取全部256条，核对官方reward。Top8按归因分数降序；不重新训练或生成轨迹。','', '## 第二题67：失败位置','', 'Turn0工具缺失；turn1恢复距离工具；turn2判断续航；turn3加满油；turn4启动引擎。以下为官方首先报告的失败位置，不是各轮独立正确率，也不保证是轨迹中按时间最早的错误。','', '| 模型 | 成功 | 恢复工具t1 | 续航t2 | 加油t3 | 启动t4 |','|---|---:|---:|---:|---:|---:|']
 for v in ['q67_random','q67_f','q67_k']:
  x=a[v];lines.append(f"| {v} | {x['successes']} | "+' | '.join(str(x['first_failure_turns'].get(str(t),0)) for t in range(1,5))+' |')
 lines+=['','状态不匹配错误不一定在错误字符串中写轮号。本分析使用官方 checker 已执行结果前缀的最后一项定位（len(execution_result)-1），并对有明确轮号的错误交叉核验。','', '## 第二题67：行为计数（覆盖全部轨迹）','','| 行为 | random100 | F128 | K128 |','|---|---:|---:|---:|']
 for key,label in [('restore_no_estimate','t1未尝试estimate_distance'),('feasibility_no_call','t2未尝试续航检查'),('fuel_no_call','t3未尝试加油'),('fuel_status_lookup','t3查询车辆状态'),('fuel_39_5_attempt','t3尝试正确增量39.5')]:lines.append('| '+label+' | '+' | '.join(str(a[v]['flags'][key]) for v in ['q67_random','q67_f','q67_k'])+' |')
 lines.append('| t3曾尝试加50 | '+' | '.join(str(a[v]['fuel_amount_trajectory_counts']['50']) for v in ['q67_random','q67_f','q67_k'])+' |')
 lines+=['','工具文档规定 fillFuelTank.fuelAmount 是额外加入量、油箱容量50；初始油量10.5，故目标增量39.5。尝试值计数不代表最终成功，错误后重试可使两种值同时出现。','', '## Q1对照','','| 模型 | 成功 | 首败t5 | t5未调用客服 |','|---|---:|---:|---:|']
 for v in ['q1_random','q1_f','q1_k']:
  x=a[v];lines.append(f"| {v} | {x['successes']} | {x['first_failure_turns'].get('5',0)} | {x['flags']['support_no_call']} |")
 lines+=['','## 长度与循环','','| 模型 | 平均输出tokens | 平均模型调用 | 发生长度截断的轨迹 |','|---|---:|---:|---:|']
 for v,x in a.items():lines.append(f"| {v} | {x['mean_output_tokens']:.0f} | {x['mean_model_calls']:.2f} | {x['length_capped_trajectories']} |")
 lines+=['','## Top8逐条阅读','', '内容概括是事后人工检查，不是逐turn因果归因。一个训练轨迹可同时含正确调用、必要澄清和低效重试。']
 for v in ['q67_f','q67_k','q1_f','q1_k']:
  g=r['selection'][v];lines+=['',f'### {v}','','| 排名 | meta_index | 类型 | 内容与问题 |','|---|---:|---|---|']
  for x in g['top8']:
   note=notes.get(str(x['meta_index']),x['first_user'].replace('\n',' ')[:180]);lines.append(f"| {x['rank']} | {x['meta_index']} | {x['traj_type']} | {note.replace('|','/')} |")
  lines+=['',f"完整top100类型：{g['selected100_types']}；归因分数与历史原始梯度范数Spearman={g['score_norm_spearman']:.4f}。"]
 lines+=['','## 解释边界','', '第二题F/K top8重叠2条（8235、2069），完整top100重叠60条。K top8中有6条conversation，F有3条；但完整top100中K为50条、F为55条，不能用top8的对话比例解释全部收益。','', '更合理的描述是：高分轨迹中存在查询状态、沿用前文信息、计算参数、处理前置条件等结构相似性，但不是按车辆/燃油关键词集中召回。F的燃油价格轨迹6752虽词面相近，任务与油箱操作不同；K第一名1449反复澄清，说明“高分=不澄清”不成立。','', 'Q1 F的top8包含消息广播8576、直接通讯1787，与执行客服消息有结构相关性；这只能形成假设，不能证明其导致了成功率提升。','', '官方参考在第二题续航检查中传入distance=780，但工具文档要求miles；成功轨迹中存在传入约484.67/485的情况。不能把不等于reference参数直接算失败，也不能把通过等同于所有调用和叙述均正确。此处不改变官方评分。','', 'Top8只是100条训练数据的一部分，F/K各有40条不同的训练轨迹。需要移除/替换或匹配对照实验才能确证哪几条轨迹贡献了收益。未在本次分析额外启动这些实验。','', '![First failure turns](failure-turns.png)','', '[完整统计](result.json) · [全部top8原文](TRAINING_CASES.md) · [失败/成功配对样例](EVAL_CASES.md)','']+['- '+s for s in r['limitations']]
 recovery=json.loads((RUN/'fuel-recovery.json').read_text());r['fuel_recovery']=recovery
 detail=['## 最主要行为差别：报错后的纠正','','| 行为 | random100 | F128 | K128 |','|---|---:|---:|---:|']
 for key,label in [('first_50','首次加油传50'),('first_39_5','首次加油传39.5'),('50_then_39_5','先传50后改39.5'),('50_without_39_5','传50后未改39.5'),('50_then_39_5_whole_task_success','先传50后改39.5且整题成功')]:detail.append('| '+label+' | '+' | '.join(str(recovery[v]['counts'][key]) for v in ['q67_random','q67_f','q67_k'])+' |')
 detail+=['','在首次传50的轨迹中，随后改成39.5的比例分别为82/177=46.3%、83/187=44.4%、108/180=60.0%。K的首次错误并没有减少；更突出的变化是从工具报错继续查询、修正参数。','', '固定配对样例r004：random100调用fillFuelTank(50)失败后，声称油箱似乎已经满了，询问用户是否想查询。K128也先调用fillFuelTank(50)，随后实际调用displayCarStatus(option=fuel)，再调用fillFuelTank(39.5)，成功完成。','', '这是后验行为分解，不是对训练轨迹的因果消融。F top8也有错误后重试的轨迹，不能仅凭K的4577具有纠错过程就断言它带来收益。','']
 lines=lines[:4]+detail+lines[4:]
 (RUN/'REPORT.md').write_text('\n'.join(lines)+'\n');r['metrics'].update({v+'_fuel_state_mismatches':a[v]['state_mismatch_fields'].get('fuelLevel',0) for v in ['q67_random','q67_f','q67_k']});write(RUN/'result.json',r)
 write(RUN/'complete.json',{'finished_at':now(),'source_audit':'data-audit-complete.json','chart_render':'small aggregate chart on jump host'});state(RUN,'complete','succeeded')
if __name__=='__main__':main()
