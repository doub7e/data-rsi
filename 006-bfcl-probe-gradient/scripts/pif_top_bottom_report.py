"""Small manually annotated artifact audit; no model execution."""
from common import *
RUN=ROOT/'runs/pif-top-bottom-content-20261106'
def main():
 r=json.loads((RUN/'result.json').read_text());annotations=json.loads((RUN/'annotations.json').read_text());groups=json.loads((RUN/'groups.json').read_text());byid={x['meta_index']:x for x in annotations}
 lines=['# Top/bottom对照：宽泛工具工作流是否有区分力','', '**结论：bottom中也大量存在这一流程，当前审查不能支持用它解释top的优势。此前仅看top8的共同点不足以建立这种解释。**','', '固定范围：第二题67的F128/K128，各最高8条和最低8条，共32个组内位置、28条不同轨迹。没有挑选其中符合预期的样例；本次不是对完整200条的语义普查。','', '为避免仅凭工具名或先后顺序贴标签，采用更严格的判据：只读查询的返回信息被用于计算写入参数、筛选操作对象或判断后续状态修改的条件。单纯“先查后写”不够；后续用户完整重新给定操作的边界案例单列不确定。这个标准比原先“执行操作”更窄；满足它的bottom当然也满足原先宽泛描述。','', '| 配置 | 明确存在 | 不存在明确证据链 | 不确定 |','|---|---:|---:|---:|']
 for g,c in r['group_counts'].items():lines.append(f"| {g} | {c['yes']}/8 | {c['no']}/8 | {c['uncertain']}/8 |")
 lines+=['','不确定案例：F bottom的2102先查询得到比赛M4，后续用户又完整指定M4和新地点；K bottom的8889先读病房状态，但用户已明确给定状态与所有操作对象。保守口径下不计为明确存在。K top第一名1449主要是用户逐轮补齐参数后创建/修改，export查询没有明确驱动后续写操作。','', '## 明确的bottom反例','', '| ID | 第二题排名（1最好/200最差） | 真实链条 |','|---|---|---|', '| 8710 | F197 / K200 | 查询员工工资及岗位薪资区间 → 计算20%涨薪、比较晋升阈值 → 更新工资/职位 |','| 1594 | F199 | 查询坐标55.7/37.6 → 各加0.01 → 写回55.71/37.61 |','| 4357 | F195 / K197 | 查询NAV=50 → 计算增加2% → 更新为51 |','| 4579 | K198 | 查询月租2600、日租100 → 月租上调5% → 写回2730并保留日租100 |','| 4582 | F200 | 租期校验失败 → 查询最低30天 → 改结束日期重新校验 → 计算租金并提交 |','| 8406 | K196 | 按内容/标签查询笔记 → 利用返回ID、标题、正文 → 编辑和复制到Archive |','', '4582的参数修复还会改变用户给定日期，7578中存在available=False却继续改期的行为。模式存在不等于操作正确、数据优质或有正训练收益；同一条轨迹可以同时有有用和有害行为。','', '## 同一内容随probe换排名','', '8710在Q1 F排名4，但在第二题F/K分别排名197/200；4582在Q1 K排名1，但在第二题F排名200。流程特征本身没变。此现象不证明梯度排名没有信号，但说明不能用这个粗粒度标签解释排名差别。','', '同环境也有相反排名：第二题F top1的6099和bottom2的1594都是env_50_sft的动物观察环境，后者甚至有更直接的查询数值→算术更新链。','', '## probe本身的核验','', '第二题K梯度只取turn1，实际链条是查询两个城市的邮编→把返回邮编传给estimate_distance；没有油箱状态修改，也没有后续加油纠错。成功probe r018和失败probe r001都有这一调用链，所以该泛化模式也不能充分区分probe成功与失败。它们的整题reward包含后续回合结果。','', 'F包含完整轨迹；成功probe r018在加油回合查询油量后传39.5，失败r001先传50，查询到10.5后仍没有正确补加。这个具体执行差异是真实的，但不能据此推断top训练数据更富含宽泛查询/操作模式。','', '## 逐条判定','']
 for g,ranking in groups.items():
  lines += [f'### {g}','','| 组内排名 | ID | 判定 | 理由 |','|---|---:|---|---|']
  for j,x in enumerate(ranking,1):
   a=byid[x['meta_index']];lines.append(f"| {j} | {a['meta_index']} | {a['label']} | {a['rationale']} |")
  lines.append('')
 lines+=['## 解释边界','', '- 这是事后、非盲审的固定极端样本检查，不是独立人工标注或全池显著性检验。','- 样本数小且F/K组重叠，不用不适当的独立样本p值宣称“全池完全相同”。','- 可以收回“这一宽泛特征解释top收益”的叙述；不能由此收回已观察到的K模型错误恢复改善，也不能证明梯度归因无效。','- 更有辨别力的假设需要检验具体错误→新增信息→正确参数修复，以及这些行为在训练损失中的权重，并在top/bottom同时测量。此处未额外启动训练。','', '![Feature comparison](comparison.png)','', '[逐条标签与原文证据](annotations.json) · [完整轨迹](CASES.md) · [真实probe样例](probe-examples.json)']
 (RUN/'REPORT.md').write_text('\n'.join(lines)+'\n')
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 names=list(r['group_counts']);fig,ax=plt.subplots(figsize=(8,4));bottom=[0]*4
 for label,color in [('yes','#167a65'),('uncertain','#e6b65c'),('no','#c3ccd1')]:
  values=[r['group_counts'][g][label] for g in names];ax.bar(names,values,bottom=bottom,label=label,color=color);bottom=[a+b for a,b in zip(bottom,values)]
 ax.set_ylim(0,8.8);ax.set_ylabel('Manually reviewed trajectories /8');ax.set_title('Read response -> compute/decide -> state-changing action');ax.legend(ncol=3,loc='upper right');fig.tight_layout();fig.savefig(RUN/'comparison.png',dpi=180);fig.savefig(RUN/'comparison.pdf');plt.close(fig)
 write(RUN/'complete.json',{'finished_at':now(),'method':'Manual review of saved artifacts; small local extraction and chart only; no model/bulk CPU job.'});state(RUN,'complete','succeeded')
if __name__=='__main__':main()
