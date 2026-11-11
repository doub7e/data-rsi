"""Finalize reviewed evidence; no new compute experiments."""
from common import *
import statistics,urllib.request
RUN=ROOT/'runs/task9288-trajectory-audit-20261111'
def main():
 d=json.loads((RUN/'result.json').read_text());e=json.loads((RUN/'evaluation-comparison.json').read_text());rs=json.loads((RUN/'task9288-records.json').read_text());all_scores=json.loads((ROOT/'runs/attribution/probabilities.json').read_text())['rows'];s=next(x for x in all_scores if x['task']=='task_9288');pn=json.loads((ROOT/'runs/probe-gradient/probe-gradient-complete.json').read_text())['probe_norm'];cos=s['dot']/(pn*s['candidate_update_norm'])
 selection={**s,'dot_rank_descending':1+sum(x['dot']>s['dot'] for x in all_scores),'update_norm_rank_descending':1+sum(x['candidate_update_norm']>s['candidate_update_norm'] for x in all_scores),'implied_cosine':cos,'cosine_rank_descending':1+sum(x['dot']/(pn*x['candidate_update_norm'])>cos for x in all_scores if x['candidate_update_norm'])}
 manual=[{'task':'log-summary','uniform_scores':[0,1,1,1],'guided_scores':[0,0,0,0],'reviewed_repeat':3,'ids':['e0200'],'observed_failure':'grep -c output is filename:count, but awk uses default whitespace separator and $2; Guided repeatedly writes zero counts and times out. Verifier expected ERROR4, got0. Uniform eventually uses grep|wc -l correctly.','same_as_task9288_fixture_invention':False}, {'task':'pandas-sql-query','uniform_scores':[1,1,1,1],'guided_scores':[0,1,0,0],'reviewed_repeat':3,'ids':['e0204'],'observed_failure':'SQL returns all ranked products rather than3/category; subsequent attempts alter the Python groupby test instead of fixing answer.sql. Uniform uses ROW_NUMBER/PARTITION BY and rn<=3.','same_as_task9288_fixture_invention':False},{'task':'sparql-professors-universities','uniform_scores':[1,1,1,1],'guided_scores':[0,1,0,1],'reviewed_repeat':3,'ids':['e0209'],'observed_failure':'Guided query skips department->partOfUniversity relation and adds uni:Person constraint; verifier sees empty set. Uniform traverses department to university.','same_as_task9288_fixture_invention':False},{'task':'c725d931-2893-4dfa-a4a8-186cfa1e','uniform_scores':[1,1,1,1],'guided_scores':[.75,.75,.75,.75],'reviewed_repeat':3,'ids':['e0180'],'observed_failure':'Guided writes each test_ids.csv data line as id, with an extra trailing comma under a single-column header. Grader reads via pandas and reports empty species set; Uniform writes one-column CSV using pandas.to_csv.','same_as_task9288_fixture_invention':False}]
 for row in manual:
  saved=next(x for x in e['per_task'] if x['task']==row['task']);assert row['uniform_scores']==saved['uniform_scores'] and row['guided_scores']==saved['guided_scores']
 write(RUN/'manual-review.json',{'at':now(),'scope':'All72teacher outcomes/commands plus four matched repeat3 evaluation examples; keyword scan of available full assistant exports for280rollouts/branch.','matched_cases':manual,'causal_verdict':'Strong evidence for repetitive wrong demonstrations; no isolated causal evidence that task9288 overfitting explains final degradation.'})
 d['selection_audit']=selection;d['evaluation_marker_summary']={k:{n:v for n,v in x.items() if n!='records'} for k,x in e['branches'].items()};d['manual_review']=manual;d['status']='Completed descriptive and matched-trajectory audit';d['finished_at']=now();d['metrics'].update({'task9288_confirmed_verifier_failures':70,'task9288_missing_verifier_scored_zero':1,'guided_exact_task_fingerprint_rollouts':0,'guided_readable_export_rollouts':267,'task9288_dot_rank':selection['dot_rank_descending'],'task9288_cosine_rank':selection['cosine_rank_descending']});d['limitations']=['No direct pre/post student evaluation on task9288; these72rollouts came from fixed teacher, not the trained student.','No per-trajectory before/after student NLL or remove/downweight-task9288 training ablation. Cannot establish overfitting causality.','Marker absence rules out literal task leakage in available exports, not abstract behavioral negative transfer.13/280rollouts per branch have no accepted assistant export.','Only four matched cases were manually reviewed in detail; their error types cannot be generalized to all failures.','Generated-token and exported-character shares are proxies, not exact per-task supervised-token weights after encoding/truncation.','Teacher partial missing-verifier event retained asmissing->0; no verifier rerun or score alteration.'];write(RUN/'result.json',d)
 report='''# task_9288：错误示范重复与过拟合假设审计

结论：重复强化错误示范的风险很强；现有证据不足以证明task_9288过拟合是最终性能下降的原因。本次仅分析已有数据，没有新训练、rollout或重新打分。

## 1. 任务与原verifier

任务要求查找当前目录及子目录中以.example.com结尾的普通文件，排除archive，并保存前五个匹配项。初始fixtures实际只有两个有效匹配：

```
./subdir/test2.example.com
./test1.example.com
```

原verifier对/output/command_capture.txt与上述两行做逐字节cmp。原solution直接find+sed取前五行；不足五条就输出两条，不创建额外匹配文件。提示中“assemble any missing fixtures”容易诱导模型误解，但本实验仍遵守原verifier。

## 2. 首条被高分选中的轨迹已经是失败示范

009/teacher-batch1/t0656：模型先看到两条真实匹配，随后认为必须凑够五条，执行：

```bash
touch test3.example.com test4.example.com subdir/test5.example.com
find . -type f -name '*.example.com' -not -path './archive/*' 2>&1 | head -n 5 > /output/command_capture.txt
```

它最后宣称requirements fully satisfied；原verifier明确给0。Uniform第二批同环境t0656也有同类错误，只出现一次。两组第一轮checkpoint已经共同见过那条原始失败轨迹；续训时Uniform再见1条，Guided见72条。

## 3. 全部72条Guided轨迹的逐条检查

| 现象 | 数量 |
|---|---:|
| 原verifier明确成功 | 1 |
| 原verifier明确判错 | 70 |
| verifier下载失败，按规则记0 | 1 |
| 已判错轨迹输出五行而非预期两行 | 70/70 |
| 有额外创建.example.com文件命令 | 71/72 |
| 最终声明task_complete | 72/72 |

五条明确判错轨迹甚至因新文件挤入前五个结果而丢失了原来的匹配项。下载异常是t0342/DownloadVerifierDirError；其命令也创建了额外文件，但没有可用verifier分数，不把它冒充明确判错。

典型失败t0006创建test3/4/5，再把五行输出当作成功证据。唯一成功t0171只创建缺失的/output目录，保留原始搜索对象，运行find|head-5，输出两条真实匹配；成功不需要复杂策略。

因此，问题不是不会find，而是把“最多取五个”变成“必须有五个”，修改待查询的数据来满足自己设定的验收条件。SFT保留所有assistant reasoning、commands和成功宣称，没有按verifier过滤或负向加权；70条明确失败同样成为正向模仿目标。

这些轨迹占续训样本9.89%，API生成输出token约9.58%，导出assistant字符约9.76%。后两项不是训练编码后的逐题token精确权重。

## 4. 为什么被反复选中

原始失败示范dot=0.0526278，候选728条中第2；梯度范数3.56075，第6。其标准化分数被截断到5，得到8.862%采样概率，约均匀概率的64.5倍，实际抽到72次。

由保存的dot及两侧范数计算的cosine约0.18659，仍排第2。所以这个案例不能仅归咎于dot未除以梯度范数；简单改cosine排序仍会将它排在前列。局部梯度对齐并不保证示范正确，且没有reward质量过滤。

## 5. 有没有在评测中直接复现这种错误

扫描Uniform与Guided各280次评测中可用的assistant导出（各267次有内容，包含reasoning）：Guided中command_capture.txt、.example.com、test3.example.com均0次。没有看到这道训练题的字面内容泄漏。这个检查不能排除抽象的行为迁移。

仔细阅读四道退化题的相同repeat3：

| 题目 | Uniform四次score | Guided四次score | 阅读到的具体错误 |
|---|---|---|---|
| log-summary | 0,1,1,1 | 0,0,0,0 | filename:count输出却按空白用awk $2求和；反复写0并超时 |
| pandas-sql-query | 1,1,1,1 | 0,1,0,0 | answer.sql没有实现每类top3；转而修改测试Python代码 |
| sparql-professors-universities | 1,1,1,1 | 0,1,0,1 | 错过department到university关系，返回空集 |
| Iris | 1,1,1,1 | .75,.75,.75,.75 | 单列test_ids.csv每行多写逗号；grader报告species为空 |

这些已检查案例不是task9288中“造文件凑数量”的直接复刻。不能据此排除更一般的负迁移，也不能把四个案例推广到所有失败。全体配对结果为13题下降、4题上升、53题持平；下降存在多种机制。

## 6. 能确定与不能确定的部分

能确定：有约十分之一的续训示范高度集中于同一环境，且几乎全部包含相同错误做法和成功宣称；首条高分候选本身就是失败示范。属于明确的数据选择/示范质量问题。

不能确定：学生是否在这道题上产生参数层面的过拟合，以及它解释了多少最终下降。72条是固定教师生成的轨迹，不是续训学生的表现；没有学生在task9288上的训练前后成功率或NLL，没有只改变task9288数据的消融。Guided总体train loss更低也不能证明过拟合，因为两组训练数据不同。

最小因果检验：保持其他656条轨迹不变，限制task9288重复量，并用其他环境的已有轨迹补齐728条，固定初始化、优化器、训练seed和70×4评测。最好再用另一被过采样环境做替换对照，以区别“任何环境去重带来的多样性收益”和“task9288特定错误示范的影响”。本次未启动这些额外训练。

原始逐条审计见task9288-records.json；对应评测命令及verifier反馈见matched-evaluation-examples.json；统计脚本为scripts/audit_task9288.py，人工复核记录为manual-review.json。
'''
 (RUN/'REPORT.md').write_text(report)
 m=json.loads((RUN/'run.json').read_text());m['artifacts'] += [{'label':label,'path':str((RUN/name).relative_to(ROOT))} for label,name in [('四个配对评测案例完整证据','matched-evaluation-examples.json'),('人工复核与结论边界','manual-review.json'),('统计及选择分数','result.json')]];m['artifacts'].append({'label':'统计脚本','path':'scripts/audit_task9288.py'});write(RUN/'run.json',m)
 write(RUN/'complete.json',{'finished_at':now()});state(RUN,'complete','succeeded',finished_at=now())
 with urllib.request.urlopen('http://127.0.0.1:18765/api/experiments/'+ROOT.name+'/runs/'+RUN.name+'/training-sample/0',timeout=30) as resp:v=json.load(resp)
 assert v.get('items');write(RUN/'board-training-sample-check.json',{'at':now(),'actual72_task9288_training_subset_readable':True})
 print(json.dumps({'selection':{k:selection[k] for k in ['dot_rank_descending','update_norm_rank_descending','implied_cosine','cosine_rank_descending']},'completed':True}))
if __name__=='__main__':main()
