# 已完成：环境重采样对照实验

整套实验于2026-11-08 08:49（Europe/Zurich）完成，计算作业已释放，看板32个已完成run的检查全部通过。

本轮未显示可靠收益：共同评分的54道留出题，Uniform平均score0.11776，Guided0.13151，差值+0.01375，95% task-bootstrap区间[-0.09076,0.11481]。全70题确认全通过分别为7和5；每个模型有1条原verifier未返回分数，保持null。两道有效probe题的单次评测分数均下降。

[完整报告与限制](runs/comparison/REPORT.md) · [对照图](runs/comparison/comparison.png) · [逐题结果](runs/comparison/result.json) · [再生成统计](runs/comparison/regeneration-audit.json)

---

# Teacher27B环境重采样对照

使用固定版本 OpenThoughts-Agent-v1-RL 的728环境，每环境由Qwen3.8-27B生成一条轨迹，训练原始Qwen3-8B一个epoch。随后两分支从完全相同权重开始、同样重置optimizer/scheduler，再训练一个epoch：uniform每环境一条新轨迹；guided按TB-dev固定20%题目probe与第一批轨迹梯度dot product形成概率，按环境有放回重采样728次、生成新轨迹。

[完整协议](configs/protocol.json)明确loss、reward、概率转换、seed及资源。原始score按题中心化；常量score题跳过且不补题。训练保留所有模型结果；基础设施异常单独恢复。各阶段结果、checkpoint和轨迹保存在本实验。

最终报告TB-dev全部70题、probe14题和未参与probe的56题；主要泛化比较看56题。两分支生成条数相同但tokens和片段数可能不同，完整记录。

梯度精度：已按实际DeepSpeed BF16前向对齐，并使用FP32外部累计。[数值核验](runs/gradient-bf16-contract-check/result.json)中loss完全一致、梯度cosine0.99996；[旧FP32-autocast对照](runs/gradient-precision-check/result.json)显示该条件不可视为同一梯度。正式probe/候选归因尚未启动前已完成纠正。

数据范围：[元数据核查](configs/domain-audit.json)显示RL训练集728题均标为NL2Bash/terminal-automation；TB-dev覆盖更广的类别。保持用户指定数据与固定14/56划分，不据此更换题目。最终报告会明确这个差异。

空轨迹边界：若模型在首次完整响应前耗尽输出/时间预算，官方ATIF没有assistant turn，则保留官方score与原始响应、导出0行、梯度贡献为0，不重采样。此类输出对基于Harbor SFT导出的归因不可见，详见[runtime说明](runs/study/empty-model-outcome-recovery.json)。

评分口径更新：用户明确要求“按原verifier来”。缺失分数保留null；probe448次中443有原始分数，4道题因分数不完整排除，2道完整且非恒分题用于平均梯度。均匀评测69/70有分数，已评分均值0.116313；确认全通过7/70，另1条无评分。旧补零统计在`runs/verifier-authority-audit/previous-derived-records/`保留。
