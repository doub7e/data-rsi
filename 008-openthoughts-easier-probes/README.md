# OTA 简单任务：score-based probe

**已完成（2026-11-06 02:26 CET）：48条筛查 + 96条独立正式rollout。** 三题主划分cosine分别为 log-summary 0.03684、broken-python −0.03152、Iris −0.05357；三题目标token保留率均100%。完整结果、图表和轨迹已通过看板API核验，本轮作业已释放。[最终分析](runs/easy-probes-20261105/ANALYSIS.md)。

原始 Qwen3-8B，复用 007 已验证的训练 loss 和模型。用户明确改为原始 score−组内平均 score；不二值化，也不除以奖励标准差。满分数仅作为描述性指标。

六题各8条筛查，优先选择分数有变化、方差较大的三题；另采独立32条计算 split-half 与逐层 cosine。正式采样不替换失败，不按分数重新划分 halves。保留原训练全assistant含reasoning监督、32K截断和有效目标token归一化。

保存每个8-rollout块的 U=Σ∇CE_sum、W=Σscore∇CE_sum 的全参数Gram；g=(mean(score)U−W)/T。原二元成功/失败分解不再适用，报告为score加权项与baseline项，不混淆。

[变更与历史归档](runs/easy-probes-20261105/score-migration.json) · [当前协议](runs/easy-probes-20261105/protocol.json)

此前 Iris 的部分得分曾触发二元读取断言；已修复并保留失败日志、原始结果。独立资源guard在控制器退出且rollout workers结束后释放推理GPU，避免空占。
