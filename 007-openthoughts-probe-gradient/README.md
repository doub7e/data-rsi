# OpenThoughts-Agent：原始Qwen3-8B的probe reward gradient

**已完成 3×32 条 rollout 及梯度分析。** 两题 0/32；fix-git 2/32，两个可定义划分 cosine 0.724 / 0.483，失败项对一致性贡献很大。见 [最终分析](runs/base-probes-20261105/ANALYSIS.md)。本轮自有计算作业已释放。

本节点是独立branched实验，不修改BFCL分支和既有1K/10K训练评测记录。用户明确选择原始`Qwen/Qwen3-8B`，revision `b968826d9c46dd6066d109eabc6255188de91218`；不使用SFT checkpoint求梯度。

研究3题：SWE-Verified-100的`sympy__sympy-13480`、OT-TBLite的`pandas-etl`、TB2的`fix-git`，每题32条新rollout。历史1K/10K结果仅用于选择耗时短、环境可用且曾有成败变化的候选任务；新的base结果不用于筛题或成功/失败配平。这不是随机benchmark样本。

## Loss如何与训练一致

直接使用原OTA镜像中的LF fork `d20b86665a2753efdcc0f4adb7503ce87f8bdb80`、Transformers4.57.6、Liger0.6.2及实际`CustomSeq2SeqTrainer.compute_loss`：

- `qwen3_nothink`是普通Template，**不会删除已有assistant reasoning**，已通过包含两个reasoning回合的labels测试。
- `train_on_prompt=false`、`mask_history=false`、`packing=false`、`cutoff_len=32768`。所有assistant回合监督，user/环境消息屏蔽；不采用BFCL分支的仅当前turn监督。
- 使用同一个`SupervisedDatasetProcessor`做模板渲染、逐turn长度分配、32K截断及labels构造，不手工拼ChatML或提前调用推理Jinja代替训练预处理。
- Trainer实际接受`num_items_in_batch`；Liger先求有效shifted-token CE的和，再除以该逻辑batch的有效目标token数。**不把每行平均loss再等权平均。**
- 诊断调用同一个loss求token-sum梯度，再以每半全部有效target token数归一化；物理microbatch1，不执行optimizer step。单GPU省掉ZeRO/optimizer，不声称分布式训练数值逐位等价。

验证产物：

- [实际源码与版本审计](runs/loss-audit-20261105/result.json)
- [GPU loss/梯度一致性](runs/loss-parity-20261105/result.json)：loss相对误差`1.82e-8`，梯度cosine`0.999986`；原始真实训练行前反传正常。
- [数据处理和估计公式验证](runs/contract-cpu-20261105/result.json)：官方完整preprocess输出与直接encoder的IDs/labels完全一致；两段assistant reasoning均有监督；32K截断通过；三种分组的Gram计算与直接reward中心化一致。
- [Harbor与导出验证](runs/harness-check-20261105/result.json)：3个真实旧轨迹导出、配置验证、独立call种子及位置/关键字参数调用测试通过。

## 从rollout到训练样本

沿用已验证的Terminus2 JSON parser、原生RunAI CPU sandbox、32K上下文和摘要机制；每个模型调用（含摘要调用）使用不重叠seed，保存实际请求/响应和可得的token/logprob信息。

对每个main、continuation、summarization轨迹文件，仅导出其最后一个完整conversation，不导出每一步的所有累积前缀。调用固定Harbor导出函数，按10K数据预处理规则用两个换行合并相邻user。保留reasoning、错误及超时轨迹，不设置teacher数据的最少5turn筛选。

部分续段/摘要conversation包含copied context；为保持`mask_history=false`训练loss，不擅自屏蔽这些assistant历史。逐文件记录copied assistant数量。只跳过没有任何新agent生成的纯复制文件，并记录原因。分词时统计截断前后的target数量、零target行和每个rollout的训练样本数；零target行贡献零梯度但rollout仍保留在reward分母中。

## 梯度定义及解释边界

令`C_i`为第i条rollout导出的训练样本集合，`CE_sum(s)`是原训练loss中未归一化的有效target-token交叉熵，`T_H`为该half中所有样本的有效target token总数：

`g_H = - sum_i_in_H [(R_i - mean_H R) * sum_s_in_C_i grad CE_sum(s)] / T_H`

每半独立计算reward均值。报告4+4、16+16的cosine，以及四个固定8条块组成的三种16+16划分；同时报告完整参数和各decoder层、embedding、lm_head的结果。全局cosine不使用逐层cosine平均。常量reward组梯度为零，cosine记undefined，不补采样。

**这是reward加权SFT-loss梯度，不是严格on-policy reward score gradient。** 训练式重放保留历史reasoning并按32K截断；原始推理可能移除历史reasoning或做摘要重置，且SFT样本可能含复制上下文。必须结合截断/复制审计解释结果。原始请求和轨迹保留，可供后续另做严格采样上下文的对照；本阶段不混算两种目标。

## 执行和记录

- 推理固定4×H200，gradient作业1×H200、512G CPU RAM；完整参数向量只在worker内累计，不进行随机投影。保存小型block Gram足够重建这次全部cosine。
- 每题32条新rollout、最多8个CPU沙箱并发；诊断agent预算1200秒、verifier上限1800秒、timeout_multiplier1。与旧正式评测时间预算不同，不能当正式benchmark复现成绩。
- 正式奖励读取verifier；明确的model预算超时可记0并保留轨迹；认证/基础设施/未分类异常停止检查，不伪装成模型失败。
- 只复用已有immutable沙箱镜像，不新建镜像，不使用Daytona。沙箱没有workspace PVC或用户凭证；可信controller使用既有短时token。
- 独立控制器：`data-rsi-ota-base-probes-1105.service`。4H200占满时排队，不取消其他分支作业。训练loss和harness预检是启动门槛；每题结束自动更新报告与看板。

[冻结完整协议](runs/base-probes-20261105/protocol.json) · 看板搜索`OpenThoughts-Agent` / `Base8B`。

分half的单位是完整rollout；同一rollout的所有续段/摘要样本一起归入同一半，不把相关上下文拆到两半制造伪重复。基础设施/API连接中断最多恢复两次，沿用已有native resume逻辑验证已完成trial结果hash不变；不重抽模型失败样本。

运行时并发更新：为减少最后少量trial拖尾造成GPU空等，允许最多2题并行生成，每题8个沙箱，总上限16；仍共享同一个4×H200原始模型服务。不改变样本数、种子、loss或评分规则。详见concurrency-update.json。
