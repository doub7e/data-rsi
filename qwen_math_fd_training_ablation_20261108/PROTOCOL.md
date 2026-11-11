# Qwen Math：FD精度是否改善训练收益

本轮从同一Qwen3-4B Base重新开始，不依赖缺失的历史N64 step20 checkpoint，也不直接对比历史47.27分数。

| 组别 | FD计算精度 | Richardson h |
|---|---|---|
| bf16-h1em5 | BF16 | 1e-5 |
| fp32-h1em5 | FP32 | 1e-5 |
| fp32-h1em6 | FP32 | 1e-6 |

共同配置：5 step，每步16个prompt×64 responses；每个probe fold32 responses；BF16 probe梯度；学习率1e-5；score/loss residual clip1；sigmoid(3s)；identity geometry；同质probe fallback0；response上限16384。三组使用相同镜像、H200资源类型、torch.compile开启设置、初始权重、prompt顺序及seed。各模型更新后按自身策略生成on-policy rollout，不能承诺训练后文本逐位相同。

训练数据是新冻结的80条teacher-screened Math prompt。固定随机候选顺序、每题8次teacher回答、保留1..7次正确。首批384题得到78题；补充阶段核验到的连续前缀已证明前80个合格题确定，未知尾部无论何种结果都不会改变成员。所有已发布筛选记录保留，未必要的尾部任务明确标记cancelled。详见screening-prefix-request.json和select-training-fixed/result.json。

生产smoke实际完成1次optimizer更新及128条rollout，随后原文件日志器因Tensor序列化异常退出。原失败保留；CPU核验了两rank optimizer step、checkpoint、rollout和真实console指标，修复日志器后通过真实Tensor/NumPy测试，没有为补日志重跑GPU。FP32导出通过逐张量roundtrip校验。

每组5-step checkpoint和Base都评测AIME24、AIME25、HMMT25 Feb、HMMT25 Nov，各30题×32次生成，16384-token cap，统一的显式trial seeds。使用原Math evaluator的判题表达式；Math-Verify在独立进程运行，避免与Hydra的ANTLR版本冲突。

主比较为FP32 h1e-6减BF16 h1e-5；同h的FP32组用于区分dtype与步长因素。报告逐题配对差值和分benchmark的问题级bootstrap区间。只有一个训练seed、五步，不把这些区间解释为跨训练seed的稳定优势。

控制器最多同时使用8张本实验GPU，保留所有失败/取消记录，不自动重试科学计算。训练完成、精确FP32导出完成后才放行评测；最终统计检查三组实际参数、代码、数据与运行环境一致性。完成后自动核验看板、清理本实验作业。无自动20-step扩展。

所有配置、源代码快照、训练曲线、实际rollout、checkpoint、逐题评测预测和报告均保存在本实验目录；见dispatch-plan.json及各runs目录。最终报告将写入REPORT.md。

## 2026-11-09 用户授权八卡重跑

三组新run均为8×H200、生成TP4、编译开启、log-prob每卡4条；保持80题、N64、5step及方法设置。旧四卡BF16在2step后OOM，失败证据保留；两项旧未启动FP32取消，由train8-*替代。首次实际生成后，使用实际FSDP worker验证每卡4条2048+16384 tokens的log-prob与entropy，成功后才开始首个optimizer update。该门槛不含已初始化的Adam状态，不当作整个训练显存保证。新控制器时间窗为最多12小时，无自动扩展20steps。
