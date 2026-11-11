# 真实模型重复梯度与HF/vLLM log-prob诊断

重复梯度检查通过预注册阈值；HF/vLLM存在非零、带较大尾部的log-prob差异，不能宣布精确的on-policy数值等价。后续消融固定原HF计算方式，保留这一限制。

| 轨迹 | 重复梯度cosine | 相对L2差 | 最低逐层cosine |
|---|---:|---:|---:|
| r001 | 0.99996053 | 0.008889 | 0.99990720 |
| r002 | 0.99996471 | 0.008407 | 0.99991759 |
| r023 | 0.99996275 | 0.008632 | 0.99990652 |

使用同一实际轨迹、同一原始Qwen3-4B权重，完整重算两次未加reward权重的log-prob梯度；全向量保留在显存中比较。参数FP32、BF16 autocast、SDPA、train模式和gradient checkpointing均与原实验相同。没有通过强制确定性更换原算法后再声称原设置确定。

## Likelihood对照

| 数据 | HF模式 | tokens | 平均绝对差(nats) | P99绝对差 | 最大绝对差 |
|---|---|---:|---:|---:|---:|
| fresh_actual_decode | mixed_train_logprobs | 756 | 0.015279 | 0.248622 | 0.491200 |
| fresh_actual_decode | mixed_eval_logprobs | 756 | 0.015279 | 0.248622 | 0.491200 |
| fresh_actual_decode | bf16_eval_logprobs | 756 | 0.017321 | 0.244383 | 0.665080 |
| recorded_forced_prefill | mixed_train_logprobs | 20771 | 0.016398 | 0.241802 | 1.586628 |
| recorded_forced_prefill | mixed_eval_logprobs | 20771 | 0.016398 | 0.241802 | 1.586628 |
| recorded_forced_prefill | bf16_eval_logprobs | 20771 | 0.019766 | 0.289261 | 2.316736 |

`recorded_forced_prefill`通过vLLM prompt_logprobs对原始记录的精确tokens做teacher forcing；`fresh_actual_decode`直接记录vLLM新生成tokens的decode log-prob，再让HF计算同一序列。强制prefill关闭prefix cache，避免混合缓存返回值；这与原在线推理的缓存设置不同，故另做实际decode对照。

HF train/eval两种模式的这些log-prob完全相同。全BF16模型重新从权重加载，以保持RoPE频率buffer的FP32精度，没有对整模型简单.bfloat16()后连buffer一起降精度。全BF16并未整体降低绝对误差，因此没有据此改写后续梯度方式。

vLLM自身decode和prefill的平均绝对差为0.008677、P99为0.156273、最大为0.373229 nats。它表明后端执行路径本身也存在差异，但不能用来自动解释或豁免所有HF/vLLM误差。

## 对后续实验的含义

- 重复同一轨迹的方向接近1，因此观测到的重复计算数值波动不足以解释此前约0.01的split-half cosine。
- log-prob平均误差不大，但P99超过事先设定的0.2调查阈值，存在超过1 nat的个别记录token差异。未发现明显token错位，不能因此宣称误差对弱probe信号无影响。
- 后续四种loss-mask使用同一HF设置、同一组轨迹、同一split、同一reward baseline。它们比较的是这一固定梯度估计器，不是已经证明精确无偏的vLLM策略梯度。
- 新第二题额外记录每次生成的实际decode log-prob，可在梯度计算时累计HF/vLLM NLL差异，无需额外模型前向。

[数值结果](result.json) · [尾部与区域统计](likelihood-followup.json) · [HF运行配置](hf-runtime.json) · [vLLM运行配置](vllm-runtime.json) · [HF BF16 buffers](hf-bf16-buffers.json)
