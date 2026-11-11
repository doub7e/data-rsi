# Qwen3-8B / official OpenThoughts 1K

用户于 2026-10-31 授权：优先 4×H100，显存不足则 4×H200；保持排队，不取消现有 10K 实验。

- 数据：`open-thoughts/OpenThoughts-Agent-SFT-1K`，revision `704dc6620eb61c0e4bef5a4943416866081beea6`。使用作者发布的 1,000 条轨迹，不从 10K 任意抽样。
- 模型与训练镜像复用 10K 的固定版本。7 epochs、32,768 cutoff、全局 batch 96、每卡 batch 1、梯度累积24、lr4e-5、ZeRO3、BF16、FA2、Liger、gradient checkpointing；训练超参数与10K一致。
- 每10步保存完整可恢复 checkpoint；保留最近2个。数据下载、格式适配和 tokenization 在 RunAI worker 内完成。
- H100 80GB 可行性依据：现有10K 4×H200已经遍历超过一个 epoch，180秒采样观察到各卡约55GiB占用。此为保守启动依据，不是H100实测峰值保证。
- `scripts/monitor_training.py` 在跳板机仅做轻量控制，提交H100并保留排队。若训练真实出现CUDA OOM，归档失败日志后自动排4×H200，不缩短上下文、不减少batch。
- 当前训练运行：`runs/repro-1k-4gpu-20261031/`。原始数据、数据修复明细、各次尝试日志、5秒GPU采样、checkpoint和最终训练指标都在本节点内，并已注册看板。
- 运行入口：`python scripts/monitor_training.py`；单独提交可用 `bash scripts/submit_training.sh h100`。不要重复提交已有任务。

10K诊断记录位于相邻节点 `002-openthoughts-qwen3-8b/runs/gpu-utilization-20261031/`。当前10K约128秒/optimizer step、735步，预计约27小时；论文表19的墙钟时间注明24 nodes，不能与4 GPU直接比较。正文另报10K约160 GH200 GPU-hours，硬件与数据长度影响使其仅供量级参考。
