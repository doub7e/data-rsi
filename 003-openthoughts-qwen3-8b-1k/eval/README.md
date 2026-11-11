> 2026-10-31 修正：SWE 也必须使用 JSON parser，与发布的 SFT 轨迹一致。先前按仓库当前 preset 使用 XML 导致格式纠错循环；旧 1K XML 运行已停止并保留为无效配置记录。当前有效 1K 运行是 `runs/eval-repro-1k-json-20261031/`，并先执行真实 SWE 任务的短时格式验证。

# 1K checkpoint evaluation

用户于 2026-10-31 授权先评测 1K。最终 checkpoint：`runs/repro-1k-4gpu-20261031/checkpoint`，epoch 7 / step 77，保持原始权重文件权限。

- 推理：4×H200、vLLM 0.11.0 固定镜像、TP=4、BF16、32K context、16K output cap；服务名 `ota-qwen3-8b-1k`。RunAI 以训练用户 UID/GID 读取共享文件，缓存写入容器 /tmp。
- Sandbox：Daytona；Harbor 在独立 RunAI CPU worker 中执行。跳板机只运行轻量控制服务 `data-rsi-1k-evaluation.service`。
- 评测：SWE Verified-100、OT-TBLite 100、TB2 89，各三次独立重复，共 867 trials。按当前 Daytona 总配额分别使用 2 / 1 / 1 并发。所有结果保存在 `runs/eval-repro-1k-20261031/`。
- 先检查模型 API、32 并发短生成，再执行 Oracle smoke 和 Terminus-2 简单任务 smoke；这些测试不计入正式分母。GPU 排队/模型加载期间不创建沙箱。
- 三项任务原始文件通过只读使用的相对链接复用 10K 节点内的固定版本；配置、运行结果、轨迹和日志全部独立保存在本节点。
- 1K 与 10K 控制器通过 workspace 的文件锁串行使用 Daytona，避免两项实验共同超额。10K 训练继续，训练完成后评测等待账户锁。
- 每项任务完成后删除沙箱；模型服务持续不可用60秒时停止任务并清理。终止时只清理本次标签下的沙箱；10分钟空闲自动停止并删除作为兜底。
- 论文 Table 11 的数值属于 **10K** 训练模型。本次报告将其明确标记为不同数据规模的参考，不把它冒充1K复现目标。

入口：本目录 `pipeline.py`（仅集群控制）；worker入口 `run_daytona_worker.sh`。版本见 `provenance.json`、`serving-image.json`、`requirements-daytona.lock.txt` 和 `benchmarks/manifest.json`。
