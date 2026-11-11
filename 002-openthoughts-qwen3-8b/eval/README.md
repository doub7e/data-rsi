> 2026-10-31 修正：SWE 也必须使用 JSON parser，与发布的 SFT 轨迹一致。先前按仓库当前 preset 使用 XML 导致格式纠错循环；旧 1K XML 运行已停止并保留为无效配置记录。当前有效 1K 运行是 `runs/eval-repro-1k-json-20261031/`，并先执行真实 SWE 任务的短时格式验证。

# 训练后 Daytona 评测

2026-10-31：用户授权暂用 Daytona，并要求推理保持 **4×H200**。任务 CPU 环境放在 Daytona，Harbor 控制程序放在 RunAI CPU worker；跳板机仅提供看板与轻量集群生命周期监控。

## 执行顺序

1. `pipeline.service` / `pipeline.py` 等待训练七个 epoch 成功且最终 checkpoint 文件齐全。不会在训练期间运行正式评测。
2. 检查 `runs/daytona-worker-preflight-20261031b/` 的真实云端 Oracle→Verifier smoke 已成功。
3. 归档训练日志并释放训练作业，申请 **4×H200** 推理作业；满额保留排队，此时不创建 Daytona 任务沙箱。
4. vLLM 0.11.0，TP=4、BF16、32K context、16K output cap、max_num_seqs=64、prefix caching。保持原解码设置。
5. 轻量 API readiness 通过后，启动 0 GPU / 4 CPU / 16GB 的 RunAI Harbor worker；它先完成 32 并发短生成检查，保存吞吐和时延，然后做一次 Daytona smoke。
6. 从固定版本的原始任务执行 SWE100 / OT-TBLite / TB2，各三次独立重复，共 **867 trials**。已有镜像直接使用；缺少镜像由 Daytona 构建。环境启动最多 2 个并发。现阶段不走 RunAI/Kaniko 的 200 镜像预构建路径，原构建授权保留但不会触发。
7. 每个 benchmark 目标并发 32，依据实测或用户提供的账户 CPU/RAM/disk 总配额、最耗资源任务和 10% 余量调低；不降低任务规定的资源。未指定的任务资源明确设为 2 vCPU / 4GiB / 10GiB，作为需报告的环境适配。实际并发保存在每次运行配置中。
8. 保存所有轨迹、verifier 输出、错误、统计、论文对比到实验目录；清理 Daytona 沙箱并释放 RunAI worker 和推理作业。

## 生命周期和凭据

- Key 从 `/home/shuli/.codex/secrets/daytona_api_key` 读取，供 worker 使用的副本位于共享目录 `.codex/secrets/daytona_api_key`，权限 600。作业参数只包含文件路径；不把 key 放入配置、日志或任务沙箱。
- `daytona_env.py` 仅补充生命周期保护、运行标签和资源默认值，不改变 agent 或 verifier。每个沙箱记录 ID、CPU/RAM/disk 和时间；只清理本次运行标签下的沙箱。
- 每个 trial 设置 `delete=true`；10 分钟空闲自动停止、停止后立即删除。Harbor 对活跃 trial 保持心跳，因此等待 LLM 时不会被误删；worker 中断后空闲策略兜底。
- 模型 API 连续不可用 60 秒，worker 中止 Harbor 并执行清理。模型启动、GPU 排队和推理压力检查阶段不分配 benchmark 沙箱。
- `daytona-cleanup.json` 必须显示无剩余沙箱；失败时保留 `daytona-cleanup-error.json`，不能把评测标为成功。
- 一次 Oracle smoke 只确认云端接口/日志/删除链路；正式 4×H200 推理速度仍需训练完成后实测。

## 固定版本与比较

- Author Harbor: `marin-community/harbor@2b53c2beea0e06d0a864b9cf81cfc4fbd576f7e4`。
- Daytona SDK: `0.227.0`；完整云端 smoke 已通过（1 trial / reward 1 / 0 errors / 0 残留沙箱）；worker 使用 `requirements-daytona.txt`，每次保存完整 package freeze。
- vLLM 镜像 digest 见 `serving-image.json`；CPU worker 复用已验证的训练镜像，在临时 venv 安装隔离的评测依赖，不请求 GPU。
- 任务 revision / ID 见 `benchmarks/manifest.json`，原始任务不改写。模型仍按当前作者 preset 使用 SWE100 XML / 其他 JSON parser，temperature 0.6。
- Table 11 参考值：SWE100 24.3±1.9、TBLite 15.6±1.5、TB2 7.9±1.4；固定分母，错误按零计入，SE 由三次准确率计算。
- 训练的相邻 user message 合并、当前作者 harness 版本、缺省资源显式化及配额导致的并发差异均须披露，不能宣称逐位复现。

旧 Podman 实现与验证见 `README-local-backend-history.md`。本地 Podman 服务保持禁用。

## 当前账户配额与实际并发

用户于 2026-10-31 提供：总计 10 vCPU / 10GiB RAM / 30GiB 存储，单沙箱最多 4 vCPU / 8GiB / 10GiB。API key 可创建/删除沙箱，但 quota 查询返回 403；记录在 `daytona-account-quota.json`。按保守资源上界计算，SWE100 为 2 并发，TBLite 和 TB2 为 1 并发；推理 GPU 数仍为 4。并发与论文的 32 不同，最终报告必须保留该差异。
