# 原生 RunAI CPU sandbox 验证

本实验与 `003-openthoughts-qwen3-8b-1k` 中正在运行的 Daytona 评测分离。所有执行、gold patch 和 verifier 都在 RunAI CPU workers/pods 中，不调用现有模型服务，也不改变原评测配置。

## 测试方法

- 固定任务：`astropy__astropy-13236`。镜像从固定的公开 SWE digest 构建，执行原始 Dockerfile，再添加终端工具 tmux；最终 digest 见 `images/manifest.json`。
- Harbor 使用与原评测相同的作者 fork / dependency lock。`native/runai_env.py` 实现环境启动/清理、远程命令、文件上传/下载。
- 任务 Pod 不挂共享工作目录；CPU controller 独立挂载实验目录。仅 controller 使用自动同步的短期 access-token kubeconfig，refresh token 留在跳板机。
- Sandbox 请求 GPU=0，CPU=1、RAM=4096Mi，并申请10240Mi ephemeral-storage；CPU/RAM有硬上限。启动时检查资源约束确实落入 Pod spec。
- 依次跑1、4、16、32并发；每个trial使用Oracle gold patch和原始verifier。前一档全部reward=1且无错误才增加并发。每个沙箱独立运行并清理。
- 记录启动、验证、整体耗时、控制请求耗时、实际同时就绪的沙箱峰值。每档配置、输出与统计保存在 `runs/native-capacity-20261031/`。

## 结论边界

这是单个真实任务的重复环境验证，验证CPU sandbox容量及Harbor完整链路，不是模型得分，也不直接测量高并发下的模型延迟。相同镜像会共享节点缓存，不能据此宣称289种环境首次启动具有相同速度。生产迁移还需要覆盖不同任务镜像/特殊环境能力，并单独验证推理高并发吞吐。

跳板机仅运行轻量凭据同步。测试结束后停止同步服务、清理controller和任务jobs，保留实验文件。既有Daytona评测继续。

## Single credential publisher (2026-11-01)

`data-rsi-native-auth.service` is the only refresh-capable automation service. It explicitly reads `/home/shuli/.kube/config` and holds a process lock. Dashboard and host lifecycle monitors use the same token-only kubeconfig already used by CPU workers. Service drop-ins are retained under `scripts/systemd/`; human `runai login` continues to update the original kubeconfig. A copied kubeconfig containing a refresh token is not an isolated credential.

Sanitized publication/authentication transition events are saved under `runs/auth-diagnosis-20261101/auth-refresh-events.jsonl`; they contain timestamps, publisher PID and token issue/expiry times, never token values. This removes the confirmed background refresh race surface; it does not prove the editor caused the earlier server-side session failure.
