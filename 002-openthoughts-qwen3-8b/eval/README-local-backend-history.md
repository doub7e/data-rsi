> 2026-10-31 更新：当前机器是跳板机，禁止在其上执行批量 CPU 评测。已禁用 `data-rsi-reproduction.service` 与本地 `data-rsi-podman.service`；训练仍在 RunAI 继续。RunAI CPU-only 探针成功调度，但默认策略拒绝 user namespace / Podman 嵌套容器。评测入口暂时受阻，待管理员支持的嵌套容器模板或原生 RunAI sandbox 适配就绪后恢复。此前的训练后构建授权仍有效，不等于授权修改集群安全策略。

# Training → evaluation → paper comparison

当前状态：用户已明确授权“训练完成后开始”，记录在 `build-authorization.json`。`data-rsi-reproduction.service` 已停用，等待 CPU worker 评测后端迁移。构建脚本和控制器都验证完整训练成功及最终 checkpoint；全局最多 2 个 CPU 构建作业，不占 GPU。先前审批阻止的提前并发构建没有执行。

2026-10-31 Docker-in-Docker 实测：CPU-only 作业 `ota-dind-probe-1031` 在 gpu231 上运行 Docker 28.3.3（未申请 GPU）。默认 daemon 在 overlay / iptables 权限处失败；使用 `vfs` 并禁用 bridge / iptables 后 daemon 可启动，但导入本地 BusyBox rootfs 和拉取 hello-world 均报 `unshare: operation not permitted`，未能进入容器执行阶段。Pod 为默认 capabilities、`RuntimeDefault` seccomp、`allowPrivilegeEscalation: true`；没有申请 privileged，也没有挂载宿主机 Docker socket。记录见 `../runs/docker-dind-probe-20261031/`。结论仅针对当前默认 Pod 安全配置，不代表管理员提供的 DinD 模板不可行。

## 验证与版本

- Author Harbor fork: `marin-community/harbor@2b53c2beea0e06d0a864b9cf81cfc4fbd576f7e4`，取自 OT-Agent 固定 commit 的 uv.lock。
- Sandbox: rootless Podman，私有 Unix socket `/run/user/223974/data-rsi-podman.sock`，VFS 图层目录 `/tmp/data-rsi-podman`。镜像/容器是可重建缓存；评测结果、日志、配置全部在本实验目录。
- 使用 Docker CLI 28.3.3 和 Compose 2.39.2 连接 Podman API；`rootless_env.py` 仅用 native Podman cp 处理 rootless UID 映射，模型 agent 与 verifier 没有改写。
- 已通过完整 smoke：`smoke/results/sandbox-check-1030b/result.json`，1 trial、0 errors、reward 1。初次 API cp 的失败结果也保留，并登记在看板。
- 模型服务镜像固定于 `serving-image.json`，GPU 上尚未验证；将在训练完成后的最终 checkpoint 上检查 API，再开始正式评测。
- `harbor-template.json` 已通过当前 JobConfig schema 验证。比较统计测试保证固定分母、错误按零计入、恰好三次独立重复和正确 SE。

## 数据与环境

`benchmarks/manifest.json` 记录作者三个 dataset 的固定 revision 及完整 task ID 列表：100 / 100 / 89 个任务。`environment-plan.json` 显示：TB2 的 89 个任务有公开镜像；SWE100 与 TBLite 合计 200 个任务需要构建。

`prebuild.py` 对已有镜像直接复用，对缺失镜像通过 RunAI/Kaniko CPU 作业构建并推送到 `registry.rcp.epfl.ch/shuli/ota-eval`，最多 **2 个并发**。不在登录主机执行镜像构建，不占 GPU。所有镜像对应的源码 revision、任务 ID、构建日志和 digest 写入 `builds/`。失败不会被静默排除后再报告缩小分母的结果。

批量执行前必须有用户授权。`pipeline.py` 检查 `build-authorization.json` 的 `allowed` 字段；缺失时报告 blocked，不运行构建器。此文件只应在收到明确授权后创建，不能通过修改文件绕过审批。

## 运行顺序

控制器为 `pipeline.service`，已安装为用户级 `data-rsi-reproduction.service`。执行顺序：

1. 等待完整 7-epoch 训练成功并验证最终 checkpoint；归档训练 Pod 日志，释放训练 job。
2. 复用或构建缺失任务环境，要求 289 个任务全部就绪。
3. RunAI 排队申请 1×H200，使用最终 checkpoint 启动 vLLM，32K context / 16K 输出上限。通过授权的 kubectl WebSocket port-forward 在本机 18080 访问 API（已核实 GET pods/portforward 权限）。
4. 生成检查成功后运行三个 benchmark，每项三个独立 repeats。每个 fire 固定任务集，`n_attempts=1`；三次构成论文的三次随机重复。
5. 保留每次 trial 的模型轨迹、verifier 输出、错误与配置。聚合为 `runs/eval-repro-10k-20261030/result.json` 和 `comparison.md`，自动被看板读取。
6. 结束或出错时归档并清理模型服务 job。

核心评测配置：Terminus-2；context 32768；output cap 16384；proactive summarization 2048；8B timeout multiplier 2；agent cap 7200s。并发 4，SWE100 使用作者当前 preset 的 XML parser，其余 JSON。Temperature=0.6；原论文没有完整披露解码参数，明确记录这一选择。

## 比较解释

对照论文 [2606.24855 Table 11](https://arxiv.org/pdf/2606.24855)：SWE-Verified-100 24.3±1.9、OT-TBLite 15.6±1.5、TB2.0 7.9±1.4（百分数）。我们的均值来自三个独立重复，SE 按三个 repeat accuracy 的样本标准差 / sqrt(3)。错误与超时按零计入固定任务分母。

硬件、梯度累积、数据格式规范化、软件版本、sandbox 和并发度均有适配，报告中会列明；不能宣称逐位或完全同环境复现。原论文 full SWE-Verified（500任务）与跨 harness 取最高分的表格不作为这个三项对照的分母。

服务：`data-rsi-podman.service` 已启动。该服务只提供本地容器 API，不会自动发起批量任务。看板独立运行，不依赖这个评测控制器。
