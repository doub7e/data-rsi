# FACET reproduction — environment and data preflight

状态：已完成公开资源与本地构建配置检查；尚未训练、评测、构建或发布镜像。集群实测受 RunAI 登录失效阻塞。

本节点的当前可运行入口是只读预检（无需 GPU）：

```bash
cd /mnt/cvlab/scratch/cvlab/home/shuli/data-rsi/001-facet-reproduction
python3 scripts/preflight.py --output evidence/preflight.json
```

预检会记录本地构建配方、CLI 可用性、RunAI 连接结果、HF 官方数据集与文件列表。依赖仅 Python 标准库；集群/公开元数据读取失败返回非零状态。它不是训练或模型兼容性测试。

## 结论

- 现有 Archer/ASFT 构建配方不适合直接复用。建议为 FACET SFT 准备独立、固定版本的新 RunAI 镜像；推理可先使用支持 Qwen3.5 的官方镜像并固定 digest。具体版本须经过 H200 smoke test 后锁定。
- 没有找到作者公开发布的 1,200 条 SFT trajectories。已下载并检查公开任务 ZIP，包含 6,020 个 Harbor tasks，而不是 teacher 训练轨迹。
- 可以先验证公开 4B checkpoint 的评测链路；严格重训复现仍需要原始轨迹及预处理配置，或明确采用重新采样的近似复现。

## 环境核查

| 对象 | 已查到的配置 | 判断 |
| --- | --- | --- |
| `../../docker/Dockerfile` | CUDA devel 12.6.1；Python 3.10 venv；Torch 2.5.1/cu124 | 旧环境；不能凭 CUDA 标签认定可运行 Qwen3.5 |
| `../../docker/archer-requirements.txt` | Transformers 4.51.3、vLLM 0.7.3、FlashAttention 2.7.3；无 LLaMA-Factory/Harbor | 需要更新模型支持及训练栈 |
| `../../docker/Dockerfile.asft` | 继承 Archer；附加 DeepSpeed 和旧版 TRL | 不解决 Python/Qwen3.5 支持问题 |
| `../../LINEAGE-final/docker/Dockerfile` | 默认 `vllm/vllm-openai:v0.11.0`；主要增加系统工具 | 也不能据此认定具备 Qwen3.5/LLaMA-Factory 支持 |
| 本机非登录 shell PATH | 有 runai/kubectl/python3；未找到 docker/uv | 这是 PATH 探测，非全盘安装清单 |
| 集群 | `runai list` 与 `kubectl get pods` 返回 `invalid_grant: Token is not active` | 无法验证当前镜像实际内容、GPU 空闲量、驱动、容器权限 |

以上镜像结论基于磁盘上的构建配方，不是 registry 当前 digest 或运行中 Pod 的包清单。详细快照见 [preflight.json](evidence/preflight.json)。

建议分开准备三个环境：

1. **SFT GPU 环境**：Python 3.11–3.13 中选定一个版本（建议 3.12）；兼容 Qwen3.5 的 LLaMA-Factory、Transformers、PyTorch、DeepSpeed；CUDA/FlashAttention 与 Torch ABI 配套。Python 范围来自 FACET 合成代码，统一采用它是工程选择，不是论文对 SFT 的原始版本声明。论文没有给出这些包的精确版本，不能声称任意最新组合就是原环境。
2. **推理 GPU 环境**：支持 Qwen3.5 的 vLLM 或 SGLang，提供 Harbor 能访问的模型 API；固定模型 revision、镜像 digest、chat template 和 reasoning 配置。
3. **任务 runner/sandbox 环境**：Harbor + Terminus-2，能够构建、启动、执行和销毁每个任务的容器。Harbor 默认使用 Docker，也支持其他后端；新建 RunAI image 本身不等于获得容器运行能力。需核实集群允许的后端、权限、网络和文件映射。Kaniko 可负责构建，不能替代任务执行后端。

如果只做 SFT 和 benchmark，不必先重跑 skill 收集/任务合成。若复现合成流水线，上游要求 Python >=3.11,<3.14、uv、Docker，声明依赖为 LiteLLM >=1.89,<1.90、PyYAML >=6、tqdm >=4.66，并提供 uv.lock；还需要模型 API 和 scenario–skill 输入。

RunAI PVC 映射：本机 `/mnt/cvlab/scratch/cvlab/home/shuli/data-rsi` → 容器 `/scratch/cvlab/home/shuli/data-rsi`，PVC `cvlab-scratch` 挂载 `/scratch`。后续 GPU 工作使用 RunAI；预检可先用 1×H200，不能把它等同于原论文训练规模。

## 论文中的训练与评测设定

来源：[论文 Appendix D / Table 9](https://arxiv.org/html/2608.18580v1#A4)。

| 项目 | 原文设定 |
| --- | --- |
| 模型 | Qwen3.5-4B / 9B / 27B |
| 训练 | LLaMA-Factory，全参数 SFT，BF16、ZeRO-3，8×H200 |
| 超参 | 3 epochs；effective batch 64；LR 1e-5；cosine；warmup ratio 0.1；长度 32768 |
| 评测 | Terminal-Bench 2.1；Terminus-2；每任务 3 次独立尝试；每次 2 小时；temperature 1.0 |
| 上下文 | FACET 模型 32768；其他模型使用官方最大值，严格复现时要保留这个差别 |

还缺：原始训练 YAML、训练数据序列化/模板、loss mask、packing/truncation、seed、checkpoint 选择、软件与 benchmark commit、完整采样与 thinking 设置。作者仓库 scripts/ 目前只有预留说明，公开 configs/ 只有三种合成策略。关于 thinking mode 的 [issue #2](https://github.com/StoKou/FACET-Terminal/issues/2) 尚无回复。不能用默认配置填补后宣称精确复现。

## 1.2k trajectories 来源与获取路径

[论文 §3.1、Appendix A.2](https://arxiv.org/html/2608.18580v1#A1.S2) 描述：DeepSeek-V4-Pro + Terminus-2 在 6,078 个验证任务上 rollout；6,074 个记录中有 1,270 个 reward=1；最终选取 1,200 条完整成功轨迹做 SFT。仅写了完整成功筛选，没有明确说明排除其余 70 条的具体规则/ID。

已核查 [GitHub](https://github.com/StoKou/FACET-Terminal)、[项目主页](https://stokou.github.io/FACET-Terminal/)、[HF 官方组织](https://huggingface.co/FACET-Terminal)、数据集文件和讨论区。HF API 目前仅列出一个官方数据集：

- [FACET-Terminal-Tasks-6k](https://huggingface.co/datasets/FACET-Terminal/FACET-Terminal-Tasks-6k)：6,020 个公开任务。
- `data/train-*.parquet` 是任务字段表，`train` split 不代表 SFT trajectories；`solution` 是 `solve.sh`，不是模型多轮交互记录。
- `FACET-Terminal-Tasks.zip` 是完整任务环境、测试和参考解；执行任务需要它，Parquet 中不包含所有 fixtures。

压缩包已保存至 `data/FACET-Terminal-Tasks.zip`（158,889,886 bytes），固定 revision：`b2d02645932e3989c8332a41e57d0f0855de7002`。

SHA256：`7355ea41afac98cd0b322fd9b317b5b0fc61eddf3a3b01091fbeb3d75d9555a6`。

实际检查到 6,020 份 `task.toml`、6,020 份 instruction、6,020 个 solution 文件、12,040 个测试文件、81,511 个环境文件。含 trajectory/messages/rollout 等名字的匹配项全部位于 environment fixtures；没有独立的 teacher rollout/SFT bundle。仅查看 ZIP 元数据，未执行包内代码；未进行全量构建或 oracle 验证。详情见 [task_archive.json](evidence/task_archive.json)。公开版本与论文的 6,078 个任务数量不同，不能直接假定两个任务集合完全一致。

获取训练轨迹的两条路径：

1. **精确复现优先**：联系作者 Kou Shi (`stokou@mail.ustc.edu.cn`)，索取 1,200 条原始训练轨迹、原始 task ID 映射、1,270→1,200 筛选规则、训练预处理与 LLaMA-Factory YAML、Harbor/Terminus-2/benchmark revision、teacher 与评测 thinking 参数。本次未发送邮件或 issue。
2. **近似复现**：对公开任务用相同 teacher 和 Terminus-2 重新 rollout，保存全部多轮交互、reasoning、工具输入输出和 verifier reward，只保留完整成功样本后转换 SFT 格式。不能把 `solve.sh` 直接替代成功轨迹。任务发布版本、teacher 服务及随机采样的变化意味着这不等于原始训练集；成本和成功数量须先用小样本测量。

## 下一步顺序

1. 恢复 RunAI 登录（通常由用户执行 `runai login` 完成交互认证），验证集群访问。
2. 拿到原始轨迹与配置；同时可准备公开 4B checkpoint 的评测 smoke test。
3. 锁定训练/推理候选版本，确认 H200 驱动兼容、模型加载、短训练和 Harbor 单任务执行，再固化训练镜像；使用 RunAI+Kaniko 构建。
4. 在协议缺口明确后扩大至论文规模。本节点目前没有伪装成可完整训练的占位配置。

参考：[FACET pyproject](https://github.com/StoKou/FACET-Terminal/blob/main/facet/pyproject.toml)、[Qwen3.5-4B 官方模型卡](https://huggingface.co/Qwen/Qwen3.5-4B)、[vLLM Qwen3.5 recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)、[Harbor 安装](https://docs.harborframework.com/getting-started/installation)、[Harbor sandbox 后端](https://docs.harborframework.com/sandboxes/pre-integrated-sandboxes)。
