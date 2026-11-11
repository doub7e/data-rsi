> 2026-10-31 更新：用户已授权 Daytona 沙箱；推理固定 4×H200（TP=4），Harbor 执行在 RunAI CPU worker。详见 [eval/README.md](eval/README.md)。下方较早的本地后端计划已被此配置替代。

> 2026-10-31 更新：当前机器是跳板机，禁止在其上执行批量 CPU 评测。已禁用 `data-rsi-reproduction.service` 与本地 `data-rsi-podman.service`；训练仍在 RunAI 继续。RunAI CPU-only 探针成功调度，但默认策略拒绝 user namespace / Podman 嵌套容器。评测入口暂时受阻，待管理员支持的嵌套容器模板或原生 RunAI sandbox 适配就绪后恢复。此前的训练后构建授权仍有效，不等于授权修改集群安全策略。

# 当前正式复现（2026-10-30）

看板：[本地实验记录](http://127.0.0.1:18765)（SSH 转发访问；服务代码在 sibling `experiment-board/`，数据仍在本目录）。

正式训练已提交为 `ota-qwen3-8b-10k-4h200-1030`，4×H200，资源不足保留排队。配置以论文 2606.24855 Appendix C / Tables 18–19 为准：7 epochs、global batch 96、32K、LR 4e-5、cosine/warmup 0.1、weight decay 0.04、max grad norm 1.0、qwen3_nothink、disable_shuffling。与下文早期环境 smoke test 配方不同。

- [正式运行记录](runs/repro-10k-4h200-20261030/run.json)
- [训练配置](configs/reproduction-10k-4h200.yaml)
- [数据适配清单](runs/repro-10k-4h200-20261030/preprocessing.json)：10000 → 10000；合并 817 行的连续 user 消息，共 937 个边界；不丢行、不删除 assistant reasoning。
- [比较协议](runs/repro-10k-4h200-20261030/protocol.json)：SWE-Verified-100 / OT-TBLite / TB2.0，对照论文 Table 11 的 24.3 / 15.6 / 7.9，不能混用 full SWE-Verified 的分母。
- [评测配置与执行说明](eval/README.md)

**尚未完成：正式训练、32K 容量验证及正式评测。** 用户已明确授权在训练完成后准备缺失镜像（最多 2 个 CPU 作业并发）。后续控制器现已启用并等待训练；批量构建和正式评测尚未开始。

---

# OpenThoughts-Agent-SFT-10K × Qwen3-8B

2026-10-30：镜像已构建、推送并完成 GPU 验证。在 gpu303 的 2×H200 上通过 FlashAttention 前向/反向、Qwen3-8B 两步全参数 ZeRO-3 训练、checkpoint 保存及重载前向。Job Succeeded，exit code 0，已归档并清理。验证使用 2048 cutoff，尚未执行 32k 显存测试或完整 10K 数据集训练。

发布镜像：`registry.rcp.epfl.ch/shuli/data-rsi-sft:qwen3-cu128-torch280-v1`

固定 digest：`sha256:d15905bbc491f0f5a7c65e36524b2b83e345e7b7031153824e64667804081d1a`

[构建报告](evidence/build-report.json) · [registry manifest](evidence/image-manifest.json) · [成功构建日志](evidence/build/1030b.log) · [GPU 验证结果](evidence/gpu-validation/validation.json) · [GPU 日志](evidence/gpu-validation/run.log)

重新运行双 H200 验证（须监控至结束并清理 job）：

```bash
bash scripts/submit_image_smoke.sh data-rsi-sft-smoke-<unique-suffix>
```

该入口使用固定 digest、真实缓存 Qwen3-8B 权重、8 条格式有效的样本、2 步 full SFT、ZeRO-3、FA2/Liger、2048 cutoff，并验证模型保存及重新加载。它不等于全量 10K 训练或 32k 显存测试。本次验证中上述 GPU 运算、双卡训练与重载均已通过。训练 loss 为 1.3406、1.3477，所有 rank 均无 NaN；重载前向 loss 有限。重载曾出现 Mistral regex 警告，但保存前后的 pre-tokenizer、词表/merges、normalizer、decoder 及抽查原始 token IDs 均一致，见 [tokenizer 核对](evidence/gpu-validation/tokenizer-comparison.json)。未根据该警告改写 Qwen tokenizer。

## 确定采用的镜像方案

已发布 tag：`registry.rcp.epfl.ch/shuli/data-rsi-sft:qwen3-cu128-torch280-v1`。

| 组件 | 固定版本/选择 |
| --- | --- |
| Base | `nvidia/cuda:12.8.1-devel-ubuntu24.04`，已固定 manifest digest |
| Python / 编译器 | Ubuntu Python 3.12 / GCC 13 系列 |
| PyTorch / torchvision / Triton | 2.8.0 CUDA 12.8 / 0.23.0 / 3.4.0 |
| Transformers / Tokenizers | 4.57.6 / 0.22.2 |
| LLaMA-Factory | `mlfoundations/llama-factory@d20b86665a2753efdcc0f4adb7503ce87f8bdb80` |
| DeepSpeed / FlashAttention-2 | 0.17.6 / 2.8.3 |
| Liger / Accelerate | 0.6.2 / 1.10.1 |
| Datasets / PyArrow | 4.1.1 / 21.0.0 |
| PEFT / TRL / TorchAO | 0.17.1 / 0.22.2 / 0.13.0 |

这些版本是本项目固定的组合，不是作者原始运行环境的逐项还原。依赖解析、镜像构建、动态库导入及双 H200 短训练已通过；32k/full-dataset 容量与性能仍需另测。

构建文件：[Dockerfile](docker/Dockerfile)、[直接依赖](docker/requirements.in)、[传递依赖锁](docker/requirements.lock)。FA2 与 DeepSpeed 在 Torch 安装后单独安装，其运行依赖也纳入锁文件。FA2 使用官方 Python 3.12 / Torch 2.8 / CXX11 ABI TRUE wheel，并校验固定 SHA256；最初源码编译任务已取消，最终构建使用该 wheel。Dockerfile 中的 pip check、导入检查和 pip-freeze 均已执行成功，完整包清单位于镜像 `/opt/image-manifest/pip-freeze.txt`。apt 系统包未逐一锁版本，基础镜像已锁 digest，发布后还应记录最终镜像 digest。

选择独立环境的原因：

- Archer 的 Transformers 4.51.3 **已经满足 Qwen3 的基本支持下限**，不应沿用前一个 Qwen3.5 任务的兼容性结论。但其旧 Torch/FA/vLLM 组合和缺失的 LLaMA-Factory 不等于完整的本次训练栈。
- 参考仓库使用定制 LLaMA-Factory 子模块，应锁定 gitlink 而不是 `git submodule update --remote`。上游 OT-Agent commit 为 `3bd1917e62c9d03d73063b433f5c442c279c0563`。
- 当前 OT-Agent README 提及 `[sft]` extra，但其 pyproject 实际没有这个 extra。总 uv.lock 包含 DeepSpeed 0.19.0，而 LF fork 的 deepspeed extra 限定 >=0.17.2,<=0.18.0；不能直接把总锁文件作为训练镜像。
- SFT 离线读取现成轨迹，不需要在训练 Pod 内安装/启动 vLLM、Ray、Harbor、Daytona 或 Docker daemon。LF 本身声明的一些 GUI/API/数据库 Python 依赖保留以避免破坏包依赖，但不需要这些服务的凭据，也不启动这些服务。
- 本 tag 专门面向 Qwen3。Transformers 4.57.6 不覆盖 Qwen3.5；后续 FACET/Qwen3.5 使用另一个经过验证的 tag。

## 数据可用性：已检查完整 10,000 行

[官方数据集](https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-SFT-10K) 包含真实多轮 `conversations: [{role, content}]`。其说明指定 GLM-4.7-AWQ teacher、Terminus-2、至少 5 个 model turns，属于 SFT 数据规模实验中的 10K 档，而非 FACET 数据。

两个 Parquet 已下载到 `data/`，固定 revision `d0f898f7b65290c9312f59089a7e9265fdaed00c`。总计约 188 MB。全量扫描结果见 [data-audit.json](evidence/data-audit.json)：

| 检查 | 结果 |
| --- | --- |
| 行数 | 10,000 |
| user / assistant 消息 | 127,359 / 126,422 |
| assistant 同时包含 think 开闭标签 | 119,622 |
| 每行消息数 | 10–287 |
| 严格 user/assistant 交替 | 9,183 行 |
| 不符合交替格式 | 817 行，包含连续 user 消息 |
| result=null | 6,366 行；不代表 verifier 成功 |
| AgentTimeoutError | 3,516 行 |
| ContextLengthExceededError | 85 行 |
| 其他错误 | 33 行 |

不能把这个数据集称为“10K 条全部成功的轨迹”，也不能仅用 `result` 判定任务成功。训练复现不应擅自删除所有 timeout 样本。`trace_source` 的实际值以 `main`、空值和 summarization 变体为主，不能照数据卡描述直接当作四种任务来源标签。

当前 LF ShareGPT converter 会跳过非交替消息，所以原样注册会先损失上述 817 行（后续还有 tokenization/truncation 的影响）。建议显式合并相邻 user 观察消息，保留顺序和内容边界，保存转换 manifest，再核验 all-assistant loss mask；本次没有修改原始 Parquet，也没有假定原作者采用了这一修复。严格复现时需进一步对齐作者的预处理。

随机抽样 128 行（seed 42），用固定 Qwen3-8B tokenizer 与 LF-style ChatML wrappers 计算未截断长度：中位数 16,874，P90 27,661，最大 39,196；4/128 超过 32,768。这是抽样长度检查，不是完整训练预处理结果；不应据此声称全量截断比例恰为 3.125%。

## 训练方式

[参考 Qwen3-8B 配置](https://github.com/open-thoughts/OpenThoughts-Agent/blob/3bd1917e62c9d03d73063b433f5c442c279c0563/sft/lf_configs/qwen3/32k_base.yaml) 明确选择 `Qwen/Qwen3-8B`，与用户指定模型一致，并非 `Qwen3-8B-Base`。配置是 full SFT、BF16、ZeRO-3、FA2、Liger、gradient checkpointing、32k、LR 4e-5、7 epochs、cosine/warmup 0.1、beta2 0.98、max_grad_norm 1e-4。基础 preset global batch 为 16，另有 96 的 preset；尚未确认哪一个对应公开 10K checkpoint 的最终运行。

训练数据映射示意见 [dataset_info.example.json](evidence/dataset_info.example.json)。正式运行使用固定 revision 的本地转换产物，并设置 `template=qwen3`、`enable_thinking=true`、`train_on_prompt=false`、`mask_history=false`；这使监督针对所有 assistant 回合，保留数据里的 reasoning，屏蔽环境观察消息。不要用推理用的 Qwen Jinja 模板提前格式化整个数据集：它对历史 reasoning 的处理可能与 LF `encode_multiturn` 不同。

上游 `global_batch_size` 等字段经过 HPC launcher 转译。RunAI 直接调用 LF CLI 时应转换为 `per_device_train_batch_size × gradient_accumulation_steps × GPU数`；不要原样把集群 YAML 当最终 CLI 配置。正式配置应关闭上游默认 `push_to_hub`；本次不包含发布或数据库注册操作。

资源建议（估计，尚未实测）：先 1×H200 做短序列、少量样本的环境 smoke test；全参数 32k 正式运行从单节点 4×H200、microbatch=1、ZeRO-3 起步，根据测量调整。单卡能加载 8B 不代表能稳定训练 32k；参数/梯度/Adam 状态之外还有长序列激活与通信缓冲。LoRA 可降低需求，但不等价于参考 full-SFT 配方。

## 可运行检查与构建边界

CPU 数据预检（不执行数据中的 shell 命令）：

```bash
python scripts/audit_data.py
```

该脚本需 pyarrow、tokenizers；首次运行下载固定 revision 的 Parquet，并读取 Qwen 模型 revision 下载 tokenizer。当前结果记录了使用的 model revision 与数据文件 hash。

训练镜像已通过 RunAI+Kaniko 构建，context 为本节点 `docker/`；重建入口是 `scripts/submit_image_build.sh <unique-job-name>`。本机到 PVC 的路径前缀由 `/mnt/cvlab/scratch` 映射为 `/scratch`。RunAI 登录已恢复，H200 驱动 580.159.04。本次完成情况：

1. 已完成：构建、pip check、import 检查与 registry digest 核验。
2. 已完成：H200 CUDA/FA2/Liger、Qwen3-8B 短训练、保存及重载。
3. 单节点双卡 ZeRO-3/NCCL 训练检查已完成；32k 显存测试尚未执行。
4. 核验转换后样本数、assistant mask、截断比例后启动正式训练。

[Qwen3 官方兼容性说明](https://huggingface.co/Qwen/Qwen3-8B)、[OT-Agent 仓库](https://github.com/open-thoughts/OpenThoughts-Agent)、[固定 LF fork](https://github.com/mlfoundations/llama-factory/tree/d20b86665a2753efdcc0f4adb7503ce87f8bdb80)。官方数据卡链接的 8B-SFT-10K checkpoint 在此次匿名请求中返回 HTTP 401，故没有将该 checkpoint 的可下载性或训练配置视为已验证。

排队策略：按用户要求，H200 满载时保留验证任务排队，不因资源暂时不足而取消，也不自动改用其他 GPU 池。完成后归档结果并清理作业。
