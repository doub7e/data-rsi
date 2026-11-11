# DataSense-Bench BFCL Tool-Use track 复现计划

日期：2026-11-02。状态：计划与离线核验完成，尚未提交 GPU 作业或调用 selector API。

目标：复现数据选择后的训练收益与组间排序，而不是单独跑 BFCL leaderboard。先验证官方发布分组，再开展独立、盲测的新选择实验。所有结果留在本实验目录，由 experiment-board 读取。

## 1. 固定复现协议

| 项目 | 原协议 / 本计划 |
|---|---|
| 学生模型 | Qwen/Qwen3-4B；不使用现有 Qwen3-8B checkpoint |
| 原始数据 | XXHStudyHard/EnvScaler-SFT-Traj-9K 的两个指定 JSON 文件 |
| 数据行 | 9,022 完整轨迹，排除固定 60 个 validation，候选池 8,962 |
| 选择 | 五组，每组 50 条完整轨迹，组间不重叠，共 250 条；g1→g5 是预测顺序 |
| 训练样本 | 选完完整轨迹后才展开：每个 assistant 回合一个样本；历史思考删除，当前回复思考保留，仅当前回复计算 loss |
| 固定训练 | 全参数 SFT，LR=5e-6，2 epochs，cosine，warmup=0.1，BF16，seed=42，cutoff=16,384 |
| 批量 | per-device batch=1，accumulation=16，2 GPU，effective batch=32 |
| 模板 / loss | qwen3，mask_history=true；直接使用官方展开和训练入口 |
| BFCL 范围 | 仅 V3 四类 multi-turn：base/miss_func/miss_param/long_context，各200例；无 composite、无其他单轮 leaderboard 类别 |
| 评测重复 | 每 checkpoint 三次800例，共2,400 case executions，温度0.7 |
| 推理 | 一个 vLLM server / checkpoint，TP=1，65,536上下文，YaRN factor=2，原始上下文32,768；不修改训练权重 |
| 思考历史 | BFCL_DROP_HISTORY_THINK=1，与训练展开方式一致 |
| 指标 | s1、相对本地 random-50 的增益、五组 Spearman、best-group accuracy |

选择三次指三个独立 agent 会话；训练 seed 固定42；e1/e2/e3 是评测重复标识。这三层不可混淆。公开 eval.py 没有为 e1/e2/e3 显式注入三个数字随机种子，执行前记录包内和 vLLM 的实际 seed 行为，不自行把 tag 当 seed 或悄悄更改协议。

## 2. 已冻结的来源

- DataSense-Bench/DataSense-Bench: `ca38f2606b3c9c4111d0516381f044797a81436c`，位于 `upstream/`。
- Qwen/Qwen3-4B: `1cfa9a7208912126459214e8b04321603b3df60c`（本次规划所查版本，论文未给出同等粒度的模型 revision）。
- EnvScaler-SFT-Traj-9K: `0279d9ac792a12bfd28f8659de07948b4698df63`。
- BFCL 数据来自 RUC-NLPIR/EnvScaler: `96ae8b02dc0187c911b8e2101e7bb6904271597b`；逐文件匹配官方 `data_manifest.json`。
- 官方运行环境：Python3.12 / CUDA12.8 / torch2.8.0+cu128 / transformers4.56.1。
- train：LLaMA-Factory `97b32d3133b501432141a82949d5c7bc4d94f23a`，TRL0.22.2，PEFT0.18.1，datasets4.0.0。
- eval：vLLM0.11.0，bfcl-eval2026.3.23，附带的5个 evaluator patch 文件。

注意：评测文件保存为 BFCL_v4_multi_turn_* 名称，但 prepare_eval.py 下载的是固定 BFCL V3 内容，并通过 SHA256 验证。不能因文件名改用当前 V4 数据。

## 3. 阶段与验收条件

### P0 — 离线协议和参考结果核验（已完成）

- 从官方 scores.csv 重算六位 selector 的指标。
- 验证18份 assignment CSV，各有250个唯一 index，五组各50条，meta_index 与 pool 映射一致。
- 验证候选池8962条与60条 validation 无交集；核对 random-50 的50个索引映射。
- 核验输出：`runs/protocol-audit-20261102/`。这只证明公开文件一致，不代表已重跑训练或评测。
- 已重算：Astra g1=36.4861%，Spearman=0.8333，best-group=1；DeepSeek g1=36.5278%。

### P1 — 环境与数据准备

- 使用 RunAI CPU 作业下载模型和原始 JSON；不在跳板机做大规模处理。
- 使用官方 build_pool.py 重建候选池；对全部9022条原始轨迹验证生成的训练文本与发布文件一致，任何差异立即停止。
- 比对重建池与官方 bfcl_pool.csv、bfcl_validation.csv；保存源文件SHA256、行顺序和tokenizer模板。
- 材料化官方 random-50：必须使用 `results/baselines/bfcl_random50.json`，不是 build_pool.py 辅助生成的 random200。
- random-50 应展开为954条训练样本，validation_overlap=0；若序列化方式相同，比对发布的 train-data SHA256。
- 用官方 post-check 验证并材料化 Astra run1 g1。记录每组展开样本数、tokens、截断率；50条轨迹不是50次模型回复。
- BFCL 评测环境安装作者补丁，并用 prepare_eval.py/check_eval.py 验证四类数据与结果格式。
- 为未训练 base 模型做明确的 base-eval 入口：复用相同推理配置与检查，不能往原始模型目录伪造 training-complete.txt。

### P2 — 最小完整校准（推荐先执行）

执行三个 checkpoint 的完整三重复评测：
1. 原始 Qwen3-4B（无训练）。
2. 官方 random-50（一次固定子集训练）。
3. 官方 Astra run1 g1（一个50轨迹组）。

- 两个训练作业，每个从同一原始模型独立初始化。
- 所有模型都跑800×3，而非用缩小测试集宣称复现；小样本 smoke 只检查流程。
- 对照参考 base31.67%、random34.75%，同时保留分项分数、全部case IDs、预测、执行/状态验证输出。
- 参考 baseline 文件明确说其聚合来自另一台机器，未附逐次原始记录。因此不是逐token精确匹配目标。
- 如 base 或 random 有约2个百分点以上的明显偏差，或分类结果出现异常集中失败，先审计数据、模板、YaRN、历史思考和patch。2pp是工程排查阈值，不是显著性检验或硬性科研结论。
- 报告本机真实训练时长、GPU峰值显存、评测吞吐、GPU-hours；以此替代下方初步预算后再扩展。

### P3 — 复现一位 selector 的完整结果（建议 Astra）

- 使用官方 Astra run1/run2/run3 的五组分配，得到15个checkpoint。
- 每个checkpoint固定训练seed42，每个评测800×3，共36,000次case execution。
- 与 P2 的 base/random 合并，共16次训练、17个模型评测、40,800次case execution；P2已完成的相同assignment/model/config可复用。
- 组顺序始终保留作者提交的 g1..g5，不能按评测结果重新排序。
- 先对每个checkpoint的三次评测求均值，再在每个选择run内计算Spearman/best-group，最后平均三个run；标准差按选择run计算，不能把9次评测混作9个独立选择。
- tie按官方计算器处理；全部五组相等的Spearman未定义，只从该指标均值剔除。
- 这一阶段是固定选择结果的后验复现，不声称重现了 agent 自主选择过程。

### P4 — 新的盲测选择实验

- 新建三个互不继承上下文的 selector CLI 会话；推荐先测一个 selector。
- 每次按论文1×H100 80GB、总4h；如改用H200，必须标注预算硬件变化，不能声称严格同预算复现。
- 首次提交无效可重试一次，两次合用4h；失败按base分数、rho=-1、best-group=0记入统计。
- selector只看到候选池、base模型和固定recipe；不得看到BFCL题目、标准答案、官方分组、发布分数或本地评测结果。
- 使用独立只读数据/模型挂载，不能挂载整个共享workspace；控制器持有集群凭证，selector不持有。
- 使用作者训练禁用guard，同时保留工具日志及选择前后模型哈希。guard只能减少误用，不能单凭关键词扫描证明完全隔离。
- 冻结assignment和方法报告后才启动训练/评测。当前已经读过论文结果的研究会话不能直接充当这个盲测selector。
- 禁止把官方assignment直接用于新agent的提示或调参。新选择结果不与不同assignment的训练checkpoint混用。

### P5 — 可选全表复现

六位selector × 三次选择 × 五组 = 90个组模型；加random一次训练和base一次评测，共91次训练、92个评测模型、220,800次case execution。先以P2实测预算决定是否展开，不作为首次执行默认范围。

## 4. RunAI资源和镜像计划

- BFCL工具环境是评测包内的本地模拟工具/API状态，不需要Daytona，也不需要为800例构建800个容器。
- 训练申请2×H200、16 CPU、128GiB RAM。保持batch=1/卡、accumulation=16、global32；原论文2×B200→2×H200作为明确硬件适配。显存/速度先通过smoke验证，不假定一定与B200时长相同。
- 评测申请1×H200、16 CPU、64GiB RAM；每个checkpoint一个vLLM服务，e1/e2/e3共享此服务。采用官方默认每重复64线程；先检查CPU、KV和请求排队再决定是否做单独标注的吞吐调整。
- 最小阶段最多用2张GPU；较大阶段可独立并行多个单GPU评测作业，避免把4B模型固定做4卡TP。
- 只监听pod本地模型服务即可，评测clients与服务同作业运行，不需要每次工具执行调用kubectl，减少凭证过期影响面。
- 两个隔离的依赖环境/镜像：datasense-train、datasense-bfcl-eval。共用CUDA12.8基础层，但不能直接复用现有OTA的LLaMA-Factory fork或未校验版本的vLLM环境。
- 不安装Harbor/Daytona profile到本track；官方README给出的三profile是仓库通用说明。
- 实际构建时走RunAI CPU+Kaniko→Harbor registry，记录不可变digest；当前阶段不构建镜像、不提交GPU。
- 使用现有单一凭证发布服务；所有本地控制命令用token-only KUBECONFIG，不增加refresh token消费者。

## 5. 成本与时间

论文中位数：50轨迹训练0.29h，使用2×B200；每checkpoint三次完整BFCL评测共享一个GPU，合计3.30h，含模型启动、不含排队。选择阶段另有最多4 H100-hours/次。

以下“论文时间换算”混合了不同阶段硬件，仅表达资源量，不能等同为实测H200性能或美元账单：

| 范围 | 训练次数 | 评测模型数 | 论文训练资源量 | 论文评测资源量 | 初步H200预算（待P2校准） |
|---|---:|---:|---:|---:|---:|
| P0 | 0 | 0 | 0 | 0 | 0 |
| P2 base+random+一个g1 | 2 | 3 | 1.16 B200·h | 9.90 单GPU·h | 12–25 GPU·h |
| P3 一位selector三次选择，含base/random | 16 | 17 | 9.28 B200·h | 56.10 单GPU·h | 70–120 GPU·h |
| P5 六位selector，含base/random | 91 | 92 | 52.78 B200·h | 303.60 单GPU·h | 400–650 GPU·h |

- H200预算是规划预留，非已测吞吐；原论文训练不含加载，实际准备/排队/首次编译另计。
- P2/P3复用发布assignment，无selector API账单；无需GLM教师rollout或Daytona账单。
- 新selector三次选择最多另加12 H100·h；作者历史API估算Astra每次至少$9.35、Kimi每次$1.38，仅为论文固定价格假设，不是当前账单承诺。
- 若复现全部18次新选择，最多再加72 H100·h及API费用。
- 不知道EPFL GPU内部计价，所以不把GPU-hours伪装成实际美元支出。
- 单次选择的五组即使训练很快，也需要约16.5单GPU小时的完整评测；完整六agent表格不是“几十分钟”项目。

## 6. 稳定性、完整性与比较边界

- 每阶段/每checkpoint在启动前注册run.json，状态区分排队、训练、评测、重试、失败、正常结束。
- 每次只恢复缺失或确证基础设施失败的case；保留原始尝试，不因模型分数低而重跑。
- 官方客户端存在非常长的请求timeout；增加基础设施心跳/进度检查，停滞时记录为未完成并调查，不能直接把缺失结果算成模型失败或任意缩短科研协议。
- 每个重复必须四类各200个唯一ID，三重复共2400个case；校验聚合脚本不接受缺失/重复/未解决记录。
- 对训练保存checkpoint、完整trainer config、有效batch、展开样本数、token统计、峰值显存和image digest。
- 对评测保存64K inference配置视图、patched evaluator hashes、版本、所有原始回答与得分。
- base/random本机重测为主要比较基线；论文聚合为参考，不将硬件和库差异掩盖为数据选择收益。
- 区别于OTA的BFCL-Parity：本计划是BFCL V3四类多轮800例，不是123例Parity，也不是最新V4综合榜。

## 7. 下一步推荐

执行顺序：P1准备 → P2校准 → 根据实测成本执行P3 → 再做P4盲测选择。首次GPU执行只覆盖P2，不自动展开90个模型。当前交付物是计划、版本清单、可运行的离线核验入口与已核验参考结果。

来源：
- 论文：https://arxiv.org/abs/2610.12190
- 官方BFCL指南：https://github.com/DataSense-Bench/DataSense-Bench/blob/ca38f2606b3c9c4111d0516381f044797a81436c/code/bfcl/README.md
- 官方训练配方：同仓库 code/bfcl/selection/recipe/sft.yaml
- 官方BFCL准备与执行：同仓库 code/bfcl/post-training/prepare_eval.py、eval.py
- 基线说明：同仓库 results/baselines/baseline-aggregates-user-20260926.json
