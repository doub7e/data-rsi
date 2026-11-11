**Qwen Math：这轮 BF16 finite differences 实验做了什么、为什么做、说明了什么**

更新时间：2026-11-09 12:54 CET。下文先解释科学问题和结果；任务编号、运行状态与原始文件放在文末，便于查证。

**先说明本轮实际回答到哪一步**

这轮实验支持：当前默认的 BF16 Richardson FD 能得到相关信号，但与直接梯度相比有明显偏差，这些偏差会传到 token 权重及加权梯度；增加 rollout 数不能消除这种数值误差。FP32 加合适步长在已测样本上明显更准确。不过，rollout 采样本身也可能很不稳定，数值更准确是否带来训练收益，仍需要训练对照。

本轮没有拿到历史最优 N64、20-step 的那份 checkpoint，也没有拿到当时冻结的 screened1600 题目顺序。实际使用 Qwen3-4B 初始模型，以及另一份 MOPD matchedN8 step20 checkpoint。后者只用于检查“训练后权重是否有不同数值行为”。因此，本轮能回答已测模型与样本上的数值可靠性，尚未完成用户最初要求的最优 checkpoint 验证。历史 Math Avg 47.27 没有在本轮复测。

**理解这些实验，先看训练时究竟在估计什么**

训练要给回答里的每个 token 一个权重。它先对同一道题生成一批回答并判对错，用这些回答的梯度构造一个“更倾向于答对”的参数方向。下面把这批回答叫作 probe rollouts，N 就是构造一次方向用了多少条回答。

接着，训练检查某个待加权 token：如果沿这个方向微调模型参数，它的 log-prob 会怎样变化？将这个方向导数乘上截断后的 teacher–student log-prob 差，就得到 score，再用 w=sigmoid(3s) 转成 0 到 1 的训练权重。

这里有两个不同的可靠性问题。第一，给定同一个方向，FD 算出的导数准不准？第二，重新抽一批 rollouts，方向本身会不会大幅改变？实验必须分开回答。否则，FD 与参考不一致时，无法知道是算术近似出了问题，还是两边用的 rollout 方向不同。

“直接梯度参考”的做法，是对同一个 token 用 autograd 求参数梯度，再与同一个方向做内积；覆盖整个回答时，使用已经与该方法核对一致的 JVP。它绕过了 FD 的扰动与相减，提供固定样本、固定方向下的参考值。它的“精确”只针对当前这个样本目标，不代表我们知道总体奖励的真实梯度。

历史训练每题生成64条回答，分成两折，因此每折构造方向用32条。**本轮 N32 是最贴近该训练实际 probe 数量的条件。**

**实验一：固定 rollout 方向，直接检验 BF16 FD 算得准不准**

为什么做：这是最核心的数值问题。必须让 BF16、FP32 和 autograd 看完全相同的方向与 token，才能把结果差异归到数值计算路径。

怎么做：先在初始模型上选三个数学题，覆盖不同 student 成功率：约47%、73%和7%。每题预先生成2048条 probe 回答；取预定子集构造 N16、N32、N128、N1024 的方向。固定方向后，对同一批 token 分别计算直接梯度参考、BF16 Richardson 和 FP32 Richardson。这里两种 FD 都用原默认相对步长 h=1e-5；teacher–student residual 也固定，避免比较时目标跟着变。

看到了什么：下表先在题内平均重复，再对三题等权平均。“score相对误差”是整个 score 向量的误差范数除以参考范数；50%不等于一半 token 算错。“权重平均绝对误差”则直接衡量0到1训练权重被改动了多少。

| 构造方向的回答数 N | BF16 score相对误差 | FP32 score相对误差 | BF16权重平均绝对误差 | FP32权重平均绝对误差 |
|---:|---:|---:|---:|---:|
| 16 | 50.15% | 4.12% | 0.05544 | 0.00130 |
| 32 | 57.30% | 10.09% | 0.05642 | 0.00128 |
| 128 | 53.49% | 6.04% | 0.04883 | 0.00274 |
| 1024 | 46.12% | 2.59% | 0.05981 | 0.00123 |

说明什么：默认 BF16 FD 有明显数值偏差，而且从16增加到1024条回答，并没有使这个偏差消失。N32下平均权重差约0.056，已经直接涉及训练实际使用的量。

为什么高相关性仍不够：BF16与直接梯度的 Pearson 可达0.89–0.97，但拟合斜率约0.53–0.56，表现为明显的幅度压缩。即便排序大体相关，送入sigmoid的数值与最终权重也可能不对。

边界：主表来自少量题目和预选 token，不能将其平均误差解释为整个训练数据集的误差。低成功率的第三题未通过本轮独立 teacher 筛选，仅作为压力测试；实验二增加了符合筛选规则的新题。

**实验二：确认误差是否只来自挑出的极端 token 或少数题目**

为什么做：如果只看几个大 score token，相关性可能被极端点撑高；如果只测选过的题，也可能没有代表性。

怎么做：先将主实验的 token 分成随机位置与大 residual 位置，另检查删除最大绝对参考 score 的 token 后是否仍有误差。再对两个完整回答的所有515、885个 token做重复检查。最后按预先冻结的题目顺序，选第一个尚未使用、满足原 teacher 筛选规则的新题，检查其回答前2048个 token；该回答实际有2304个 token，因此明确报告为 prefix。

看到了什么：主实验 N32 只看随机位置，BF16 score相对误差仍为52.1%，权重平均绝对误差为0.0402。删除最大的参考 score token后，权重误差仍存在。独立新题的2048-token prefix上，BF16 h=1e-5 的 score误差为46.43%，权重误差为0.02706；采用此前扫描得到的 FP32 h=1e-6，分别为0.130%和0.000110。对参考 score绝对值大于0.1的位置，符号一致率分别为87.98%和100%。

说明什么：BF16问题在随机位置、完整短回答和独立新题的长prefix上都出现，不能仅用“几个极端点影响统计”解释。独立题提供了补充验证，但总题目数仍少，尚不能给出跨训练数据的总体可靠性保证。

**实验三：问题来自 BF16 本身、FD步长，还是参数扰动失真？**

为什么做：发现BF16不准后，需要知道如何修复。直接换FP32可能仍受FD截断误差影响；简单缩小BF16步长也可能让舍入误差更严重。训练前后的权重还可能有不同的舍入行为。

怎么做：保持模型、方向与target不变，扫描FD步长；用BF16直接autograd作额外对照；测量施加扰动后真正改变了多少参数、实际位移是否沿请求方向。另在辅助训练后checkpoint上重复，并对比保留FP32 master和先舍入到BF16的参数中心。

看到了什么：

- 同一个完整515-token回答，FP32 h=1e-5 的score误差为59.0%，h=1e-6降到1.15%。说明只改精度而保留原步长，仍可能留下较大误差。
- 辅助训练后checkpoint的一组固定方向中，FP32 h=1e-5、1e-6、1e-7的误差分别为59.9%、21.9%、0.62%；继续缩小到3e-8，误差又升到6.45%。最佳步长并非越小越好，也并非所有checkpoint通用。
- 同一个515-token回答，BF16 autograd相对FP32参考的误差为6.78%，BF16 FD则为55.2%。因此，BF16与FP32算术目标的差异不能解释全部FD偏差。
- Base的一个N32条件下，BF16扰动只改变0.1235%的参数，沿请求方向的实际投影只有请求值的0.477倍。辅助训练后master的同类条件下，实际位移范数为请求值的8.85倍，方向cosine仅0.103。

说明什么：BF16量化后的实际参数位移会偏离请求位移，Richardson外推并没有自动消除这一问题。FP32加步长校准是有证据支持的修复方向；目前不能把本轮Base上的1e-6宣布为训练全过程的统一最优值。score误差与sigmoid权重误差也要同时看，因为饱和权重可能掩盖score偏差。

补充检查：直接复用已经保存的前向结果，比较普通中心差分和Richardson，N32下普通中心差分的权重误差还略小于Richardson。它说明“用了Richardson”本身不能证明BF16估计准确，尚未构成训练方法优劣的比较。

**实验四：这些 score 和权重误差，会不会真的改变训练梯度？**

为什么做：score算得不准，未必影响最终优化。需要检验它经过sigmoid后，会把实际加权梯度改动多少。

怎么做：冻结同一个回答、teacher–student residual和probe方向。分别用直接梯度、BF16 FD、FP32 FD生成token权重，再用相同的FP32反向计算加权梯度。只替换权重来源，其他计算相同。

| 回答范围 | 用什么计算权重 | 与直接梯度权重所得梯度的cosine | 加权梯度相对误差 |
|---|---|---:|---:|
| 完整515 tokens | BF16，h=1e-5 | 0.88347 | 49.31% |
| 完整515 tokens | FP32，h=1e-6 | 0.999989 | 0.481% |
| 独立新题2048-token prefix | BF16，h=1e-5 | 0.95711 | 29.46% |
| 独立新题2048-token prefix | FP32，h=1e-6 | 0.999998 | 0.174% |

说明什么：在已测回答上，BF16误差足以显著改变最终加权梯度；校准FP32保留了更接近参考的梯度。因此这值得进入训练消融。这里测的是固定回答下的梯度和，尚未执行Adam更新，不能据此声称Math准确率会提高。

**实验五：如果FD完全精确，16/32条rollout构造的方向本身可靠吗？**

为什么做：用户关心训练时实际估计器是否可靠。即使FD没有误差，如果换一批rollouts就得到不同方向，token权重仍会不稳定。

怎么做：固定模型和target，反复换用不同probe回答。两个完整回答上，N16各重复32次、N32各16次、N128各8次。直接用autograd计算各次方向对应的权重，先绕开FD误差；再观察方向和token权重随采样如何变化。另取独立的1024条回答作为参考，但明确它自身也有采样噪声。

看到了什么：随着N16增加到N128，与独立1024参考方向的cosine均值，一题从0.364升至0.727，另一题从0.161升至0.369，说明增加样本通常有帮助。不过，即便N128，第二题各token的精确权重在重复采样间仍有明显波动：权重标准差均值为0.189，第90百分位为0.486。

更极端的独立新题，student只答对9/2048条。将其分成两个独立1024集合，分别只有5、4条答对；两边都用FP32计算，方向cosine仅0.014。小N下大部分集合一条也答不对，按原训练规则无法构造方向，使用fallback权重0。

说明什么：增加N能减轻某些题上的采样噪声，但1024也不能自动视为真值；teacher筛选合格也不保证student能提供足够的正负样本。FD数值误差与probe采样误差是两件都需要处理的事。

边界：重复实验只有两个主要完整回答，跨N的子集有嵌套，不能当成大量独立题目或跨训练seed证据。部分方向比较还包含probe精度差异；上述新题的两个1024集合都用FP32，才是精度匹配的采样对照。

**实验六：计算probe方向时使用BF16，会不会已经引入另一层误差？**

为什么做：此前检验FD时冻结了probe方向。但真实训练的方向本身也用BF16梯度构造，修好FD未必修好整个估计器。

怎么做：给BF16与FP32完全相同的probe回答和对错标签，分别构造方向。target端统一使用FP32直接梯度，避免把FD误差再混进来。

看到了什么：辅助训练后checkpoint上，N32仅切换probe梯度精度，就使最终权重产生约0.0814的平均绝对差。初始模型上的差异通常较小，但随题目和N而变化。

说明什么：probe梯度精度也需要单独审计。当前“FP32 FD”的训练候选仍保留BF16 probe，它只修复FD这一环，不能被描述成整个估计器已经接近真实总体梯度。

**实验七：以上现象会不会只是独立诊断脚本或某张GPU的问题？**

为什么做：独立脚本的数值结论，需要在接近实际训练的分片与attention路径上复现，才能指导训练修改。

怎么做：使用真实Qwen3-4B和双GPU FULL_SHARD，检查参数缓存失效与恢复；覆盖实际BF16 FlashAttention2、FP32切换SDPA的生产路径；另在H100重放同一方向。

看到了什么：FSDP/FA2控制中，BF16 h=1e-5 的score误差约61.7%，FP32 h=1e-6约0.0397%。H100重放分别约61.9%和0.0419%。所有100个主/独立验证条件的原参数storage恢复检查通过，恢复后log-prob差异为0。

说明什么：结果支持问题在实际分片与精度切换路径上仍存在，并非只出现在单个独立SDPA脚本或单张卡上。该控制还没有覆盖整个trainer、长序列packing和optimizer；BF16与FP32使用的attention kernel也不同，不能把全部差异严格归为单独一种舍入操作。

**实验八：数值更准确，最终训练会不会更好？目前还没有结果**

为什么做：BF16压缩score有时会使权重更接近某个有噪声的参考，类似一种收缩作用。所以“更接近直接梯度”不自动推出“训练更好”。最终需要保持其他训练设置相同，比较模型表现。

怎么设计：三组分别用BF16 h=1e-5、FP32 h=1e-5、FP32 h=1e-6。第一组与第二组隔离精度变化，第二组与第三组隔离步长变化。共同使用Base初始化、固定80题顺序及seed、每步16题×64回答、共5步，其余沿用clip1、sigmoid3s、identity、fallback0。更新后每组各自生成on-policy回答。计划在AIME24、AIME25、HMMT February和November上比较Base与三个checkpoint，每模型120题×32次生成。

实际进展：4×H200的BF16基线记录了2步，随后因CUDA显存不足失败，5步训练及最终评测没有完成。后续登记了三组8×H200重跑。重新查询集群发现，8卡BF16作业也在11月9日06:14 CET失败；日志显示Disk quota exceeded，发生在写入runtime记录、进入训练之前。核对时磁盘监控状态停在更早的queued；已依据集群状态与日志补记失败，保留旧状态历史，不能报告仍在排队。两组8卡FP32尚无已完成结果。

说明什么：截至本次重写，训练实验尚未回答数值修复是否带来准确率提升。当前已有的是数值与固定回答梯度的证据；训练收益仍待资源/存储问题解决后完成对照。

**证据强度与尚缺的验证**

已测证据足以支持优先研究“FP32 FD加步长校准”，并要求把probe噪声作为独立问题报告。尚不能据此宣布BF16训练无效、FP32训练胜出、N128全局最优，或最优N64 step20 checkpoint已经验证通过。

最直接的缺口仍是：最优历史checkpoint及原训练题上的数值复验；更多独立题目的统计；三组训练与最终评测。本轮只有少量题目和一个计划训练seed，不能替代这些证据。

**原始结果与执行留存**

- [原技术报告及全部数值表](REPORT.md)：保留当时生成的技术记录，本篇补充实验逻辑与更新状态。
- [主实验逐条件结果](runs/comparison/comparisons.json)；[同一步长配对表](runs/comparison/paired_primary.csv)；[步长校准配对表](runs/comparison/paired_calibration.csv)。
- [完整回答采样重复：515 tokens](runs/dense-repeats-base-homotopy/dense_summary.json)；[885 tokens](runs/dense-repeats-base-zeros/dense_summary.json)。
- [515-token加权梯度](runs/full-response-weighted-base/result.json)；[新题2048-token加权梯度](runs/prefix-weighted-screened-validation-v2/result.json)。后者在较早技术报告中未展开，此处直接使用其原始结果。
- [协议和方法定义](protocol.json)；[资源统计](compute-accounting.json)；[参数恢复核验](final-condition-audit.json)。

诊断共38个顶层run：36成功、1取消、1失败，run内重试另行保留。主要失败涉及写权限、依赖、token编码接口、B300软件兼容及prefix检查；无效数据未进入有效结果。软件检查98通过、9跳过。可核实GPU声明的26个终止容器合计9.657 allocated GPU-hours，另5个归档容器缺少声明，因此不是完整成本。诊断已结束，相关作业已清理。

主诊断统计包括100个模型/题目/样本量/重复/精度组合，其中20个因奖励全同而没有方向；这些按原规则使用fallback0，未当作零误差。五组数据合计10240条probe回答及80条独立target回答。两个完整回答的重复检查合计112次。数量用于追溯执行覆盖，不能当成100个独立题目或112次独立训练。

**附录A：已完成数值诊断的逐run记录**


起止时间来自当前worker state，不代表所有重试的总耗时；详细重试见attempts/与cluster/pods/。表中succeeded是该run的完成状态，不等于最优checkpoint验证已经补齐。

| run | 状态 | 内容 | 开始CET | 结束CET | 证据 |
|---|---|---|---|---|---|
| audit-aux-homotopy | succeeded | FD可靠性 · 辅助 MOPD step20 · 同伦题 · N16/32/128/1024 三方法 | 11-08 02:46 | 11-08 06:27 | [记录](runs/audit-aux-homotopy/result.json) |
| audit-base-coin | succeeded | FD可靠性 · Base · 源文本预选硬币题 · N16/32/128/1024 三方法 | 11-08 02:30 | 11-08 03:45 | [记录](runs/audit-base-coin/result.json) |
| audit-base-homotopy | succeeded | FD可靠性 · Qwen3-4B Base · 同伦题 · N16/32/128/1024 三方法 | 11-08 01:59 | 11-08 02:35 | [记录](runs/audit-base-homotopy/result.json) |
| audit-base-zeros | succeeded | FD可靠性 · Base · 不可数零和题 · N16/32/128/1024 三方法 | 11-08 02:33 | 11-08 02:51 | [记录](runs/audit-base-zeros/result.json) |
| auxiliary-master-merge | succeeded | FD可靠性 · 辅助 MOPD step20 · 保留 FP32 master 合并 | 11-08 01:29 | 11-08 01:32 | [记录](runs/auxiliary-master-merge/result.json) |
| bank-aux-homotopy | succeeded | FD可靠性 · 辅助 MOPD step20 FP32 master · 同伦题 · 2048 probes +16 targets | 11-08 01:51 | 11-08 02:43 | [记录](runs/bank-aux-homotopy/result.json) |
| bank-base-coin | succeeded | FD可靠性 · Base · 源文本预选硬币题 · 2048 probes +16 targets | 11-08 01:53 | 11-08 02:11 | [记录](runs/bank-base-coin/result.json) |
| bank-base-homotopy | succeeded | FD可靠性 · Qwen3-4B Base · 同伦题 · 2048 probes +16 targets | 11-08 01:47 | 11-08 01:55 | [记录](runs/bank-base-homotopy/result.json) |
| bank-base-zeros | succeeded | FD可靠性 · Base · 不可数零和题 · 2048 probes +16 targets | 11-08 01:51 | 11-08 01:57 | [记录](runs/bank-base-zeros/result.json) |
| bank-screened-validation | succeeded | FD可靠性 · 新教师筛选合格题 · 独立2048 probes | 11-08 09:42 | 11-08 10:05 | [记录](runs/bank-screened-validation/result.json) |
| bank-teacher-screening | succeeded | FD可靠性 · 最佳配方teacher筛题检查 · 10题×8rollouts | 11-08 03:48 | 11-08 04:30 | [记录](runs/bank-teacher-screening/result.json) |
| bf16-target-ad-base | succeeded | FD可靠性 · 全515tokens · BF16 AD surrogate 与 FD误差分离 | 11-08 03:48 | 11-08 03:53 | [记录](runs/bf16-target-ad-base/result.json) |
| center-audit-aux-homotopy | succeeded | FD可靠性 · 同一训练后master · BF16格点中心敏感性 | 11-08 04:05 | 11-08 04:11 | [记录](runs/center-audit-aux-homotopy/result.json) |
| center-calibration-aux | succeeded | FD可靠性 · 训练后权重小步长校准（固定方向） | 11-08 09:41 | 11-08 09:46 | [记录](runs/center-calibration-aux/result.json) |
| comparison | succeeded | Qwen Math · FD可靠性总报告 · 主80条件+独立20条件及完整prefix控制 | 11-08 09:41 | 11-08 09:41 | [记录](runs/comparison/result.json) |
| comparison-interim-01 | succeeded | FD可靠性 · 阶段性汇总 · 数值误差、probe噪声与权重 | 11-08 02:56 | 11-08 02:56 | [记录](runs/comparison-interim-01/result.json) |
| dense-repeats-base-homotopy | succeeded | FD可靠性 · Base同伦题 · 全515 tokens · 56组独立block重复 | 11-08 03:16 | 11-08 03:29 | [记录](runs/dense-repeats-base-homotopy/result.json) |
| dense-repeats-base-zeros | succeeded | FD可靠性 · Base零和题 · 全885 tokens · 56组独立block重复 | 11-08 03:16 | 11-08 03:30 | [记录](runs/dense-repeats-base-zeros/result.json) |
| environment-preflight | cancelled | Qwen Math · 1×H200 环境与缓存资产预检 | 未记录 | 11-08 02:03 | [记录](runs/environment-preflight/state.json) |
| environment-preflight-b300 | succeeded | FD可靠性 · B300 环境兼容性预检 | 未记录 | 11-08 01:14 | [记录](runs/environment-preflight-b300/result.json) |
| environment-preflight-cpu | succeeded | FD可靠性 · CPU 容器依赖与资产预检 | 未记录 | 11-08 01:16 | [记录](runs/environment-preflight-cpu/result.json) |
| environment-preflight-h100 | succeeded | FD可靠性 · H100 环境兼容性预检 | 未记录 | 11-08 01:19 | [记录](runs/environment-preflight-h100/result.json) |
| fsdp-cache-smoke | succeeded | FD可靠性 · 两 GPU FSDP 生产缓存路径校验 | 11-08 01:37 | 11-08 01:38 | [记录](runs/fsdp-cache-smoke/result.json) |
| fsdp-real-base-n32 | succeeded | FD可靠性 · 实际 Qwen3-4B · N32 BF16 probe · 双 GPU FSDP | 11-08 03:02 | 11-08 03:04 | [记录](runs/fsdp-real-base-n32/result.json) |
| fsdp-real-base-n32-fa2 | succeeded | FD可靠性 · 实际4B双GPU · FlashAttention2 BF16 与 FP32切换 | 11-08 03:10 | 11-08 03:12 | [记录](runs/fsdp-real-base-n32-fa2/result.json) |
| full-prefix-base-homotopy | succeeded | FD可靠性 · Base 同伦题 · 全515 tokens exact JVP 数值对照 | 11-08 02:36 | 11-08 02:40 | [记录](runs/full-prefix-base-homotopy/result.json) |
| full-prefix-screened-validation | succeeded | FD可靠性 · 新筛选题完整冻结prefix · JVP与BF16/FP32 FD | 11-08 10:42 | 11-08 10:45 | [记录](runs/full-prefix-screened-validation/result.json) |
| full-response-weighted-base | succeeded | FD可靠性 · Base全515-token回答 · 五种权重的完整响应梯度 | 11-08 02:59 | 11-08 03:01 | [记录](runs/full-response-weighted-base/result.json) |
| generation-runtime-check | succeeded | FD可靠性 · vLLM 离线生成依赖与词表校验 | 11-08 01:36 | 11-08 01:37 | [记录](runs/generation-runtime-check/result.json) |
| grader-staging | succeeded | FD可靠性 · 固定版本 Math verifier 离线依赖准备 | 11-08 01:20 | 11-08 01:20 | [记录](runs/grader-staging/result.json) |
| numerical-smoke-b300 | succeeded | FD可靠性 · Qwen3-4B Base · B300 数值实现烟测 | 11-08 01:19 | 11-08 01:21 | [记录](runs/numerical-smoke-b300/result.json) |
| prefix-weighted-screened-validation | failed | FD可靠性 · 新筛选题完整冻结prefix · 加权梯度方向 | 11-08 10:46 | 11-08 10:46 | [记录](runs/prefix-weighted-screened-validation/failure.json) |
| prefix-weighted-screened-validation-v2 | succeeded | FD可靠性 · 新题2048/2304 token prefix · 加权梯度（显式范围） | 11-08 10:50 | 11-08 10:53 | [记录](runs/prefix-weighted-screened-validation-v2/result.json) |
| replay-base-homotopy-hopper | succeeded | FD可靠性 · 同一1024参考方向 · Hopper跨硬件重放 | 11-08 02:54 | 11-08 02:57 | [记录](runs/replay-base-homotopy-hopper/result.json) |
| source-inventory | succeeded | FD可靠性 · 原始 Math 题目与历史干预行号核验 | 11-08 01:16 | 11-08 01:17 | [记录](runs/source-inventory/result.json) |
| validation-screened | succeeded | FD可靠性 · 教师筛选合格新题 · N16/32/128/1024 | 11-08 10:06 | 11-08 11:34 | [记录](runs/validation-screened/result.json) |
| weighted-base-homotopy-n1024 | succeeded | FD可靠性 · Base同伦题 · N1024 BF16 probe · 子集加权梯度 | 11-08 02:45 | 11-08 02:48 | [记录](runs/weighted-base-homotopy-n1024/result.json) |
| weighted-base-homotopy-n32 | succeeded | FD可靠性 · Base同伦题 · N32 BF16 probe · 子集加权梯度 | 11-08 02:41 | 11-08 02:44 | [记录](runs/weighted-base-homotopy-n32/result.json) |


**附录B：训练节点状态，及实时核对发现的不一致**

以下为本次读取的磁盘记录。8卡BF16已根据集群终止状态补记失败并保留旧queued历史；其失败日志显示启动写入时磁盘配额超限。其他节点未完成时不能据此推断科学结论。

| run | 状态 | 已记录进度 |
|---|---|---|
| comparison | planned | — |
| eval-base | planned | 0/3840 |
| eval-bf16-h1em5 | planned | 0/3840 |
| eval-fp32-h1em5 | planned | 0/3840 |
| eval-fp32-h1em6 | planned | 0/3840 |
| evaluation-scorer-check | succeeded | 1/1 |
| export-smoke-training | succeeded | 5/5 |
| export-train-bf16-h1em5 | planned | — |
| export-train-fp32-h1em5 | planned | — |
| export-train-fp32-h1em6 | planned | — |
| prepare-data | failed | — |
| prepare-data-extended | succeeded | 640/640 |
| prepare-data-selected | succeeded | 384/384 |
| prepare-smoke | succeeded | 2/2 |
| runtime-preflight | failed | — |
| runtime-preflight-isolated | failed | 1/2 |
| runtime-preflight-native | failed | — |
| runtime-preflight-pytorch | succeeded | 1/1 |
| screen-0 | failed | — |
| screen-0-fixed | succeeded | 1536/1536 |
| screen-1 | failed | — |
| screen-1-fixed | succeeded | 1536/1536 |
| screen-extra-0 | cancelled | 384/1024 |
| screen-extra-1 | cancelled | 448/1024 |
| select-training | cancelled | — |
| select-training-fixed | succeeded | 80/80 |
| smoke-training | failed | 0/1 |
| smoke-training-eager | failed | 0/1 |
| smoke-training-h200 | failed | 0/1 |
| train-bf16-h1em5 | failed | 2/5 |
| train-fp32-h1em5 | cancelled | — |
| train-fp32-h1em6 | cancelled | — |
| train8-bf16-h1em5 | failed（集群核实并补记） | 0/5 |
| train8-fp32-h1em5 | planned | 0/5 |
| train8-fp32-h1em6 | planned | 0/5 |
| verify-smoke-checkpoint | succeeded | 1/1 |

8卡失败的集群终止时间：2026-11-09T05:14:42Z（06:14 CET）。本报告仅核实和记录状态，未据此启动新计算或清理数据。
