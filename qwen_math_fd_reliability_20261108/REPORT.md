# Qwen Math：BF16 Richardson FD 可靠性审计

更新：2026-11-08T10:36:51.886925+00:00。数值表由已保存的逐条件结果生成。

## 当前结论与适用范围

**当前 BF16 Richardson、h_rel=1e-5 有有效信号，但不能视为精确方向导数的可靠替代。** 增加 probe 数量减轻采样噪声，不能消除固定方向下的 BF16 数值误差。FP32 FD 可显著降低权重误差，但也需要按 checkpoint 和方向校准步长；仅将 dtype 改为 FP32、保留原步长，并不保证 score 精确。

本次采用已完成 Qwen Math 20-step 最优配置的数值规则：N64、score/loss residual clip=1、sigmoid(3s)、identity geometry、同质 probe fallback weight=0；历史 Math Avg 47.27%，未重新评测。**最优 N64 step20 权重和原始 frozen screened1600 顺序不可访问。** 主实验是 Qwen3-4B 初始化模型；Aux 为本地另一 MOPD matchedN8 step20 的 FP32 master，只作训练后数值控制，不能冒充最优 checkpoint。最优 checkpoint 上的直接验证仍未完成。

## 实验定义

N 是构造一次 outcome-gradient 的 response 数；训练 N64 的两折每折 N32。每个主 bank 含2048个独立 probe responses 和16个独立 target responses。前1024为A参考，后1024为B；B内三组不交叠128条中嵌套抽16/32/128，B1024整体另作一次。A1024是有采样噪声的独立参考，不是真实总体梯度。

固定 checkpoint、target、clipped BF16 teacher-current residual及单位 probe 方向，比较 FP32 autograd 内积与 BF16/FP32 Richardson。probe 梯度本身的 BF16/FP32 精度另行比较。每个主条件选16个随机位置及8个大 residual 位置（可能重叠），最多2048-token target prefix；probe response保留全长、不按正确性筛选。另对515/885-token完整 response做密集重复验证。

Richardson 使用 ±δ、±2δ 四次前向，δ=h_rel‖θ‖₂；方向已归一化。相减组合用高精度算术，但模型权重及前向按所标BF16/FP32执行。exact指固定样本目标的autograd，而非真实奖励的总体导数。所有结果使用原项目Math-Verify，不重新标注、不删除失败或截断样本。生成使用vLLM0.32镜像，数值审计使用torch2.8/cu128、transformers4.57.6；完整vocab与canonical prompt tokenization已核对，但这不是原历史运行的软件栈逐位重放。

具体定义：`g_N = mean_i[(R_i − mean(R)) ∇θ sum_t log πθ(y_it|x,y_i,<t)]`，`v=g_N/‖g_N‖₂`；`u_t=clip(logπ_teacher−logπ_student,−1,1)`在比较中冻结，`s_t=u_t〈∇θlogπθ(y_t|prefix),v〉`，`w_t=sigmoid(3s_t)`。这采用OPD上升方向的符号；同质probe不定义v，实际权重为0。每个response内部使用token log-prob总和，没有按response长度归一化。

## 主实验覆盖

| 模型/题目 | probes | 正确率 | 平均response长度 | 截断 | 方向条件 |
|---|---:|---:|---:|---:|---:|
| Base/homotopy | 2048 | 46.58% | 496.3 | 0 | 20/20 |
| Base/zeros | 2048 | 73.34% | 782.2 | 0 | 20/20 |
| Base/coin | 2048 | 7.47% | 1690.7 | 0 | 20/20 |
| Aux/homotopy | 2048 | 91.99% | 4717.3 | 0 | 20/20 |

80/80主条件完成；40组B侧probe集合中3组同质（Aux的三个N16），对应6个精度条件。它们的score/rank未定义，实际fallback0纳入practical-weight分析，不能从数值表中缺失误解成表现优良。

## 同方向、同一步长：Base三题的数值误差

以下固定BF16 probe方向，h=1e-5；先题内平均重复，再对题目等权。N16/32/128各9个条件，N1024为3个。位置为24-token union，非随机总体。相对L2为‖s_FD−s_exact‖/‖s_exact‖；sign只统计|s_exact|>0.1。

| N | FD | Pearson | Spearman | 相对L2 | 校准斜率 | sign一致率 | weight MAE |
|---:|---|---:|---:|---:|---:|---:|---:|
| 16 | BF16 | 0.93005 | 0.81083 | 0.50152 | 0.53503 | 0.90448 | 0.055445 |
| 16 | FP32 | 0.99876 | 0.99308 | 0.04118 | 0.99566 | 1 | 0.0012969 |
| 32 | BF16 | 0.88541 | 0.80404 | 0.57296 | 0.52763 | 0.87797 | 0.056416 |
| 32 | FP32 | 0.99192 | 0.9997 | 0.10092 | 0.93808 | 1 | 0.0012782 |
| 128 | BF16 | 0.90885 | 0.84699 | 0.53488 | 0.53455 | 0.88357 | 0.048826 |
| 128 | FP32 | 0.9976 | 0.99377 | 0.060412 | 0.99188 | 1 | 0.0027379 |
| 1024 | BF16 | 0.97308 | 0.84105 | 0.46116 | 0.5593 | 0.90833 | 0.059805 |
| 1024 | FP32 | 0.99984 | 0.9991 | 0.025906 | 1.0145 | 1 | 0.0012251 |

N32（对应训练N64的每折probe数）分位置层查看，同为h=1e-5、同BF16 probe方向。Base为三题，Aux为一题；random为预先冻结的16个随机位置，extreme为8个大原始residual位置，两者不是独立题目。

| 模型 | 位置层 | FD | Pearson | 相对L2 | weight MAE |
|---|---|---|---:|---:|---:|
| Base | random | BF16 | 0.75771 | 0.52061 | 0.040159 |
| Base | random | FP32 | 0.99922 | 0.020355 | 0.00052251 |
| Base | extreme | BF16 | 0.86066 | 0.58739 | 0.088931 |
| Base | extreme | FP32 | 0.99207 | 0.10085 | 0.0027896 |
| Aux | random | BF16 | 0.99561 | 0.11542 | 0.044654 |
| Aux | random | FP32 | 0.99999 | 0.012046 | 0.00046505 |
| Aux | extreme | BF16 | 0.90734 | 0.37584 | 0.1584 |
| Aux | extreme | FP32 | 0.98714 | 0.1142 | 0.00030694 |

从同一批已保存的±h前向事后派生central-2作为控制：N32、h=1e-5时，Base三题的weight MAE为0.04611（Richardson为0.05642），Aux为0.04898（Richardson为0.08257）。这组对照没有额外GPU调用；它显示Richardson外推本身并未消除BF16误差，不构成独立计时或训练收益比较。

高Pearson并不保证尺度或最终权重准确。BF16斜率约0.53–0.56，存在明显幅度压缩；单靠增加N不能修复。

## 配对步长校准

以下只使用每题repeat0的同一BF16 probe方向，比较BF16 h=1e-5与FP32 h=1e-6。后者由本次扫描选择，属于事后校准，不是独立验证的通用默认值。

| N | BF16相对L2 | FP32相对L2 | BF16 weight MAE | FP32 weight MAE | 删除最大绝对exact token后的BF16 MAE | 对应FP32 MAE |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.43158 | 0.00071852 | 0.050177 | 0.00015354 | 0.052358 | 0.00016022 |
| 32 | 0.44768 | 0.00097397 | 0.070406 | 9.4709e-05 | 0.073467 | 9.8827e-05 |
| 128 | 0.49148 | 0.00072381 | 0.057889 | 0.00016618 | 0.060406 | 0.0001734 |
| 1024 | 0.46116 | 0.00042486 | 0.059805 | 0.00011709 | 0.062405 | 0.00012218 |

删除单个极端token后权重误差依然存在。随机位置、极端位置、删除最大点、各题各重复的完整表见[主汇总](runs/comparison/report.md)与[配对CSV](runs/comparison/paired_calibration.csv)。

## 完整response上的probe重复采样

每题N16有32次、N32有16次、N128有8次；同一N内probe块不重叠，跨N嵌套，不能当作独立题目重复。BF16 FD h=1e-5，FP32控制 h=1e-6。

| 题目 | N | 重复/同质 | probe方向cos(A)均值±SD | BF16数值weight MAE | FP32数值weight MAE | exact-N对A权重MAE（含fallback0） | BF16对A权重MAE（含fallback0） |
|---|---:|---:|---|---:|---:|---:|---:|
| homotopy | 16 | 32/0 | 0.364±0.048 | 0.025151 | 5.6269e-05 | 0.081801 | 0.077378 |
| homotopy | 32 | 16/0 | 0.484±0.043 | 0.025442 | 5.5932e-05 | 0.078327 | 0.074101 |
| homotopy | 128 | 8/0 | 0.727±0.029 | 0.02573 | 6.4871e-05 | 0.062606 | 0.061728 |
| zeros | 16 | 32/1 | 0.161±0.052 | 0.056822 | 0.00016634 | 0.20893 | 0.20463 |
| zeros | 32 | 16/0 | 0.225±0.052 | 0.055081 | 0.00016776 | 0.17873 | 0.17473 |
| zeros | 128 | 8/0 | 0.369±0.033 | 0.053454 | 0.00015091 | 0.15454 | 0.15842 |

不依赖A参考的直接重复稳定性：先对每个token计算不同probe重复下的权重标准差，再汇总token均值和第90百分位。这里包含fallback0；百分位是token间分布，不是置信区间。各N重复数不同，仅作描述性比较。

| 题目 | N | exact-B权重SD均值 / P90 | BF16权重SD均值 / P90 |
|---|---:|---:|---:|
| homotopy | 16 | 0.0883 / 0.4586 | 0.0816 / 0.4513 |
| homotopy | 32 | 0.0860 / 0.4451 | 0.0804 / 0.4467 |
| homotopy | 128 | 0.0708 / 0.3653 | 0.0683 / 0.3607 |
| zeros | 16 | 0.2585 / 0.4859 | 0.2458 / 0.4788 |
| zeros | 32 | 0.2107 / 0.4885 | 0.1978 / 0.4783 |
| zeros | 128 | 0.1894 / 0.4856 | 0.1840 / 0.4759 |

即便N128，部分token权重仍随probe大幅变化；平均weight MAE或方向cos不能代表每个token都稳定。

排除接近零的score后，完整response上符号一致率如下。每列均要求其参考score绝对值>0.1；FD列参考为本次B方向exact，采样列参考为A方向exact，token集合与分母不同，不能把两列误差直接相加。只对非同质fold计算score指标。

| 题目 | N | BF16 FD vs exact-B sign | FP32 FD vs exact-B sign | exact-B vs exact-A sign |
|---|---:|---:|---:|---:|
| homotopy | 16 | 0.89986 | 1 | 0.58106 |
| homotopy | 32 | 0.89165 | 1 | 0.59468 |
| homotopy | 128 | 0.8935 | 1 | 0.6745 |
| zeros | 16 | 0.90377 | 1 | 0.58347 |
| zeros | 32 | 0.90809 | 1 | 0.62817 |
| zeros | 128 | 0.9081 | 1 | 0.68174 |

同一批probe记录、同一FP32 target autograd，仅切换probe梯度精度得到的weight MAE：

| 模型/题目 | N16 | N32 | N128 | N1024 |
|---|---:|---:|---:|---:|
| audit-base-homotopy | 0.0037822 | 0.0040193 | 0.0057683 | 0.00079092 |
| audit-base-zeros | 0.013689 | 0.022884 | 0.0040841 | 0.0028207 |
| audit-base-coin | 0.014637 | 0.0095691 | 0.014315 | 0.0069731 |
| audit-aux-homotopy | 同质，未定义 | 0.081447 | 0.056948 | 0.078748 |

Aux训练后权重上，probe计算精度本身也是显著误差来源；不能将它混入FD误差，也不能只修复FD而假定整体估计器已经准确。图中的cos(A)比较包含独立probe采样与精度两种差异；上表使用相同probe记录来隔离精度变化。

probe噪声在这些条件下往往大于FD引入的平均权重误差。BF16幅度压缩有时使权重更接近有噪声A参考，因此“数值更精确”不能直接推导“训练表现更好”。N128提高方向一致性，但这些少量题目不能确定全局最优N。

## 算术、FSDP与实际加权梯度控制

| 控制 | 相对L2 | weight MAE | 说明 |
|---|---:|---:|---|
| bf16_ad_surrogate_vs_fp32_jvp | 0.067829 | 0.0038981 | 全515 tokens，同一A1024方向 |
| bf16_fd_h1e-05_vs_fp32_jvp | 0.55213 | 0.035842 | 全515 tokens，同一A1024方向 |
| bf16_fd_h1e-05_vs_bf16_ad_surrogate | 0.56835 | 0.0333 | 全515 tokens，同一A1024方向 |
| full-response FP32 FD h=1e-05 vs JVP | 0.59044 | 0.0018946 | 全515 tokens，同一A1024方向，eager |
| full-response FP32 FD h=1e-06 vs JVP | 0.011538 | 7.0261e-05 | 全515 tokens，同一A1024方向，eager |
| H100 replay fp32 h=1e-06 | 0.00041866 | 8.0651e-05 | 同一A1024方向，24位置，H100上autograd参考 |
| H100 replay bf16 h=1e-05 | 0.61868 | 0.073192 | 同一A1024方向，24位置，H100上autograd参考 |
| fsdp-real-base-n32: fp32 h=1e-06 | 0.0004881 | 6.9073e-05 | 实际4B、2 GPU FULL_SHARD、自身方向autograd |
| fsdp-real-base-n32: bf16 h=1e-05 | 0.64495 | 0.030416 | 实际4B、2 GPU FULL_SHARD、自身方向autograd |
| fsdp-real-base-n32-fa2: fp32 h=1e-06 | 0.00039733 | 0.00012015 | 实际4B、2 GPU FULL_SHARD、自身方向autograd |
| fsdp-real-base-n32-fa2: bf16 h=1e-05 | 0.61684 | 0.028423 | 实际4B、2 GPU FULL_SHARD、自身方向autograd |

BF16 autograd surrogate仍远好于BF16 FD；FD误差并非仅由BF16算术目标与FP32目标不同解释。FSDP/FA2复现同样问题，排除了只在standalone SDPA出现的解释。FA2控制的BF16路径使用FlashAttention2，FP32路径按原生产full_precision_scores上下文切换为SDPA；它检验实际精度切换路径，包含kernel变化，不能称为全程FP32 FlashAttention2。仍未覆盖完整trainer packing、optimizer及训练轨迹。

全515-token response的加权梯度（同A1024方向、sum reduction、固定residual）：

| 权重来源 | 与exact-weighted梯度cos | 梯度相对L2 | weight MAE | 固定A方向投影 |
|---|---:|---:|---:|---:|
| exact_jvp | 1 | 0 | 0 | 366.75 |
| fp32_fd_h1e-5 | 0.99605 | 0.089495 | 0.0018946 | 366.18 |
| fp32_fd_h1e-6 | 0.99999 | 0.0048138 | 7.0262e-05 | 366.74 |
| bf16_fd_h1e-5 | 0.88347 | 0.49306 | 0.035842 | 322.49 |
| unweighted | 0.67171 | 1.1058 | 0.50177 | -109.07 |

这说明误差可改变实际加权更新方向；但这只是固定response的梯度和线性代理投影，不能解释为实际Adam更新或Math准确率变化。

## 训练后master与量化中心

Aux保留源DTensor中的FP32 master，未在合并时提前转BF16。固定同一Aux checkpoint、方向、target及residual，分别以原master和BF16-rounded master为中心，各自对照自身FP32 autograd。

| 中心 | FD | h | 相对L2 | weight MAE |
|---|---|---:|---:|---:|
| master | fp32 | 1e-05 | 0.59909 | 0.0068088 |
| master | fp32 | 1e-06 | 0.21862 | 0.0002098 |
| master | fp32 | 1e-07 | 0.006177 | 0.00045083 |
| master | bf16 | 1e-05 | 0.60123 | 0.10029 |
| master | bf16 | 1e-06 | 0.30323 | 0.20364 |
| master | bf16 | 1e-07 | 1.2566 | 0.46269 |
| bf16 | fp32 | 1e-05 | 0.5414 | 0.009938 |
| bf16 | fp32 | 1e-06 | 0.31673 | 0.00033515 |
| bf16 | fp32 | 1e-07 | 0.0064192 | 0.00082935 |
| bf16 | bf16 | 1e-05 | 0.6311 | 0.1043 |
| bf16 | bf16 | 1e-06 | 0.71925 | 0.13694 |
| bf16 | bf16 | 1e-07 | 0.83065 | 0.32806 |
| master | fp32 | 3e-07 | 0.03522 | 9.5765e-05 |
| master | fp32 | 3e-08 | 0.064479 | 0.0022667 |
| master | fp32 | 1e-08 | 0.20065 | 0.0059231 |
| master | bf16 | 3e-07 | 0.15749 | 0.36168 |
| master | bf16 | 3e-08 | 1.3448 | 0.27945 |
| master | bf16 | 1e-08 | 9.3979 | 0.37739 |
| bf16 | fp32 | 3e-07 | 0.045015 | 0.00026745 |
| bf16 | fp32 | 3e-08 | 0.064078 | 0.0021416 |
| bf16 | fp32 | 1e-08 | 0.19237 | 0.0039504 |
| bf16 | bf16 | 3e-07 | 0.53045 | 0.33542 |
| bf16 | bf16 | 3e-08 | 2.3537 | 0.45463 |
| bf16 | bf16 | 1e-08 | 8.2194 | 0.41354 |

固定N32 repeat0 BF16 probe方向、h=1e-5的实际+δ扰动（相对同一路径零偏移的量化权重，统计全部参数）：

| 模型 | 改变参数占比 | 实际范数/请求范数 | 投影/请求位移 | 与请求方向cos |
|---|---:|---:|---:|---:|
| audit-base-homotopy | 0.1235% | 0.71796 | 0.47694 | 0.6643 |
| audit-aux-homotopy | 0.4248% | 8.8457 | 0.90685 | 0.10252 |

这一控制直接显示BF16扰动并不沿请求方向等比例移动。这里的实际位移排除了零偏移量化误差，不是简单把master到BF16的舍入距离计入扰动。

Base权重位于BF16网格，Aux权重绝大多数不在网格上。小扰动跨越量化边界的行为不同，不能只用初始化模型推断训练中表现。Aux中FP32 h=1e-6仍可能有较大score误差；sigmoid饱和可掩盖大score误差，所以同时报告score和weight。上述小步长扫描是特定方向上的事后校准。

## 教师筛选与独立验证

原主题的独立8次teacher screen结果：homotopy 5/8、zeros 3/8，符合原训练[1/8,7/8]规则；coin 8/8不符合，作为低student成功率压力测试单独理解。新题按照冻结顺序选首个尚未使用且合格的row46947（7/8，一次截断保留），没有根据FD误差或utility选题。

- `bank-screened-validation`：succeeded。

- `validation-screened`：succeeded。

- `center-calibration-aux`：succeeded。

- `full-prefix-screened-validation`：succeeded。

- `prefix-weighted-screened-validation-v2`：succeeded。

新题student bank：9/2048正确，平均2428.5tokens，0次截断。teacher筛选合格不保证student probe具有足够的正负样本。

新题单独列出，不混入原三题均值。下表只用每个N的repeat0同一BF16方向，采用前述Base校准选出的FP32 h=1e-6与原BF16 h=1e-5。若repeat0同质则如实显示fallback0，不借用别的repeat补数。

| N | 已完成BF16方向/其中同质 | repeat0正确数 | BF16相对L2 | FP32相对L2 | BF16 weight MAE | FP32 weight MAE |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 3/3 | 0 | fallback0 | fallback0 | 未定义 | 未定义 |
| 32 | 3/3 | 0 | fallback0 | fallback0 | 未定义 | 未定义 |
| 128 | 3/1 | 0 | fallback0 | fallback0 | 未定义 | 未定义 |
| 1024 | 1/0 | 4 | 0.42923 | 0.00098594 | 0.034161 | 0.00013991 |

新题所有已完成的非同质BF16 probe条件，固定h=1e-5的逐重复结果（24个抽样位置；同质条件仍见上表，未替换或丢弃）：

| N | repeat | probe正确数 | cos(A) | BF16相对L2 | FP32相对L2 | BF16 weight MAE | FP32 weight MAE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1 | 1 | -0.017288 | 0.4084 | 0.0075055 | 0.10774 | 0.0041076 |
| 128 | 2 | 1 | 0.03224 | 0.50558 | 0.014827 | 0.0083982 | 0.00056105 |
| 1024 | 0 | 4 | 0.017457 | 0.42923 | 0.0048561 | 0.034161 | 0.002328 |

cos(A)还包含BF16 probe与FP32 A的精度差异，A本身也仅有5条成功response；不能把它当无噪声真值，或把该列偏差全部归因于采样。后续训练对照应同时报告student成功率及fallback比例。

同精度的直接采样对照：新题A1024含5条成功、B1024含4条成功，两者都用FP32计算，方向cos仅为0.01408。因此低成功率题目上，即使1024 rollouts也不能自动作为稳定梯度或真值；这项差异没有混入BF16 probe精度变化。

新题完整冻结prefix验证：2048个位置，复用A1024方向。原response共2304 tokens，此处只覆盖已冻结范围。先通过24个冻结位置的JVP/backward一致性门槛，再进行FD比较；不是新增独立probe。

| FD | h | 完整prefix相对L2 | weight MAE | sign一致率（参考绝对值>0.1） |
|---|---:|---:|---:|---:|
| fp32 | 1e-05 | 0.022431 | 0.0012659 | 0.99773 |
| fp32 | 1e-06 | 0.0013046 | 0.00011017 | 1 |
| fp32 | 1e-07 | 0.012602 | 0.001135 | 1 |
| bf16 | 1e-05 | 0.46428 | 0.027059 | 0.87982 |
| bf16 | 1e-06 | 2.8074 | 0.11091 | 0.48753 |
| bf16 | 1e-07 | 26.496 | 0.13378 | 0.4966 |

加权prefix检查的首次尝试被旧脚本的完整response校验拒绝，发生在模型加载前；失败保存在 `runs/prefix-weighted-screened-validation`。修正尝试显式启用prefix模式并核对2048/2304范围，默认模式仍拒绝prefix，未改变数据或五组权重方案。

## 建议与尚缺证据

1. 保留当前BF16结果作为有用但有偏差的估计器；不要以高Pearson或Richardson名称证明其精确。

2. 后续训练消融应分开比较probe数量与数值方法。优先在真实最优checkpoint上，以少量固定target做FP32/JVP或autograd校准，查看score误差、非饱和权重、sign、去极端点及更新方向，再确定FD步长；不能全局硬编码本次Base优选的1e-6。

3. 若改用FP32 FD，先进行相同rollout/seed的短程训练对照，再判断最终收益。当前实验没有证明accuracy改善；也没有证明BF16应直接弃用。

4. 必须补上最优N64 step20 checkpoint及原frozen训练prompt上的验证。当前资产缺失使这一要求未完成，Aux不是替代证据。

## 复现与产物

分支 `inf-token-reweight`，基础revision `8634274d5ecf076d96bb976eee1239232263b33c`。每个run保存source快照及SHA、spec、runtime、真实job映射、逐条件结果、参数恢复校验与终止日志。诊断运行在RunAI，未在本机使用CUDA。

- [完整主实验报告与统计定义](runs/comparison/report.md)
- [主表原始CSV](runs/comparison/comparisons.csv)
- [同一步长配对](runs/comparison/paired_primary.csv)
- [同条件步长校准配对](runs/comparison/paired_calibration.csv)
- [完整response重复：homotopy](runs/dense-repeats-base-homotopy/dense_summary.json)
- [完整response重复：zeros](runs/dense-repeats-base-zeros/dense_summary.json)
- [实验协议](protocol.json)

![固定case的步长扫描](runs/comparison/figures/error_vs_h.png)

![probe方向一致性](runs/comparison/figures/gradient_cosine_vs_n.png)

![实际权重与独立A的偏差](runs/comparison/figures/practical_weight_stability_vs_n.png)

## 计算与验证记录

已归档且GPU数可核实的26个终止容器合计9.657 allocated GPU-hours。此数不含排队、不等于CUDA活跃时间或账单；还有资源声明缺失的早期容器和当前运行任务未计入，不能视为完整总成本。详见[逐容器记录](compute-accounting.json)。

FP32 FD在实际2-GPU FSDP/FA2的单target微基准中四次前向约0.65秒，BF16约0.33秒（h=1e-5）；不含probe构造、模型加载及训练其余部分，不能据此声称完整训练只有相同倍率开销。

515-token完整response的FP32 eager JVP约2.15秒，所选24-token reverse-mode oracle约20.61秒；已验证两者一致。它是本次诊断的有效工具，尚未验证16K序列/FSDP训练集成，不作为已完成的生产替换。

[本机软件检查](software-tests-final.log)：107个测试，9个因本机无GPU依赖跳过；核心数值测试另在GPU worker执行通过。各run的board-verification.json及[真实样本/报告/图像API核验](artifact-api-verification.json)保存实际看板读取证据。早期容器启动与tokenization失败保存在attempts，未混入有效bank。
