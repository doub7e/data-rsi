# 预条件投影：失败模式与top训练轨迹

分析范围：Q1 和第二题67，各比较随机100、F128、K128，共1536条已完成评测；每组读取全部256条，核对官方reward。Top8按归因分数降序；不重新训练或生成轨迹。

## 最主要行为差别：报错后的纠正

| 行为 | random100 | F128 | K128 |
|---|---:|---:|---:|
| 首次加油传50 | 177 | 187 | 180 |
| 首次加油传39.5 | 65 | 61 | 73 |
| 先传50后改39.5 | 82 | 83 | 108 |
| 传50后未改39.5 | 95 | 104 | 72 |
| 先传50后改39.5且整题成功 | 79 | 75 | 104 |

在首次传50的轨迹中，随后改成39.5的比例分别为82/177=46.3%、83/187=44.4%、108/180=60.0%。K的首次错误并没有减少；更突出的变化是从工具报错继续查询、修正参数。

固定配对样例r004：random100调用fillFuelTank(50)失败后，声称油箱似乎已经满了，询问用户是否想查询。K128也先调用fillFuelTank(50)，随后实际调用displayCarStatus(option=fuel)，再调用fillFuelTank(39.5)，成功完成。

这是后验行为分解，不是对训练轨迹的因果消融。F top8也有错误后重试的轨迹，不能仅凭K的4577具有纠错过程就断言它带来收益。

## 第二题67：失败位置

Turn0工具缺失；turn1恢复距离工具；turn2判断续航；turn3加满油；turn4启动引擎。以下为官方首先报告的失败位置，不是各轮独立正确率，也不保证是轨迹中按时间最早的错误。

| 模型 | 成功 | 恢复工具t1 | 续航t2 | 加油t3 | 启动t4 |
|---|---:|---:|---:|---:|---:|
| q67_random | 144 | 19 | 0 | 93 | 0 |
| q67_f | 131 | 22 | 3 | 100 | 0 |
| q67_k | 168 | 7 | 4 | 75 | 2 |

状态不匹配错误不一定在错误字符串中写轮号。本分析使用官方 checker 已执行结果前缀的最后一项定位（len(execution_result)-1），并对有明确轮号的错误交叉核验。

## 第二题67：行为计数（覆盖全部轨迹）

| 行为 | random100 | F128 | K128 |
|---|---:|---:|---:|
| t1未尝试estimate_distance | 18 | 18 | 5 |
| t2未尝试续航检查 | 13 | 12 | 7 |
| t3未尝试加油 | 12 | 7 | 1 |
| t3查询车辆状态 | 156 | 141 | 179 |
| t3尝试正确增量39.5 | 148 | 144 | 182 |
| t3曾尝试加50 | 177 | 187 | 180 |

工具文档规定 fillFuelTank.fuelAmount 是额外加入量、油箱容量50；初始油量10.5，故目标增量39.5。尝试值计数不代表最终成功，错误后重试可使两种值同时出现。

## Q1对照

| 模型 | 成功 | 首败t5 | t5未调用客服 |
|---|---:|---:|---:|
| q1_random | 132 | 111 | 117 |
| q1_f | 155 | 93 | 94 |
| q1_k | 127 | 117 | 121 |

## 长度与循环

| 模型 | 平均输出tokens | 平均模型调用 | 发生长度截断的轨迹 |
|---|---:|---:|---:|
| q67_random | 8618 | 16.21 | 1 |
| q67_f | 8807 | 16.43 | 1 |
| q67_k | 8622 | 16.65 | 1 |
| q1_random | 8611 | 13.75 | 0 |
| q1_f | 8573 | 13.78 | 1 |
| q1_k | 9005 | 13.79 | 3 |

## Top8逐条阅读

内容概括是事后人工检查，不是逐turn因果归因。一个训练轨迹可同时含正确调用、必要澄清和低效重试。

### q67_f

| 排名 | meta_index | 类型 | 内容与问题 |
|---|---:|---|---|
| 1 | 6099 | non_conversation | 动物观测记录：创建位置/物种、查重、添加和修改观察记录；偏多步事务，没有车辆工具。 |
| 2 | 685 | conversation | 股票交易：查询市场/权限/报价，卖出后由前文金额计算可买股数并执行；有查询→计算→更新的结构。 |
| 3 | 7568 | non_conversation | 酒店改期：查可用性、换房、取消旧预订再创建；同时连续尝试多个冲突ID，是高分但含低效行为的反例。 |
| 4 | 8235 | non_conversation | 软件本地化：启用locale、读取翻译、审批、创建新key和译文；F/K共同top8，具有启用后继续执行的结构。 |
| 5 | 4398 | non_conversation | 任务管理批处理：查项目任务、按状态完成/激活、重分配和创建；偏条件判断与状态修改。 |
| 6 | 2069 | conversation | 用户维护：查moderator且未登录者并锁定，查旧inactive账户并删除；F/K共同top8，使用查询结果决定操作。 |
| 7 | 6752 | non_conversation | 燃油价格数据清理：检查来源可靠性、替换价格记录、刷新汇总；词面有fuel，但不涉及车辆油箱/加油量。 |
| 8 | 918 | conversation | 研究参与者注册：先多轮追问缺失姓名/联系/唯一ID，得到信息后执行，再管理入组状态；并非统一鼓励少澄清。 |

完整top100类型：{'non_conversation': 45, 'conversation': 55}；归因分数与历史原始梯度范数Spearman=-0.0256。

### q67_k

| 排名 | meta_index | 类型 | 内容与问题 |
|---|---:|---|---|
| 1 | 1449 | conversation | 测试用例管理：连续追问描述、前置条件、步骤、期望结果和时间戳，补齐后创建并继续操作；K第一名却有大量澄清，反驳K仅召回立即执行样本。 |
| 2 | 4577 | non_conversation | 租房：查房源、检查条件、激活、补齐日历、计算租金、提交确认；查询与前置条件处理相似，但也有日期调整与重复ID尝试。 |
| 3 | 41 | conversation | 任务管理对话：查用户任务和状态，完成、分配、改优先级、移项目；多轮复用前文对象。 |
| 4 | 8235 | non_conversation | 软件本地化：启用locale、读取翻译、审批、创建新key和译文；F/K共同top8，具有启用后继续执行的结构。 |
| 5 | 2782 | conversation | 调度系统：暂停、查任务/执行记录、按状态和时间筛选删除、恢复调度；前置状态与条件判断。 |
| 6 | 957 | conversation | 商品目录：查集合商品，删除、按前文价格算折扣、更新价格及分类；将已知数值变成操作参数。 |
| 7 | 4036 | conversation | 广告管理：查广告主/活动，按前文预算转移金额，审批与激活；同时包含根据推断日期拒绝激活等混合行为。 |
| 8 | 2069 | conversation | 用户维护：查moderator且未登录者并锁定，查旧inactive账户并删除；F/K共同top8，使用查询结果决定操作。 |

完整top100类型：{'non_conversation': 50, 'conversation': 50}；归因分数与历史原始梯度范数Spearman=0.0510。

### q1_f

| 排名 | meta_index | 类型 | 内容与问题 |
|---|---:|---|---|
| 1 | 3548 | conversation | Hi! First, I’d like to make sure that Elena Rodriguez (ID PH001) had active coverage on the service date for claim CLM005. Can you confirm that for me? |
| 2 | 4341 | conversation | Hey! Could you please check if the ticker symbol “GLOBEQ2” is already in use on the NYSE market? |
| 3 | 8576 | non_conversation | 消息系统：读取未解决消息并更新状态，然后自行撰写允许自拟内容的广播；与Q1执行消息操作的行为较接近。 |
| 4 | 8710 | non_conversation | 1. Change employee **E003**’s employment status from *on_leave* to **active**.   2. For every employee in department **D001** (Engineering) whose current job title is **Software En |
| 5 | 2561 | conversation | Hey! To get started, can you pull up all predictions made by model‑001 from January 14 2023 through January 16 2023, inclusive? |
| 6 | 1787 | conversation | 通讯系统：查外部联系人、创建直接会话、安排事件和发送消息；与Q1客服/消息任务有结构相关性，但不是同一环境。 |
| 7 | 2298 | conversation | Hey! Could you check what’s going on with table T005 for me? I know order O003 was served there—can you pull up its details and tell me the current status of that table? |
| 8 | 1176 | conversation | Hi! I’d like to look into Alice Thompson’s scheduled reminders—specifically for her Lisinopril medication. Can we check what time that reminder is currently set for? |

完整top100类型：{'non_conversation': 45, 'conversation': 55}；归因分数与历史原始梯度范数Spearman=-0.0643。

### q1_k

| 排名 | meta_index | 类型 | 内容与问题 |
|---|---:|---|---|
| 1 | 4582 | non_conversation | 1. The landlord of property **P100** (Seattle apartment) must update the availability calendar so that every day from **2025‑05‑15** through **2025‑06‑13** (a 30‑day period) is mar |
| 2 | 1831 | conversation | Hey there! Can you help me check the current status of the **METARhub** data feed in our system? Let me know if it's running or not. |
| 3 | 752 | conversation | Hey! Before we move forward, could you check if the city “Oslo” is already registered in your repository? |
| 4 | 7216 | non_conversation | 1. Identify all members whose `membership_status` is either **suspended** or **canceled** and whose `last_updated` timestamp is more than 180 days older than the current date (2025 |
| 5 | 4596 | non_conversation | Create a new public playlist for user **USR001** named **“Luna Nova Essentials”**. Populate this playlist with the three most popular tracks released by **Luna Nova (artist_id = AR |
| 6 | 3548 | conversation | Hi! First, I’d like to make sure that Elena Rodriguez (ID PH001) had active coverage on the service date for claim CLM005. Can you confirm that for me? |
| 7 | 4577 | non_conversation | 租房：查房源、检查条件、激活、补齐日历、计算租金、提交确认；查询与前置条件处理相似，但也有日期调整与重复ID尝试。 |
| 8 | 1015 | conversation | Hi there! Could you please start by authenticating me as player “AriaStorm”? |

完整top100类型：{'non_conversation': 48, 'conversation': 52}；归因分数与历史原始梯度范数Spearman=-0.0482。

## 解释边界

第二题F/K top8重叠2条（8235、2069），完整top100重叠60条。K top8中有6条conversation，F有3条；但完整top100中K为50条、F为55条，不能用top8的对话比例解释全部收益。

更合理的描述是：高分轨迹中存在查询状态、沿用前文信息、计算参数、处理前置条件等结构相似性，但不是按车辆/燃油关键词集中召回。F的燃油价格轨迹6752虽词面相近，任务与油箱操作不同；K第一名1449反复澄清，说明“高分=不澄清”不成立。

Q1 F的top8包含消息广播8576、直接通讯1787，与执行客服消息有结构相关性；这只能形成假设，不能证明其导致了成功率提升。

官方参考在第二题续航检查中传入distance=780，但工具文档要求miles；成功轨迹中存在传入约484.67/485的情况。不能把不等于reference参数直接算失败，也不能把通过等同于所有调用和叙述均正确。此处不改变官方评分。

Top8只是100条训练数据的一部分，F/K各有40条不同的训练轨迹。需要移除/替换或匹配对照实验才能确证哪几条轨迹贡献了收益。未在本次分析额外启动这些实验。

![First failure turns](failure-turns.png)

[完整统计](result.json) · [全部top8原文](TRAINING_CASES.md) · [失败/成功配对样例](EVAL_CASES.md)

- Official errors report first failure only; per-turn function occurrence statistics cover all generated turns and are attempts, not necessarily successful execution.
- Keyword counts are text proxies, not manually verified semantic labels.
- Same seed does not hold intermediate states fixed.
- Top8 semantic resemblance cannot establish causal influence; training uses all selected100.
- Question67 was chosen after prior diagnostics; both studies are single-task adaptation.

## 后续top/bottom对照修正

[完整对照审查](../pif-top-bottom-content-20261106/REPORT.md)发现，同样的查询→计算/判断→操作流程在bottom8中普遍存在。此前top8中的语义共同点不能作为高分或训练收益的充分解释。第二题K的实际梯度范围仅为距离工具恢复的turn1，后续加油错误恢复是评测行为，不应混为同一个归因机制。
