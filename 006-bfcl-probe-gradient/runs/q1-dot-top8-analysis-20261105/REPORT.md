# Dot top8 audit — base Qwen3-4B

Q1: flight booking, insurance, invoice, temporarily missing support tool, tool restoration and contact request, cancellation. Critical distinction: execute a generic support message versus ask the user to supply exact message text.

| Setting | Positive dot /200 | Conversation /top8 | Mean norm /top8 | Dot–norm Spearman /200 | Cosine top8 overlap |
|---|---:|---:|---:|---:|---:|
| f008 | 200 | 8 | 45.40 | 0.962 | 3 |
| f032 | 0 | 0 | 19.80 | -0.801 | 2 |
| f128 | 0 | 0 | 22.56 | -0.832 | 6 |
| k008 | 200 | 8 | 42.63 | 0.692 | 2 |
| k032 | 63 | 3 | 33.54 | 0.071 | 8 |
| k128 | 0 | 0 | 17.45 | -0.922 | 4 |

## f008

| Rank | meta_index | Env | Type | Dot | Cosine | Norm | Cosine rank |
|---|---:|---|---|---:|---:|---:|---:|
| 1 | 3100 | env_101_sft | conversation | 49621.82 | 0.16617 | 50.13 | 1 |
| 2 | 3620 | env_118_sft | conversation | 45588.61 | 0.16182 | 47.29 | 9 |
| 3 | 2937 | env_96_sft | conversation | 45151.02 | 0.16303 | 46.49 | 6 |
| 4 | 3633 | env_118_sft | conversation | 44077.40 | 0.15558 | 47.56 | 34 |
| 5 | 1298 | env_40_sft | conversation | 40918.61 | 0.16387 | 41.91 | 3 |
| 6 | 2054 | env_64_sft | conversation | 40073.78 | 0.15420 | 43.62 | 42 |
| 7 | 713 | env_22_sft | conversation | 39714.90 | 0.15148 | 44.01 | 70 |
| 8 | 1019 | env_31_sft | conversation | 39440.80 | 0.15700 | 42.17 | 25 |

## f032

| Rank | meta_index | Env | Type | Dot | Cosine | Norm | Cosine rank |
|---|---:|---|---|---:|---:|---:|---:|
| 1 | 7894 | env_108_sft | non_conversation | -820.98 | -0.01781 | 14.41 | 10 |
| 2 | 8710 | env_131_sft | non_conversation | -929.47 | -0.01847 | 15.73 | 16 |
| 3 | 5131 | env_22_sft | non_conversation | -996.93 | -0.01010 | 30.85 | 1 |
| 4 | 7087 | env_80_sft | non_conversation | -1013.07 | -0.01843 | 17.19 | 15 |
| 5 | 6010 | env_48_sft | non_conversation | -1180.81 | -0.01512 | 24.42 | 3 |
| 6 | 6922 | env_76_sft | non_conversation | -1192.62 | -0.01996 | 18.68 | 33 |
| 7 | 7720 | env_102_sft | non_conversation | -1303.72 | -0.01867 | 21.83 | 19 |
| 8 | 4582 | env_6_sft | non_conversation | -1328.80 | -0.02719 | 15.28 | 162 |

## f128

| Rank | meta_index | Env | Type | Dot | Cosine | Norm | Cosine rank |
|---|---:|---|---|---:|---:|---:|---:|
| 1 | 6456 | env_60_sft | non_conversation | -105.94 | -0.00226 | 29.41 | 1 |
| 2 | 7720 | env_102_sft | non_conversation | -134.38 | -0.00386 | 21.83 | 3 |
| 3 | 7242 | env_85_sft | non_conversation | -166.78 | -0.00368 | 28.41 | 2 |
| 4 | 4958 | env_17_sft | non_conversation | -204.90 | -0.00555 | 23.15 | 4 |
| 5 | 6953 | env_77_sft | non_conversation | -237.17 | -0.00642 | 23.14 | 5 |
| 6 | 7894 | env_108_sft | non_conversation | -267.10 | -0.01162 | 14.41 | 10 |
| 7 | 8710 | env_131_sft | non_conversation | -308.33 | -0.01228 | 15.73 | 14 |
| 8 | 6010 | env_48_sft | non_conversation | -336.44 | -0.00864 | 24.42 | 6 |

## k008

| Rank | meta_index | Env | Type | Dot | Cosine | Norm | Cosine rank |
|---|---:|---|---|---:|---:|---:|---:|
| 1 | 2054 | env_64_sft | conversation | 1592.08 | 0.02466 | 43.62 | 4 |
| 2 | 1015 | env_31_sft | conversation | 1457.79 | 0.02316 | 42.53 | 8 |
| 3 | 713 | env_22_sft | conversation | 1448.55 | 0.02224 | 44.01 | 15 |
| 4 | 2937 | env_96_sft | conversation | 1430.65 | 0.02080 | 46.49 | 30 |
| 5 | 3615 | env_118_sft | conversation | 1399.97 | 0.02307 | 41.01 | 9 |
| 6 | 489 | env_15_sft | conversation | 1358.72 | 0.02216 | 41.43 | 17 |
| 7 | 3503 | env_114_sft | conversation | 1348.87 | 0.02198 | 41.47 | 18 |
| 8 | 3837 | env_123_sft | conversation | 1348.65 | 0.02253 | 40.45 | 13 |

## k032

| Rank | meta_index | Env | Type | Dot | Cosine | Norm | Cosine rank |
|---|---:|---|---|---:|---:|---:|---:|
| 1 | 3775 | env_122_sft | conversation | 378.45 | 0.01058 | 39.15 | 2 |
| 2 | 6819 | env_72_sft | non_conversation | 356.79 | 0.01017 | 38.39 | 3 |
| 3 | 1449 | env_45_sft | conversation | 305.14 | 0.01070 | 31.21 | 1 |
| 4 | 1761 | env_54_sft | conversation | 271.73 | 0.00855 | 34.79 | 5 |
| 5 | 8844 | env_134_sft | non_conversation | 249.14 | 0.00877 | 31.08 | 4 |
| 6 | 8576 | env_127_sft | non_conversation | 193.32 | 0.00631 | 33.55 | 6 |
| 7 | 5185 | env_23_sft | non_conversation | 160.07 | 0.00608 | 28.81 | 7 |
| 8 | 6237 | env_54_sft | non_conversation | 152.72 | 0.00534 | 31.30 | 8 |

## k128

| Rank | meta_index | Env | Type | Dot | Cosine | Norm | Cosine rank |
|---|---:|---|---|---:|---:|---:|---:|
| 1 | 7894 | env_108_sft | non_conversation | -352.42 | -0.04611 | 14.41 | 2 |
| 2 | 4582 | env_6_sft | non_conversation | -374.49 | -0.04621 | 15.28 | 3 |
| 3 | 6922 | env_76_sft | non_conversation | -490.73 | -0.04954 | 18.68 | 7 |
| 4 | 4579 | env_6_sft | non_conversation | -514.20 | -0.04852 | 19.98 | 5 |
| 5 | 8710 | env_131_sft | non_conversation | -522.49 | -0.06261 | 15.73 | 47 |
| 6 | 7087 | env_80_sft | non_conversation | -541.85 | -0.05944 | 17.19 | 28 |
| 7 | 5547 | env_34_sft | non_conversation | -559.36 | -0.05240 | 20.13 | 10 |
| 8 | 8987 | env_139_sft | non_conversation | -559.56 | -0.05806 | 18.17 | 21 |

[Readable trajectories](CASES.md) · [Full statistics](result.json)

- Semantic analogies are not per-turn gradient attribution.
- Only100 selected trajectories, not only top8, were used in training.
- The six probe estimators have different norms; raw scores should not be compared across configurations.
- Observed rank correlations describe this fixed candidate pool, not a universal relationship.

## 人工阅读：F8 top8

重点读取表现最好的F8以及K32；其余配置的全部top8原文也保存在CASES.md。共有36条不同候选。以下语义联系属于事后观察，不代表已计算出对应turn的梯度贡献。

| F8 rank | meta_index | 内容及与probe的联系 |
|---|---:|---|
| 1 | 3100 | 用户注册、验证、改名、更新密码、删除用户。跨轮沿用对象身份与状态。用户允许时生成占位密码hash，但后续更改密码又追问具体值；策略并非始终自主补参数。 |
| 2 | 3620 | 启用locale，补缺失翻译并设pending，更新及批准翻译，随后禁用locale。启用后继续操作的顺序与probe有结构类比。 |
| 3 | 2937 | 查找文档、归档、创建索引、修改权限、清理日志及会话。跨轮沿用查询返回的ID和修改后的状态。 |
| 4 | 3633 | 启用西班牙语，查询并批准翻译，禁用日语。也是可用性变更后的后续动作。 |
| 5 | 1298 | 激活Polygon投资网络、同步资产、修改多个用户持仓、更新健康度。与probe的状态/约束检查及顺序操作相似。 |
| 6 | 2054 | 查询后删除或修改账户角色与状态，创建新用户，记录登录时间。先确认条件再更改持久对象。 |
| 7 | 713 | 先查市场状态、盘后权限、实时数据订阅；后查询用户Alex失败，用户多次确认ID，模型反复要求确认并最终停止。包含类似probe失败的追问停滞，不能称作“正确执行”正例。 |
| 8 | 1019 | 鉴权后购买装备、支付两种货币、换装、用药、修改成就。与订票/保险/取消类似跨轮操作链；也会追问缺失的装备ID和价格。 |

这些状态变化都不是Q1里工具定义从缺失到恢复的同一事件。训练任务涉及环境对象或权限状态，不能混同。

F8的200候选dot全部正；dot与norm的Spearman=0.9622。F8 top100与纯norm top100重合92条，top8平均norm45.40；全候选conversation平均34.72、non_conversation平均28.42。最强可见规律是大梯度及conversation格式，而非订票或客服领域相似。不能由此断言轨迹长度导致norm更大。

## 人工阅读：K32 top8的混合行为

- 8576（第6）：用户允许自行组织促销内容，教师生成message_content并调用send_broadcast_message。与probe成功行为“构造通用消息并执行”很接近，但训练用户明确授权自由文案，Q1没有同样明确的措辞。
- 1761（第4）：用户允许指定合适instrument_id；教师也自行填充asset_id和占位current_price，继续执行。包含参数补全行为，不能保证每个默认值都正确。
- 3775（第1）：修改笔记标题、内容、标签时追问具体内容。
- 1449（第3）：创建测试用例、执行记录和更新结果时，多次要求用户补齐必填信息。

因此高分集合同时含“补参数并执行”和“缺参数先问”。不能把完整轨迹的正dot解释成其中每个动作都有利。

K32 top8与cosine top8集合完全一致；top100重合97条。新加入6922/7894/8235，移除685/1281/3452。因此dot K32相对cosine的成功率变化不能归因于新的top8内容；实际改变位于选择边界，且整个训练序列/步数随数据变化。

## 其余设置及解释边界

F32/F128/K128的200候选dot全部为负，top8是最不负，而非正向对齐；三者top8均为non_conversation，平均norm分别19.80/22.56/17.45。K128 dot与norm的Spearman=-0.9218，明显偏向小范数候选。F8/K8则全部为正，top8均为conversation。对不同probe的原始dot数值不作横向大小比较。

结论：有跨轮ID绑定、可用状态变化、顺序工具操作及部分自由文本参数补全的共同点，但没有统一“学会Q1关键行为”的证据。F8的大梯度偏好和K32未变化的top8，是比语义类比更强的可验证事实。下一步若解释收益来源，应区分norm-only选择效应、top100边界变化和实际关键turn贡献；本次仅分析，没有额外启动这些实验。
