# multi_turn_base_22：token/turn梯度消融

固定128条轨迹、原64+64分组、每半独立reward均值。关键用户回合为0-based turn 3。所有模式保持实际输入上下文，只改变监督token位置；没有重新采样或按结果配平。

| 模式 | 监督tokens | 全参数cosine | A范数 | B范数 |
|---|---:|---:|---:|---:|
| 全部tokens | 926324 | -0.007456 | 1869.8453 | 2055.4358 |
| 去掉reasoning | 75513 | 0.018194 | 450.7406 | 441.7758 |
| 仅关键turn（含reasoning） | 183399 | 0.023043 | 811.4432 | 837.1104 |
| 仅关键turn且去掉reasoning | 23962 | 0.004615 | 217.8957 | 227.8562 |

![消融与逐层cosine](ablation.png)

## 逐层结果

| 模式 | Decoder层最小cosine | 最大cosine | 中位数 | 正值层数/36 |
|---|---:|---:|---:|---:|
| 全部tokens | -0.059694 | 0.150405 | 0.010812 | 23/36 |
| 去掉reasoning | -0.059583 | 0.138666 | 0.020211 | 31/36 |
| 仅关键turn（含reasoning） | -0.136231 | 0.426674 | 0.016153 | 22/36 |
| 仅关键turn且去掉reasoning | -0.138999 | 0.211860 | -0.003897 | 16/36 |

[完整逐层表](layer-cosines.csv)同时给出平方范数占比，避免把小范数层的高cosine误作全局强信号。全局cosine由完整向量计算，不是逐层cosine平均。

## 解释边界

- “去掉reasoning”保留工具调用、普通可见回答和EOS，不等于仅保留结构化工具调用。Reasoning仍留在当前调用上下文中，只不作为预测目标。
- 关键turn中的所有模型调用均保留；其他turn贡献为零。缺失目标token的轨迹仍计入64条及原reward baseline，不会从分母删除。
- 去掉reasoning或只选关键turn会改变梯度目标，是诊断消融，不能默认视为原完整policy-gradient估计的无偏降方差版本。
- HF/vLLM存在已量化的log-prob数值差异；本比较固定原HF方法，不声称精确on-policy等价。
- 第一题关键turn来自此前定性审计；第二题关键turn在新rollout生成前按任务语义固定。没有搜索哪一层/哪个turn的cosine最高后再只报告该处。
- 只报告一个预先固定split；尚未据此重新选择训练数据或验证下游训练收益。

## 实际decode likelihood对照

第二题记录了实际decode log-prob。按相同mask汇总的HF/vLLM NLL差异见[likelihood-aggregate.json](likelihood-aggregate.json)；这些是轨迹求和的差异，不能当作逐token绝对误差。
