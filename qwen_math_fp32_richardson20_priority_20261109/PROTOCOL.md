**最高优先级：FP32 Richardson，sigmoid(3s)，fallback0，20-step**

先训练FP32 Richardson h1e-5，8H200，N64，每步16题，20步；score/loss clip1、identity、seed27115、16K response。以Base初始化，不做5步pilot前置。

用户确认旧/volume不可访问，原随机1600清单未进入Git。复用Git中206个n16合格题及已完成的n8筛选，只补缺失筛选。全部来源、prompt hash和固定320题顺序留存。新题集不冒充历史47.27的原题集。

顺序：存储/依赖检查→准备和补齐筛选→冻结320题→20步FP32训练→FP32导出→四项n32评测及Base对照→报告。其他FD任务保持后置。
