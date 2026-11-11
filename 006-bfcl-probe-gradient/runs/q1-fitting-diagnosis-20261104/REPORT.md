# 续训拟合/泛化诊断

同一批48条轨迹、完全相同token输入和目标，比较起点与三个续训checkpoint。均使用原生位置编码、16K cutoff、FP32权重和BF16 autocast；包含完整当前reasoning。

| 模型 | 分组 | 轨迹数 | 起点NLL/token | 续训后NLL/token | 每轨迹平均配对变化 |
|---|---|---:|---:|---:|---:|
| random50 | selected_seen | 16 | 0.37745 | 0.28381 | -0.09173 |
| random50 | unselected_seen | 16 | 0.39684 | 0.41477 | +0.01706 |
| random50 | heldout | 16 | 0.50490 | 0.54459 | +0.03893 |
| k032 | selected_seen | 16 | 0.39613 | 0.29165 | -0.10136 |
| k032 | unselected_seen | 16 | 0.37963 | 0.39575 | +0.01566 |
| k032 | heldout | 16 | 0.50490 | 0.54166 | +0.03598 |
| f128 | selected_seen | 15 | 0.38268 | 0.28039 | -0.10165 |
| f128 | unselected_seen | 17 | 0.39099 | 0.40788 | +0.01705 |
| f128 | heldout | 16 | 0.50490 | 0.54094 | +0.03527 |

NLL越低，表示更贴合EnvScaler的教师输出；它不等同于Q1工具任务成功率。16条heldout均来自官方val60，未进入这两阶段SFT；这不是完整泛化benchmark。
