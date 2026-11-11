# Guided与Uniform训练轨迹分布

对已保存728+728条轨迹进行描述统计；没有新生成或重新分词。

## 环境频次

{
  "1": {
    "trajectories": 72,
    "trajectory_fraction": 0.0989010989010989,
    "probability_mass_top_probability": 0.08862036076174189
  },
  "5": {
    "trajectories": 175,
    "trajectory_fraction": 0.2403846153846154,
    "probability_mass_top_probability": 0.23198640413021804
  },
  "10": {
    "trajectories": 212,
    "trajectory_fraction": 0.29120879120879123,
    "probability_mass_top_probability": 0.2772952457633073
  },
  "20": {
    "trajectories": 259,
    "trajectory_fraction": 0.3557692307692308,
    "probability_mass_top_probability": 0.34344963747098367
  },
  "50": {
    "trajectories": 354,
    "trajectory_fraction": 0.48626373626373626,
    "probability_mass_top_probability": 0.45029948474502207
  },
  "100": {
    "trajectories": 460,
    "trajectory_fraction": 0.6318681318681318,
    "probability_mass_top_probability": 0.5532168101651865
  },
  "histogram_environment_counts": {
    "0": 400,
    "1": 188,
    "2": 84,
    "3": 31,
    "4": 10,
    "5": 3,
    "6": 4,
    "7": 1,
    "9": 2,
    "13": 1,
    "18": 1,
    "19": 1,
    "53": 1,
    "72": 1
  },
  "never_sampled": 400,
  "sampled_once": 188,
  "sampled_twice_or_more": 140,
  "repeat_draws_beyond_first": 400,
  "empirical_effective_environments": 51.75625,
  "probability_effective_environments": 53.73383878439908
}

## 最常抽到的环境

| 环境 | 抽取数 | 概率 | Guided满分 | Uniform满分 |
|---|---:|---:|---:|---:|
| task_9288 | 72 | 0.088620 | 1/72 | 0.0 |
| task_3480 | 53 | 0.088620 | 42/53 | 1.0 |
| task_2507 | 19 | 0.021310 | 6/19 | 0.0 |
| task_9487 | 18 | 0.019181 | 17/18 | 1.0 |
| task_7374 | 13 | 0.014254 | 12/13 | 1.0 |
| task_4995 | 9 | 0.011097 | 9/9 | 1.0 |
| task_701 | 9 | 0.006548 | 9/9 | 1.0 |
| task_3366 | 7 | 0.001537 | 6/7 | 1.0 |
| task_6646 | 6 | 0.009716 | 6/6 | 1.0 |
| task_1670 | 6 | 0.007217 | 6/6 | 1.0 |
| task_7646 | 6 | 0.006795 | 5/6 | 1.0 |
| task_5341 | 6 | 0.003625 | 1/6 | 1.0 |
| task_4680 | 5 | 0.008364 | 5/5 | 1.0 |
| task_4755 | 5 | 0.007804 | 3/5 | 1.0 |
| task_9586 | 5 | 0.001865 | 0/5 | 0.0 |

## 长度与质量

```json
{
  "guided": {
    "trajectories": 728,
    "unique_environments": 328,
    "full_passes": 526,
    "nonfull_passes": 202,
    "missing_verifier": 1,
    "assistant_calls": {
      "count": 728,
      "mean": 4.7760989010989015,
      "min": 1,
      "p25": 4.0,
      "p50": 4.0,
      "p75": 5.0,
      "p90": 7.0,
      "p95": 7.649999999999977,
      "p99": 11.0,
      "max": 21
    },
    "generated_output_tokens": {
      "count": 728,
      "mean": 6948.79532967033,
      "min": 595,
      "p25": 3328.75,
      "p50": 5438.5,
      "p75": 9213.75,
      "p90": 13591.3,
      "p95": 16225.699999999997,
      "p99": 23393.29000000002,
      "max": 51621
    },
    "training_encoding": {
      "at": "2026-11-09T04:32:12.647001+00:00",
      "dataset_sha256": "f64be282f139097948a01bdff4778e9567aa0a88c62f9e5546d090501a24339a",
      "sft_rows": 728,
      "input_tokens": 6491949,
      "target_tokens": 4882779,
      "original_target_tokens": 4931792,
      "retained_target_fraction": 0.9900618274250009,
      "truncated_rows": 6,
      "zero_target_rows": 0,
      "all_source_rows_preserved": true,
      "encoder": "original SupervisedDatasetProcessor; all assistant reasoning retained"
    },
    "mean_training_target_tokens": 6707.114010989011,
    "mean_training_input_tokens": 8917.512362637362
  },
  "uniform": {
    "trajectories": 728,
    "unique_environments": 728,
    "full_passes": 565,
    "nonfull_passes": 163,
    "missing_verifier": 0,
    "assistant_calls": {
      "count": 728,
      "mean": 4.976648351648351,
      "min": 2,
      "p25": 4.0,
      "p50": 5.0,
      "p75": 5.0,
      "p90": 7.0,
      "p95": 8.0,
      "p99": 11.730000000000018,
      "max": 17
    },
    "generated_output_tokens": {
      "count": 728,
      "mean": 6875.861263736264,
      "min": 1040,
      "p25": 3334.0,
      "p50": 5316.5,
      "p75": 9177.0,
      "p90": 13508.00000000001,
      "p95": 15857.6,
      "p99": 24484.84000000001,
      "max": 46883
    },
    "training_encoding": {
      "at": "2026-11-07T02:05:19.552343+00:00",
      "dataset_sha256": "18302319a659c9daa24c8afe8040bfebc9c8752be9a1ae08d0bb4522b4bd88e5",
      "sft_rows": 728,
      "input_tokens": 6503600,
      "target_tokens": 4837691,
      "original_target_tokens": 4877741,
      "retained_target_fraction": 0.9917892319415893,
      "truncated_rows": 4,
      "zero_target_rows": 0,
      "all_source_rows_preserved": true,
      "encoder": "original SupervisedDatasetProcessor; all assistant reasoning retained"
    },
    "mean_training_target_tokens": 6645.179945054945,
    "mean_training_input_tokens": 8933.516483516483
  }
}
```

## 口径与限制

- Generated output token counts are sums of recorded API completion_tokens including reasoning; they are NOT per-row training tokenization lengths. Exact aggregate training token counts come from the original encoder audit.
- Repeated environment means newly generated teacher trajectory with a distinct seed, not duplicating an identical stored trajectory.
- Uniform has only one outcome per environment; a single-task success comparison is noisy.
- Descriptive concentration and success-rate differences do not establish the cause of evaluation degradation.


## 高频失败环境

{
  "top_environment": "task_9288",
  "top_environment_trajectory_percent": 9.89010989010989,
  "top_environment_failure_share": 0.35148514851485146,
  "descriptive_success_rate_excluding_top_environment": 0.8003048780487805,
  "top_two_trajectories": 125,
  "top_probability_relative_to_uniform": 64.5156226345481
}

上述剔除后的成功率仅为描述性统计，没有重新训练或进行因果验证。
