# Probe-gradient pilot results

**Post-experiment finding (2026-11-03):** Unterminated `<think>` blocks leaked into later inference history. A full token audit and gradient decomposition confirmed a material effect on sampled cosine signs. Original scores are preserved, but the strict no-history-reasoning policy was not fully implemented. See the reasoning-context diagnostic report; a corrected-harness experiment has not yet been run.

Main metric excludes all 32 probe tasks: 768 held-out tasks, 3 repeats per model.

| Model | Held-out accuracy | Repeat SD | Probe accuracy | All800 accuracy |
|---|---:|---:|---:|---:|
| random100 | 36.50% | 0.30 pp | 47.92% | 36.96% |
| top100 | 35.94% | 0.47 pp | 48.96% | 36.46% |
| bottom100 | 36.07% | 1.13 pp | 46.88% | 36.50% |

## Paired comparisons

- top100-minus-bottom100: -0.13 pp; paired task bootstrap 95% CI [-1.909722222222222, 1.6059027777777777] pp.
- top100-minus-random100: -0.56 pp; paired task bootstrap 95% CI [-2.213541666666667, 1.0416666666666665] pp.
- bottom100-minus-random100: -0.43 pp; paired task bootstrap 95% CI [-2.083333333333333, 1.215277777777778] pp.

Primary top-vs-bottom: inconclusive at the paired-task 95% interval.
Exploratory top-vs-random: inconclusive at the paired-task 95% interval.

## Selection audit

| Group | Trajectories | Expanded rows | Original100 overlap | Mean cosine | Supervised tokens |
|---|---:|---:|---:|---:|---:|
| top100 | 100 | 1999 | 47 | -0.019590 | 1141914 |
| bottom100 | 100 | 1634 | 53 | -0.026688 | 1001194 |

Probe: 32 questions × 8 repeats; 15 questions have nonzero empirical reward variance. Original random100 expands to 1853 samples.

All 200 cosine scores were negative. Top100 means relatively better alignment, not a positive predicted gain. The ranking/sign was not flipped after observing this.

## Limits

- Benchmark-feedback pilot, not blind DataSense selection.
- Probe coverage excludes long_context.
- Held-out tasks can share tool environments with probes; this is not an unseen-environment test.
- One training seed; equal trajectories but unequal expanded sample/token budgets.
- Cosine uses raw SGD direction; the fixed training optimizer is AdamW.

## Category breakdown (held-out tasks; exploratory)

| Category | Tasks | Random100 | Top100 | Bottom100 |
|---|---:|---:|---:|---:|
| base | 184 | 42.03% | 43.48% | 39.86% |
| miss_func | 192 | 36.81% | 36.28% | 38.72% |
| miss_param | 192 | 30.38% | 28.47% | 30.21% |
| long_context | 200 | 37.00% | 35.83% | 35.67% |

These subgroup values do not replace the predeclared overall top-vs-bottom comparison. Probe sampling used temperature 1/full softmax; formal evaluation used temperature 0.7 and the original defaults.

## Validation timing

Pre-scoring tests checked tiny-Qwen gradients against standard HF loss and checked a descent update. The planned directional finite-difference check was completed only during the final audit (relative error 3.08e-5), not before scoring. The original protocol is archived; ranking and experimental results were not changed.

## Completion audit

Independent CPU replay reproduced all 256 probe rewards and all 7200 formal evaluation outcomes exactly. The 768-task exclusion, model means, per-category results and paired bootstrap intervals were independently recomputed. Candidate membership, disjoint ranking splits, actual SFT sample counts/configuration/checkpoints, and saved probe-gradient hashes passed verification. See [audit result](../completion-audit/result.json).
