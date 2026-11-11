# Token-intervention handoff (2026-10-20)

## Start here

This branch is `inf-token-reweight`. The verified, lightweight results are
in this directory; raw generations, per-position JSON, model weights, and
datasets are intentionally not committed. The original shared-filesystem
outputs are under `/volume/pt-dev/users/qshan/G-OPD-qz-token-reweight-outputs`.
Do not treat these single-token interventions as a training result or as
held-out benchmark improvement.

## What was measured

For one student response and token position `t`, let
`u_t = log q(y_t | prefix) - log p_old(y_t | prefix)`. The recent Qwen study
selects 30 positions with the largest `|u_t|` before clipping. It uses the
project's math verifier without manual relabeling. From 1024 independent
student responses to the same prompt, form a centered binary-outcome policy
gradient using **summed**, not length-normalized, response log-probabilities:

`g_R = mean_i [(R_i - mean(R)) grad log p(y_i | x)]`.

For scoring and intervention, set `u'_t = clip(u_t, -1, 1)`. The exact score is
`s_t = <g_R / ||g_R||, u'_t grad log p(y_t | prefix)>`. This is the dot product
with the **OPD update direction**, not with its loss gradient. Each token is
updated independently from the same original checkpoint with pure FP32 SGD:
`theta' = theta + 5e-6 u'_t grad log p(y_t | prefix)`. No Adam, gradient
normalization, gradient clipping, weight decay, or accumulation across tokens.

For each position, baseline and updated models answer the **same source prompt**
under 4096 paired random seeds (temperature=1, top_p=1, thinking disabled).
Utility in percentage points is `100 * (n01 - n10) / 4096`, where `n01` is
wrong-to-correct and `n10` correct-to-wrong. The intervals are **pointwise 95%
Tango matched-pair score CIs**, not multiplicity-adjusted. Paired seeds reduce
sampling noise but do not require identical baseline/updated text. Truncated
responses are graded by the unchanged verifier; format and length effects can
therefore contribute to utility.

## High-resolution Qwen results

| Student | Prompt / target response | Max tokens | Probe split-half cosine | Pearson | Spearman | Pearson / Spearman without largest `|s|` |
|---|---|---:|---:|---:|---:|---:|
| Base | Homotopy group of `S^2 x S^1`; 324 tokens, verifier-correct | 16384 | Not measured for the 1024-response probe | 0.956 | 0.839 | 0.825 / 0.822 |
| OPD step 10 | Sum of uncountably many zeros; 1486 tokens, verifier-correct | 16384 | 0.682 | 0.989 | 0.706 | 0.972 / 0.675 |
| OPD step 20 | Klein-bottle triangulation; 2602 tokens, verifier-correct | 8192 | 0.440 | 0.736 | 0.401 | 0.681 / 0.337 |

Base uses **1024-probe scores** paired by exact token position
with already-completed **SGD 5e-6 utilities**. The score-plan directory name
contains `lr1em6`, but those lr=1e-6 utilities are *not* used here. Do not
substitute the older 256-probe Base split-half cosine; none was measured for
the 1024-probe scores.

The three prompts differ, so the row differences are **not** a checkpoint
trend. Step-20's split-half cosine **failed the predeclared 0.5 gate**; the
experiment proceeded only with explicit user authorization. Probe truncations
were 163/1024 at step 20 and 0/1024 at step 10. Step-20 had 6/30 pointwise
Tango CIs excluding zero (3 gains, 3 losses), with utilities ranging from
-10.62 to +5.30 pp. Examples: `recall` (position 22) scored +41.97 and had
+5.30 pp utility; `But` (position 586) scored -38.81 and had -10.62 pp;
`concept` (position 482) scored -28.46 and had -5.08 pp. These are causal
effects under this one prompt and verifier, not semantic judgments about the
token strings alone.

- [Base and step-10 full tables](base_ckpt10_report.md)
- [Step-20 full table](ckpt20_report.md)
- [Base and step-10 coordinates](base_ckpt10_coordinates.csv)
- [Step-20 full-precision coordinates](ckpt20_coordinates.csv)
- [Step-20 score/utility plot](ckpt20_score_utility.png)

## Broad SmolLM check: important counterevidence

The separate SmolLM MixSFT / math-teacher study tested **six checkpoints x 10
responses x 10 top-absolute-residual positions = 600 interventions**. It used
thinking enabled, *unclipped* residual, SGD 1e-6, and only 256 paired rollouts
per position, so its scores and utilities are **not directly comparable** to
the Qwen study above. Mean within-response Spearman across the ten prompts was
-0.085 (Base), +0.096 (step 4), +0.056 (step 8), +0.032 (step 12), +0.049
(step 16), and +0.077 (step 20). Prompt-bootstrap 95% intervals all included
zero. This does not establish reliable cross-prompt token ranking. Few token
utilities individually excluded zero at 256 pairs. Do not claim the Qwen
single-response correlations generalize from these data.

- [600-position utility-only report](smollm_600_utility_report.md)
- [600-position utility CSV](smollm_600_utility_positions.csv)
- [Utility box plot](smollm_600_utility_boxplot.svg)
- [Frozen six-checkpoint protocol](smollm_600_protocol.md)

## Code, artifacts, and rerun notes

- `tools/qwen_mid_checkpoint_probe.py`: 1024-response probe, split-half
  cosine, exact FP32 token scores, and frozen `plan.json`.
- `tools/run_qwen_mid_probe_pipeline.py`: eight-GPU bank generation and score.
- `tools/run_smol_math_dense_oracle.py`: independent FP32 SGD intervention,
  vLLM weight reload, paired evaluation, per-position saved output. It also
  supports `--reuse-baseline`, `--pending-offset`, and
  `--gradient-checkpointing` for long prefixes.
- `tools/run_qwen_checkpoint_interventions.py`: simple group launcher; with
  `--shards 16`, run offsets 0 and 8 on separate eight-GPU workers. For split
  groups, run summarization manually after all 30 results exist.
- `tools/summarize_qwen_topresidual_sgd.py` validates frozen plan, local
  baseline fingerprints, paired seed lists, update magnitude and IPC transfer;
  `tools/dense_oracle_stats.py` computes Tango intervals.
- `tools/report_base_ckpt10_interventions.py` joins Base's 1024-probe score
  plan to its 5e-6 utility study, checking target and residual equality.

Original step-20 directory:
`/volume/pt-dev/users/qshan/G-OPD-qz-token-reweight-outputs/qwen_ckpt20_row2670_probe1024_max8192`.
The other two source directory names are in the full-table report. Re-run
summarization with:

```bash
python3 tools/summarize_qwen_topresidual_sgd.py \
  --study /volume/pt-dev/users/qshan/G-OPD-qz-token-reweight-outputs/qwen_ckpt20_row2670_probe1024_max8192 \
  --lr 5e-6
```

GPU hosts are **offline**. Stage model checkpoints, teacher, dataset,
dependencies and source outputs on the shared filesystem before GPU submission;
never rely on an on-GPU download. The historical compatible environment was
`/volume/pt-dev/users/qshan/.venvs/open-mopd-smollm3/bin/python` with
`PYTHONPATH=<repo>/verl`, HF offline flags, FP32 and TF32 disabled. The
step-10 and step-20 student checkpoints were in
`/volume/pt-dev/users/qshan/G-OPD/outputs/singleteacher_opd_math_math_n32_unclipped_unweighted_curriculum1600_v1_seed27115_step50/merged_hf_model_opd_math_math_n32_unclipped_unweighted_curriculum1600_v1_seed27115_step50_checkpoint{10,20}`.

The original long-prefix intervention sometimes OOMed during backward. Its
successful retries used activation checkpointing (all model dropout zero),
not a change to score, lr, seeds or generation settings. Early orchestration
also reserved eight GPUs while waiting for baselines and the host was
auto-stopped for low utilization. **Stage baselines before submitting** and
do not reserve GPUs for dependency polling. Keep 16 local baselines and
position results disjoint to avoid duplicate writes. Do not run a new GPU
experiment solely from this handoff without checking the model paths and
environment on the destination machine.

## Verification at handoff

- The saved summaries checked 30/30 positions for each Qwen response, matched
  baseline fingerprints and all 4096 paired seeds, and verified actual vLLM
  parameter transfer for nonzero updates.
- The Base/step-10 coordinate CSV has 60 rows; the step-20 CSV has 30; the
  broad SmolLM utility CSV has 600.
- `python3 -m unittest discover -s tests -p test_dense_oracle.py -k
  PairedStatisticsTests` passed 6/6. The full test module could not run on
  this CPU host because its PyTorch-dependent tests need `torch`, which is not
  installed here. The GPU training/inference paths were not re-executed for
  this documentation-only handoff.
