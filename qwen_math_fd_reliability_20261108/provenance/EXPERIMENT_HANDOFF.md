# G-OPD Token Reweighting: Experiment Handoff

Last updated: 2026-10-20

This is the entry point for continuing the project on another machine. The current research line is **single-teacher OPD with outcome-gradient-guided token weighting**. Early MOPD, Crossmask, ExOPD, teacher-selection, and direction-only experiments are historical context, not the active baseline.

## 1. Repository State

- Repository: `doub7e/OPD-experiments`
- Branch: `inf-token-reweight`
- Upstream: `RUCBM/G-OPD`
- Student: Qwen3-4B Base
- Main teachers: Qwen3-4B Math-RL and Code-RL checkpoints
- Default comparison: vanilla OPD without ExOPD extrapolation
- Detailed selected method: [method_n32_raw_sigmoid.md](method_n32_raw_sigmoid.md)
- Broad historical table: [all_results_summary.md](all_results_summary.md)
- Paper-oriented evidence: [research_reference.md](research_reference.md)

GPU workers cannot access the internet. Models, datasets, packages, evaluation code, and tokenizer assets must be cached on the shared filesystem before submission. Use explicit local paths and Hugging Face/W&B offline flags.

Large outputs are intentionally not versioned. This branch does not commit checkpoints, merged models, raw generations, caches, or JSON result files. Compact results are reproduced below and in Markdown reports.

## 2. Research Question

For a student-sampled token, ordinary sampled-token OPD uses

$$
u_t = \log q(y_t\mid x,y_{<t})-\log p_{\mathrm{old}}(y_t\mid x,y_{<t})
$$

and the local direction

$$
d_t^{\mathrm{OPD}}=u_t\nabla_\theta\log p_\theta(y_t\mid x,y_{<t}).
$$

The hypothesis is that teacher feedback is not equally useful at every sampled token position. We estimate whether each OPD direction aligns with an outcome-improving direction, then use that score to downweight unreliable positions.

## 3. Selected N=32 Method

The selected mainline is the historical **N32 Raw Sigmoid** configuration. Important details follow; do not infer them from older experiment names.

### Data and curriculum

1. Use the Math RL teacher to generate answers for candidate Math prompts.
2. Keep prompts whose teacher accuracy is in `[1/8, 7/8]`.
3. The frozen 800-prompt pool mixes an older 320-prompt, 16-rollout screen and 480 prompts screened with 8 rollouts.
4. Sample 640 prompts with seed `27115`, then sort by estimated teacher accuracy descending.
5. Train 20 steps with 32 prompts per step and 32 student responses per prompt: 1,024 trajectories per step.

The frozen parquet is `prompt_screens/math_pool800_random640_sorted_seed27115/train.parquet`, SHA256 `3b3f8f197183a235d971c7f686e08d77b6df1802acdb4fc1dbb3ca61e88589aa`.

### Two-fold outcome probe

Split each prompt's 32 responses into two deterministic folds of 16. For probe fold (F):

$$
J_F(\theta)=\frac1{|F|}\sum_{i\in F}(R_i-\bar R_F)
\sum_t\log p_\theta(y_{i,t}\mid x,y_{i,<t}),
\qquad g_F=\nabla_\theta J_F(\theta).
$$

Fold A scores fold B's OPD tokens and vice versa, so a target response does not enter its own probe objective. Response log probability is a token sum, not a length-normalized mean.

### Token score

Normalize the outcome gradient and estimate the directional derivative using BF16 four-forward Richardson finite differences:

$$
v_F=\frac{g_F}{\lVert g_F\rVert_2},
\qquad
s_t=u_t\left\langle v_F,\nabla_\theta\log p_\theta(y_t\mid x,y_{<t})\right\rangle.
$$

The relative FD scale is `h=1e-5`. The production path perturbs master parameters at `+h`, `-h`, `+2h`, and `-2h`, clears FSDP caches after writes/restores, and applies Richardson correction.

The selected method uses sampled-token OPD, current pre-update `p_old`, unclipped residual for score and loss, identity geometry rather than Adam-aware preconditioning, and no ExOPD.

### Weighting and fallback

$$
w_t=\sigma\left(3(s_t+1)\right).
$$

This is conservative downweighting centered at `s=-1`: `w(-2)=0.047`, `w(-1)=0.5`, and `w(0)=0.953`. It never boosts above vanilla weight 1.

If a probe fold is all-correct or all-wrong, its gradient is zero and the opposite target fold gets weight zero. It does **not** fall back to unweighted OPD. Zero-weight folds must not dilute valid folds through an incorrect denominator or distributed gradient scaling.

## 4. Correctness Fixes to Preserve

1. **Current residual reference:** use `log q - log p_old/current` for training and oracle diagnostics. An old oracle path used `log q - log r_initial`; those results are implementation-mismatched.
2. **Clipped versus unclipped score:** clipping the OPD loss does not automatically clip score residual. State both choices explicitly.
3. **FSDP cache restoration:** clear gathered/unsharded caches after each FD write and restore. A historical bug could reuse stale perturbed weights.
4. **Fallback reduction:** homogeneous folds with zero weights must not dilute valid folds.
5. **GPU work balance:** distribute probe/FD work by estimated token load. This removes multi-GPU idle time without intentionally changing the score.
6. **Diagnostic storage:** write large diagnostics under `/volume/pt-dev/users/qshan`, not a small system filesystem.

The historical score near 48 from an **unclipped-score plus clipped-loss hybrid** is evidence for that hybrid only. It is not a corrected clipped-score result.

## 5. Main Training Results

Math evaluation uses AIME24, AIME25, HMMT Feb, and HMMT Nov, each with 30 questions x 32 generations and a 16,384-token response cap.

### N32 raw-sigmoid mainline

| Method | AIME24 | AIME25 | HMMT Feb | HMMT Nov | Math Avg |
|---|---:|---:|---:|---:|---:|
| N32 unweighted, same 640-prompt curriculum, step 20 | 61.04 | 54.58 | 31.25 | 36.88 | 45.94 |
| N32 raw sigmoid, historical Richardson, step 20 | 60.42 | 55.83 | 33.44 | 38.13 | **46.95** |

The gain is `+1.02` Math Avg, but this comparison combines weighting with zero-weight fallback. It is not yet a weighting-only causal comparison.

### Work-balanced FD ablation

| Configuration | Step 10 | Step 20 |
|---|---:|---:|
| Historical unbalanced Richardson-4 | 44.56 | **46.95** |
| Work-balanced Richardson-4 | 43.67 | 46.41 |
| Work-balanced central-2 | 43.96 | 46.85 |

Central two-forward FD was much cheaper in the FD portion and finished only `0.10` point below the historical result, but these are single runs.

### Later clipped score/loss follow-ups

These weighted arms use clipped score and loss, `sigmoid(3s)`, screened random prompts, and fallback-drop. The N=1 controls use unscreened random data, so these are not pure N ablations.

| Domain | Configuration | Step | Main score |
|---|---|---:|---:|
| Math | N=1 clip-loss-only, unweighted random | 20 | 46.90 |
| Math | N=16 clipped score/loss sigmoid | 20 | 45.70 |
| Math | N=32 clipped score/loss sigmoid | 20 | 46.98 |
| Math | N=64 clipped score/loss sigmoid | 20 | **47.27** |
| Math | N=128, resumed from N32 | 20 | 46.88 |
| Code | N=1 clip-loss-only, unweighted random | 20 | 59.99 |
| Code | N=16 clipped score/loss sigmoid | 20 | **61.33** |
| Code | N=32 clipped score/loss sigmoid | 20 | 60.87 |
| Code | N=64 clipped score/loss sigmoid | 20 | 60.83 |

`Coding Avg` is the mean of HumanEval+, MBPP+, and LCBv6:

| Configuration | HumanEval+ | MBPP+ | LCBv6 | Coding Avg |
|---|---:|---:|---:|---:|
| N=1 clip control | 85.52 | 68.45 | 26.00 | 59.99 |
| N=16 weighted | 86.43 | 68.98 | 28.57 | **61.33** |
| N=32 weighted | 86.59 | 67.72 | 28.29 | 60.87 |
| N=64 weighted | 87.65 | 67.13 | 27.71 | 60.83 |

Do not claim a universal optimum N: screening, restart history, prompt order, and fallback behavior differ across some rows.

### SG-OPD reproduction

The branch also implements SG-OPD as a separate comparison. The run used `N=8`: 128 prompts per step, 8 responses per prompt, 1,024 trajectories per step, 20 steps, random unscreened data, no residual clipping, and no token score/reweighting. Parameters were `lambda_high=1.8`, `beta=1`, `phi_tau=1`, PTS ratio `0.125`, and PTS SFT alpha `1.0` through step 6, `0.25` at step 7, then zero.

| Domain | AIME24 / HumanEval+ | AIME25 / MBPP+ | HMMT Feb / LCBv6 | HMMT Nov | Average |
|---|---:|---:|---:|---:|---:|
| Math SG-OPD N8 | 59.17 | 55.21 | 30.31 | 37.19 | 45.47 |
| Code SG-OPD N8 | 87.04 | 67.92 | 27.86 | N/A | 60.94 |

Both SG-OPD training and evaluation jobs completed. It did not exceed the strongest local token-reweighting result in these single runs, but data recipes are not fully matched.

## 6. Token Intervention / Oracle Evidence

The oracle studies ask whether an independently applied token-position OPD update changes paired answer accuracy in the direction predicted by the score.

### Multi-prompt evidence

The breadth-oriented experiment used 32 Math prompts, 64 target responses, 512 independently updated token positions, a separate 256-response reward probe, and 256 paired evaluation seeds per position. It used clipped residual, pure SGD at `2.5e-6`, and Richardson `h=1e-4`.

- High-score minus low-score mean effect: **+0.592 pp**.
- 95% prompt-cluster CI: **[+0.185, +1.019] pp**.
- Bonferroni-allocated 97.5% CI: **[+0.130, +1.078] pp**.
- Mean within-response Spearman: `0.110`, exploratory 95% CI `[0.008, 0.214]`.
- Signed score did not clearly beat `-abs(score)`: difference `0.111`, 97.5% CI `[-0.062, 0.286]`.
- Every individual position's conservative 95% interval crossed zero.

The defensible claim is group-level separation of extreme score bins, not reliable per-token classification.

### High-precision fixed-case evidence

A fixed 35-position response was evaluated with 4,096 paired seeds per position and fresh-state Adam interventions:

| Adam LR | Within-response Spearman |
|---:|---:|
| `1e-8` | 0.268 |
| `2.5e-8` | 0.340 |
| `5e-8` | 0.698 |
| `1e-7` | 0.749 |
| `2e-7` | 0.802 |
| `5e-7` | **0.833** |
| `1e-6` | 0.628 |

At `2e-7`, multiple positive and negative effects were individually resolved. At `5e-7`, rank correlation was strongest but updates were highly nonlinear. At `1e-6`, several positive-score positions became significantly harmful. A second prompt did not reproduce the strong association, and optimizer-consistent Adam scoring did not reliably fix the discrepancy.

The synthesis is:

1. harmful and beneficial token-level OPD interventions both exist;
2. score has aggregate predictive signal and can rank a carefully measured fixed case;
3. signal strength depends on prompt, response, probe, and update scale;
4. score is not an error-free sign classifier and lacks broad calibration.

## 7. Evaluation Protocols

Math:

- AIME24, AIME25, HMMT Feb 2025, HMMT Nov 2025;
- 30 questions each, 32 generations each unless labeled otherwise;
- temperature 1, top-p 1, response cap 16,384;
- arithmetic mean of four pass@1 values.

Code:

- EvalPlus HumanEval+ and MBPP+ **plus** scores, not `base`;
- LiveCodeBench release-v6;
- efficient protocol uses `n=4` with DP8;
- `Coding Avg = mean(HumanEval+, MBPP+, LCBv6)`.

Do not mix 16K results with the separate 32K response-length diagnostic or use the early incorrect LCB evaluation.

## 8. Important Files

Core implementation:

- `verl/verl/workers/actor/dp_actor.py`
- `verl/verl/trainer/ppo/ray_trainer.py`
- `verl/verl/utils/probe_gradient.py`
- `verl/verl/utils/fd_work_balance.py`
- `verl/verl/utils/crossfit_fallback.py`
- `verl/verl/utils/sgopd.py`
- `verl/verl/utils/checkpoint_pause.py`

Launch and analysis helpers:

- `scripts/run_singleteacher_train20.sh`
- `scripts/run_math_n32_balanced_fd_ablation.sh`
- `scripts/run_math_n32_followup.sh`
- `tools/eval_math_checkpoint_local.py`
- `tools/eval_code_checkpoint_local.py`
- `tools/run_clipped_sigmoid3s_n32_resume10_to20.py`
- `tools/run_sgopd_n8_step20_eval.py`

Compact evidence:

- `eval_outputs/math_n32_balanced_fd_ablation_v1/results.md`
- `eval_outputs/math_token_oracle_multiprompt32_base_lr2p5em6_v1/findings.md`
- `eval_outputs/math_token_oracle_base_two_prompts_adam_v1/findings.md`
- `eval_outputs/math_token_oracle_adam_larger_lr_fd_v1/findings.md`
- `eval_outputs/math_token_oracle_adam_lr5em7_fd_v1/findings.md`
- `eval_outputs/math_token_oracle_adam_lr1em6_fd_v1/findings.md`

JSON files referenced by those reports remain local and are deliberately not pushed.

## 9. Offline Artifacts

- Student: `Qwen/Qwen3-4B`, revision `1cfa9a7208912126459214e8b04321603b3df60c`
- Math teacher: `Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500`, revision `05d82d02780d4a6f8295b2909dbbd89e8a8b5aaa`
- Code teacher: `Keven16/Qwen3-4B-Non-Thinking-RL-Code-Step300`, revision `a3735900473592c2d33fc27706ff6fda30abda67`
- Dataset: `Keven16/G-OPD-Training-Data`

The old shared filesystem rooted these under `/volume/pt-dev/users/qshan`. Update `ROOT`, model paths, `HF_HOME`, benchmark roots, and Python paths in a new environment. Never assume a GPU worker can download a missing artifact.

## 10. Recommended Next Experiments

The highest-value next step is a matched fresh-seed comparison:

1. Freeze one prompt set and order for all arms.
2. Run at least three seeds of vanilla OPD, zero-fallback-only, and N32 sigmoid weighting.
3. Keep trajectories, N, response cap, optimizer, residual clipping choice, and evaluation seeds identical.
4. Report mean/variation plus fallback and nonzero-weight ratios.
5. Separately repeat the extreme-bin oracle comparison on new prompts without selecting prompts or LRs by observed token effects.

The unresolved causal question is whether gains come from signed score, suppressing large/noisy updates, or fallback/curriculum regularization. Existing evidence supports some signed information but has not excluded alternatives.

## 11. Starting on Another Machine

```bash
git clone https://github.com/doub7e/OPD-experiments.git
cd OPD-experiments
git checkout inf-token-reweight
```

Then read this file and `method_n32_raw_sigmoid.md`, verify all offline artifacts, update absolute paths, run focused CPU tests, and perform a one-step GPU smoke test. Inspect score distribution, fallback ratio, nonzero weights, cache restoration, and optimizer-step behavior before a full run.

```bash
pytest -q tests/test_sgopd.py tests/test_sgopd_teacher_rollouts.py
pytest -q tests/test_balanced_resume_validation.py tests/test_checkpoint_pause.py
python -m py_compile tools/eval_math_checkpoint_local.py tools/eval_code_checkpoint_local.py
git status --short
```

The local Experiment Board contains some stale historical records marked running. Treat bridge heartbeats and job logs as authoritative and audit the board before resuming an old task.
