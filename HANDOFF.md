# Collaborator handoff

Updated **2026-11-11**. This is the repository-level entry point for taking over the project. It is a dated snapshot, not a live job monitor.

## Start here

| Question | Read first |
| --- | --- |
| What is this project? | This document, then [README.md](README.md) and [AGENTS.md](AGENTS.md) |
| What is the latest completed agent experiment? | [Cap3 comparison](011-ota-cap3-resampling/runs/comparison/REPORT.md) |
| How were the probe and training targets defined? | [Full70 protocol](010-ota-full70-r4/configs/protocol.json), [loss implementation](010-ota-full70-r4/scripts/loss_contract.py) |
| Why did uncapped Guided perform poorly? | [Training-distribution audit](010-ota-full70-r4/runs/guided-data-distribution-20261110/REPORT.md), [task_9288 audit](010-ota-full70-r4/runs/task9288-trajectory-audit-20261111/REPORT.md) |
| What work may still be running? | The separate [math coordinator](qwen_math_fp32_richardson20_priority_20261109/COORDINATOR.md); verify live state before taking action |

**Current takeaway:** the completed OTA experiments have not established an advantage over Uniform continuation. A cap of three trajectories per environment improved the Guided point estimate, but its confidence interval versus uncapped Guided still includes zero. The mathematical-gradient work is a separate branch of research and must not be treated as finished just because OTA is finished.

## Research question and workstreams

The broad question is whether reward-gradient information from benchmark rollouts can identify better training examples or environments. The workspace includes preliminary reproductions, gradient diagnostics, sample selection, environment resampling, and a separate mathematical reasoning/finite-difference program.

Each immediate experiment directory is an experiment node. Each execution or analysis has a `runs/<run-id>/run.json` manifest. Directory numbers are historical identifiers, not pipeline stages that should all be restarted in order.

| Node | Purpose and handoff status |
| --- | --- |
| `001-facet-reproduction` | FACET resource/environment preflight. Not a completed FACET training reproduction. |
| `002-openthoughts-qwen3-8b` | Original 10K SFT reproduction. Training completed: 7 epochs, 735 steps. Its older README contains stale startup text; consult run results for actual outcomes. |
| `003-openthoughts-qwen3-8b-1k` | Original 1K SFT reproduction. Training completed: 7 epochs, 77 steps. Its evaluation history is separate from the later TB-dev study. |
| `004-runai-native-eval` | Native RunAI sandbox adapter and infrastructure. Reused by the OTA experiments. |
| `005-datasense-bfcl` | Qwen3-4B / EnvScaler / BFCL baseline and harness setup. |
| `006-bfcl-probe-gradient` | BFCL selection, gradient reliability, continuation, dot-product and projected-influence experiments. Read the particular run report; the top-level README does not summarize every later ablation. |
| `007-openthoughts-probe-gradient` | Completed base-Qwen3-8B task-level probe diagnostics, including fix-git. |
| `008-openthoughts-easier-probes` | Completed score-based easier-task probe diagnostics. |
| `009-ota-environment-resampling` | Preliminary generated-data experiment; produced the first-epoch and Uniform checkpoints reused later. Its probe/evaluation protocol differs from 010/011. |
| `010-ota-full70-r4` | Completed full70 x4 evaluation, probe-gradient scoring and uncapped Guided resampling. |
| `011-ota-cap3-resampling` | Completed capped-probability / maximum-three-occurrences continuation and evaluation. Latest OTA comparison. |
| `qwen_math_fd_reliability_20261108` | Completed numerical-reliability diagnostics on accessible checkpoints. Not a reproduction of the unavailable historical best checkpoint. |
| `qwen_math_fd_training_ablation_20261108` | Earlier short-training pilot; intentionally held in favor of the priority experiment. Do not restart its old controller. |
| `qwen_math_fp32_richardson20_priority_20261109` | Separate priority FP32 Richardson training program. At the 2026-11-11 12:59 CET check, retry pod `opd-p20-b300r2-1111a-0-0` was Running; the latest saved progress was 2/20, timestamped 12:26 CET. No final result was established by this handoff. |

The math program's code also depends on the separate `OPD-experiments-inf-token-reweight` worktree. Its coordinator owns its resource/queue decisions. Do not cancel its jobs, reset its training-winner claim, or revive a superseded queue from an older note.

## The latest OTA experiment, precisely

### Data and model lineage

This is **not** the original 1K/10K reproduction in nodes 002/003.

- Student: `Qwen/Qwen3-8B`.
- Fixed teacher: `Qwen/Qwen3.8-27B`.
- Environment source: `open-thoughts/OpenThoughts-Agent-v1-RL`; the prepared pool used here contains 728 environments.
- Evaluation/feedback source: `open-thoughts/OpenThoughts-TB-dev`, 70 tasks.
- First epoch: one teacher trajectory per environment, then full-parameter SFT for one epoch.
- Uniform continuation: one fresh teacher trajectory per environment, then one more SFT epoch from that first-epoch checkpoint.
- Guided continuation: use benchmark feedback to score the first batch's trajectories, resample their environments, generate 728 fresh teacher trajectories, then train one epoch from the **same first-epoch checkpoint**.
- Cap3 continuation: reuse the same probe and dot scores, change the sampling rule, generate another fresh batch of 728 trajectories, and again start from the same first-epoch checkpoint. It does not continue from the uncapped Guided model.

Both continuation branches reset the optimizer and scheduler. Teacher failures are retained as SFT demonstrations; there is no success-only filter. The three later training stages each used 728 SFT rows and 23 optimizer steps.

The original 009 experiment used a smaller probe subset and single-repeat final evaluation. Do not mix its metrics or missing-score policy with the full70 x4 results below.

### Probe gradient and loss contract

010 evaluated the first-epoch checkpoint on all 70 tasks, four rollouts per task. Fifty tasks scored zero on all four attempts and had no centered reward signal. The remaining **20 tasks / 80 rollouts** contributed to the probe. Rewards include partial verifier scores, not just binary success.

The implemented surrogate is:

```text
rbar_q = mean score over the four rollouts of question q
T_q    = total retained supervised target tokens across those rollouts

g_q     = -sum_i[(r_i - rbar_q) * grad CE_sum(C_i)] / T_q
g_probe = mean_q(g_q), over the 20 nonconstant questions

d_j     = -grad CE_sum(first_teacher_trajectory_j) / T_j
score_j = dot(g_probe, d_j)
```

`C_i` can contain multiple exported conversations because of continuation/summarization. The 80 probe rollouts yielded 196 samples and approximately 2.253 million retained supervised tokens. One rollout had no valid supervised tokens; its gradient contribution was zero, while its score still participated in the question mean.

The implementation uses the training LLaMA-Factory encoder, masks and Liger/Trainer loss. Important settings are `mask_history=false`, `train_on_prompt=false`, `packing=false`, cutoff 32768, and retained assistant reasoning. The name `qwen3_nothink` does **not** mean existing reasoning text is removed from the labels. The reduction is based on supervised token sums/counts, not an unweighted mean of individual sample losses.

Gradients use BF16 model forward/backward and FP32 accumulation over the full trainable parameter space. No projection or preconditioning was used in 010/011. This is a reward-weighted **training-loss surrogate**, not a claim of an exact on-policy policy gradient: training-style replay, history and truncation can differ from rollout inference context. There was no new split-half stability measurement for the four-rollout-per-question probe in 010.

### Sampling rules

For 010, with 728 candidate scores:

```text
z_i = clip((score_i - mean(score)) / population_std(score), -5, 5)
p_i = 0.1 / 728 + 0.9 * softmax(z)_i
```

Temperature is one. The original plan drew 728 times with replacement and no count cap. It covered 328 environments; one environment appeared 72 times.

For 011:

```text
q_i = min(3 / 728, lambda * p_i), with sum_i(q_i) = 1
```

The excess probability mass is redistributed by water filling. Randomly permuted systematic dependent rounding of `728*q` then produces integer counts whose sum is exactly 728 and whose maximum is three. **This is not independent multinomial sampling.** A probability ceiling alone would not guarantee the requested count ceiling.

The frozen cap3 plan covered 526 environments: 384 occurred once, 82 twice, and 60 three times. `task_9288` occurred three times. See [sampler implementation](011-ota-cap3-resampling/scripts/select_cap3.py) and [frozen sampling result](011-ota-cap3-resampling/runs/sampling/result.json).

### Training and evaluation settings

The OTA continuation recipe is one epoch, full SFT, BF16 / DeepSpeed ZeRO-3, global batch 32 (4 GPUs x microbatch 1 x accumulation 8), learning rate 4e-5, cosine schedule, warmup 0.1, weight decay 0.04, gradient norm cap 1, and training seed 42. Exact revisions, image digests, optimizer settings and limits are in the [011 protocol](011-ota-cap3-resampling/configs/protocol.json).

Rollouts use the pinned Harbor / Terminus-2 JSON harness with native RunAI CPU sandboxes and 4xH200 model inference. The configured context is 32K with summarization. Task-specific resolved timeouts still apply; the global 1800-second cap should not be interpreted as every task receiving an identical time budget.

All four reported models have 70 tasks x 4 attempts. Uniform, uncapped Guided and cap3 use matching task-repeat evaluation seeds. The first-epoch probe seeds differ from the post-training seeds. The prior three models' completed results are reused in 011 rather than rerun.

Missing terminal verifier rewards count as zero under the user's explicit protocol. Preserve `raw_score=null`, the exception, and the imputation flag. Do not describe an imputed zero as a score returned by the verifier. Do not silently discard a trial or reroll model failures to obtain successes.

## Results and what they do not establish

| Condition | Mean verifier score | Full passes | Full-pass rate | Missing verifier scores counted as zero |
| --- | ---: | ---: | ---: | ---: |
| First epoch | 0.1021 | 23/280 | 8.21% | 2 |
| Uniform continuation | 0.1468 | 36/280 | 12.86% | 4 |
| Uncapped Guided | 0.0931 | 18/280 | 6.43% | 2 |
| Cap3 Guided | 0.1374 | 31/280 | 11.07% | 5 |

| Mean-score comparison | Difference | Paired task-bootstrap 95% interval |
| --- | ---: | --- |
| Uncapped Guided minus Uniform | -0.0537 | [-0.0968, -0.0170] |
| Cap3 minus uncapped Guided | +0.0443 | [-0.0023, +0.0950] |
| Cap3 minus Uniform | -0.0094 | [-0.0634, +0.0438] |

Bootstrap units are the 70 questions after averaging each question's four attempts, not 280 independent questions. Only one training seed was used. All 70 benchmark questions supplied selection feedback: these are **feedback-used results, not independent held-out generalization**. Intervals containing zero do not establish equivalence or a reliable improvement.

Cap3 changes coverage and the joint sampling distribution, and it uses newly generated teacher trajectories. It is not an isolated ablation of `task_9288` or a pure deletion of duplicated samples.

### The task_9288 finding

The first trajectory selected for this environment was already a verifier failure, but its dot score ranked second of 728. It interpreted “take the first five matches” as “create enough files to have five matches,” although the existing fixtures contained two matching files and the original verifier expected those two paths.

Among the 72 uncapped Guided trajectories, one passed, 70 were explicitly rejected, and one lacked a verifier score after a download error. All 70 confirmed failures produced five lines instead of the expected two. These failures and their completion claims were retained as positive SFT targets. The environment occupied 9.89% of the continuation examples.

This establishes repeated incorrect demonstrations, **not causal proof of student overfitting**. There is no isolated remove/downweight-task9288 training experiment or direct pre/post student test on that environment. Four manually inspected degraded evaluation cases showed different errors (log aggregation, SQL top-3, graph-query traversal and CSV formatting). The [audit report](010-ota-full70-r4/runs/task9288-trajectory-audit-20261111/REPORT.md) separates evidence from hypotheses.

## Code, Git records and shared-storage artifacts

The public repository tracks code, frozen configurations, documentation, run manifests and selected lightweight results. The original workspace is `/mnt/cvlab/scratch/cvlab/home/shuli/data-rsi`; RunAI sees the corresponding path under `/scratch/cvlab/home/shuli/data-rsi`.

A clone alone is **not** a runnable copy of the original storage. These remain outside Git:

- Model weights, optimizer state, checkpoints and full gradient tensors. No Git LFS is used for them.
- Raw datasets, prepared benchmark environments, seed files and full rollout traces.
- Downloaded runtimes, build contexts, CLI binaries and external source snapshots.
- Credentials and live process/control state.
- The sibling `../experiment-board` application, which was explicitly excluded from publication.

Some historical Markdown links point to unversioned files and only work on the original shared storage. A `run.json` source path is a pointer, not proof that the corresponding artifact was included in the Git commit.

Important original-server artifact locations (untracked):

| Artifact | Location relative to `data-rsi/` |
| --- | --- |
| Fixed teacher | `009-ota-environment-resampling/assets/teacher/` |
| Shared first-epoch checkpoint | `009-ota-environment-resampling/runs/sft-epoch1/checkpoint/` |
| Uniform final checkpoint | `009-ota-environment-resampling/runs/sft-uniform/checkpoint/` |
| Uncapped Guided final checkpoint | `010-ota-full70-r4/runs/sft-guided/checkpoint/` |
| Cap3 final checkpoint | `011-ota-cap3-resampling/runs/sft-guided/checkpoint/` |
| Prepared RL and dev tasks | `009-ota-environment-resampling/tasks/native/{rl,dev}/` |
| Generated data and raw trials | Each experiment's `runs/teacher-guided/` and `runs/eval-guided/` |

On 2026-11-09 the user authorized removal of four merged 010 gradient shard files and seven old optimizer/resume checkpoint directories in 002/003/009, freeing approximately 869.73 GiB. Final model weights and the merged probe gradient were retained. **Do not assume old optimizer-state checkpoints still exist.** The incomplete 010 checkpoint from the quota failure was preserved under its training run's `recovery/` directory.

## Operations and experiment-board workflow

### Before taking over a job

1. Read the local `AGENTS.md`, frozen protocol, latest run results and coordinator notes.
2. Inspect the actual RunAI job and pod plus recent output. A manifest's static `status: planned`, an old README, or a cached dashboard status is not authoritative live state.
3. Identify the controller already owning the job before starting another one. Retain queued jobs; do not cancel/recreate them simply because GPUs are busy.
4. Restore required datasets/models/runtime and obtain authorized registry/cluster access before launching from a fresh clone. Follow the pinned revisions rather than replacing images with `latest`.

The original machine is a jump host. Keep GPU workloads, bulk CPU evaluation and image builds on RunAI. OTA serving/training uses 4xH200 and up to32 CPU sandboxes; math has separate explicit resource decisions. A teacher/inference server and a task sandbox are different jobs: the model produces commands while the sandbox executes them and runs the verifier.

Original-server read-only checks:

```bash
export KUBECONFIG=/mnt/cvlab/scratch/cvlab/home/shuli/.codex/secrets/native-runai-kubeconfig.json
kubectl get runaijobs -n runai-cvlab-shuli
kubectl get pods -n runai-cvlab-shuli
systemctl --user status data-rsi-native-auth.service
```

Only the existing native-auth service refreshes from the original login configuration. Other processes consume the short-lived token-only configuration. Do not add competing refresh consumers, copy credentials into the repository, or share the credential file with another account. If interactive login is needed, the account owner runs `runai login`; no token should be pasted into a discussion.

### Board access and registration

The original board listens on `127.0.0.1:18765` and is managed by `experiment-board.service`. Access it through SSH forwarding:

```bash
ssh -N -L 18765:127.0.0.1:18765 <user>@<jump-host>
```

Open `http://127.0.0.1:18765` locally. The board code is a separate, unpublished sibling project. It reads experiment files; results must not be copied into the board application.

Unless explicitly exempted, register every experiment before execution and keep it in the board. Use a new experiment/run ID for a new method rather than rewriting an old result. Sources are relative to the experiment directory:

| Field | Purpose |
| --- | --- |
| `sources.status` | Atomically updated process state |
| `sources.metrics` | Training curves / step records |
| `sources.results` | Final metrics and structured results |
| `sources.progress` | Current phase progress |
| `sources.training_data` | Actual selected training conversations |
| `sources.trajectories` | Raw evaluation job/trial tree |
| `sources.evaluation_plan` | Frozen trial IDs and complete denominator |

Check the actual API and at least one real training/evaluation sample before claiming recording is complete. On the original server, the `experiment-board-recording` skill and its `verify_board_run.py` helper define the workflow. If those local skills are unavailable on a collaborator's machine, follow `AGENTS.md` and validate the equivalent API views.

### Completed OTA controllers

010 and 011 are finished; their jobs have been released. Their controller scripts and historical service names remain for provenance:

- `data-rsi-ota-r4-1108.service`: 010.
- `data-rsi-ota-c3-1111.service`: 011.

**Do not restart these completed controllers to start a new experiment.** Create a new node and unique job/service names. Read the current math coordinator separately before touching any `opd-*` workload.

## Known failure modes and recovery evidence

| Failure mode | What to preserve or check |
| --- | --- |
| Inference pod replacement | Pause clients using the stale endpoint, preserve completed results, then reconnect to the replacement pod. An unchanged job name does not imply an unchanged pod UID/IP. |
| Storage quota exhausted | `df` free space is insufficient evidence of writable quota. Preserve the failed attempt; verify checkpoint completeness before resume. 010 had to restart its continuation from the first-epoch model because checkpoint10 was incomplete. |
| Missing verifier result | Apply the experiment's frozen policy and retain raw exceptions. For 010/011 the rule is missing-as-zero; do not retroactively apply it to older protocols. |
| API reply exists but no assistant step was committed | A terminal sandbox failure can leave a raw response and a user-only trajectory. Preserve the response, export zero SFT rows, and do not fabricate an assistant turn or reroll the outcome. |
| Historical BFCL reasoning leakage | An older handler retained unterminated reasoning in later prompts. Later strict-v2 diagnostics repaired this; do not pool the original pilot with corrected-harness results without accounting for the change. |
| Apparent success from a recreated process | Check real checkpoint/step/result artifacts, not exit code alone. The math priority coordinator documents a duplicate-claim case that exited0 without completing training. |

011 initially stopped at279/280 because `e0167` triggered the empty-export edge case. A CPU-only metadata repair accepted its existing missing score as0, preserved its original raw result and all279 prior record hashes, and performed **zero new rollouts**. See [repair record](011-ota-cap3-resampling/runs/eval-finalization-repair/result.json).

## Suggested next investigations — not queued work

No further OTA training is implied or authorized by this handoff. Useful next questions are:

- Does selectively limiting/replacing task9288, while preserving the other trajectories, change performance? Current cap3 changes many environments at once.
- Is the average probe stable with more rollouts or independent fixed splits? Four attempts per question do not establish directional reliability.
- Does a prespecified quality filter or a different influence approximation help under matched data/token budgets?
- Do conclusions replicate over training seeds and on an independent holdout that supplies no selection feedback?

Before implementing any of these, freeze the comparison and budget, register it in the board, and keep the existing Uniform/Guided/cap3 records unchanged.

## Maintaining this repository

The remote is `doub7e/data-rsi`, branch `main`, and is public. The initial import is commit `821637a`.

Enable the local commit guard after cloning:

```bash
git config core.hooksPath .githooks
python3 -m unittest discover -s tools -p 'test_check_staged.py'
python3 tools/check_staged.py
```

Never add checkpoint files, optimizer/gradient arrays, credentials or raw large artifacts, including through Git LFS. The guard checks staged files; it does not make untracked artifacts portable. Stage only your intended changes, particularly while another experiment is updating its own files.

When handing off again, update this document with a dated status, actual job ownership, evidence links, methodological changes, and remaining uncertainties. Preserve the historical failures and keep hypotheses separate from measured results.
