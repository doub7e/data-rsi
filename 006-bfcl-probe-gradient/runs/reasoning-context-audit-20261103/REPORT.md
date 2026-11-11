# Probe history-reasoning diagnosis

The recorded probe prompts and gradient replay are consistent. Full CPU re-tokenization/decode of all2535 captured prompts found zero mismatches. Probe gradients directly replay those prompt IDs plus the current completion, with zero prompt tokens directly supervised.

A separate, real harness defect was found: truncated outputs with an opening `<think>` but no closing `</think>` are stored as ordinary assistant content. Clearing `reasoning_content` does not remove this content from later prompts.

-22 originating outputs were all cut at4096 tokens (`finish_reason=length`).
-They affected148 later model calls across22 rollouts and7 questions.
-21 affected calls, across5 rollouts/3 questions, had nonzero advantage and contributed to the probe gradient. Their history tokens are loss-masked, but still alter the current conditional distribution and its gradients.
-The original formatter exactly reproduced all148 recorded affected prompts. An isolated content-stripping patch removes the unclosed thoughts while preserving non-assistant content in all148 cases; nominal and disabled-flag fixtures also passed. The original experimental harness has not been overwritten.

## Gradient decomposition

Let G be the saved aggregate probe gradient and L the recomputed contribution from those21 calls, using the original advantages and normalization. The diagnostic compares G with G-L; it does not clean old prompts and claim a new on-policy estimator.

Norm(G)=595.620207; norm(L)=109.752130; norm(L)/norm(G)=18.43%. Cosine(G,G-L)=0.982886.

The12 candidates were fixed by original ranking positions1,2,10,25,50,75,100,101,125,150,175,200 before the diagnostic.11/12 turned positive. Mean cosine shift was+0.032718; maximum original-score recomputation difference was4.06e-5.

| Original rank | Saved cosine | Cosine after omitting leaking-call gradients |
|---|---:|---:|
| 1 | -0.009730 | 0.019862 |
| 2 | -0.012044 | 0.021710 |
| 10 | -0.015327 | 0.018642 |
| 25 | -0.017801 | 0.016556 |
| 50 | -0.020343 | 0.008872 |
| 75 | -0.021729 | 0.013176 |
| 100 | -0.023269 | 0.009030 |
| 101 | -0.023372 | 0.010223 |
| 125 | -0.024427 | 0.008698 |
| 150 | -0.025796 | 0.008934 |
| 175 | -0.028309 | 0.002890 |
| 200 | -0.038522 | -0.006616 |

## Interpretation and next controlled experiment

This identifies a substantial negative-alignment contribution, but does not prove the behavior of all200 candidates after omission or that fixing the harness improves downstream scores. Omitting selected call gradients is diagnostic only. A corrected study must fix history handling during rollout, regenerate trajectories/rewards under that context policy, recompute the probe/ranking, and evaluate under the same corrected handler. Cleaning only gradient replay would create the inference/gradient context mismatch that was originally suspected.

The original reported scores remain the outcomes of the original handler. They do not test the intended strict no-history-reasoning policy. The earlier categorical statement that every historical reasoning block was removed was incorrect for unterminated blocks. The previous completion audit replayed scores but did not cover this parser edge case.

See `../reasoning-token-replay-20261103/result.json` for full token audit, `../history-leak-gradient-audit-20261103/result.json` for decomposition, and `proposed-fix/history-think.patch` for the isolated proposed fix. No corrected-policy rollout or retraining has been performed.

## Integration follow-up (2026-11-03)

The shared BFCL launcher now installs `strict-v2` via `005-datasense-bfcl/scripts/history_protocol.py`; legacy-v1 can be selected explicitly. This is a runtime overlay on the pinned immutable image. The fresh `probe194-scaling-20261103` job verified the effective formatter hash and all148 affected prompt roundtrips inside the actual evaluation container before inference. Archived original rollouts, ranking and evaluations are unchanged; they have not been reclassified as corrected-policy results.
