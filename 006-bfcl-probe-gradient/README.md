# BFCL probe-gradient pilot

**Post-experiment finding (2026-11-03):** Unterminated `<think>` blocks leaked into later inference history. A full token audit and gradient decomposition confirmed a material effect on sampled cosine signs. Original scores are preserved, but the strict no-history-reasoning policy was not fully implemented. See the reasoning-context diagnostic report; a corrected-harness experiment has not yet been run.

Completed pilot (2026-11-03): no demonstrated selection gain in this fixed-pool/seed experiment.
On 768 held-out BFCL tasks × 3 repeats, random100 scored 36.50%, top100 35.94%, and bottom100 36.07%.
Top-minus-bottom was −0.13 percentage points (paired-task 95% CI −1.91 to +1.61).
All 200 cosine scores were negative. Equal trajectory counts had unequal expanded sample/token budgets.

- [Detailed results and limitations](runs/comparison/REPORT.md)
- [Comparison figure](runs/comparison/comparison.png)
- [Independent completion audit](runs/completion-audit/result.json)
- [Frozen ranking](data/ranking.json)

All compute jobs have finished; the experiment board retains training curves, trajectories and results.

See [PROTOCOL.md](PROTOCOL.md) for the frozen method. This experiment uses benchmark feedback and is not a blind DataSense submission.

The user service `data-rsi-probe-gradient.service` runs the lightweight host controller. All rollout, gradient, SFT, evaluation and bulk analysis jobs run on RunAI. The original random100 experiment005 continues independently.

Stages: fixed32 probe questions ×8 seeded rollouts → full-parameter reward-centered probe gradients → cosine scores for a nested random200 pool → top100 and bottom100 SFT → three full BFCL repeats per model → paired comparison on768 non-probe tasks against random100.

Run records and state are under `runs/`; final analysis will be under `runs/comparison/`. Do not interpret a registered run as completed. Probe samples with zero reward variance contribute zero gradient; low variance triggers the predeclared extension to16 repeats on the same questions.

Shared read-only inputs: experiment005's pinned model, validated pool, fixed recipe and immutable container images. All new results belong here.

Runtime compatibility repair: the pinned BFCL wheel lacks the local `Qwen/Qwen3-4B-FC` registration used by the upstream launcher. `005.../scripts/bfcl_registry.py` registers it by copying the existing local Qwen3-8B-FC config and changing only model identity. All three compared models use this same alias and the same official patched Qwen handler. The initial zero-result startup failure is retained under `runs/probe-rollouts/attempts/`.

## Probe194 scaling follow-up — 2026-11-03

`runs/probe194-scaling-20261103` generated128 fresh strict-v2 rollouts (50successes), with fixed outcome-blind halves and per-call seeds. Full-parameter split-half cosines at total8/16/32/64/128 rollouts are0.039204/0.016245/0.013944/0.042350/0.013312. Three fixed64-vs64 partitions at128 give0.013312, -0.014341, 0.033127: no strong directional agreement was observed at this scale. All1726 actual prompts passed history/token checks; four unclosed-think outputs had50clean subsequent calls. This does not replace the original32-probe study or establish downstream selection utility. See the run REPORT.md, figures and verification.json; all128 trajectories are registered in the board.

## Two-probe gradient diagnosis and ablations — 2026-11-03

Real-trajectory replay cosines are~0.99996; HF/vLLM logprobs have nonzero numerical differences. For128rollouts per probe, [all, no reasoning, key turn all, key turn visible] split-half cosines are [0.01331,0.09829,0.08587,0.38522] on miss_func194 and [-0.00746,0.01819,0.02304,0.00461] on base22. The first-probe global improvement did not replicate on the second. An exploratory shared local feature is block32 under key-turn-all (~0.411/~0.427); it requires independent validation. See `runs/probe22-ablation-20261103/ANALYSIS.md`, comparison figures and layer CSV. No new SFT or candidate ranking was performed.
