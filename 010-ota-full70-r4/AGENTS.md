# Full70 ×4 environment resampling

- Complete the user-authorized first-epoch280rollouts → reward probe gradients →728 candidate dots →728 newly regenerated teacher trajectories →1epoch continuation →280guided evals, plus Uniform supplement210 to its original70.
- New experiment; do not mutate any009 raw outputs/checkpoints or reuse its old guided model/results as this run. Record all stages in the board using experiment-board-recording.
- Latest user explicitly requires missing verifier rewards to score0. Keep raw_score/null, error and imputed flag; do not claim imputed0 is a verifier output.
- All70 tasks are feedback-used. No heldout claims. Constant-score questions have zero centered signal and are excluded without adaptive replacement.
- Same BF16 model forward, actual LLaMA-Factory encoder/masks/Liger loss and FP32 gradient accumulation as training. Raw full-parameter dot, no projection.
- Reuse exact009 first-epoch and Uniform checkpoints read-only. Only reused first70 Uniform outcomes are imported, with immutable source provenance; new repeats use distinct seeds. First-epoch probe seeds differ from post-training eval seeds; Uniform/newGuided share task-repeat seeds.
- Keep4H200 inference and4H200 training, up to4 single-H200 probe shards or2 candidate workers, max8 requested GPUs total. Keep queued when busy.32 native sandbox concurrency, one inference server at a time.
- Shared token-only kubeconfig; only existing auth publisher refreshes. Heavy CPU/GPU work on RunAI. Never touch sibling workloads.
