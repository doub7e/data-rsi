# OpenThoughts-Agent probe reward gradients

- Branched experiment: user explicitly selected original Qwen3-8B, not1K/10K checkpoints. Do not modify other branches' BFCL or attribution jobs/configurations.
- Match the actual OpenThoughts-Agent SFT pipeline: pinned LF fork, qwen3_nothink,32K, train_on_prompt=false, mask_history=false, packing=false, preserved assistant reasoning. Inspect actual trainer loss reduction before freezing the reward-gradient estimator.
- All results belong in this experiment and must be registered/verified in the experiment board.
- Use existing native RunAI sandboxes; no Daytona and no batch CPU evaluation on the jump host. Reuse immutable task images, never expose shared storage or user credentials to sandbox pods.
- Agent inference uses4×H200 as previously requested for this experiment family; separate gradient workers use measured-fit H200 resources. Preserve queued jobs.
- Record exact rollout-to-training-row adaptation, summarization segments, truncation and sample weighting. Do not silently drop long/failed trajectories or remove reasoning. Export only the final full conversation of each context segment, not every cumulative prefix. Copied context inherited by these SFT rows must be counted and disclosed; do not change mask_history to remove its supervision.
- Reward-weighted training loss gradients and exact on-policy score gradients are different estimators; do not label them equivalent if template/context/normalization differ.
