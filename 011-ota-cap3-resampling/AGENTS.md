# OTA cap3 resampling

- User authorizes max3 training trajectories per environment, fresh728teacher rollouts, continuation1epoch from exact009 first-epoch model, and complete70x4evaluation.
- Preserve009/010 models, scores, trajectories; read-only baseline reuse. No rerunning probe or gradients.
- Same frozen scores, teacher, harness, loss, training hyperparameters and evaluation seeds as010. Change sampling cap only; teacher generations necessarily have new seeds. Retain failure trajectories, raw verifier scores and missing-as-zero policy.
- Register every stage in board. Results stay here; do not edit prior experiments.
- Native RunAI sandboxes32;4H200 per GPU job, maximum4requested GPUs here, sequential serving/training. Retain queued workloads. Shared token-only kubeconfig; only existing auth publisher refreshes.
- Never run GPU or bulk CPU experiments on jump host. Small control/metadata analyses allowed. Do not touch sibling jobs.
