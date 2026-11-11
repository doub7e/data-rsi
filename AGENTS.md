# Data Self-Improvement workspace

- **Default user requirement:** Unless the user explicitly opts out, every experiment must be clearly recorded in the experiment board. Apply `/home/shuli/.codex/skills/experiment-board-recording/SKILL.md` without asking whether to register. Registration, live state/progress, relevant results/charts and applicable trajectories must be verified through the actual board API; writing files alone is not sufficient. This includes small computational diagnostics and failed experiments, but not ordinary conceptual Q&A or software unit tests.

- Each immediate experiment directory is a runnable experiment node. Keep reusable dashboard code in the sibling `experiment-board/` project.
- Register every experiment with `experiment.json`; register every run before launching it in `runs/<run-id>/run.json`.
- Keep ALL experiment configurations, logs, status snapshots, metrics, checkpoints, evaluation artifacts and comparisons inside that experiment directory. Never put experiment results in the dashboard code directory.
- The dashboard scans these manifests. Use `sources.metrics` for LF trainer_log.jsonl (or generic step-based JSONL), `sources.status` for process state JSON, and `job.name/namespace` for RunAI status.
- For trajectory viewing, record `sources.training_data` for prepared JSONL conversations and `sources.trajectories` for the evaluation jobs directory, relative to the experiment directory. The board also recognizes `runs/<id>/data/train.jsonl` and `runs/<id>/jobs` by convention. Keep the raw data and trajectories in the experiment, not the board project.
- For evaluation progress, declare `sources.evaluation_plan` pointing to a run-owned JSON file with `jobs: [{id, label, total_trials}]`. IDs must match job directories beneath `sources.trajectories`. Include all planned repeats; omit smoke/diagnostic jobs so progress has a stable denominator.
- State writes must be atomic. Record exit status, attempts/restarts, actual sample count, immutable model/data/image revisions, exact hyperparameters and evaluation protocol. Do not silently drop rows or replace failed evaluation trials with successes.
- GPU execution uses RunAI H200. Preserve queued jobs when the pool is full; do not cancel just for resource shortage or silently use another GPU class.
- For long training, save resumable checkpoints and preserve per-attempt logs. After training, perform the requested evaluation and update the same experiment's records and comparison. Do not mark evaluation complete before actual results exist.
- The user prioritizes a stable, small, generic dashboard over project-specific features.

- This machine is a jump host. Do not run bulk CPU experiments, benchmark sandboxes, or image builds here. Keep only lightweight UI/control/status polling here; run CPU evaluation on allocated RunAI workers (or an explicitly approved external sandbox provider).

- Host automation and monitoring must use `KUBECONFIG=/mnt/cvlab/scratch/cvlab/home/shuli/.codex/secrets/native-runai-kubeconfig.json` (short-lived token only). Only `data-rsi-native-auth.service` refreshes from `/home/shuli/.kube/config`; interactive `runai login` remains on the original config. Do not introduce another refresh-capable background consumer.
