# DataSense-Bench BFCL Tool-Use reproduction

Current execution: user-authorized random 100 EnvScaler trajectories (seed 42),
Qwen3-4B full SFT on 2 H200, followed by BFCL V3 800 cases × 3 repeats on 1 H200.
The fixed validation 60 trajectories are excluded. No selection-agent API is used.

The lightweight host controller is `scripts/random100_controller.py`, managed by
`data-rsi-datasense-random100.service`. Preparation, builds, training and evaluation
run on RunAI. Logs and status live in `runs/random100-s42-20261102/` and
`runs/eval-random100-s42-20261102/`. This 100-trajectory extension does not modify
the released five-group, 50-trajectory assignments. Execution state must be read
from those run records; the existence of this README is not evidence of completion.

Read [PLAN.md](PLAN.md) for the full protocol, stages, resources and budget. `upstream/` is pinned to official commit `ca38f2606b3c9c4111d0516381f044797a81436c`.

Recompute the released reference metrics without GPU or extra Python packages:

```bash
bash scripts/recompute_reference.sh
```

All generated artifacts belong under this experiment's `runs/`. The experiment board discovers `experiment.json` and run manifests. Future selector sessions must not have access to this full checkout: it includes evaluation tasks/protocol and released assignments/scores.

## Versioned history-reasoning fix (2026-11-03)

The shared `scripts/bfcl_registry.py` entrypoint now defaults to `BFCL_HISTORY_PROTOCOL=strict-v2`. When `BFCL_DROP_HISTORY_THINK=1`, it removes unterminated reasoning blocks from assistant content as well as parsed reasoning fields. The immutable image is unchanged; this is a hash-checked runtime formatter overlay with regression fixtures. Set `BFCL_HISTORY_PROTOCOL=legacy-v1` explicitly to reproduce the earlier formatter. Existing saved rollouts and reported results remain unchanged. The fresh probe194 scaling run in experiment006 verifies148 previously affected prompt roundtrips and checks every newly generated prompt/token sequence.
