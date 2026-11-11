#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python3 "$root/upstream/code/bfcl/metric-calculation/calculate.py" \
  "$root/upstream/results/scores.csv" \
  --references "$root/upstream/results/baselines/baseline-aggregates-user-20260926.json" \
  --output "$root/runs/protocol-audit-20261102/released-metrics.json"
