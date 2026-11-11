#!/usr/bin/env bash
set -euo pipefail
root=/scratch/cvlab/home/shuli/data-rsi/002-openthoughts-qwen3-8b
cd "$root/eval"
export UV_CACHE_DIR=/tmp/ota-uv-cache
export HARBOR_DISABLE_TELEMETRY=1 OTEL_SDK_DISABLED=true
export DATA_RSI_EVAL_EXECUTION_LOCATION=runai-cpu
export PYTHONPATH="$root/eval"
uv venv --python /usr/bin/python3.12 /tmp/ota-daytona-venv
uv pip install --python /tmp/ota-daytona-venv/bin/python -r requirements-daytona.txt
export PATH="/tmp/ota-daytona-venv/bin:$PATH"
exec python -u daytona_worker.py "${1:-evaluate}"
