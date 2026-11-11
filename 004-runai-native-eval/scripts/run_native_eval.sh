#!/usr/bin/env bash
set -euo pipefail
root=/scratch/cvlab/home/shuli/data-rsi/004-runai-native-eval
cd "$root"
export PATH="$root/bin:$PATH"
export KUBECONFIG=/scratch/cvlab/home/shuli/.codex/secrets/native-runai-kubeconfig.json
export NATIVE_AUTH_GUARD=1
export PYTHONPATH="$root/native:$root/scripts"
export UV_CACHE_DIR=/tmp/native-uv-cache HARBOR_DISABLE_TELEMETRY=1 OTEL_SDK_DISABLED=true
uv venv --python /usr/bin/python3.12 /tmp/native-eval-venv
uv pip install --python /tmp/native-eval-venv/bin/python -r requirements.lock.txt
export PATH="/tmp/native-eval-venv/bin:$PATH"
exec python -u scripts/native_eval_worker.py "$1"
