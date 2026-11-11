#!/usr/bin/env bash
set -euo pipefail
root=/scratch/cvlab/home/shuli/data-rsi/007-openthoughts-probe-gradient
native=/scratch/cvlab/home/shuli/data-rsi/004-runai-native-eval
export PATH="$native/bin:$PATH"
export KUBECONFIG=/scratch/cvlab/home/shuli/.codex/secrets/native-runai-kubeconfig.json
export NATIVE_AUTH_GUARD=1
export PYTHONPATH="$root/scripts:$native/native:$native/scripts"
export UV_CACHE_DIR=/tmp/ota-probe-uv-cache HARBOR_DISABLE_TELEMETRY=1 OTEL_SDK_DISABLED=true
uv venv --python /usr/bin/python3.12 /tmp/ota-probe-venv
uv pip install --python /tmp/ota-probe-venv/bin/python -r "$native/requirements.lock.txt"
export PATH="/tmp/ota-probe-venv/bin:$PATH"
if [ "$1" = "--preflight" ]; then
  exec python -u "$root/scripts/check_harness.py"
fi
exec python -u "$root/scripts/rollouts.py" "$1"
