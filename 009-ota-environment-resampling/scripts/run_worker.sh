#!/usr/bin/env bash
set -euo pipefail
root=/scratch/cvlab/home/shuli/data-rsi/009-ota-environment-resampling
native=/scratch/cvlab/home/shuli/data-rsi/004-runai-native-eval
export PATH="$native/bin:$PATH"
export KUBECONFIG=/scratch/cvlab/home/shuli/.codex/secrets/native-runai-kubeconfig.json
export NATIVE_AUTH_GUARD=1
export PYTHONPATH="$root/scripts:$native/native:$native/scripts"
export UV_CACHE_DIR=/tmp/ota-resample-uv HARBOR_DISABLE_TELEMETRY=1 OTEL_SDK_DISABLED=true
if [ -x /opt/harbor/bin/python ]; then
  export PATH="/opt/harbor/bin:$PATH"
else
  uv venv --python /usr/bin/python3.12 /tmp/ota-resample-venv
  uv pip install --python /tmp/ota-resample-venv/bin/python -r "$native/requirements.lock.txt"
  export PATH="/tmp/ota-resample-venv/bin:$PATH"
fi
if [ "$1" = "--worker-check" ]; then
  exec python -u "$root/scripts/check_worker.py"
fi
if [ "$1" = "--preflight" ]; then
  exec python -u "$root/scripts/harness_smoke.py"
fi
exec python -u "$root/scripts/rollouts.py" "$1"
