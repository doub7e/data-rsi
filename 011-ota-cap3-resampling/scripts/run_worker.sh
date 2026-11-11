#!/usr/bin/env bash
set -euo pipefail
root=/scratch/cvlab/home/shuli/data-rsi/011-ota-cap3-resampling
native=/scratch/cvlab/home/shuli/data-rsi/004-runai-native-eval
export PATH="/opt/harbor/bin:$native/bin:$PATH"
export KUBECONFIG=/scratch/cvlab/home/shuli/.codex/secrets/native-runai-kubeconfig.json
export NATIVE_AUTH_GUARD=1
export PYTHONPATH="$root/scripts:$native/native:$native/scripts"
export HARBOR_DISABLE_TELEMETRY=1 OTEL_SDK_DISABLED=true
exec /opt/harbor/bin/python -u "$root/scripts/rollouts.py" "$1"
