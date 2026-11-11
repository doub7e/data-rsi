# Source this only for the CPU evaluation controller. GPU serving runs in RunAI.
export PATH=/mnt/cvlab/scratch/cvlab/home/shuli/data-rsi/002-openthoughts-qwen3-8b/eval/bin:/mnt/cvlab/scratch/cvlab/home/shuli/data-rsi/002-openthoughts-qwen3-8b/eval/.venv/bin:$PATH
export DOCKER_HOST=unix:///run/user/223974/data-rsi-podman.sock
export DOCKER_CONFIG=/mnt/cvlab/scratch/cvlab/home/shuli/.docker
export HARBOR_DISABLE_TELEMETRY=1
export OTEL_SDK_DISABLED=true
