#!/usr/bin/env bash
set -euo pipefail
root=/scratch/cvlab/home/shuli/data-rsi/002-openthoughts-qwen3-8b
checkpoint="$root/runs/repro-10k-4h200-20261030/checkpoint"
export HF_HOME=/scratch/cvlab/home/shuli/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
exec python3 -m vllm.entrypoints.openai.api_server \
  --model "$checkpoint" --served-model-name ota-qwen3-8b \
  --host 0.0.0.0 --port 8000 --dtype bfloat16 \
  --tensor-parallel-size 4 --max-model-len 32768 --max-num-seqs 64 \
  --gpu-memory-utilization 0.90 --enable-prefix-caching \
  --reasoning-parser qwen3 --generation-config auto \
  --override-generation-config '{"max_new_tokens":16384}'
