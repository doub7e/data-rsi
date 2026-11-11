#!/usr/bin/env bash
set -euo pipefail
export HF_HOME=/scratch/cvlab/home/shuli/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
exec python3 -m vllm.entrypoints.openai.api_server \
  --model /scratch/cvlab/home/shuli/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \
  --served-model-name ota-easy-qwen3-8b-base --host 0.0.0.0 --port 8000 \
  --dtype bfloat16 --tensor-parallel-size 4 --max-model-len 32768 --max-num-seqs 64 \
  --gpu-memory-utilization 0.90 --enable-prefix-caching --reasoning-parser qwen3 \
  --generation-config auto --override-generation-config '{"max_new_tokens":16384}'
