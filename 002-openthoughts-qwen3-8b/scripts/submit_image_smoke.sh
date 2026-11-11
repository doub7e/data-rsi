#!/usr/bin/env bash
set -euo pipefail
run_id="${1:?Supply a unique RunAI job/run ID}"
runai submit --name "$run_id" \
  --image registry.rcp.epfl.ch/shuli/data-rsi-sft@sha256:d15905bbc491f0f5a7c65e36524b2b83e345e7b7031153824e64667804081d1a \
  --node-pool h200 --gpu 2 --cpu 16 --memory 128G --large-shm --backoff-limit 0 \
  --existing-pvc claimname=cvlab-scratch,path=/scratch \
  --command -- /bin/bash /scratch/cvlab/home/shuli/data-rsi/002-openthoughts-qwen3-8b/scripts/run_image_smoke.sh "$run_id"
