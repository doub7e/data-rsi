#!/usr/bin/env bash
set -euo pipefail
runai submit --name ota-qwen3-8b-10k-4h200-1030 \
  --image registry.rcp.epfl.ch/shuli/data-rsi-sft@sha256:d15905bbc491f0f5a7c65e36524b2b83e345e7b7031153824e64667804081d1a \
  --node-pool h200 --gpu 4 --cpu 32 --memory 256G --large-shm --backoff-limit 2 \
  --existing-pvc claimname=cvlab-scratch,path=/scratch \
  --command -- /opt/sft/bin/python /scratch/cvlab/home/shuli/data-rsi/002-openthoughts-qwen3-8b/scripts/run_reproduction.py
