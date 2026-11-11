#!/usr/bin/env bash
set -euo pipefail
pool="${1:-h100}"
case "$pool" in h100|h200) ;; *) echo 'Expected h100 or h200' >&2; exit 2;; esac
runai submit --name "ota-qwen3-8b-1k-4${pool}-1031" \
  --image registry.rcp.epfl.ch/shuli/data-rsi-sft@sha256:d15905bbc491f0f5a7c65e36524b2b83e345e7b7031153824e64667804081d1a \
  --node-pool "$pool" --gpu 4 --cpu 32 --memory 256G --large-shm --backoff-limit 0 \
  --environment "OTA_GPU_CLASS=${pool^^}" \
  --existing-pvc claimname=cvlab-scratch,path=/scratch \
  --command -- /opt/sft/bin/python /scratch/cvlab/home/shuli/data-rsi/003-openthoughts-qwen3-8b-1k/scripts/run_training.py
