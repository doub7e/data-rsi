#!/usr/bin/env bash
set -euo pipefail
job_name="${1:?Supply a unique RunAI build job name}"
image_ref="${IMAGE_REF:-registry.rcp.epfl.ch/shuli/data-rsi-sft:qwen3-cu128-torch280-v1}"
context=/scratch/cvlab/home/shuli/data-rsi/002-openthoughts-qwen3-8b/docker
runai submit --name "$job_name" \
  --image gcr.io/kaniko-project/executor@sha256:4e7a52dd1f14872430652bb3b027405b8dfd17c4538751c620ac005741ef9698 \
  --gpu 0 --cpu 8 --memory 32G \
  --environment DOCKER_CONFIG=/scratch/cvlab/home/shuli/.docker \
  --existing-pvc claimname=cvlab-scratch,path=/scratch \
  --command -- /kaniko/executor \
  --dockerfile="$context/Dockerfile" --context="dir://$context" \
  --destination="$image_ref" --cache=true \
  --cache-repo=registry.rcp.epfl.ch/shuli/data-rsi-sft/cache \
  --snapshot-mode=redo --log-format=text
