#!/usr/bin/env bash
set -euo pipefail
[[ $# == 1 ]] || exit 2
EXP=/scratch/cvlab/home/shuli/data-rsi/qwen_math_fp32_richardson20_priority_20261109
RUN="$EXP/runs/$1"
mkdir -p "$RUN/logs"
export HF_HOME=/scratch/cvlab/home/shuli/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline
export USER=shuli LOGNAME=shuli VLLM_USE_FLASHINFER_SAMPLER=0 TORCH_COMPILE_DISABLE=0
export FD_GRADER_OVERLAY=/scratch/cvlab/home/shuli/data-rsi/qwen_math_fd_training_ablation_20261108/runtime/grader
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$RUN/source/tools:$RUN/source/verl"
cd "$RUN/source"
FD_PYTHON=$(command -v python || command -v python3)
echo "PRIORITY=1 RUN_ID=$1 OUTPUT=$RUN"
"$FD_PYTHON" "$RUN/source/tools/fd_reliability_job.py" --experiment "$EXP" --run-id "$1" 2>&1 | tee -a "$RUN/logs/${HOSTNAME:-worker}.log"
