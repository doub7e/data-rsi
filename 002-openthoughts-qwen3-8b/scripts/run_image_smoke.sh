#!/usr/bin/env bash
set -euo pipefail
cd /scratch/cvlab/home/shuli/data-rsi/002-openthoughts-qwen3-8b
run_dir="$PWD/outputs/${1:?Supply a unique run ID}"
if [[ -e "$run_dir" ]]; then
  echo "Refusing to overwrite validation run: $run_dir" >&2
  exit 1
fi
mkdir -p "$run_dir"
exec > >(tee "$run_dir/run.log") 2>&1
export HF_HOME=/scratch/cvlab/home/shuli/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export TORCH_EXTENSIONS_DIR="$run_dir/torch_extensions"
python scripts/prepare_image_smoke.py "$run_dir"
FORCE_TORCHRUN=1 NPROC_PER_NODE=2 llamafactory-cli train "$run_dir/train.yaml"
python scripts/verify_image_smoke.py "$run_dir"
