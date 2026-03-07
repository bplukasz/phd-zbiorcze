#!/usr/bin/env bash
set -euo pipefail

# Wspólne cache (żeby nie puchło w ~/.cache w losowych miejscach)
export EDX_ROOT="${HOME}/edx"
export XDG_CACHE_HOME="${EDX_ROOT}/caches/xdg"
export HF_HOME="${EDX_ROOT}/caches/huggingface"
export TRANSFORMERS_CACHE="${EDX_ROOT}/caches/huggingface/transformers"
export TORCH_HOME="${EDX_ROOT}/caches/torch"
export WANDB_DIR="${EDX_ROOT}/logs/wandb"
export MPLCONFIGDIR="${EDX_ROOT}/caches/matplotlib"

# Stabilniejsze logowanie w SSH/tmux
export PYTHONUNBUFFERED=1

# PYTHONPATH: root projektu, żeby `from shared.utils import ...` działał
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Ścieżka do danych (CelebA lub inny dataset)
# Nadpisz przez: export DATA_DIR=/twoja/sciezka  lub  make train-e001 DATA_DIR=...
export DATA_DIR="${DATA_DIR:-${EDX_ROOT}/datasets/celeba/img_align_celeba}"

mkdir -p \
  "${XDG_CACHE_HOME}" "${HF_HOME}" "${TRANSFORMERS_CACHE}" \
  "${TORCH_HOME}" "${WANDB_DIR}" "${MPLCONFIGDIR}" \
  "${EDX_ROOT}/datasets"
