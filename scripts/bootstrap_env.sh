#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# instalacja z constraints z Jupyterowego env
CONSTRAINTS="${HOME}/Projekty/common-env/edgexpert-ml.txt"
if [[ ! -f "${CONSTRAINTS}" ]]; then
  echo "Brak constraints: ${CONSTRAINTS}"
  echo "Najpierw zrob: pip freeze z /home/bplukasz/jupyterlab/.venv"
  exit 1
fi

# Filtrujemy constraints - usuwamy torch i powiazane pakiety CUDA,
# zeby pip mogl sam dobrac wlasciwy wariant (+cu130)
FILTERED_CONSTRAINTS="$(mktemp)"
grep -viE '^(torch|torchvision|torchaudio|triton|nvidia-|xformers)[^a-z]' \
  "${CONSTRAINTS}" > "${FILTERED_CONSTRAINTS}"

pip install \
  --upgrade \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  -c "${CONSTRAINTS}" \
  -r requirements.in
#pip install --extra-index-url https://download.pytorch.org/whl/cu130 -c "${FILTERED_CONSTRAINTS}" -r requirements-dev.in

rm -f "${FILTERED_CONSTRAINTS}"

# szybki sanity check GPU/torch
python scripts/gpu_check.py

echo "OK. Interpreter: ${PROJECT_DIR}/.venv/bin/python"
