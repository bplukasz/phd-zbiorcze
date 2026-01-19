#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
EXP_NAME="e001-01-wavelets-baseline"

echo "🚀 Pushing experiment: $EXP_NAME"

echo "[1/3] Dataset version ($EXP_NAME-lib)"
kaggle datasets version -p "$ROOT/dataset" --dir-mode zip -m "update $EXP_NAME code"
#kaggle datasets create -p "${ROOT}/dataset" --dir-mode zip

echo "[2/3] Push script kernel"
kaggle kernels push -p "$ROOT/kernels/script"

echo "[3/3] Push notebook kernel"
kaggle kernels push -p "$ROOT/kernels/notebook"

echo "✅ Done: $EXP_NAME"


