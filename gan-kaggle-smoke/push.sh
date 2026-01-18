#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "[1/3] Dataset version (ganlab-lib)"
kaggle datasets version -p "$ROOT/ganlab_ds" --dir-mode zip -m "update ganlab code"

echo "[2/3] Push script kernel"
kaggle kernels push -p "$ROOT/kernels/script"

echo "[3/3] Push notebook kernel"
kaggle kernels push -p "$ROOT/kernels/notebook"

echo "OK"
