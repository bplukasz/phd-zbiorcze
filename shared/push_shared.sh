#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
EXP_NAME="shared-lib"

echo "🚀 Pushing shared library"

echo "Dataset found -> updating $EXP_NAME"
kaggle datasets version -p "${ROOT}" --dir-mode zip -m "update shared code"

echo "✅ Done: shared-lib"
