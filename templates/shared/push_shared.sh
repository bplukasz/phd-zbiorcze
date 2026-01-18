#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "🚀 Pushing shared library"
kaggle datasets version -p "$ROOT" --dir-mode zip -m "update shared code"

echo "✅ Done: shared-lib"

