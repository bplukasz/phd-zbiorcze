#!/bin/bash
# Pipeline script dla E002-01 GAN Cluster Metrics

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SRC_DIR="${SCRIPT_DIR}/dataset/src"

echo "============================================"
echo "E002-01: GAN Cluster Metrics Pipeline"
echo "============================================"

# 1. Precompute real features
echo ""
echo "Step 1: Precompute real features..."
python "${SRC_DIR}/eval/precompute_real.py" \
    --config "${SRC_DIR}/configs/e11_precompute_cifar10.yml"

# 2. Training
echo ""
echo "Step 2: Training..."
python "${SRC_DIR}/train.py" \
    --config "${SRC_DIR}/configs/cifar10_baseline.yml"

# 3. Znajdź ostatni checkpoint
CKPT=$(ls -t /kaggle/working/runs/*/checkpoints/ckpt_*.pt 2>/dev/null | head -1 || echo "")

if [ -z "$CKPT" ]; then
    echo "BŁĄD: Brak checkpointów!"
    exit 1
fi

echo ""
echo "Found checkpoint: $CKPT"

# 4. Evaluacja E11
echo ""
echo "Step 3: E11 evaluation..."
python "${SRC_DIR}/eval/eval_e11.py" \
    --config "${SRC_DIR}/configs/e11_eval_cifar10.yml" \
    --ckpt "$CKPT"

# 5. Raport
echo ""
echo "Step 4: Generating report..."
python "${SRC_DIR}/eval/report.py" \
    --e11_dir /kaggle/working/e11_results \
    --out /kaggle/working/report.csv

echo ""
echo "============================================"
echo "Pipeline completed!"
echo "Results:"
echo "  - Training: /kaggle/working/runs/"
echo "  - E11 cache: /kaggle/working/e11_cache/"
echo "  - E11 results: /kaggle/working/e11_results/"
echo "  - Reports: /kaggle/working/report*.csv"
echo "============================================"

