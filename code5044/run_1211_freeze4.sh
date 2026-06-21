#!/usr/bin/env bash
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate speedsci

cd "$(dirname "$0")"

DATA_ROOT="${1:-/home/cyp/speedsci/AFAC2026/v1/framework/data/rec_data}"
OUT_DIR="${2:-/home/cyp/speedsci/AFAC2026/code504/output_1211_freeze4}"

mkdir -p "$OUT_DIR"

python -u train_1211_freeze4.py \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUT_DIR" \
  2>&1 | tee "$OUT_DIR/run_1211_freeze4.log"

