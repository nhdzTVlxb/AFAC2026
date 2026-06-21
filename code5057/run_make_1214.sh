#!/usr/bin/env bash
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate speedsci

cd "$(dirname "$0")"

SAMPLE="${1:-/home/cyp/speedsci/AFAC2026/v1/framework/data/rec_data/sample_submission.csv}"
OUT="${2:-/home/cyp/speedsci/AFAC2026/code5057/output1214_ssl_transfer_x_stable/A2.csv}"

python -u make_1214_fusion.py \
  --component-dir /home/cyp/speedsci/AFAC2026/code5057/components \
  --sample "$SAMPLE" \
  --output "$OUT"

