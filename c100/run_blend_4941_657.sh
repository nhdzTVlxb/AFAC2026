#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
A2_4941="${A2_4941:?set A2_4941=/path/to/0.4941/A2.csv}"
A2_657="${A2_657:-/home/cyp/speedsci/AFAC2026/afac3_eval_pack/task2_feature_model_submit/output657_lgb_source_rank_features/A2.csv}"
OUT_DIR="${OUT_DIR:-$ROOT/blend_4941_657_outputs}"

cd "$ROOT"
python blend_4941_657.py \
  --a2_4941 "$A2_4941" \
  --a2_657 "$A2_657" \
  --out_dir "$OUT_DIR"

echo "Blend outputs:"
ls -lh "$OUT_DIR"
