#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/home/cyp/speedsci/AFAC2026/afac3_eval_pack/A推荐/A推荐}"
OUT_DIR="${OUT_DIR:-$ROOT/output_4941_source657}"

cd "$ROOT"
mkdir -p "$OUT_DIR"

python make_4941_source657.py \
  --src train_lgb_xgb.py \
  --out train_lgb_xgb_source657.py \
  --lgb_weight "${LGB_WEIGHT:-1}" \
  --xgb_weight "${XGB_WEIGHT:-2}"

DATA_ROOT="$DATA_ROOT" python train_lgb_xgb_source657.py --save_to "$OUT_DIR"

echo "done: $OUT_DIR"
ls -lh "$OUT_DIR"
