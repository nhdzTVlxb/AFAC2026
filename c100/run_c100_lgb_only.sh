#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/home/cyp/speedsci/AFAC2026/afac3_eval_pack/A推荐/A推荐}"
OUT_DIR="${OUT_DIR:-$ROOT/output_c100_lgb_only}"

cd "$ROOT"
rm -f train_lgb_xgb_source657.py

python make_4941_source657.py \
  --src train_lgb_xgb.py \
  --out train_lgb_xgb_source657.py \
  --lgb_weight 1 \
  --xgb_weight 2

DATA_ROOT="$DATA_ROOT" \
OUT_DIR="$OUT_DIR" \
MAX_CANDIDATES=100 \
N_JOBS="${N_JOBS:-3}" \
PRED_BATCH_SIZE="${PRED_BATCH_SIZE:-120}" \
LGB_DEVICE="${LGB_DEVICE:-gpu}" \
LGB_ONLY_PIPELINE=1 \
LGB_ONLY_FINAL=1 \
bash run_4941_source657.sh

echo "LGB-only c100 output:"
ls -lh "$OUT_DIR"
