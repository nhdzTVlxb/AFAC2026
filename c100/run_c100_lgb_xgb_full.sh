#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/home/cyp/speedsci/AFAC2026/afac3_eval_pack/A推荐/A推荐}"
OUT_DIR="${OUT_DIR:-$ROOT/output_c100_lgb_xgb_full}"

cd "$ROOT"
rm -f train_lgb_xgb_source657.py

python make_4941_source657.py \
  --src train_lgb_xgb.py \
  --out train_lgb_xgb_source657.py \
  --lgb_weight "${LGB_WEIGHT:-1}" \
  --xgb_weight "${XGB_WEIGHT:-2}"

DATA_ROOT="$DATA_ROOT" \
OUT_DIR="$OUT_DIR" \
MAX_CANDIDATES=100 \
N_JOBS="${N_JOBS:-3}" \
PRED_BATCH_SIZE="${PRED_BATCH_SIZE:-120}" \
LGB_DEVICE="${LGB_DEVICE:-gpu}" \
XGB_DEVICE="${XGB_DEVICE:-cuda}" \
XGB_TREE_METHOD="${XGB_TREE_METHOD:-hist}" \
bash run_4941_source657.sh

echo "Full LGB/XGB c100 output:"
ls -lh "$OUT_DIR"
