#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AFAC_ROOT="${AFAC_ROOT:-/home/cyp/speedsci/AFAC2026}"
V1_FRAMEWORK="${V1_FRAMEWORK:-${AFAC_ROOT}/v1/framework}"

cd "${ROOT}"

mkdir -p A分类 A推荐
rm -f A分类/A分类 A推荐/A推荐
ln -s "${V1_FRAMEWORK}/data/cls_data" A分类/A分类
ln -s "${V1_FRAMEWORK}/data/rec_data" A推荐/A推荐

python - <<'PY'
import xgboost
print("xgboost ok")
PY

if [[ ! -f A1.csv ]]; then
  python train_cls.py 2>&1 | tee train_cls.log
  python predict_cls_deg.py 2>&1 | tee predict_cls_deg.log
fi

python train_final.py 2>&1 | tee train_final_xgb.log

rm -f prediction_afac3_xgb.zip
zip prediction_afac3_xgb.zip A1.csv A2.csv

mkdir -p "${V1_FRAMEWORK}/output_afac3_xgb/submission"
cp A1.csv "${V1_FRAMEWORK}/output_afac3_xgb/submission/A1.csv"
cp A2.csv "${V1_FRAMEWORK}/output_afac3_xgb/submission/A2.csv"
cp prediction_afac3_xgb.zip "${V1_FRAMEWORK}/output_afac3_xgb/prediction.zip"
ls -lh "${V1_FRAMEWORK}/output_afac3_xgb/prediction.zip"
