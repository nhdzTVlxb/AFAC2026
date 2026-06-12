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

A1_PATH="${V1_FRAMEWORK}/output203/submission/A1.csv"
if [[ ! -f "${A1_PATH}" ]]; then
  echo "ERROR: missing fixed A1 ${A1_PATH}"
  echo "Run 195-206 first, or change A1_PATH in this script."
  exit 1
fi

OUTPUT122_A2="${V1_FRAMEWORK}/output122/submission/A2.csv"
if [[ ! -f "${OUTPUT122_A2}" ]]; then
  echo "WARNING: ${OUTPUT122_A2} not found. GRU fusion variants will fallback to AFAC A2."
fi

if [[ ! -f A2.csv ]]; then
  echo "AFAC base A2.csv not found, running train_task_b_final.py once..."
  python train_task_b_final.py
fi

python run_task2_only_207_218.py \
  --root "${ROOT}" \
  --out_root "${V1_FRAMEWORK}" \
  --a1_path "${A1_PATH}" \
  --output122_a2 "${OUTPUT122_A2}" \
  2>&1 | tee run_207_218.log

echo "===== Done 207-218 ====="
cat "${V1_FRAMEWORK}/ablation_207_218_summary.md"
