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

FIXED_A2="${V1_FRAMEWORK}/output_afac3/submission/A2.csv"
if [[ ! -f "${FIXED_A2}" ]]; then
  echo "ERROR: missing ${FIXED_A2}"
  echo "Run run_afac3_best.sh first, or copy your preferred A2.csv there."
  exit 1
fi

python run_afac_task1_181_194.py \
  --root "${ROOT}" \
  --out_root "${V1_FRAMEWORK}" \
  --fixed_a2 "${FIXED_A2}" \
  2>&1 | tee run_181_194.log

echo "===== Done 181-194 ====="
cat "${V1_FRAMEWORK}/ablation_181_194_summary.md"
