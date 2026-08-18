#!/usr/bin/env bash
# PatchTST feature-space arms, matched to the XGB/agro arms run on the same
# window so the two architectures can be compared directly.
#
#   window : s2002-s2022  (FY1 -> target years 2003-2023)
#   crop   : corn, CONUS, all counties
#   target : yield (absolute). Nulls are recomputed from loo_all_errors.csv
#            afterwards, so no --anomaly_detrend is needed here.
#   seed   : 42 on every arm, matching the XGB side.
#
# Soil moisture is deliberately excluded: its phenology parquet starts at
# season 2003, so with --phenology_lag_years 1 it would silently drop the s2002
# fold and put that arm on a different sample set from the others.
#
# Requires code at commit 23202d5 or newer (git pull first).
# Usage:  DATA_DIR=... RESULTS_DIR=... bash run_tst_arms.sh
# NB: no `set -u` — conda's activation scripts reference unbound vars.
set -o pipefail

DATA_DIR="${DATA_DIR:?set DATA_DIR to the processed-data dir on this machine}"
RESULTS_DIR="${RESULTS_DIR:?set RESULTS_DIR to where results should be written}"
LOG_DIR="${RESULTS_DIR}/_tst_arm_logs"
mkdir -p "${LOG_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ag-deeplearning
cd "$(dirname "$0")"

COMMON=(--study_years 2002 2022
        --data_dir "${DATA_DIR}"
        --results_dir "${RESULTS_DIR}"
        --model_type tst
        --crop corn
        --aoi CONUS
        --target yield
        --seed 42
        --epochs 100
        --lr 0.001)

run_arm () {           # run_arm <name> <extra args...>
  local name="$1"; shift
  echo "=== [$(date '+%F %H:%M:%S')] START ${name} ==="
  python run_loo_experiment.py "${COMMON[@]}" "$@" \
      > "${LOG_DIR}/${name}.log" 2>&1
  local st=$?
  if [ $st -ne 0 ]; then
    echo "=== [$(date '+%H:%M:%S')] FAILED ${name} (exit ${st}) — see ${LOG_DIR}/${name}.log ==="
    tail -20 "${LOG_DIR}/${name}.log"
  else
    echo "=== [$(date '+%H:%M:%S')] DONE ${name} ==="
    grep -E "^   Overall|mean R.anom" "${LOG_DIR}/${name}.log"
  fi
}

run_arm A_climate
run_arm B_ndvi        --use_phenology ndvi
run_arm C_ndvi_fpar   --use_phenology ndvi,fpar
run_arm D_yhist5      --yield_history --yield_history_len 5

echo "=== [$(date '+%F %H:%M:%S')] ALL ARMS COMPLETE ==="
echo
echo "Copy these back for comparison (the whole run dirs, not just the logs):"
ls -d "${RESULTS_DIR}"/loo_corn_tst_*2002_2022*/ 2>/dev/null
