#!/usr/bin/env bash
# PatchTST feature-space arms, matched to the XGB/agro arms run on the same
# window so the two architectures can be compared directly.
#
#   window : s2002-s2022  (FY1 -> target years 2003-2023)
#   crop   : corn, CONUS, all counties
#   target : yield_zscore — the model predicts (yield - county mean)/county sd,
#            with the baseline fit per fold on TRAINING rows only and the
#            prediction inverted back to kg/ha before scoring, so RMSE stays in
#            yield units and comparable to a --target yield run.
#            NOTE: predicting z=0 for every row reproduces the county-mean null
#            exactly, so that null is the FLOOR here, not an achievement.
#            --anomaly_detrend makes that baseline a per-county LINE in year
#            rather than a flat mean. CONUS maize carries a ~+127 kg/ha/yr
#            technology trend which is ~24% of flat-mean anomaly variance on
#            this window and is structurally unpredictable under LOO (the model
#            cannot know which year it is forecasting). Without it, a quarter of
#            the target being predicted is guaranteed noise.
#            Set TARGET=yield to run the absolute-yield family instead; both
#            keep --anomaly_detrend so the two are scored against the SAME
#            (detrended) climatology null and their r_anom / R2anom columns
#            stay comparable. Under --target yield the flag only changes that
#            scoring null, not what the model learns.
#   seed   : 42 on every arm, matching the XGB side.
#
# Soil moisture is deliberately excluded: its phenology parquet starts at
# season 2003, so with --phenology_lag_years 1 it would silently drop the s2002
# fold and put that arm on a different sample set from the others.
#
# Requires code at commit 23202d5 or newer (git pull first).
# Usage:  DATA_DIR=... RESULTS_DIR=... bash run_tst_arms.sh
#         DATA_DIR=... RESULTS_DIR=... TARGET=yield bash run_tst_arms.sh
#
# NOTE ON COST: this is 6 arms x N seeds x 21 folds x 100 epochs PER TARGET.
# At the default 5 seeds that is 630 fold-trainings per target family, 1,260
# for both. Time ONE arm first (SEEDS=42 ARMS=A_climate) and multiply before
# committing the GPU. Dial it down with SEEDS="42 43 44" if needed — three
# seeds still gives a usable spread, one gives none.
# (As of commit ce437f5 --study_years selects the training pool as well as the
# held-out folds, so each fold trains on ~32k county-years from the 21 study
# years, matching the tabular runs.)
# NB: no `set -u` — conda's activation scripts reference unbound vars.
set -o pipefail

DATA_DIR="${DATA_DIR:?set DATA_DIR to the processed-data dir on this machine}"
RESULTS_DIR="${RESULTS_DIR:?set RESULTS_DIR to where results should be written}"
TARGET="${TARGET:-yield_zscore}"
# Seed replicates. One run = one replicate; runs sharing a config hash differ
# only in seed and are pooled by aggregate_seed_runs.py into mean +/- sd.
SEEDS="${SEEDS:-42 43 44 45 46}"
# Optional subset for timing a single arm, e.g. ARMS="A_climate".
ARMS="${ARMS:-A_climate B_ndvi C_ndvi_fpar D_yhist5 F_ndvi_yhist5 E_full}"
LOG_DIR="${RESULTS_DIR}/_tst_arm_logs_${TARGET}"
AGG_DIR="${RESULTS_DIR}/agg_${TARGET}"
mkdir -p "${LOG_DIR}" "${AGG_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ag-deeplearning
cd "$(dirname "$0")"

COMMON=(--study_years 2002 2022
        --data_dir "${DATA_DIR}"
        --results_dir "${RESULTS_DIR}"
        --model_type tst
        --crop corn
        --aoi CONUS
        --target "${TARGET}"
        --anomaly_detrend
        --epochs 100
        --lr 0.001)

run_arm () {           # run_arm <name> <extra args...>
  local name="$1"; shift
  case " ${ARMS} " in *" ${name} "*) ;; *) echo "--- skipping ${name}"; return 0 ;; esac
  local ok=0
  for seed in ${SEEDS}; do
    local log="${LOG_DIR}/${name}_seed${seed}.log"
    echo "=== [$(date '+%F %H:%M:%S')] START ${name} seed ${seed} ==="
    python run_loo_experiment.py "${COMMON[@]}" --seed "${seed}" "$@" > "${log}" 2>&1
    if [ $? -ne 0 ]; then
      echo "=== [$(date '+%H:%M:%S')] FAILED ${name} seed ${seed} — see ${log} ==="
      tail -20 "${log}"
    else
      ok=$((ok+1))
      echo "=== [$(date '+%H:%M:%S')] DONE ${name} seed ${seed} ==="
      grep -E "^   Overall|mean R.anom" "${log}"
    fi
  done

  if [ "${ok}" -lt 2 ]; then
    echo "!!! ${name}: only ${ok} seed(s) succeeded — nothing to pool"
    return 0
  fi
  # Every seed of one arm shares a config hash (the seed is excluded from it by
  # design), so read it off any successful log and pool that group. Reading the
  # hash beats globbing folder names, which vary with the feature-set tags.
  local hash
  # head -1 is load-bearing: grep -m1 caps matches PER FILE, so globbing several
  # seed logs would otherwise yield one hash per file and a multi-line variable.
  hash=$(grep -hm1 "^Config hash" "${LOG_DIR}/${name}"_seed*.log 2>/dev/null \
         | head -1 | sed -E "s/.*:[[:space:]]*([0-9a-f]+).*/\\1/")
  if [ -z "${hash}" ]; then
    echo "!!! ${name}: could not read a config hash; pool manually"
    return 0
  fi
  echo "--- pooling ${ok} seed(s) of ${name} (cfg-${hash})"
  python aggregate_seed_runs.py \
      --runs_dir "${RESULTS_DIR}" --config_hash "${hash}" \
      --out_dir "${AGG_DIR}/${name}" > "${LOG_DIR}/${name}_aggregate.log" 2>&1 \
    && grep -E "mean|sd|RMSE" "${LOG_DIR}/${name}_aggregate.log" | tail -5 \
    || { echo "!!! aggregate failed for ${name}"; tail -15 "${LOG_DIR}/${name}_aggregate.log"; }
}

# The arms form a 3 x 2 grid: {no phenology, NDVI, NDVI+fPAR} crossed with
# {no yield history, 5yr yield history}. That makes every effect separable —
# phenology at fixed history, and history at fixed phenology.
#
#                      no yield history      + 5yr yield history
#   no phenology       A_climate             D_yhist5
#   NDVI               B_ndvi                F_ndvi_yhist5
#   NDVI + fPAR        C_ndvi_fpar           E_full
YH=(--yield_history --yield_history_len 5)

run_arm A_climate
run_arm B_ndvi        --use_phenology ndvi
run_arm C_ndvi_fpar   --use_phenology ndvi,fpar
run_arm D_yhist5                                "${YH[@]}"
run_arm F_ndvi_yhist5 --use_phenology ndvi      "${YH[@]}"
run_arm E_full        --use_phenology ndvi,fpar "${YH[@]}"

echo "=== [$(date '+%F %H:%M:%S')] ALL ARMS COMPLETE (target=${TARGET}, seeds: ${SEEDS}) ==="
echo
echo "Pooled per-arm results (mean +/- sd across seeds):"
ls -d "${AGG_DIR}"/*/ 2>/dev/null
echo
echo "Copy BOTH back for comparison — the pooled dirs above and the raw run dirs:"
ls -d "${RESULTS_DIR}"/loo_corn_tst_*2002_2022*/ 2>/dev/null | head -40
echo "  (raw run dirs: $(ls -d "${RESULTS_DIR}"/loo_corn_tst_*2002_2022*/ 2>/dev/null | wc -l) total)"
