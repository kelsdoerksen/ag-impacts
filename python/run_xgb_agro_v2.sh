#!/usr/bin/env bash
# Agro arm re-run after dropping season_days_obs / season_days_missing from the
# feature set (they are per-county constants, hence pure county identity).
# 5 seeds, so it is comparable to the pooled control.

DATA_DIR="/Users/kdoerkse/Documents/UCT/AgImpacts/CONUS_processed_data"
RESULTS_DIR="/Users/kdoerkse/Documents/UCT/AgImpacts/results"
LOG_DIR="${RESULTS_DIR}/_xgb_agro_logs"
PY="/opt/anaconda3/envs/ag-deeplearning/bin/python"

mkdir -p "${LOG_DIR}"
cd "$(dirname "$0")" || exit 1

for seed in 1 2 3 4 5; do
  echo "=== [$(date '+%H:%M:%S')] agro v2 seed ${seed} ==="
  "${PY}" run_loo_experiment.py \
      --study_years 1980 2009 \
      --data_dir "${DATA_DIR}" \
      --results_dir "${RESULTS_DIR}" \
      --model_type xgb --crop corn --feature_mode aggregated \
      --target yield_anomaly --agro_features --seed "${seed}" \
      > "${LOG_DIR}/agro_v2_seed${seed}.log" 2>&1 || echo "FAILED seed ${seed}"
done
echo "=== [$(date '+%H:%M:%S')] done ==="
