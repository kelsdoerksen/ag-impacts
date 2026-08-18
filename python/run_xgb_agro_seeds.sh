#!/usr/bin/env bash
# Seed replicates for the agro-vs-control comparison, so "does the agronomic
# block help?" is answered with mean +/- std rather than one draw. Seed 1 of
# each arm is already on disk from run_xgb_agro_conus.sh, so this fills in 2-5.
#
# Pool afterwards with:
#   python aggregate_seed_runs.py --runs_dir <RESULTS_DIR> --config_hash <hash>

DATA_DIR="/Users/kdoerkse/Documents/UCT/AgImpacts/CONUS_processed_data"
RESULTS_DIR="/Users/kdoerkse/Documents/UCT/AgImpacts/results"
LOG_DIR="${RESULTS_DIR}/_xgb_agro_logs"
PY="/opt/anaconda3/envs/ag-deeplearning/bin/python"

mkdir -p "${LOG_DIR}"
cd "$(dirname "$0")" || exit 1

common=(--study_years 1980 2009
        --data_dir "${DATA_DIR}"
        --results_dir "${RESULTS_DIR}"
        --model_type xgb
        --crop corn
        --feature_mode aggregated
        --target yield_anomaly)

for seed in 2 3 4 5; do
  echo "=== [$(date '+%H:%M:%S')] seed ${seed}: agro ==="
  "${PY}" run_loo_experiment.py "${common[@]}" --seed "${seed}" --agro_features \
      > "${LOG_DIR}/agro_anomaly_seed${seed}.log" 2>&1 \
      || echo "FAILED agro seed ${seed}"
  echo "=== [$(date '+%H:%M:%S')] seed ${seed}: control ==="
  "${PY}" run_loo_experiment.py "${common[@]}" --seed "${seed}" \
      > "${LOG_DIR}/plain_anomaly_seed${seed}.log" 2>&1 \
      || echo "FAILED control seed ${seed}"
done

echo "=== [$(date '+%H:%M:%S')] done ==="
