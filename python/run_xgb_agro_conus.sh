#!/usr/bin/env bash
# Leave-one-year-out XGBoost over all of CONUS, 1980-2009 (30 folds), corn.
#
# Three arms, so the agronomic block is judged against a control rather than in
# isolation:
#   1. agro  + yield_anomaly  — the block, scored on weather skill
#   2. plain + yield_anomaly  — same everything, no --agro_features (the control)
#   3. agro  + yield          — absolute yield, comparable to the existing TST runs
#
# Read R2anom, not R2: against absolute yield most of the variance is the gap
# between counties, which weather cannot explain and a county baseline already
# captures. R2anom > 0 means the run beat "predict this county's normal yield".

DATA_DIR="/Users/kdoerkse/Documents/UCT/AgImpacts/CONUS_processed_data"
RESULTS_DIR="/Users/kdoerkse/Documents/UCT/AgImpacts/results"
LOG_DIR="${RESULTS_DIR}/_xgb_agro_logs"
PY="/opt/anaconda3/envs/ag-deeplearning/bin/python"
SEED=1

mkdir -p "${LOG_DIR}"
cd "$(dirname "$0")" || exit 1

common=(--study_years 1980 2009
        --data_dir "${DATA_DIR}"
        --results_dir "${RESULTS_DIR}"
        --model_type xgb
        --crop corn
        --feature_mode aggregated
        --seed "${SEED}")

echo "=== [$(date '+%H:%M:%S')] 1/3 agro + yield_anomaly ==="
"${PY}" run_loo_experiment.py "${common[@]}" --target yield_anomaly --agro_features \
    > "${LOG_DIR}/agro_anomaly.log" 2>&1 || echo "FAILED — see ${LOG_DIR}/agro_anomaly.log"

echo "=== [$(date '+%H:%M:%S')] 2/3 control (no agro) + yield_anomaly ==="
"${PY}" run_loo_experiment.py "${common[@]}" --target yield_anomaly \
    > "${LOG_DIR}/plain_anomaly.log" 2>&1 || echo "FAILED — see ${LOG_DIR}/plain_anomaly.log"

echo "=== [$(date '+%H:%M:%S')] 3/3 agro + yield ==="
"${PY}" run_loo_experiment.py "${common[@]}" --target yield --agro_features \
    > "${LOG_DIR}/agro_yield.log" 2>&1 || echo "FAILED — see ${LOG_DIR}/agro_yield.log"

echo "=== [$(date '+%H:%M:%S')] done — logs in ${LOG_DIR} ==="
