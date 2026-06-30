#!/usr/bin/env bash
# PatchTST LOO experiment (1980-2009, target=yield) per crop with a dispersed
# 150-county sample, 100 epochs @ lr=1e-3 (config that yields usable R2).
# cassava excluded (no CONUS yield data). Crops run smallest-first so partial
# completion still yields the most crops. Plots are generated after each crop.
# NB: no `set -u` — conda's activation scripts reference unbound vars.

DATA_DIR="/Users/kdoerkse/Documents/UCT/AgImpacts/CONUS_processed_data"
RESULTS_DIR="/Users/kdoerkse/Documents/UCT/AgImpacts/results"
LOG_DIR="${RESULTS_DIR}/_tst_logs"
mkdir -p "${LOG_DIR}"

CROPS=(rice sorghum soy corn wheat)

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ag-deeplearning
cd "$(dirname "$0")"

for crop in "${CROPS[@]}"; do
  echo "=== [$(date '+%H:%M:%S')] START ${crop} ==="
  python run_loo_experiment.py \
      --study_years 1980 2009 \
      --data_dir "${DATA_DIR}" \
      --results_dir "${RESULTS_DIR}" \
      --model_type tst \
      --crop "${crop}" \
      --target yield \
      --sample_counties 150 \
      --sample_seed 42 \
      --epochs 100 \
      --lr 0.001 \
      > "${LOG_DIR}/${crop}.log" 2>&1
  status=$?
  if [ $status -ne 0 ]; then
    echo "=== [$(date '+%H:%M:%S')] FAILED ${crop} (exit ${status}) — see ${LOG_DIR}/${crop}.log ==="
    continue
  fi
  # newest experiment dir for this crop
  exp_dir=$(ls -dt "${RESULTS_DIR}"/loo_${crop}_tst_*/ 2>/dev/null | head -1)
  echo "=== [$(date '+%H:%M:%S')] DONE ${crop} (ok) -> ${exp_dir} ==="
  python make_crop_plots.py "${exp_dir%/}" >> "${LOG_DIR}/${crop}.log" 2>&1 \
    && echo "    plots ok for ${crop}" \
    || echo "    plots FAILED for ${crop} — see ${LOG_DIR}/${crop}.log"
done
echo "=== [$(date '+%H:%M:%S')] ALL CROPS COMPLETE ==="
