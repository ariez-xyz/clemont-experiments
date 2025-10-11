#!/bin/bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

export INPUT_CSV="../data/RobustTrees/predictions/higgs/train_pred.csv"
export RESULTS_BASE="../results/quantitative/robusttrees_higgs"
export RESULTS_DIR="${RESULTS_BASE}/results"
export LOGS_DIR="${RESULTS_BASE}/logs"
export BATCH_SIZES="10000,50000,100000"
export WALLTIMES="06:00:00,48:00:00"
export EPSILONS="0.01,0.025,0.05,0.1"
export MAX_K_VALUES=",1024"
export WORK_SCRIPT="slurm_quant_higgs_work.sh"

mkdir -p "${RESULTS_DIR}" "${LOGS_DIR}"

if [[ ! -f "${INPUT_CSV}" ]]; then
  echo "Input CSV not found: ${INPUT_CSV}" >&2
  exit 1
fi

unset SLURM_EXPORT_ENV
pushd .. >/dev/null
source activate.sh
popd >/dev/null

num_batch_sizes=$(echo "${BATCH_SIZES}" | tr ',' '\n' | wc -l | tr -d ' ')
num_walltimes=$(echo "${WALLTIMES}" | tr ',' '\n' | wc -l | tr -d ' ')
num_epsilons=$(echo "${EPSILONS}" | tr ',' '\n' | wc -l | tr -d ' ')
num_maxk=$(echo "${MAX_K_VALUES}" | tr ',' '\n' | wc -l | tr -d ' ')
num_tasks=$((num_batch_sizes * num_walltimes * num_epsilons * num_maxk))

sbatch \
  --job-name=quant_higgs \
  --output="${LOGS_DIR}/quant_higgs-%A-%a.log" \
  --cpus-per-task=8 \
  --time=50:00:00 \
  --mem=96G \
  --array=1-${num_tasks} \
  --export=ALL \
  "${WORK_SCRIPT}"
