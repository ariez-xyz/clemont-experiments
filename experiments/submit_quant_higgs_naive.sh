#!/bin/bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

INPUT_CSV="../data/RobustTrees/predictions/higgs/train_pred.csv"
RESULTS_BASE="../results/quantitative/robusttrees_higgs"

INPUT_CSV=$(realpath "${INPUT_CSV}")
RESULTS_BASE=$(realpath -m "${RESULTS_BASE}")

export INPUT_CSV
export RESULTS_BASE
export RESULTS_DIR="${RESULTS_BASE}/results"
export LOGS_DIR="${RESULTS_BASE}/logs"
export WALLTIMES="00:10:00,06:00:00,20:00:00"
export WORK_SCRIPT="slurm_quant_higgs_naive.sh"

mkdir -p "${RESULTS_DIR}" "${LOGS_DIR}"

if [[ ! -f "${INPUT_CSV}" ]]; then
  echo "Input CSV not found: ${INPUT_CSV}" >&2
  exit 1
fi

unset SLURM_EXPORT_ENV
pushd .. >/dev/null
source activate.sh
popd >/dev/null

num_walltimes=$(echo "${WALLTIMES}" | tr ',' '\n' | wc -l | tr -d ' ')

sbatch \
  --job-name=quant_higgs \
  --output="${LOGS_DIR}/quant_higgs-%A-%a.log" \
  --cpus-per-task=8 \
  --time=22:00:00 \
  --mem=192G \
  --array=1-${num_walltimes} \
  --export=ALL \
  "${WORK_SCRIPT}"
