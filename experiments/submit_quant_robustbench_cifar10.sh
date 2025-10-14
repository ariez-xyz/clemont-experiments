#!/bin/bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

INPUT_CSVS=(
  "../data/RobustBench/predictions/cifar10-Bartoldson2024Adversarial_WRN-94-16-combined.csv"
  "../data/RobustBench/predictions/cifar10-Standard-combined.csv"
)
MODEL_LABELS=(robust baseline)
RESULTS_BASE="../results/quantitative/robustbench_cifar10"
WORK_SCRIPT="slurm_quant_robustbench_cifar10_work.sh"

RESULTS_BASE=$(realpath -m "${RESULTS_BASE}")
mkdir -p "${RESULTS_BASE}/logs"

for idx in "${!INPUT_CSVS[@]}"; do
  INPUT_CSVS[$idx]=$(realpath "${INPUT_CSVS[$idx]}")
  if [[ ! -f "${INPUT_CSVS[$idx]}" ]]; then
    echo "Input CSV not found: ${INPUT_CSVS[$idx]}" >&2
    exit 1
  fi
  mkdir -p "${RESULTS_BASE}/${MODEL_LABELS[$idx]}"
  INPUT_CSVS[$idx]="${INPUT_CSVS[$idx]}"
done

export INPUT_CSVS_CSV=$(IFS=','; echo "${INPUT_CSVS[*]}")
export MODEL_LABELS_CSV=$(IFS=','; echo "${MODEL_LABELS[*]}")
export RESULTS_BASE

unset SLURM_EXPORT_ENV
pushd .. >/dev/null
source activate.sh
popd >/dev/null

num_tasks=${#MODEL_LABELS[@]}

sbatch \
  --job-name=quant_cifar10 \
  --output="${RESULTS_BASE}/logs/quant_cifar10-%A-%a.log" \
  --cpus-per-task=8 \
  --time=24:00:00 \
  --mem=96G \
  --array=1-${num_tasks} \
  --export=ALL,INPUT_CSVS_CSV,MODEL_LABELS_CSV,RESULTS_BASE \
  "${WORK_SCRIPT}"
