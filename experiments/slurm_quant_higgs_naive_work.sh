#!/bin/bash
set -euo pipefail

export OMP_NUM_THREADS=7
export MKL_NUM_THREADS=7
export OPENBLAS_NUM_THREADS=7

IFS=',' read -ra WALLTIMES_ARRAY <<< "${WALLTIMES}"
num_walltimes=${#WALLTIMES_ARRAY[@]}

array_index=$((SLURM_ARRAY_TASK_ID - 1))
wt_index=$((array_index % num_walltimes))

walltime=${WALLTIMES_ARRAY[$wt_index]}
walltime_label=$(echo "${walltime}" | tr ':' '-')

results_dir_combined="${RESULTS_DIR}/naive/wt_${walltime_label}"
mkdir -p "${results_dir_combined}"

echo "Running naive quant_runner on ${INPUT_CSV}" >&2
echo "walltime: ${walltime}" >&2

cmd=(
  python quant_runner.py
  --input-csv "${INPUT_CSV}"
  --preds-csv none
  --ignore-cols "pred,label"
  --pred-cols "prob_0.0,prob_1.0"
  --backend faiss
  --frnn-metric linf
  --out-metric tv
  --frnn-threads 7
  --walltime "${walltime}"
  --results-dir "${results_dir_combined}"
  --display-stride 5000
  --initial-k 9999999
)

srun "${cmd[@]}"
