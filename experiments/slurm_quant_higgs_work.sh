#!/bin/bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

export OMP_NUM_THREADS=7
export MKL_NUM_THREADS=7
export OPENBLAS_NUM_THREADS=7

IFS=',' read -ra BATCH_SIZES_ARRAY <<< "${BATCH_SIZES}"
IFS=',' read -ra WALLTIMES_ARRAY <<< "${WALLTIMES}"
IFS=',' read -ra EPS_ARRAY <<< "${EPSILONS}"
IFS=',' read -ra MAXK_ARRAY <<< "${MAX_K_VALUES}"
num_batch_sizes=${#BATCH_SIZES_ARRAY[@]}
num_walltimes=${#WALLTIMES_ARRAY[@]}
num_epsilons=${#EPS_ARRAY[@]}
num_maxk=${#MAXK_ARRAY[@]}

if [[ ${num_batch_sizes} -eq 0 || ${num_walltimes} -eq 0 || ${num_epsilons} -eq 0 || ${num_maxk} -eq 0 ]]; then
  echo "BATCH_SIZES, WALLTIMES, EPSILONS, or MAX_K_VALUES not configured" >&2
  exit 1
fi

array_index=$((SLURM_ARRAY_TASK_ID - 1))
bs_index=$((array_index % num_batch_sizes))
wt_index=$(((array_index / num_batch_sizes) % num_walltimes))
eps_index=$(((array_index / (num_batch_sizes * num_walltimes)) % num_epsilons))
maxk_index=$((array_index / (num_batch_sizes * num_walltimes * num_epsilons)))

if [[ ${maxk_index} -ge ${num_maxk} ]]; then
  echo "Invalid array index ${SLURM_ARRAY_TASK_ID} for provided combinations" >&2
  exit 1
fi

batch_size=${BATCH_SIZES_ARRAY[$bs_index]}
walltime=${WALLTIMES_ARRAY[$wt_index]}
epsilon=${EPS_ARRAY[$eps_index]}
max_k_value=${MAXK_ARRAY[$maxk_index]}
walltime_label=$(echo "${walltime}" | tr ':' '-')
epsilon_label=$(echo "${epsilon}" | tr '.' '_')
maxk_label=$( [[ -n "${max_k_value}" ]] && echo "maxk_${max_k_value}" || echo "maxk_none" )

results_dir_combined="${RESULTS_DIR}/eps_${epsilon_label}/${maxk_label}/batch_${batch_size}_wt_${walltime_label}"
mkdir -p "${results_dir_combined}"

echo "Running quant_runner on ${INPUT_CSV}" >&2
echo "Batch size: ${batch_size}, walltime: ${walltime}, epsilon: ${epsilon}, max_k: ${max_k_value:-<unset>}" >&2

srun \
  python quant_runner.py \
    --input-csv "${INPUT_CSV}" \
    --preds-csv none \
    --ignore-cols "pred,label" \
    --pred-cols "prob_0.0,prob_1.0" \
    --backend kdtree \
    --batchsize "${batch_size}" \
    --frnn-metric linf \
    --out-metric tv \
    --frnn-threads 7 \
    --epsilon "${epsilon}" \
    --walltime "${walltime}" \
    --results-dir "${results_dir_combined}" \
    --display-stride 5000 \
    --save-points \
    $( [[ -n "${max_k_value}" ]] && printf "--max-k %s" "${max_k_value}" )
