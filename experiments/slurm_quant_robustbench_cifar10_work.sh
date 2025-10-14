#!/bin/bash
set -euo pipefail

export OMP_NUM_THREADS=7
export MKL_NUM_THREADS=7
export OPENBLAS_NUM_THREADS=7

IFS=',' read -ra INPUT_CSVS_ARRAY <<< "${INPUT_CSVS_CSV}"
IFS=',' read -ra MODEL_LABELS_ARRAY <<< "${MODEL_LABELS_CSV}"

num_inputs=${#INPUT_CSVS_ARRAY[@]}
num_labels=${#MODEL_LABELS_ARRAY[@]}

if [[ ${num_inputs} -eq 0 || ${num_inputs} -ne ${num_labels} ]]; then
  echo "INPUT_CSVS_CSV and MODEL_LABELS_CSV mismatch" >&2
  exit 1
fi

array_index=$((SLURM_ARRAY_TASK_ID - 1))
if [[ ${array_index} -lt 0 || ${array_index} -ge ${num_inputs} ]]; then
  echo "Invalid array index ${SLURM_ARRAY_TASK_ID} for ${num_inputs} tasks" >&2
  exit 1
fi

input_csv=${INPUT_CSVS_ARRAY[$array_index]}
model_label=${MODEL_LABELS_ARRAY[$array_index]}

results_dir="${RESULTS_BASE}/${model_label}"
mkdir -p "${results_dir}"

cmd=(
  python quant_runner.py
  --input-csv "${input_csv}"
  --preds-csv none
  --ignore-cols "pred,label"
  --pred-cols "prob_0,prob_1,prob_2,prob_3,prob_4,prob_5,prob_6,prob_7,prob_8,prob_9"
  --backend faiss
  --frnn-metric linf
  --out-metric tv
  --epsilon 0.314
  --walltime 22:00:00
  --results-dir "${results_dir}"
  --display-stride 100
  --initial-k 64
  --frnn-threads 7
)

printf 'Running quant_runner on %s (%s)\n' "${input_csv}" "${model_label}" >&2
srun "${cmd[@]}"
