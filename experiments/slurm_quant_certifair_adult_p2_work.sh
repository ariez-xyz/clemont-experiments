#!/bin/bash
set -euo pipefail

export OMP_NUM_THREADS=7
export MKL_NUM_THREADS=7
export OPENBLAS_NUM_THREADS=7

if [[ -z "${RUN_CONFIG_FILE:-}" || -z "${RESULTS_BASE:-}" ]]; then
  echo "RUN_CONFIG_FILE and RESULTS_BASE must be exported" >&2
  exit 1
fi

if [[ ! -f "${RUN_CONFIG_FILE}" ]]; then
  echo "Run configuration file not found: ${RUN_CONFIG_FILE}" >&2
  exit 1
fi

array_index=$((SLURM_ARRAY_TASK_ID - 1))
if [[ ${array_index} -lt 0 ]]; then
  echo "Invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi

line=$(sed -n "$((array_index + 1))p" "${RUN_CONFIG_FILE}")
if [[ -z "${line}" ]]; then
  echo "No configuration for array index ${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi

IFS='|' read -r description input_csv subdir extra_args <<< "${line}"

if [[ -z "${input_csv}" || -z "${subdir}" ]]; then
  echo "Malformed configuration line: ${line}" >&2
  exit 1
fi

results_dir="${RESULTS_BASE}/${subdir}"
mkdir -p "${results_dir}"

cmd=(
  python quant_runner.py
  --input-csv "${input_csv}"
  --preds-csv none
  --pred-cols "p1(>50K),p0(<=50K)"
  --ignore-cols "row_id,pred,label"
  --results-dir "${results_dir}"
  --display-stride 5000
  --frnn-threads 7
  --frnn-metric l2
  --out-metric tv
  --static
  --walltime 12:00:00
)

if [[ -n "${extra_args}" ]]; then
  # shellcheck disable=SC2206
  extra_parts=( ${extra_args} )
  cmd+=("${extra_parts[@]}")
fi

printf 'Running %s\n' "${description}" >&2
printf 'Command: %s\n' "${cmd[*]}" >&2

srun "${cmd[@]}"
