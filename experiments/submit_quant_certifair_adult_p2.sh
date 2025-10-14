#!/bin/bash
set -euo pipefail

FAIR_MODEL="../data/Certifair/predictions_tacas/adult-global-P2-combined.csv"
BASE_MODEL="../data/Certifair/predictions_tacas/adult-base-P2-combined.csv"
RESULTS_BASE="../results/quantitative/certifair"
WORK_SCRIPT="slurm_quant_certifair_adult_p2_work.sh"
CONFIG_FILE_NAME="certifair_adult_p2_runs.tsv"

FAIR_MODEL=$(realpath "${FAIR_MODEL}")
BASE_MODEL=$(realpath "${BASE_MODEL}")
RESULTS_BASE=$(realpath -m "${RESULTS_BASE}")
CONFIG_FILE="${RESULTS_BASE}/${CONFIG_FILE_NAME}"
LOGS_DIR="${RESULTS_BASE}/logs"

mkdir -p "${RESULTS_BASE}" "${LOGS_DIR}"

if [[ ! -f "${FAIR_MODEL}" ]]; then
  echo "Input CSV not found: ${FAIR_MODEL}" >&2
  exit 1
fi
if [[ ! -f "${BASE_MODEL}" ]]; then
  echo "Input CSV not found: ${BASE_MODEL}" >&2
  exit 1
fi

: > "${CONFIG_FILE}"

printf 'Fair model (no max-k)|%s|fair_no_maxk|\n' "${FAIR_MODEL}" >> "${CONFIG_FILE}"
printf 'Baseline model (no max-k)|%s|baseline_no_maxk|\n' "${BASE_MODEL}" >> "${CONFIG_FILE}"

for epsilon in $(seq -f "%.3f" 0.005 0.005 0.100); do
  epsilon_token=${epsilon/./p}
  printf 'Fair model epsilon %s (max-k 64)|%s|fair_eps_%s_maxk64|--epsilon %s --max-k 64\n' \
    "${epsilon}" "${FAIR_MODEL}" "${epsilon_token}" "${epsilon}" >> "${CONFIG_FILE}"
done

printf 'Fair model (k-grow-factor 1.05)|%s|fair_kgrow_1p05|--k-grow-factor 1.05\n' "${FAIR_MODEL}" >> "${CONFIG_FILE}"
printf 'Fair model (k-grow-factor 1.1)|%s|fair_kgrow_1p1|--k-grow-factor 1.1\n' "${FAIR_MODEL}" >> "${CONFIG_FILE}"
printf 'Fair model (k-grow-factor 1.2)|%s|fair_kgrow_1p2|--k-grow-factor 1.2\n' "${FAIR_MODEL}" >> "${CONFIG_FILE}"

NUM_TASKS=$(grep -c "" "${CONFIG_FILE}")
if [[ "${NUM_TASKS}" -eq 0 ]]; then
  echo "No runs configured" >&2
  exit 1
fi

unset SLURM_EXPORT_ENV
pushd .. >/dev/null
source activate.sh
popd >/dev/null

export RUN_CONFIG_FILE="${CONFIG_FILE}"
export RESULTS_BASE

sbatch \
  --job-name=quant_certifair \
  --output="${LOGS_DIR}/quant_certifair-%A-%a.log" \
  --cpus-per-task=8 \
  --time=14:00:00 \
  --mem=16G \
  --array=1-${NUM_TASKS} \
  --export=ALL,RUN_CONFIG_FILE,RESULTS_BASE \
  "${WORK_SCRIPT}"
