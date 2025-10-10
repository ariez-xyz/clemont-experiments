#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_DIR="$PROJECT_ROOT/data/Certifair/predictions"
RESULTS_DIR="$PROJECT_ROOT/results/quantitative/certifair"

mkdir -p "$RESULTS_DIR"

declare -a DATASETS=(
  "adult-base-P2-combined.csv"
  "adult-global-P2-combined.csv"
)

for dataset in "${DATASETS[@]}"; do
  INPUT_PATH="$DATA_DIR/$dataset"
  echo "[quant-runner] processing $INPUT_PATH"
  python "$SCRIPT_DIR/quant_runner.py" \
    --input-csv "$INPUT_PATH" \
    --pred-cols "p1(>50K),p0(<=50K)" \
    --ignore-cols "row_id,pred,label" \
    --results-dir "$RESULTS_DIR" \
    --max-k 1024 \
    "$@"
done
