#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

INPUT_PATH="$PROJECT_ROOT/data/toydata/cifar10-Bartoldson2024Adversarial_WRN-94-16.diff-adv-combined.csv"
RESULTS_DIR="$PROJECT_ROOT/results/quantitative/cifar10"

mkdir -p "$RESULTS_DIR"

PRED_COLS="p[airplane],p[automobile],p[bird],p[cat],p[deer],p[dog],p[frog],p[horse],p[ship],p[truck]"
IGNORE_COLS="pred,label,pred_name,label_name"

echo "[quant-runner] processing $INPUT_PATH"
python "$SCRIPT_DIR/quant_runner.py" \
  --input-csv "$INPUT_PATH" \
  --preds-csv none \
  --pred-cols "$PRED_COLS" \
  --ignore-cols "$IGNORE_COLS" \
  --results-dir "$RESULTS_DIR" \
  --frnn-metric linf \
  --backend faiss
  "$@"
