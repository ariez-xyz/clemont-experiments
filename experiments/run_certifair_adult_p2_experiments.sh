#!/usr/bin/env bash
# Runs a selection of quantitative monitor experiments on the Certifair Adult P2 datasets.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_DIR="$PROJECT_ROOT/data/Certifair/predictions"
RESULTS_DIR="$PROJECT_ROOT/results/quantitative/certifair"

FAIR_MODEL_CSV="$DATA_DIR/adult-global-P2-combined.csv"
BASE_MODEL_CSV="$DATA_DIR/adult-base-P2-combined.csv"

mkdir -p "$RESULTS_DIR"

EXTRA_ARGS=("$@")

run_quant() {
  local description=$1
  local run_subdir=$2
  shift 2
  local output_dir="$RESULTS_DIR/$run_subdir"
  mkdir -p "$output_dir"
  printf '\n[quant-runner] %s -> %s\n' "$description" "$output_dir" >&2
  if ((${#EXTRA_ARGS[@]})); then
    python "$SCRIPT_DIR/quant_runner.py" \
      --pred-cols "p1(>50K),p0(<=50K)" \
      --ignore-cols "row_id,pred,label" \
      --results-dir "$output_dir" \
      --shuffle \
      "$@" \
      "${EXTRA_ARGS[@]}"
  else
    python "$SCRIPT_DIR/quant_runner.py" \
      --pred-cols "p1(>50K),p0(<=50K)" \
      --ignore-cols "row_id,pred,label" \
      --results-dir "$output_dir" \
      --shuffle \
      "$@"
  fi
}

# 1. Fair model without max-k cap.
run_quant "fair model (no max-k)" "fair_no_maxk" \
  --input-csv "$FAIR_MODEL_CSV"

# 2. Baseline model without max-k cap.
run_quant "baseline model (no max-k)" "baseline_no_maxk" \
  --input-csv "$BASE_MODEL_CSV"

# 3. Epsilon sweep on fair model with max-k=64.
for epsilon in $(seq -f "%.3f" 0.005 0.005 0.100); do
  epsilon_token=${epsilon/./p}
  run_quant "fair model epsilon ${epsilon} (max-k 64)" "fair_eps_${epsilon_token}_maxk64" \
    --input-csv "$FAIR_MODEL_CSV" \
    --epsilon "$epsilon" \
    --max-k 64
done

# 4. Fair model with a small k-grow-factor.
run_quant "fair model (k-grow-factor 1.05)" "fair_kgrow_1p1" \
  --input-csv "$FAIR_MODEL_CSV" \
  --k-grow-factor 1.1
