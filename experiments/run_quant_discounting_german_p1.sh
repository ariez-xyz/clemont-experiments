#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_DIR="$PROJECT_ROOT/data/Certifair/predictions_tacas"
RESULTS_DIR="$PROJECT_ROOT/results/quantitative/discounting/german"

mkdir -p "$RESULTS_DIR"

BASE_CSV="$DATA_DIR/german-base-P1-combined.csv"
FAIR_CSV="$DATA_DIR/german-global-P1-combined.csv"

for discount in 1 0.999 0.9975 0.995 0.99; do
  echo "[quant-runner] interpolating $BASE_CSV -> $FAIR_CSV (discount=$discount)"
  python "$SCRIPT_DIR/quant_runner.py" \
    --input-csv "$BASE_CSV" \
    --interpolate "$FAIR_CSV" \
    --interpolate-bins 5 \
    --discount-factor "$discount" \
    --pred-cols "p0(Bad Credit),p1(Good Credit)" \
    --ignore-cols "pred,label" \
    --max-n 1001 \
    --results-dir "$RESULTS_DIR" \
    --save-points \
    "$@"
done

for discount in 1 0.999 0.9975 0.995 0.99; do
  echo "[quant-runner] base-only $BASE_CSV (discount=$discount)"
  python "$SCRIPT_DIR/quant_runner.py" \
    --input-csv "$BASE_CSV" \
    --interpolate "$FAIR_CSV" \
    --interpolate-bins 5 \
    --interpolate-weights "0,0,0,0,0" \
    --discount-factor "$discount" \
    --pred-cols "p0(Bad Credit),p1(Good Credit)" \
    --ignore-cols "pred,label" \
    --max-n 1001 \
    --results-dir "$RESULTS_DIR" \
    --save-points \
    "$@"
done

for discount in 1 0.999 0.9975 0.995 0.99; do
  echo "[quant-runner] fair-only $BASE_CSV (discount=$discount)"
  python "$SCRIPT_DIR/quant_runner.py" \
    --input-csv "$BASE_CSV" \
    --interpolate "$FAIR_CSV" \
    --interpolate-bins 5 \
    --interpolate-weights "1,1,1,1,1" \
    --discount-factor "$discount" \
    --pred-cols "p0(Bad Credit),p1(Good Credit)" \
    --ignore-cols "pred,label" \
    --max-n 1001 \
    --results-dir "$RESULTS_DIR" \
    --save-points \
    "$@"
done
