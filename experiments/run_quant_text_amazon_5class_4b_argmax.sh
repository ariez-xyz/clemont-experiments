#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

INPUT_CSV="$PROJECT_ROOT/data/text/amazon/amazon-judge-gemma-4-26b-a4b-it_embed-pplx-embed-v1-4b_temp-t0_5class_n2000.csv"
RESULTS_DIR="$PROJECT_ROOT/results/quantitative/text_amazon/5class_4b_argmax"

if [[ ! -f "$INPUT_CSV" ]]; then
  echo "Input CSV not found: $INPUT_CSV" >&2
  exit 1
fi

mkdir -p "$RESULTS_DIR"

INPUT_COLS="$(
  python - "$INPUT_CSV" <<'PY'
import csv
import re
import sys

with open(sys.argv[1], newline="") as fh:
    fieldnames = csv.DictReader(fh).fieldnames or []

embedding_cols = [name for name in fieldnames if re.fullmatch(r"e\d+", name)]
embedding_cols.sort(key=lambda name: int(name[1:]))

if not embedding_cols:
    raise SystemExit("no embedding columns e0..eN found")

print(",".join(embedding_cols))
PY
)"

PRED_COLS="prob_1,prob_2,prob_3,prob_4,prob_5"

echo "[quant-runner] input: $INPUT_CSV"
echo "[quant-runner] embedding columns: $(awk -F, '{print NF}' <<< "$INPUT_COLS")"
echo "[quant-runner] prediction columns: $PRED_COLS"
echo "[quant-runner] results: $RESULTS_DIR"

RUN_LOG="$(mktemp)"
trap 'rm -f "$RUN_LOG"' EXIT

python "$SCRIPT_DIR/quant_runner.py" \
  --input-csv "$INPUT_CSV" \
  --preds-csv none \
  --input-cols "$INPUT_COLS" \
  --pred-cols "$PRED_COLS" \
  --results-dir "$RESULTS_DIR" \
  --frnn-metric cosine \
  --out-metric linf \
  --output-transform argmax-normalized \
  --backend faiss \
  --normalize \
  --display-stride 1000 | tee "$RUN_LOG"

REPORT_JSON="$(sed -n 's/^Saved run to //p' "$RUN_LOG" | tail -n 1)"
if [[ -z "$REPORT_JSON" ]]; then
  echo "Could not determine monitor JSON path from quant_runner output" >&2
  exit 1
fi

echo "[report] generating PDF report for $REPORT_JSON"
python "$PROJECT_ROOT/results/quantitative/report_text_monitor.py" "$REPORT_JSON" --top-k 10
