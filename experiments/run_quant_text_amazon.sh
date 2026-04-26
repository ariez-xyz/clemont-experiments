#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

INPUT_CSV="${1:-$PROJECT_ROOT/data/text/amazon/amazon_sentiment_openrouter_20260424T150523Z.csv}"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/results/quantitative/text_amazon}"

if [[ ! -f "$INPUT_CSV" ]]; then
  echo "Input CSV not found: $INPUT_CSV" >&2
  echo "Usage: $0 [monitor-ready-amazon-csv] [extra quant_runner args...]" >&2
  exit 1
fi

mkdir -p "$RESULTS_DIR"

shift $(( $# > 0 ? 1 : 0 ))

INPUT_COLS="$(
  python - "$INPUT_CSV" <<'PY'
import csv
import re
import sys

with open(sys.argv[1], newline="") as fh:
    fieldnames = csv.DictReader(fh).fieldnames or []

embedding_cols = [
    name for name in fieldnames
    if re.fullmatch(r"e\d+", name)
]
embedding_cols.sort(key=lambda name: int(name[1:]))

if not embedding_cols:
    raise SystemExit("no embedding columns e0..eN found")

print(",".join(embedding_cols))
PY
)"

echo "[quant-runner] input: $INPUT_CSV"
echo "[quant-runner] embedding columns: $(awk -F, '{print NF}' <<< "$INPUT_COLS")"
echo "[quant-runner] results: $RESULTS_DIR"

python "$SCRIPT_DIR/quant_runner.py" \
  --input-csv "$INPUT_CSV" \
  --preds-csv none \
  --input-cols "$INPUT_COLS" \
  --pred-cols "prob_0,prob_1" \
  --ignore-cols "row_id,example_id,source_row,prompt_hash,rating,rating_value,country,review_date,review_title,review_text,judge_answer,first_token,first_token_logprob,top_logprobs_json,label_logprob_floor,judge_finish_reason,judge_model_returned,judge_response_id,judge_error,logprob_0,logprob_0_source,logprob_1,logprob_1_source" \
  --results-dir "$RESULTS_DIR" \
  --frnn-metric cosine \
  --out-metric tv \
  --backend faiss \
  --normalize \
  --display-stride 1000 \
  "$@"
