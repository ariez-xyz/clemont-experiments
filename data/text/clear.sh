#!/opt/homebrew/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

mapfile -t TARGETS < <(
  {
    find "$PROJECT_ROOT/data/text/amazon" \
      \( -name 'amazon-judge-*.csv' \
      -o -name 'amazon-judge-*.json' \
      -o -name 'amazon-judge-*_witness_revised_*.csv' \
      -o -name 'amazon-judge-*_witness_revised_*.json' \) \
      -type f -print

    find "$PROJECT_ROOT/data/text/toxic-chat" \
      \( -name 'toxic-chat-judge-*.csv' \
      -o -name 'toxic-chat-judge-*.json' \
      -o -name 'toxic-chat-judge-*_witness_revised_*.csv' \
      -o -name 'toxic-chat-judge-*_witness_revised_*.json' \) \
      -type f -print

    find "$PROJECT_ROOT/results/quantitative/text_amazon" \
      \( -name 'quant_run_*.json' \
      -o -name '*_report.pdf' \
      -o -name '*_report.md' \) \
      -type f -print 2>/dev/null || true

    find "$PROJECT_ROOT/results/quantitative/text_toxic_chat" \
      \( -name 'quant_run_*.json' \
      -o -name '*_report.pdf' \
      -o -name '*_report.md' \) \
      -type f -print 2>/dev/null || true

    find "$PROJECT_ROOT/results/quantitative/text_amazon" \
      -name '*_report_assets' -type d -print 2>/dev/null || true

    find "$PROJECT_ROOT/results/quantitative/text_toxic_chat" \
      -name '*_report_assets' -type d -print 2>/dev/null || true
  } | sort
)

if (( ${#TARGETS[@]} == 0 )); then
  echo "No generated text data or monitor artifacts found."
  exit 0
fi

echo "The following generated text data and monitor artifacts will be deleted:"
printf '  %s\n' "${TARGETS[@]}"
echo
read -r -p "Type DELETE to continue: " CONFIRM
if [[ "$CONFIRM" != "DELETE" ]]; then
  echo "Aborted."
  exit 1
fi

for path in "${TARGETS[@]}"; do
  if [[ -d "$path" ]]; then
    rm -rf "$path"
  else
    rm -f "$path"
  fi
done

find "$PROJECT_ROOT/results/quantitative/text_amazon" \
  -depth -type d -empty -delete 2>/dev/null || true
find "$PROJECT_ROOT/results/quantitative/text_toxic_chat" \
  -depth -type d -empty -delete 2>/dev/null || true

echo "Deleted ${#TARGETS[@]} artifact(s)."
