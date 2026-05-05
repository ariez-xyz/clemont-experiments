#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RESULTS_ROOT="$PROJECT_ROOT/results/quantitative/text_amazon/5class_4b_probs"

MONITOR_JSON="$(
  python - "$RESULTS_ROOT" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
paths = list(root.rglob("quant_run_*.json"))
if not paths:
    raise SystemExit(1)
latest = max(paths, key=lambda path: path.stat().st_mtime)
print(latest)
PY
)" || {
  echo "No monitor quant_run_*.json found under $RESULTS_ROOT" >&2
  exit 1
}

echo "[revision-prep] monitor: $MONITOR_JSON"
python "$SCRIPT_DIR/revise_from_monitor.py" "$MONITOR_JSON" --max-workers 32
