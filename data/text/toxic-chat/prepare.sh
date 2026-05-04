#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "$SCRIPT_DIR/prepare_toxicity.py" \
  --classes 5 \
  --sample-size 100 \
  --seed 42 \
  --max-input-chars 4000
