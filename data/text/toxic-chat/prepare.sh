#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "$SCRIPT_DIR/prepare_toxicity.py" \
  --classes 5 \
  --sample-size 10000 \
  --seed 42 \
  --max-input-chars 4000 \
  --max-workers 32
