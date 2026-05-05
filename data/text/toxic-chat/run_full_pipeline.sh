#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "[pipeline:toxic-chat] prepare baseline dataset"
"$SCRIPT_DIR/prepare.sh"

echo "[pipeline:toxic-chat] monitor baseline dataset"
"$PROJECT_ROOT/experiments/run_quant_text_toxic_chat_5class_4b_probs.sh"

echo "[pipeline:toxic-chat] prepare monitor-informed revision dataset"
"$SCRIPT_DIR/prepare_revision.sh"

echo "[pipeline:toxic-chat] monitor revised dataset"
"$PROJECT_ROOT/experiments/run_quant_text_toxic_chat_5class_4b_probs_revised.sh"
