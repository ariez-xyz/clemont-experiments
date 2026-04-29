#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
URL="http://$HOST:$PORT/index.html"

cd "$ROOT_DIR"

if command -v open >/dev/null 2>&1; then
  (sleep 1; open "$URL") &
elif command -v xdg-open >/dev/null 2>&1; then
  (sleep 1; xdg-open "$URL") &
else
  printf 'Open %s in your browser.\n' "$URL" >&2
fi

printf 'Serving %s at %s\n' "$ROOT_DIR" "$URL" >&2
python3 -m http.server "$PORT" --bind "$HOST"
