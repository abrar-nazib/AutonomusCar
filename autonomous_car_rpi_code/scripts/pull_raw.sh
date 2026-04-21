#!/usr/bin/env bash
# Pull one raw (un-annotated) frame from the Pi's streamer and drop it into
# scripts/debug_out/raw.jpg. Used by local_dev.py as its default input.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$SCRIPT_DIR/debug_out"
mkdir -p "$OUT_DIR"

HOST="${PI_HOST:-raspberrypi.local}"
PORT="${PI_PORT:-8000}"
URL="http://$HOST:$PORT/raw.jpg"
DEST="$OUT_DIR/raw.jpg"

echo ">> GET $URL"
curl -sSf --max-time 5 "$URL" -o "$DEST"
echo ">> saved $DEST ($(stat -c%s "$DEST") bytes)"
