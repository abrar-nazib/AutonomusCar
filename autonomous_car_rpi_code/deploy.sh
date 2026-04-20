#!/usr/bin/env bash
# Send the autocar source tree to the Raspberry Pi via scp.
# Skips venv/ and __pycache__/. Does not copy itself.

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-abir@raspberrypi.local}"
REMOTE_DIR="${REMOTE_DIR:-~/autonomous_car_rpi_code}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ">> cleaning __pycache__ under src/ and tests/ ..."
find src tests -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

echo ">> ensuring remote dir exists: $REMOTE_HOST:$REMOTE_DIR"
ssh "$REMOTE_HOST" "mkdir -p $REMOTE_DIR"

echo ">> scp source tree to $REMOTE_HOST:$REMOTE_DIR"
scp -r \
    pyproject.toml \
    README.md \
    .gitignore \
    config \
    src \
    tests \
    "$REMOTE_HOST:$REMOTE_DIR/"

echo ">> done."
