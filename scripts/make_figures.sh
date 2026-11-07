#!/usr/bin/env bash
# Regenerate figs/*.png from results/results.csv.
#
# Pillow is installed in the project venv on Linux, so this is a plain call now. On Windows it
# needed a second interpreter because pip could not install Pillow there.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"

PY="$REPO/.venv/bin/python"
[ -x "$PY" ] || { echo "ERROR: run bash scripts/setup_linux.sh first" >&2; exit 1; }

CSV="$REPO/results/results.csv"
[ -f "$CSV" ] || { echo "ERROR: $CSV missing. Run bench/bench.py first." >&2; exit 1; }

exec "$PY" "$REPO/bench/bench.py" --figs-only
