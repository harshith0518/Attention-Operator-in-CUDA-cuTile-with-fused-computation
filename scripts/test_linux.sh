#!/usr/bin/env bash
# Run the project's test ladder on Linux, in the order PLAN.md section 9 requires.
# Stops at the first failure, because a timing on a failing kernel is worthless.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
PY="$REPO/.venv/bin/python"

export CUDA_HOME=/usr/local/cuda-12.6
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CUPY_CACHE_DIR="$REPO/tmp/cupy"
mkdir -p "$CUPY_CACHE_DIR" "$REPO/tmp/cuda_tile"

fail() { printf '\nFAILED at: %s\n' "$1" >&2; exit 1; }

printf '\n===== 1. Gate A: does cuTile run on this machine =====\n'
"$PY" "$REPO/scripts/smoke_cutile.py" || fail "smoke_cutile.py"
tail -3 "$REPO/tmp/smoke.log" 2>/dev/null

printf '\n===== 2. cuTile kernel self-check (8 cases) =====\n'
"$PY" "$REPO/kernels/cutile-fused-attn/flash.py" || fail "kernels/cutile-fused-attn/flash.py"

printf '\n===== 3. Harness: correctness then timing (quick) =====\n'
"$PY" "$REPO/bench/bench.py" --quick || fail "bench.py --quick"

printf '\n===== 4. Figures from the CSV =====\n'
"$PY" "$REPO/bench/bench.py" --figs-only || fail "bench.py --figs-only"

printf '\n===== ALL LINUX CHECKS PASSED =====\n'
