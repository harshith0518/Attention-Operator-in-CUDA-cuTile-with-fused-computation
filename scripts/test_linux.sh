#!/usr/bin/env bash
# Run the whole test ladder on Linux, in the same order as `make test`, and stop at the first failure.
# A timing taken on a kernel that fails the oracle is worthless, so nothing is timed until it passes.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
PY="$REPO/.venv/bin/python"

export CUDA_HOME="${CUDA:-/usr/local/cuda-12.6}"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CUPY_CACHE_DIR="$REPO/tmp/cupy"
mkdir -p "$CUPY_CACHE_DIR" "$REPO/tmp/cuda_tile"

fail() { printf '\nFAILED at: %s\n' "$1" >&2; exit 1; }

printf '\n===== 1. Gate A: does cuTile run on this machine =====\n'
"$PY" "$REPO/scripts/smoke_cutile.py" || fail "smoke_cutile.py"

printf '\n===== 2. cuTile kernel self-check =====\n'
"$PY" "$REPO/kernels/cutile-fused-attn/flash.py" || fail "kernels/cutile-fused-attn/flash.py"

printf '\n===== 3. closed-form checks, no oracle =====\n'
"$PY" "$REPO/scripts/closed_form_check.py" || fail "closed_form_check.py"

printf '\n===== 4. determinism, bandwidth, fp16 limits =====\n'
"$PY" "$REPO/scripts/extra_checks.py" || fail "extra_checks.py"

printf '\n===== 5. harness: oracle matrix =====\n'
"$PY" "$REPO/bench/bench.py" --tests-only || fail "bench.py --tests-only"

printf '\n===== 6. figures from the CSV =====\n'
"$PY" "$REPO/bench/bench.py" --figs-only || fail "bench.py --figs-only"

printf '\n===== ALL LINUX CHECKS PASSED =====\n'
