#!/usr/bin/env bash
# Profile one attention kernel with Nsight Compute (Linux / WSL2).
#
# Usage:  bash scripts/profile.sh [cuda|cutile] [N] [0|1] [set]
#   e.g.  bash scripts/profile.sh cuda 2048 0 detailed
#
# On WSL2 the counters are usually refused (ERR_NVGPUCTRPERM) even when the Windows permission is
# on, because hardware counters are not officially supported under WSL. The script detects that
# and says so instead of writing an empty report.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"

MODEL="${1:-cuda}"
N="${2:-2048}"
CAUSAL="${3:-0}"
SET="${4:-detailed}"

CUDA_HOME=/usr/local/cuda-12.6
NCU="$CUDA_HOME/bin/ncu"
PY="$REPO/.venv/bin/python"

[ -x "$NCU" ] || { echo "ERROR: ncu not found at $NCU" >&2; exit 1; }
[ -x "$PY" ]  || { echo "ERROR: venv python not found. Run bash scripts/setup_linux.sh" >&2; exit 1; }

export CUDA_HOME CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
mkdir -p "$REPO/profiles"

OUT="$REPO/profiles/${MODEL}_n${N}_c${CAUSAL}"
echo "profiling $MODEL N=$N causal=$CAUSAL -> $OUT.ncu-rep"

set +e
"$NCU" --set "$SET" --clock-control none --cache-control none \
       --launch-skip 1 --launch-count 1 -f -o "$OUT" \
       "$PY" "$REPO/scripts/launch_once.py" "$MODEL" "$N" "$CAUSAL" 2>&1 | tee "$REPO/tmp/ncu.log"
RC=${PIPESTATUS[0]}
set -e

if grep -q ERR_NVGPUCTRPERM "$REPO/tmp/ncu.log" 2>/dev/null; then
  cat <<'MSG'

Counter collection was refused. Two causes, in order of likelihood:
  1. WSL2: hardware counters are not supported. Profile on the Windows host instead, or accept
     that DRAM bytes cannot be measured under WSL.
  2. Missing permission: on Windows, NVIDIA Control Panel -> Desktop ->
     Manage GPU Performance Counters -> allow this user.
Everything else (SASS, disassembly, compute-sanitizer) works fine in WSL.
MSG
  exit 2
fi

echo "report: $OUT.ncu-rep"
echo "read with: $NCU --import $OUT.ncu-rep --page details"
exit $RC
