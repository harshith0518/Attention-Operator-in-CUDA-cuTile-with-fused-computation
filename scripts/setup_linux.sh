#!/usr/bin/env bash
# Set up the Linux (WSL2) environment for this project.
#
# Why Linux is now primary: the full CUDA toolkit is present at /usr/local/cuda-12.6, so nvcc,
# cuobjdump, nvdisasm and compute-sanitizer all work. On Windows the pip package had no libcu++
# headers, so only NVRTC was possible, and PyTorch would not install at all.
#
# Usage:  bash scripts/setup_linux.sh
#
# Creates ./.venv (Linux), installs the pinned intent file, then runs Gate A.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
VENV="$REPO/.venv"
PY="${PYTHON:-python3}"
LOG="$REPO/tmp/setup_linux.log"
mkdir -p "$REPO/tmp"

say() { printf '\n=== %s ===\n' "$1"; }

say "toolchain check"
if [ ! -x /usr/local/cuda-12.6/bin/nvcc ]; then
  echo "ERROR: /usr/local/cuda-12.6/bin/nvcc not found. Install the CUDA toolkit first." >&2
  exit 1
fi
echo "nvcc:    $(/usr/local/cuda-12.6/bin/nvcc --version | tail -1)"
echo "python:  $($PY -V 2>&1)"
echo "gpu:     $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"

# The project keeps the CUDA compilers reachable through the environment, because both CuPy and
# tileiras look them up rather than hard-coding a path.
export CUDA_HOME=/usr/local/cuda-12.6
export CUDA_PATH=$CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

say "create the venv"
if [ -d "$VENV" ] && [ -x "$VENV/bin/python" ]; then
  echo "existing Linux venv found, reusing it"
else
  if [ -d "$VENV" ]; then
    echo "a non-Linux venv is in the way; moving it to .venv-win"
    mv "$VENV" "$REPO/.venv-win"
  fi
  "$PY" -m venv "$VENV"
  echo "created $VENV"
fi

say "install"
"$VENV/bin/python" -m pip install --upgrade pip wheel 2>&1 | tail -2
if [ -f "$REPO/requirements.txt" ]; then
  "$VENV/bin/python" -m pip install -r "$REPO/requirements.txt" 2>&1 | tee -a "$LOG" | tail -12
else
  echo "ERROR: requirements.txt missing" >&2
  exit 1
fi

say "verify the stack"
"$VENV/bin/python" - <<'PY'
import numpy, PIL, cupy, cuda.tile as ct
print("numpy   ", numpy.__version__)
print("pillow  ", PIL.__version__)
print("cupy    ", cupy.__version__)
print("cuda.tile", getattr(ct, "__version__", "ok"))
props = cupy.cuda.runtime.getDeviceProperties(0)
free, total = cupy.cuda.Device(0).mem_info
print("device  ", props["name"].decode())
print(f"sm      {props['major']}.{props['minor']}  free {free/2**30:.1f} GiB of {total/2**30:.1f} GiB")
PY

say "Gate A"
"$VENV/bin/python" "$REPO/scripts/smoke_cutile.py" 2>&1 | tail -6

printf '\nsetup_linux.sh finished\n'
