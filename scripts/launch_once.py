#!/usr/bin/env python3
"""Launch one attention kernel once, so ncu and compute-sanitizer have a single clean target.

Usage:  python scripts/launch_once.py [cuda|cutile] [N] [0|1]

The first call warms the compilation, the second is the one a profiler should record.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import bench                                                        # noqa: E402


def main() -> int:
    model_name = sys.argv[1] if len(sys.argv) > 1 else "cuda"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    causal = bool(int(sys.argv[3])) if len(sys.argv) > 3 else False

    model = bench.make_models([model_name])[0]
    q, k, v = bench.inputs(n)

    import cupy as cp
    model(q, k, v, causal)                 # warm: pays the compile
    cp.cuda.Device(0).synchronize()
    out = model(q, k, v, causal)           # the launch a profiler records
    cp.cuda.Device(0).synchronize()

    print(f"launched {model_name} N={n} causal={int(causal)} out={tuple(out.shape)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
