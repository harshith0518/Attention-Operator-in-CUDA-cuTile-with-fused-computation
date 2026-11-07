#!/usr/bin/env python3
"""Cross-check the shipped kernels against each other and against the materialised baseline.

Two different implementations agreeing is the strongest correctness evidence available, because a
shared bug across a register-resident mma.sync kernel and a compiler-generated tile kernel is
implausible. The baseline is compared too, but it is coarser: it rounds through fp32 matmuls that CuPy
runs separately, so a larger difference is expected there.

Run:  .venv/bin/python scripts/crosscheck.py
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("CUPY_CACHE_DIR", str(ROOT / "tmp" / "cupy"))
sys.path.insert(0, str(ROOT))

import numpy as np                                                   # noqa: E402
import cupy as cp                                                    # noqa: E402

import bench.bench as B                                              # noqa: E402


def main() -> int:
    models = {m.name: m for m in B.make_models(["cuda", "cutile", "unfused"])}
    print("  models: " + ", ".join(models))

    worst = 0.0
    print(f"\n  {'N':>6}{'causal':>7}{'|CUDA-cuTile|':>16}{'|CUDA-unfused|':>16}")
    for n in (256, 1024, 2048):
        for causal in (0, 1):
            outs = {}
            for name, model in models.items():
                q, k, v = B.inputs(n)
                outs[name] = cp.asnumpy(model(q, k, v, bool(causal))).astype(np.float64)
            d = float(np.max(np.abs(outs["cuda"] - outs["cutile"])))
            b = float(np.max(np.abs(outs["cuda"] - outs["unfused"])))
            worst = max(worst, d)
            print(f"  {n:>6}{causal:>7}{d:>16.2e}{b:>16.2e}")

    # the kernels must agree tightly; this is the claim the report makes
    ok = worst <= 5e-4
    print(f"\n  worst CUDA-versus-cuTile disagreement: {worst:.2e}  "
          f"({'within 5e-04, as the report states' if ok else 'ABOVE the 5e-04 the report claims'})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
