#!/usr/bin/env python3
"""Cross-check that v1, v2 and cuTile produce the same answer, since v2 is now the shipped kernel."""

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
    models = {m.name: m for m in B.make_models(["cuda", "cuda2", "cutile"])}
    print("  " + ", ".join(models))
    worst = 0.0
    for n in (256, 1024, 2048):
        for causal in (0, 1):
            outs = {}
            for name, m in models.items():
                q, k, v = B.inputs(n)
                outs[name] = cp.asnumpy(m(q, k, v, bool(causal))).astype(np.float64)
            d = float(np.max(np.abs(outs["cuda"] - outs["cuda2"])))
            c = float(np.max(np.abs(outs["cuda2"] - outs["cutile"])))
            worst = max(worst, d, c)
            print(f"  N={n:5d} causal={causal}  |v1-v2|={d:.2e}  |v2-cuTile|={c:.2e}")
    print(f"\n  worst disagreement: {worst:.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
