#!/usr/bin/env python3
"""Run the benchmark 10 times per configuration and report the average, not a single p50.

A single run on a laptop reports p50 but hides drift. This repeats the whole timing measurement N
times, then reports the mean of the 10 p50s with its standard deviation, the min and the max. It
writes results/repeat_bench.csv so the numbers are traceable, and It does not touch results/benchmarks.csv, which is the curated table.

Usage:  .venv/bin/python scripts/repeat_bench.py [reps] [sizes] [models]
"""

import csv
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import numpy as np                                                  # noqa: E402
import cupy as cp                                                   # noqa: E402
import bench                                                        # noqa: E402

CEIL = 12.1          # measured f32-accumulate ceiling, TFLOP/s
OUT = ROOT / "results" / "repeat_bench.csv"


def main() -> int:
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    sizes = [int(s) for s in (sys.argv[2] if len(sys.argv) > 2
                              else "256,512,1024,2048").split(",")]
    which = [s.strip() for s in (sys.argv[3] if len(sys.argv) > 3
                                 else "cuda,cutile,unfused").split(",")]

    models = bench.make_models(which)
    flush = cp.zeros(8 * 1024 * 1024, cp.uint32)
    print(f"{reps} repetitions per configuration, models: {[m.name for m in models]}\n")

    rows = []
    header = f"{'model':<8}{'N':>6}{'causal':>7}{'mean us':>10}{'std':>8}{'min':>10}{'max':>10}" \
             f"{'TFLOP/s':>9}{'% ceiling':>10}"
    print(header)
    print("-" * len(header))

    for n in sizes:
        q, k, v = bench.inputs(n)
        ref = bench.reference(q, k, v, False)
        for causal in (False, True):
            allowed = bench.TOL * float(np.max(np.abs(ref)))
            for m in models:
                # Verify before timing, exactly like the main harness.
                probe = m(q, k, v, causal).astype(np.float32).get()
                err = float(np.max(np.abs(probe - ref))) if not causal else None
                if not causal and (err > allowed or not np.isfinite(probe).all()):
                    print(f"{m.name:<8}{n:>6}{0:>7}   REFUSED max_abs={err:.3e}")
                    continue

                p50s = []
                for _ in range(reps):
                    p50, _p90, _mn, _mx, _it = bench.timed(m, q, k, v, causal, flush,
                                                           min_seconds=0.5)
                    p50s.append(p50)

                mean = statistics.fmean(p50s)
                sd = statistics.pstdev(p50s) if len(p50s) > 1 else 0.0
                flops = 4.0 * bench.B * bench.H * n * n * bench.D * (0.5 if causal else 1.0)
                tflops = flops / (mean * 1e-6) / 1e12
                pct = 100.0 * tflops / CEIL
                print(f"{m.name:<8}{n:>6}{int(causal):>7}{mean:>10.1f}{sd:>8.1f}"
                      f"{min(p50s):>10.1f}{max(p50s):>10.1f}{tflops:>9.2f}{pct:>9.0f}%")
                rows.append(dict(model=m.name, N=n, causal=int(causal), reps=reps,
                                 mean_us=round(mean, 1), std_us=round(sd, 1),
                                 min_us=round(min(p50s), 1), max_us=round(max(p50s), 1),
                                 tflops=round(tflops, 2), pct_ceiling=round(pct, 1)))
        print()

    if rows:
        with OUT.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {OUT.relative_to(ROOT)}: {len(rows)} averaged rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
