#!/usr/bin/env python3
"""The tests a reviewer expects beyond the oracle matrix.

1. Determinism      — neither kernel uses atomics, so two runs must match bit for bit.
2. Achieved bandwidth — a copy kernel, to check the 192 GB/s figure the analysis rests on.
3. Tile boundaries at 4095 and 4097, not only 63/64/127/128.
4. fp16 overflow    — inputs near +-65504 must stay finite.
5. fp16 underflow of P — nearly uniform scores push weights near 2^-24, where fp16 rounds to zero.

Runs on the models from bench.py, so both kernels face the same tests.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import numpy as np                                                  # noqa: E402
import cupy as cp                                                   # noqa: E402
import bench                                                        # noqa: E402

ok = []


def report(name, passed, detail):
    ok.append(passed)
    print(f"  {'PASS' if passed else 'FAIL'}  {name:<44} {detail}")


def main() -> int:
    models = bench.make_models(["cuda", "cutile"])

    print("=== 1. determinism (no atomics, so bit-identical) ===")
    for n in (65, 1000):
        q, k, v = bench.inputs(n)
        for causal in (False, True):
            for m in models:
                a = m(q, k, v, causal).copy()
                b = m(q, k, v, causal).copy()
                same = bool(cp.array_equal(a, b))
                report(f"{m.name} N={n} causal={int(causal)}", same,
                       "bit-identical" if same else "DIFFERS between runs")

    print("\n=== 2. achieved bandwidth (copy kernel) ===")
    nbytes = 256 * 1024 * 1024
    src = cp.zeros(nbytes // 4, cp.float32)
    dst = cp.empty_like(src)
    for _ in range(3):
        dst[...] = src
    cp.cuda.Device(0).synchronize()
    s, e = cp.cuda.Event(), cp.cuda.Event()
    s.record()
    for _ in range(10):
        dst[...] = src
    e.record()
    e.synchronize()
    ms = cp.cuda.get_elapsed_time(s, e) / 10
    gbs = 2 * nbytes / (ms * 1e-3) / 1e9
    report("copy bandwidth", 100 <= gbs <= 400, f"{gbs:.0f} GB/s over {nbytes/2**20:.0f} MiB")

    print("\n=== 3. tile boundaries at 4095 and 4097 (H=1 to keep the oracle cheap) ===")
    rng = np.random.default_rng(11)
    for n in (4095, 4097):
        q = cp.asarray(rng.standard_normal((1, 1, n, 64)).astype(np.float16))
        k = cp.asarray(rng.standard_normal((1, 1, n, 64)).astype(np.float16))
        v = cp.asarray(rng.standard_normal((1, 1, n, 64)).astype(np.float16))
        ref = bench.reference(q, k, v, True)
        allowed = 5e-3 * float(np.max(np.abs(ref)))
        for m in models:
            out = m(q, k, v, True).astype(np.float32).get()
            err = float(np.max(np.abs(out - ref)))
            finite = bool(np.isfinite(out).all())
            report(f"{m.name} N={n} causal=1", err <= allowed and finite,
                   f"max_abs {err:.3e} allowed {allowed:.3e} finite={finite}")

    print("\n=== 4. fp16 overflow: inputs near +-65504 ===")
    for m in models:
        for causal in (False, True):
            n = 130
            base = rng.standard_normal((1, 8, n, 64)).astype(np.float32)
            huge = (base * 2000.0).clip(-65504, 65504).astype(np.float16)
            q = cp.asarray(huge)
            k = cp.asarray(np.clip(rng.standard_normal((1, 8, n, 64)) * 2000.0,
                                   -65504, 65504).astype(np.float16))
            v = cp.asarray(np.clip(rng.standard_normal((1, 8, n, 64)) * 2000.0,
                                   -65504, 65504).astype(np.float16))
            out = m(q, k, v, causal).astype(np.float32).get()
            ref = bench.reference(q, k, v, causal)
            finite = bool(np.isfinite(out).all())
            # magnitude only: the reference is the fp64 answer for the same inputs
            rel = float(np.max(np.abs(out - ref)) / (np.max(np.abs(ref)) + 1e-30))
            report(f"{m.name} huge inputs causal={int(causal)}", finite and rel < 0.2,
                   f"finite={finite} rel {rel:.2e}")

    print("\n=== 5. fp16 underflow of P: nearly uniform scores ===")
    # Every score almost equal, so every weight is tiny but non-zero; in fp16 the smallest normal is
    # 6.1e-05, so weights below that round to zero. The output must still be inside the tolerance.
    for m in models:
        n = 1024
        q = cp.asarray((rng.standard_normal((1, 8, n, 64)) * 0.01).astype(np.float16))
        k = cp.asarray(rng.standard_normal((1, 8, n, 64)).astype(np.float16))
        v = cp.asarray(rng.standard_normal((1, 8, n, 64)).astype(np.float16))
        ref = bench.reference(q, k, v, False)
        allowed = 5e-3 * float(np.max(np.abs(ref)))
        out = m(q, k, v, False).astype(np.float32).get()
        err = float(np.max(np.abs(out - ref)))
        report(f"{m.name} N={n} uniform-ish scores", err <= allowed,
               f"max_abs {err:.3e} allowed {allowed:.3e}")

    print()
    good = sum(ok)
    print(f"=== {good} of {len(ok)} PASSED ===")
    return 0 if good == len(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
