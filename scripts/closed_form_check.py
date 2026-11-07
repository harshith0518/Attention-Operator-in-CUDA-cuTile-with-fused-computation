#!/usr/bin/env python3
"""Closed-form checks that do not use the fp64 oracle.

The oracle and both kernels share the same causal convention, so an off-by-one in that convention
would pass all three. These cases have an answer that follows from the maths alone:

  1. V = all ones          -> every output must be exactly 1, for any Q, K and mask.
                              Also catches a mismatch between the fp32 row sum and the fp16 P.
  2. Q = 0                 -> every score is 0, so the weights are uniform and the output is the
                              running mean of V over the allowed keys.
  3. N = 1                 -> one key, weight 1, so the output equals V[0].
  4. one huge score        -> one-hot weights, so the output equals that row of V.

Also covers the tile boundaries that the main matrix skips: N = 63, 64, 127, 128.

Run:  .venv/bin/python scripts/closed_form_check.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import numpy as np                                                  # noqa: E402
import cupy as cp                                                   # noqa: E402
import bench                                                        # noqa: E402

tensor = cp.asarray

TOL = 5e-3
results = []


def check(name, out, expected, note=""):
    err = float(np.max(np.abs(out - expected)))
    scale = max(1.0, float(np.max(np.abs(expected))))
    ok = err <= TOL * scale and np.isfinite(out).all()
    results.append(ok)
    flag = "PASS" if ok else "FAIL"
    print(f"  {flag}  {name:<46} max_abs={err:.3e}  allowed={TOL * scale:.3e} {note}")


def run(models, q, k, v, causal):
    """All models must be given the same inputs; returns {model: output}."""
    out = {}
    for m in models:
        o = m(q, k, v, causal)
        out[m.name] = np.asarray(o.astype(np.float32).get() if hasattr(o, "get") else o,
                                 dtype=np.float32)
    return out


def main() -> int:
    models = bench.make_models(["cuda", "cutile"])
    rng = np.random.default_rng(7)

    print("=== closed-form checks, both models, both masks ===")

    # 1. V = ones  ->  output is exactly 1 (the weights sum to 1 by construction)
    for n in (64, 1000):
        for causal in (False, True):
            q = tensor(rng.standard_normal((1, 8, n, 64)).astype(np.float16))
            k = tensor(rng.standard_normal((1, 8, n, 64)).astype(np.float16))
            v = tensor(np.ones((1, 8, n, 64), np.float16))
            for name, out in run(models, q, k, v, causal).items():
                check(f"V=1, N={n}, causal={int(causal)} [{name}]", out,
                      np.ones_like(out), "weights must sum to 1")

    # 2. Q = 0  ->  uniform weights over the allowed keys, so the output is a running mean
    for n in (64, 128):
        for causal in (False, True):
            q = tensor(np.zeros((1, 8, n, 64), np.float16))
            k = tensor(rng.standard_normal((1, 8, n, 64)).astype(np.float16))
            v = tensor(rng.standard_normal((1, 8, n, 64)).astype(np.float16))
            vf = v.get().astype(np.float64)[0]        # drop the batch dim: B = 1
            rows = np.arange(n)[:, None]
            keys = np.arange(n)[None, :]
            allowed = np.ones((n, n), bool) if not causal else (keys <= rows)
            expected = np.einsum("ij,hjd->hid", allowed / allowed.sum(1, keepdims=True), vf)[None]
            for name, out in run(models, q, k, v, causal).items():
                check(f"Q=0, N={n}, causal={int(causal)} [{name}]", out,
                      expected.astype(np.float32), "uniform weights")

    # 3. N = 1  ->  the single key gets weight 1, so the output is V[0]
    for causal in (False, True):
        q = tensor(rng.standard_normal((1, 8, 1, 64)).astype(np.float16))
        k = tensor(rng.standard_normal((1, 8, 1, 64)).astype(np.float16))
        v = tensor(rng.standard_normal((1, 8, 1, 64)).astype(np.float16))
        expected = np.repeat(v.get().astype(np.float32), 1, axis=2)
        for name, out in run(models, q, k, v, causal).items():
            check(f"N=1, causal={int(causal)} [{name}]", out, expected, "output equals V[0]")

    # 4. one huge score  ->  one-hot weights on that key.
    # Q must be non-zero: with Q = 0 every score is 0 whatever K is, so the test would be vacuous.
    # Only the non-causal case is used, because under causal row 0 can see key 0 alone and the
    # expected answer differs per row. The N=1 case above already covers the causal single key.
    n = 64
    for causal in (False,):
        q = tensor(np.ones((1, 8, n, 64), np.float16))
        k = tensor(np.zeros((1, 8, n, 64), np.float16))
        v_ = rng.standard_normal((1, 8, n, 64)).astype(np.float16)
        kk = k.get()
        kk[..., 1, :] = 40.0                     # key 1 scores far above the rest
        k = tensor(kk)
        v = tensor(v_)
        expected = np.repeat(v_.astype(np.float32)[:, :, 1:2, :], n, axis=2)
        for name, out in run(models, q, k, v, causal).items():
            check(f"one-hot key, N={n}, causal={int(causal)} [{name}]", out, expected,
                  "output equals V[1]")

    # 5. tile boundaries the main matrix skips
    print("\n=== tile boundaries against the fp64 oracle ===")
    for n in (63, 64, 127, 128):
        q, k, v = bench.inputs(n)
        for causal in (False, True):
            ref = bench.reference(q, k, v, causal)
            for name, out in run(models, q, k, v, causal).items():
                check(f"N={n}, causal={int(causal)} [{name}]", out, ref, "vs fp64 oracle")

    print()
    if all(results):
        print(f"=== ALL {len(results)} CLOSED-FORM CHECKS PASSED ===")
        return 0
    print(f"=== {results.count(False)} of {len(results)} CHECKS FAILED ===")
    return 1


if __name__ == "__main__":
    sys.exit(main())
