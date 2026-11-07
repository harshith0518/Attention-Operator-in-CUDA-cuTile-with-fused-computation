"""Fused attention forward pass in cuTile (tile level).

One kernel per launch: QK^T, online softmax and PV stay on chip. The N x N score matrix never
reaches DRAM. This is the tile-level half of the comparison; the thread-level half is
kernels/cuda-fused-attn/flash_v2.cu.

Numerics: fp32 state, one folded base-2 scale, alpha applied once before the accumulate, p masked
twice (once on the score, once on the weight), a finite sentinel instead of -inf, and the tail
predicate built from indices.

Run:  .venv/bin/python kernels/cutile-fused-attn/flash.py
"""

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _env, _sub in (("CUPY_CACHE_DIR", "cupy"), ("CUDA_TILE_CACHE_DIR", "cuda_tile")):
    _dir = _ROOT / "tmp" / _sub
    _dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(_env, str(_dir))
_TEMP = _ROOT / "tmp" / "temp"
_TEMP.mkdir(parents=True, exist_ok=True)
for _env in ("TEMP", "TMP"):
    os.environ[_env] = str(_TEMP)

import numpy as np
import cupy as cp
import cuda.tile as ct

NEG = -1e30          # finite sentinel, fp32. -inf would poison the first rescale.


@ct.kernel
def _attn_fwd(Q, K, V, O, N, scale, causal: ct.Constant[bool],
              BQ: ct.Constant[int], BK: ct.Constant[int], D: ct.Constant[int]):
    qi, bh = ct.bid(0), ct.bid(1)
    rows = ct.arange(BQ, dtype=ct.int32)[None, :, None] + qi * BQ      # (1,BQ,1) absolute rows
    q = ct.load(Q, (bh, qi, 0), shape=(1, BQ, D), padding_mode=ct.PaddingMode.ZERO)

    m = ct.full((1, BQ, 1), NEG, ct.float32)          # keepdims: shapes never change in the loop
    l = ct.zeros((1, BQ, 1), ct.float32)
    acc = ct.zeros((1, BQ, D), ct.float32)

    nk = (N + BK - 1) // BK
    # Causal: a query tile never needs a K tile beyond its own diagonal. Rows
    # [qi*BQ, qi*BQ+BQ) can only see keys below (qi+1)*BQ, so the last useful tile is
    # ceil((qi+1)*BQ / BK). Written out here rather than left to the compiler, so the report
    # describes what the source asks for.
    if causal:
        last = min(nk, ((qi + 1) * BQ + BK - 1) // BK)
    else:
        last = nk
    for ki in range(last):
        k_t = ct.load(K, (bh, ki, 0), shape=(1, BK, D), padding_mode=ct.PaddingMode.ZERO)
        s = ct.mma(q, ct.transpose(k_t, 1, 2), ct.zeros((1, BQ, BK), ct.float32)) * scale

        cols = ct.arange(BK, dtype=ct.int32)[None, None, :] + ki * BK  # (1,1,BK) absolute keys
        keep = cols < N                                               # tail, every path
        if causal:
            keep = keep & (cols <= rows)                              # diagonal, every path
        s = ct.where(keep, s, NEG)

        m_new = ct.maximum(m, ct.max(s, axis=2, keepdims=True))
        p = ct.exp2(s - m_new) * keep          # * keep kills the phantom weight of a masked row
        alpha = ct.exp2(m - m_new)
        l = l * alpha + ct.sum(p, axis=2, keepdims=True)
        v_t = ct.load(V, (bh, ki, 0), shape=(1, BK, D), padding_mode=ct.PaddingMode.ZERO)
        acc = ct.mma(p.astype(ct.float16), v_t, acc * alpha)           # rescale BEFORE accumulate
        m = m_new

    out = (acc / l).astype(O.dtype)
    ct.store(O, (bh, qi, 0), out)


def attn(q, k, v, causal=False, BQ=64, BK=64, out=None):
    """q,k,v: cupy fp16 [B,H,N,D] -> O fp16 [B,H,N,D].

    The launch is asynchronous, like any CuPy call. Read the result with `.get()` or another
    synchronising operation. Do not add a device sync here: it would make every call block, and the
    CUDA model in bench.py does not pay that cost, so the comparison would be unfair.

    Pass `out` to reuse a buffer, so the timed path does no allocation.
    """
    B, H, N, D = q.shape
    scale = float(np.log2(np.e) / np.sqrt(D))       # base-2 softmax, one exp2
    qf = cp.ascontiguousarray(q.reshape(B * H, N, D))
    kf = cp.ascontiguousarray(k.reshape(B * H, N, D))
    vf = cp.ascontiguousarray(v.reshape(B * H, N, D))
    if out is None:
        out = cp.zeros((B * H, N, D), dtype=cp.float16)
    grid = ((N + BQ - 1) // BQ, B * H, 1)
    ct.launch(cp.cuda.get_current_stream(), grid, _attn_fwd,
              (qf, kf, vf, out, N, scale, causal, BQ, BK, D))
    return out.reshape(B, H, N, D)


def reference(q, k, v, causal=False):
    """fp64 oracle on the CPU, one head at a time. Direct: no online softmax, so it cannot share
    a bug with the kernel."""
    q = q.astype(np.float64).get() if hasattr(q, "get") else np.asarray(q, np.float64)
    k = k.astype(np.float64).get() if hasattr(k, "get") else np.asarray(k, np.float64)
    v = v.astype(np.float64).get() if hasattr(v, "get") else np.asarray(v, np.float64)
    B, H, N, D = q.shape
    out = np.zeros((B, H, N, D), dtype=np.float64)
    scale = 1.0 / np.sqrt(D)
    for b in range(B):
        for h in range(H):
            s = (q[b, h] @ k[b, h].T) * scale
            if causal:
                s = np.where(np.tril(np.ones((N, N), dtype=bool)), s, -np.inf)
            s = s - s.max(axis=1, keepdims=True)
            p = np.exp(s)
            p = p / p.sum(axis=1, keepdims=True)
            out[b, h] = p @ v[b, h]
    return out


def check(N, H=1, causal=False, tol=5e-3, seed=17, B=1, D=64):
    rng = np.random.default_rng(seed)
    q = cp.asarray(rng.standard_normal((B, H, N, D)).astype(np.float16))
    k = cp.asarray(rng.standard_normal((B, H, N, D)).astype(np.float16))
    v = cp.asarray(rng.standard_normal((B, H, N, D)).astype(np.float16))
    out = attn(q, k, v, causal=causal)
    fin = bool(cp.isfinite(out).all())
    ref = reference(q, k, v, causal=causal)
    err = float(np.max(np.abs(out.astype(np.float64).get() - ref)))
    allowed = tol * float(np.max(np.abs(ref)))
    ok = fin and err <= allowed
    print(f"  N={N:5d} H={H} causal={int(causal)}  max_abs={err:.3e}  allowed={allowed:.3e}  "
          f"finite={fin}  {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print("cuTile fused attention, fp16 in / fp32 accumulate")
    ok = True
    ok &= check(64, causal=False)
    ok &= check(64, causal=True)
    ok &= check(256, causal=False)
    ok &= check(256, causal=True)
    ok &= check(1000, causal=False)
    ok &= check(1000, causal=True)
    ok &= check(256, H=8, causal=False)
    print("ALL PASS" if ok else "FAILURES PRESENT")
    sys.exit(0 if ok else 1)
