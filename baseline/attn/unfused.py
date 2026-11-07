"""Materialised (unfused) attention: the baseline the fused kernels are compared against.

This is the honest "before" picture. It computes the same operator in three separate passes, and
every intermediate goes through global memory:

    1. S = Q @ K^T * scale          (cuBLAS, via cupy.matmul)   -> S written to global
    2. P = softmax_row(S)           (elementwise kernels)       -> P written to global
    3. O = P @ V                    (cuBLAS)                    -> O written to global

The fused kernels keep S and P on chip, so this baseline exists to measure what that costs. Traffic
accounting, at N=2048 with H=8:

    S fp32:  write 134 MB + read 134 MB
    P fp16:  write  67 MB + read  67 MB
    total    402 MB        against 8.4 MB for the fused form

Usage:
    from unfused import unfused_attn, traffic_bytes
    out = unfused_attn(q, k, v, causal=True)
"""

import numpy as np

try:
    import cupy as cp
except ImportError:                                        # pragma: no cover
    cp = None


def traffic_bytes(B, H, N, D, score_bytes=4, p_bytes=2):
    """Bytes moved by the materialised form, given the dtype of S and of P."""
    per_head = N * N
    io = 4 * B * H * N * D * 2                 # Q, K, V read once, O written once, fp16
    s = 2 * B * H * per_head * score_bytes     # write S, read S
    p = 2 * B * H * per_head * p_bytes         # write P, read P
    return io + s + p


def traffic_report(B=1, H=8, N=2048, D=64):
    fused = 4 * B * H * N * D * 2
    for label, sb in (("S fp32, P fp16", 4), ("S fp16, P fp16", 2)):
        t = traffic_bytes(B, H, N, D, sb)
        print(f"  {label:<16} {t / 1e6:7.1f} MB   vs fused {fused / 1e6:.1f} MB "
              f"  ratio {t / fused:5.1f}x")


def unfused_attn(q, k, v, causal=False, score_fp32=True, out=None):
    """q, k, v: cupy fp16 [B, H, N, D] -> O fp16 [B, H, N, D], computed without fusion."""
    B, H, N, D = q.shape
    scale = np.float32(1.0 / np.sqrt(D))

    qf = cp.ascontiguousarray(q.reshape(B * H, N, D))
    kf = cp.ascontiguousarray(k.reshape(B * H, N, D))
    vf = cp.ascontiguousarray(v.reshape(B * H, N, D))

    # 1. S = Q @ K^T, materialised. The operands are cast to fp32 because cupy.matmul on fp16
    #    inputs returns fp16, and fp16 accumulation overflows for large inputs (scores near 1.6e7 at
    #    a x100 input scale), which produced NaN. A tuned library would use fp16 operands with an
    #    fp32 accumulate; casting the operands costs extra traffic, so this baseline is an UPPER
    #    BOUND on the materialised time, not a floor.
    s = cp.matmul(qf.astype(cp.float32), kf.astype(cp.float32).transpose(0, 2, 1))
    s *= scale
    if causal:
        rows = cp.arange(N)[:, None]
        cols = cp.arange(N)[None, :]
        s = cp.where(cols <= rows, s, cp.float32(-1e30))
    if not score_fp32:
        s = s.astype(cp.float16)

    # 2. Softmax over the last axis, in fp32 arithmetic, then materialise P in fp16 for the second
    #    matmul. Two passes over S, which is exactly the traffic this baseline exists to show.
    m = s.max(axis=2, keepdims=True)
    e = cp.exp(s - m)
    p_sum = e.sum(axis=2, keepdims=True)
    p = (e / p_sum).astype(cp.float16)

    # 3. O = P @ V. Same fp32 reasoning: P @ V sums 2048 terms and overflows fp16 at large scales.
    o = cp.matmul(p.astype(cp.float32), vf.astype(cp.float32))

    if out is None:
        out = cp.empty((B * H, N, D), np.float16)
    cp.copyto(out, o.astype(np.float16))
    return out.reshape(B, H, N, D)


def reference(q, k, v, causal=False):
    """fp64 oracle, only used by the standalone self-check."""
    q = np.asarray(q, np.float64)
    k = np.asarray(k, np.float64)
    v = np.asarray(v, np.float64)
    B, H, N, D = q.shape
    out = np.empty((B, H, N, D), np.float64)
    for b in range(B):
        for h in range(H):
            s = q[b, h] @ k[b, h].T / np.sqrt(D)
            if causal:
                s = np.where(np.tril(np.ones((N, N), bool)), s, -np.inf)
            s = s - s.max(axis=1, keepdims=True)
            e = np.exp(s)
            out[b, h] = (e / e.sum(axis=1, keepdims=True)) @ v[b, h]
    return out


if __name__ == "__main__":
    print("materialised traffic at N=2048, H=8:")
    traffic_report()

    print("\nself-check against an fp64 oracle:")
    rng = np.random.default_rng(5)
    for n in (64, 130, 512):
        q = cp.asarray(rng.standard_normal((1, 8, n, 64)).astype(np.float16))
        k = cp.asarray(rng.standard_normal((1, 8, n, 64)).astype(np.float16))
        v = cp.asarray(rng.standard_normal((1, 8, n, 64)).astype(np.float16))
        for causal in (False, True):
            got = unfused_attn(q, k, v, causal).astype(np.float32).get()
            ref = reference(q.get(), k.get(), v.get(), causal)
            err = float(np.max(np.abs(got - ref)))
            allowed = 5e-3 * float(np.max(np.abs(ref)))
            print(f"  N={n:<5} causal={int(causal)}  max_abs {err:.3e}  "
                  f"allowed {allowed:.3e}  {'PASS' if err <= allowed else 'FAIL'}")
