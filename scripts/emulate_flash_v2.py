#!/usr/bin/env python3
"""Lane-level NumPy emulation of kernels/cuda-fused-attn/flash_v2.cu's index arithmetic.

It runs the same formulas the CUDA source uses, one warp at a time, lane by lane: ldmatrix addressing
and register contents (non-transposed for K and Q, transposed for V), the m16n8k16 fragment mapping for
A, B and C, the causal and tail masks, the online softmax with quad shuffles, the rescale, the P
packing, the final reduction and the store.

It does not model timing, cp.async or memory ordering -- those need the GPU harness. What it can catch
is a wrong index or lane formula, which is the error class a silently wrong answer usually comes from.
The mutation test at the end proves the check can fail.

Run:  .venv/bin/python scripts/emulate_flash_v2.py
"""

import sys

import numpy as np

BQ = BK = 64
HD = 64
SP = 72
NEG = np.float32(-1e30)


def ldsm_x4(smem, lane_addr, trans):
    """ldmatrix.x4. lane_addr[L] = (row, col) of the 8 halves lane L points at.

    Matrix = L >> 3, row within the 8x8 = L & 7. Without .trans lane L holds M[g][2t+h];
    with .trans it holds M[2t+h][g].
    """
    regs = np.zeros((32, 4, 2), np.float32)
    for mat in range(4):
        rows = []
        for r in range(8):
            R, C = lane_addr[mat * 8 + r]
            rows.append(smem[R, C:C + 8])
        m = np.stack(rows)
        for lane in range(32):
            g, t = lane >> 2, lane & 3
            for h in range(2):
                regs[lane, mat, h] = m[2 * t + h, g] if trans else m[g, 2 * t + h]
    return regs


def mma(a, b, c):
    """mma.sync.m16n8k16, fp16 in, fp32 accumulate. a[32,4,2], b[32,2,2], c[32,4] -> d[32,4]."""
    A = np.zeros((16, 16))
    B = np.zeros((16, 8))
    C = np.zeros((16, 8))
    for lane in range(32):
        g, t = lane >> 2, lane & 3
        for i in range(4):
            for h in range(2):
                A[g + 8 * (i & 1), 2 * t + h + 8 * (i >> 1)] = a[lane, i, h]
        for j in range(2):
            for h in range(2):
                B[2 * t + h + 8 * j, g] = b[lane, j, h]
        for cc in range(4):
            C[g + 8 * (cc >> 1), 2 * t + (cc & 1)] = c[lane, cc]
    D = A @ B + C
    d = np.zeros((32, 4), np.float32)
    for lane in range(32):
        g, t = lane >> 2, lane & 3
        for cc in range(4):
            d[lane, cc] = D[g + 8 * (cc >> 1), 2 * t + (cc & 1)]
    return d


def fp16(x):
    return np.asarray(x, np.float32).astype(np.float16).astype(np.float32)


def run_head(Q, K, V, causal, mutate_v_trans=False):
    """Emulate the whole grid for one head. Q, K, V: float32 [N, 64], fp16-representable."""
    N = Q.shape[0]
    nq = (N + BQ - 1) // BQ
    nk = (N + BK - 1) // BK
    out = np.zeros((N, HD), np.float32)
    lane = np.arange(32)
    g = lane >> 2
    t = lane & 3
    mi = lane >> 3
    scale_log2 = np.float32(np.log2(np.e) / np.sqrt(HD))

    for blk in range(nq):
        qi = (nq - 1 - blk) if causal else blk
        q0 = qi * BQ
        smem = np.zeros((BQ + 4 * BK, SP), np.float32)
        for r in range(BQ):
            gr = q0 + r
            if gr < N:
                smem[r, :HD] = Q[gr]

        def load_kv(kt, buf):
            for r in range(BK):
                gr = kt * BK + r
                ok = gr < N
                smem[BQ + buf * BK + r, :HD] = K[gr] if ok else 0.0
                smem[BQ + 2 * BK + buf * BK + r, :HD] = V[gr] if ok else 0.0

        n_full = min((q0 + 1) // BK, N // BK) if causal else N // BK
        last = min(nk, (q0 + BQ + BK - 1) // BK) if causal else nk

        warps = []
        for w in range(4):
            qa = [ldsm_x4(smem, [(w * 16 + (l & 15), kk * 16 + (l >> 4) * 8) for l in range(32)], False)
                  for kk in range(4)]
            warps.append({"qa": qa,
                          "m": np.full((32, 2), NEG, np.float32),
                          "l": np.zeros((32, 2), np.float32),
                          "acc": np.zeros((32, 8, 4), np.float32)})

        load_kv(0, 0)
        for kt in range(last):
            buf = kt & 1
            if kt + 1 < last:
                load_kv(kt + 1, buf ^ 1)
            kbase = BQ + buf * BK
            vbase = BQ + 2 * BK + buf * BK

            for w in range(4):
                st = warps[w]
                q_row0 = q0 + w * 16

                s = np.zeros((32, 8, 4), np.float32)
                for kk in range(4):
                    for npair in range(4):
                        addr = [(kbase + npair * 16 + ((l >> 3) >> 1) * 8 + (l & 7),
                                 kk * 16 + ((l >> 3) & 1) * 8) for l in range(32)]
                        b = ldsm_x4(smem, addr, False)
                        s[:, 2 * npair] = mma(st["qa"][kk], b[:, 0:2, :], s[:, 2 * npair])
                        s[:, 2 * npair + 1] = mma(st["qa"][kk], b[:, 2:4, :], s[:, 2 * npair + 1])

                need_mask = kt >= n_full
                for j in range(8):
                    for c in range(4):
                        v = s[:, j, c] * scale_log2
                        if need_mask:
                            key = kt * BK + j * 8 + 2 * t + (c & 1)
                            row = q_row0 + g + 8 * (c >> 1)
                            ok = key < N
                            if causal:
                                ok = ok & (key <= row)
                            v = np.where(ok, v, NEG)
                        s[:, j, c] = v

                mx = np.full((32, 2), NEG, np.float32)
                for j in range(8):
                    mx[:, 0] = np.maximum(mx[:, 0], np.maximum(s[:, j, 0], s[:, j, 1]))
                    mx[:, 1] = np.maximum(mx[:, 1], np.maximum(s[:, j, 2], s[:, j, 3]))
                for r in range(2):
                    mx[:, r] = np.maximum(mx[:, r], mx[lane ^ 1, r])
                    mx[:, r] = np.maximum(mx[:, r], mx[lane ^ 2, r])

                mnew = np.maximum(st["m"], mx)
                alpha = np.exp2(st["m"] - mnew).astype(np.float32)
                st["m"] = mnew

                ls = np.zeros((32, 2), np.float32)
                for j in range(8):
                    for c in range(4):
                        s[:, j, c] = np.exp2(s[:, j, c] - mnew[:, c >> 1]).astype(np.float32)
                        ls[:, c >> 1] += s[:, j, c]
                st["l"] = st["l"] * alpha + ls
                st["acc"] *= np.stack([alpha[:, 0], alpha[:, 0], alpha[:, 1], alpha[:, 1]],
                                      axis=1)[:, None, :]

                pa = []
                for kk in range(4):
                    regs = np.zeros((32, 4, 2), np.float32)
                    regs[:, 0, :] = fp16(s[:, 2 * kk, 0:2])
                    regs[:, 1, :] = fp16(s[:, 2 * kk, 2:4])
                    regs[:, 2, :] = fp16(s[:, 2 * kk + 1, 0:2])
                    regs[:, 3, :] = fp16(s[:, 2 * kk + 1, 2:4])
                    pa.append(regs)

                trans_v = not mutate_v_trans
                for kk in range(4):
                    for dp in range(4):
                        addr = [(vbase + kk * 16 + ((l >> 3) & 1) * 8 + (l & 7),
                                 dp * 16 + ((l >> 3) >> 1) * 8) for l in range(32)]
                        vb = ldsm_x4(smem, addr, trans_v)
                        st["acc"][:, 2 * dp] = mma(pa[kk], vb[:, 0:2, :], st["acc"][:, 2 * dp])
                        st["acc"][:, 2 * dp + 1] = mma(pa[kk], vb[:, 2:4, :], st["acc"][:, 2 * dp + 1])

        for w in range(4):
            st = warps[w]
            for r in range(2):
                st["l"][:, r] = st["l"][:, r] + st["l"][lane ^ 1, r]
                st["l"][:, r] = st["l"][:, r] + st["l"][lane ^ 2, r]
            inv = (1.0 / st["l"]).astype(np.float32)
            q_row0 = q0 + w * 16
            for ln in range(32):
                gg, tt = ln >> 2, ln & 3
                row0 = q_row0 + gg
                row1 = row0 + 8
                for nd in range(8):
                    col = nd * 8 + 2 * tt
                    if row0 < N:
                        out[row0, col] = st["acc"][ln, nd, 0] * inv[ln, 0]
                        out[row0, col + 1] = st["acc"][ln, nd, 1] * inv[ln, 0]
                    if row1 < N:
                        out[row1, col] = st["acc"][ln, nd, 2] * inv[ln, 1]
                        out[row1, col + 1] = st["acc"][ln, nd, 3] * inv[ln, 1]
    return out


def reference(Q, K, V, causal):
    N = Q.shape[0]
    s = (Q.astype(np.float64) @ K.astype(np.float64).T) / np.sqrt(HD)
    if causal:
        s = np.where(np.tril(np.ones((N, N), bool)), s, -np.inf)
    s = s - s.max(axis=1, keepdims=True)
    p = np.exp(s)
    p = p / p.sum(axis=1, keepdims=True)
    return p @ V.astype(np.float64)


def case(N, causal, rng, mutate=False, scale=2.5):
    Q = fp16(rng.standard_normal((N, HD)) * scale)
    K = fp16(rng.standard_normal((N, HD)) * scale)
    V = fp16(rng.standard_normal((N, HD)))
    got = run_head(Q, K, V, causal, mutate_v_trans=mutate)
    ref = reference(Q, K, V, causal)
    err = float(np.max(np.abs(got - ref)))
    allowed = 5e-3 * float(np.max(np.abs(ref)))
    finite = bool(np.isfinite(got).all())
    return (finite and err <= allowed), err, allowed


def main() -> int:
    rng = np.random.default_rng(7)
    all_ok = True
    print("lane-level emulation of flash_v2.cu")
    for N in (1, 63, 64, 65, 130, 200):
        for causal in (False, True):
            ok, err, allowed = case(N, causal, rng)
            all_ok &= ok
            print(f"  N={N:4d} causal={int(causal)}  max_abs={err:.3e}  allowed={allowed:.3e}  "
                  f"{'PASS' if ok else 'FAIL'}")

    ok, err, allowed = case(130, False, np.random.default_rng(7), mutate=True)
    detected = not ok
    print(f"\nmutation test (V read non-transposed; must FAIL): "
          f"max_abs={err:.3e} -> {'detected' if detected else 'NOT DETECTED'}")
    verdict = all_ok and detected
    print("\nALL PASS" if verdict else "\nFAILURES PRESENT")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
