// Fused attention forward, CUDA C++, version 2 (sm_86). CANDIDATE -- not the submitted kernel.
//
// The submitted kernel is flash.cu (v1): measured, oracle-verified, and faster than the cuTile
// kernel at N>=1024 non-causal. This file is the register-resident rewrite described in
// docs/HANDOVER-OP2.md section 9.3, kept here because its index logic is verified (see
// scripts/emulate_flash_v2.py) even though it has not been run through the oracle or timed.
//
// Changes from v1, each with its source in OPTIMIZATION_NOTES.md:
//   1. S, the softmax state and O stay in registers. v1 round-tripped S and O through shared memory
//      because a WMMA accumulator has an opaque layout. mma.sync exposes the layout, so the per-row
//      rescale can run on registers.
//   2. ldmatrix loads the Q, K and V fragments (v1 used wmma::load_matrix_sync).
//   3. K and V are double-buffered with cp.async, so the next tile loads while this one computes.
//   4. Only diagonal and tail tiles run the mask. Interior tiles run a branch-free path.
//   5. For causal attention, query tiles are launched in reverse order.
//
// Layout per warp: 16 query rows (BQ = 64, 4 warps). Block: 128 threads, 46080 B dynamic shared.
// Entry point and argument list match v1.
//
// Fragment mapping (PTX ISA, m16n8k16 .f16, lane = 4g + t):
//   A (16x16, row-major): reg i, half h -> row g + 8*(i&1), col 2t + h + 8*(i>>1)
//   B (16x8,  col-major): reg j, half h -> row 2t + h + 8*j,  col g
//   C/D (16x8):           elem c        -> row g + 8*(c>>1),  col 2t + (c&1)
// Verified empirically on this GPU by tmp/fragmap.cu (both A and B tests, 128/128 elements).

#include <cuda_fp16.h>

#define HD        64
#define BQ        64
#define BK        64
#define NTHREADS  128
#define SP        72        // padded row stride in halves; 16-byte aligned, conflict-free for ldmatrix
#define NEG       (-1e30f)  // finite sentinel; -inf would make exp2(m_old - m_new) NaN

#define SMEM_BYTES ((BQ + 4 * BK) * SP * 2)   // 46080

__device__ __forceinline__ unsigned int smem_u32(const void* p) {
    return static_cast<unsigned int>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ int imin_(int a, int b) { return a < b ? a : b; }

__device__ __forceinline__ void cp_async16(unsigned int dst, const void* src, bool valid) {
    const int n = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(dst), "l"(src), "r"(n) : "memory");
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}
template <int PENDING>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(PENDING) : "memory");
}

__device__ __forceinline__ void ldsm_x4(unsigned int* r, unsigned int addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void ldsm_x4_t(unsigned int* r, unsigned int addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}

__device__ __forceinline__ void mma16816(float* d, const unsigned int* a, unsigned int b0, unsigned int b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ unsigned int pack_h2(float lo, float hi) {
    __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<unsigned int*>(&h);
}

__device__ __forceinline__ float ex2f(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ void load_kv(const __half* __restrict__ K, const __half* __restrict__ V,
                                        size_t base, int N, int kt, __half* kbuf, __half* vbuf, int tid) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int c   = tid + i * NTHREADS;
        const int r   = c >> 3;
        const int col = (c & 7) * 8;
        const int gr  = kt * BK + r;
        const bool ok = gr < N;
        const size_t off = base + (size_t)(ok ? gr : 0) * HD + col;
        cp_async16(smem_u32(kbuf + r * SP + col), K + off, ok);
        cp_async16(smem_u32(vbuf + r * SP + col), V + off, ok);
    }
}

extern "C" __global__ void __launch_bounds__(NTHREADS, 2)
attn_fwd(const __half* __restrict__ Q, const __half* __restrict__ K, const __half* __restrict__ V,
         __half* __restrict__ O, int N, float scale_log2, int causal)
{
    extern __shared__ __align__(128) unsigned char smem_raw[];
    __half* sQ = reinterpret_cast<__half*>(smem_raw);

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g    = lane >> 2;
    const int t    = lane & 3;
    const int mi   = lane >> 3;

    const int nq = (N + BQ - 1) / BQ;
    const int qi = causal ? (nq - 1 - (int)blockIdx.x) : (int)blockIdx.x;
    const int bh = blockIdx.y;
    const int q0 = qi * BQ;
    const int nk = (N + BK - 1) / BK;
    const size_t base = (size_t)bh * N * HD;

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int c   = tid + i * NTHREADS;
        const int r   = c >> 3;
        const int col = (c & 7) * 8;
        const int gr  = q0 + r;
        const bool ok = gr < N;
        cp_async16(smem_u32(sQ + r * SP + col), Q + base + (size_t)(ok ? gr : 0) * HD + col, ok);
    }
    load_kv(K, V, base, N, 0, sQ + (BQ + 0 * BK) * SP, sQ + (BQ + 2 * BK) * SP, tid);
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();

    unsigned int qa[4][4];
    #pragma unroll
    for (int kk = 0; kk < 4; ++kk)
        ldsm_x4(qa[kk], smem_u32(sQ + (warp * 16 + (lane & 15)) * SP + kk * 16 + (lane >> 4) * 8));

    const int q_row0 = q0 + warp * 16;
    const int n_full = causal ? imin_((q0 + 1) / BK, N / BK) : N / BK;
    const int last   = causal ? imin_(nk, (q0 + BQ + BK - 1) / BK) : nk;

    float m_r[2] = {NEG, NEG};
    float l_r[2] = {0.f, 0.f};
    float acc[8][4];
    #pragma unroll
    for (int nd = 0; nd < 8; ++nd)
        #pragma unroll
        for (int c = 0; c < 4; ++c) acc[nd][c] = 0.f;

    for (int kt = 0; kt < last; ++kt) {
        const int buf = kt & 1;
        __half* kbuf = sQ + (BQ + buf * BK) * SP;
        __half* vbuf = sQ + (BQ + 2 * BK + buf * BK) * SP;

        if (kt + 1 < last)
            load_kv(K, V, base, N, kt + 1, sQ + (BQ + (buf ^ 1) * BK) * SP,
                    sQ + (BQ + 2 * BK + (buf ^ 1) * BK) * SP, tid);
        cp_async_commit();
        cp_async_wait<1>();
        __syncthreads();

        float s[8][4];
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            #pragma unroll
            for (int c = 0; c < 4; ++c) s[j][c] = 0.f;

        #pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            #pragma unroll
            for (int np = 0; np < 4; ++np) {
                unsigned int b[4];
                ldsm_x4(b, smem_u32(kbuf + (np * 16 + (mi >> 1) * 8 + (lane & 7)) * SP
                                         + kk * 16 + (mi & 1) * 8));
                mma16816(s[2 * np],     qa[kk], b[0], b[1]);
                mma16816(s[2 * np + 1], qa[kk], b[2], b[3]);
            }
        }

        const bool need_mask = kt >= n_full;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                float v = s[j][c] * scale_log2;
                if (need_mask) {
                    const int key = kt * BK + j * 8 + 2 * t + (c & 1);
                    const int row = q_row0 + g + 8 * (c >> 1);
                    const bool ok = (key < N) && (!causal || key <= row);
                    v = ok ? v : NEG;
                }
                s[j][c] = v;
            }
        }

        float mx[2] = {NEG, NEG};
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            mx[0] = fmaxf(mx[0], fmaxf(s[j][0], s[j][1]));
            mx[1] = fmaxf(mx[1], fmaxf(s[j][2], s[j][3]));
        }
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 1));
            mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 2));
        }

        float alpha[2], mnew[2];
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            mnew[r]  = fmaxf(m_r[r], mx[r]);
            alpha[r] = ex2f(m_r[r] - mnew[r]);
            m_r[r]   = mnew[r];
        }

        float ls[2] = {0.f, 0.f};
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                s[j][c] = ex2f(s[j][c] - mnew[c >> 1]);
                ls[c >> 1] += s[j][c];
            }
        }
        #pragma unroll
        for (int r = 0; r < 2; ++r) l_r[r] = l_r[r] * alpha[r] + ls[r];
        #pragma unroll
        for (int nd = 0; nd < 8; ++nd) {
            acc[nd][0] *= alpha[0];  acc[nd][1] *= alpha[0];
            acc[nd][2] *= alpha[1];  acc[nd][3] *= alpha[1];
        }

        unsigned int pa[4][4];
        #pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            pa[kk][0] = pack_h2(s[2 * kk][0],     s[2 * kk][1]);
            pa[kk][1] = pack_h2(s[2 * kk][2],     s[2 * kk][3]);
            pa[kk][2] = pack_h2(s[2 * kk + 1][0], s[2 * kk + 1][1]);
            pa[kk][3] = pack_h2(s[2 * kk + 1][2], s[2 * kk + 1][3]);
        }

        #pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            #pragma unroll
            for (int dp = 0; dp < 4; ++dp) {
                unsigned int vb[4];
                ldsm_x4_t(vb, smem_u32(vbuf + (kk * 16 + (mi & 1) * 8 + (lane & 7)) * SP
                                        + dp * 16 + (mi >> 1) * 8));
                mma16816(acc[2 * dp],     pa[kk], vb[0], vb[1]);
                mma16816(acc[2 * dp + 1], pa[kk], vb[2], vb[3]);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        l_r[r] += __shfl_xor_sync(0xffffffffu, l_r[r], 1);
        l_r[r] += __shfl_xor_sync(0xffffffffu, l_r[r], 2);
    }
    const float inv0 = 1.f / l_r[0];
    const float inv1 = 1.f / l_r[1];
    const int row0 = q_row0 + g;
    const int row1 = row0 + 8;
    #pragma unroll
    for (int nd = 0; nd < 8; ++nd) {
        const int col = nd * 8 + 2 * t;
        if (row0 < N)
            *reinterpret_cast<unsigned int*>(O + base + (size_t)row0 * HD + col)
                = pack_h2(acc[nd][0] * inv0, acc[nd][1] * inv0);
        if (row1 < N)
            *reinterpret_cast<unsigned int*>(O + base + (size_t)row1 * HD + col)
                = pack_h2(acc[nd][2] * inv1, acc[nd][3] * inv1);
    }
}
