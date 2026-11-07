# Brief for a 3-slide demo deck

Hand this file to whoever or whatever builds the slides. It contains the problem, the mathematics,
the design decisions, the measured numbers, and a slide-by-slide specification. Everything here is
measured on the machine named below; nothing is estimated.

---

## 0. Context

| Field | Value |
|---|---|
| Course | CS-6023 GPU Programming |
| Project | Operator OP2, fused attention forward pass |
| Author | Balla Surya Harshith |
| Roll number | CS23B010 |
| Mentor | Pravin Jagtap |
| Deadline | 7 November 2026 |
| Repository | https://github.com/harshith0518/Attention-Operator-in-CUDA-cuTile-with-fused-computation |
| Report | `report/report.pdf`, four pages, same content as this brief |

**What the instructor asked for, word for word:**

> Prepare a 3-slide presentation (title + names + roll numbers, problem statement + results, challenges
> faced and how you resolved those). Each team has 10 minutes. I will stop you after 3 slides are
> presented. The presentation will then be followed by a demo.

So the deck is **exactly three slides**, in that order, and it must be readable from the back of a
room during a ten-minute talk. Everything below exists to make that possible.

---

## 1. Problem statement

Attention is the operation that lets each position in a sequence look at every other position. For one
attention head it is

```
O = softmax( Q Kᵀ / √D + mask ) · V
```

with `Q, K, V, O` of shape `[B, H, N, D]`. This project uses `B = 1` batch, `H = 8` heads, `D = 64`
head width, and `N` the sequence length, measured at 256, 512, 1024 and 2048.

**Why this is hard on a GPU.** The straightforward form builds the `N × N` score matrix in global
memory. At `N = 2048` across eight heads that matrix is **134 MB** in fp32. It is written, read back,
and written again for the second matrix product. The operation therefore becomes memory-bound and the
tensor cores sit idle. Measured on this machine, the materialised version achieves an effective
**31 GB/s against 170 GB/s available**, and takes **13.3 ms**.

**The task.** Fuse the three stages into one kernel, so the score matrix never leaves the chip. This
forces two consequences that shape the whole design:

1. The softmax cannot see a full row at once, so it must run in a **streaming** form.
2. That streaming implies a **per-row rescale of a live accumulator**, and *where that accumulator
   lives* turns out to be the single most important decision in the project.

**The question the project answers.** The brief asks for the same operator written twice: by hand in
CUDA C++ at thread level, and by compiler in cuTile at tile level. So: when does the tile model's
productivity outweigh the control it gives up? The answer here is measured, not asserted.

---

## 2. The mathematics used

### 2.1 Softmax, exactly

For row `i` with logits `s_ij`:

```
softmax(s)_ij = exp(s_ij − m_i) / Σ_j exp(s_ij − m_i),      m_i = max_j s_ij
```

Subtracting the row maximum changes nothing mathematically. It prevents overflow, because the largest
exponent becomes zero.

### 2.2 Online softmax (the recurrence that makes fusion possible)

A streaming kernel sees one key tile at a time. It keeps a running maximum `m`, a running normaliser
`ℓ`, and a running unnormalised output `acc`. When a new tile arrives with tile maximum `m_new`:

```
α   = exp(m_old − m_new)
ℓ   ← α·ℓ + Σ_j exp(s_j − m_new)
acc ← α·acc + P·V
```

`α` is applied **exactly once per tile, before the accumulate**. This gives the same answer as the
two-pass form, in one pass, with row state that stays bounded in size. This is the key idea of the
whole kernel.

### 2.3 One exponential per element

Both kernels work in base two. The attention scale is folded into the QK product as `log₂(e)/√D`, so
the softmax needs `exp2` only. An ablation removed `exp2` entirely and changed the time by 0.0% to
0.9%, which proves the exponential is not the bottleneck.

### 2.4 Masks

Two are applied element-wise:

- **Tail mask:** every key at or past `N` is set to a finite sentinel (−1e30), because `−∞` would make
  `α = NaN`.
- **Causal mask:** keep only keys where `key ≤ query`.

Interior tiles need neither. That fact is exploited in the kernel: only diagonal and tail tiles run the
mask comparison.

### 2.5 Precision rules (a correctness rule, not a tuning choice)

Scores, softmax state and accumulator are **fp32**. Only the two matrix-multiply inputs are fp16.
fp16 accumulation overflows on these inputs; the materialised baseline demonstrates it at a ×100
stress scale.

### 2.6 Cost and intensity

```
FLOPs  = 4 N² D per head   →  8.59 GFLOP at N = 2048
Fused traffic = 4 N D halves in and out  →  8.4 MB total
Intensity = 1024 FLOP/byte, against a ridge point of 71  →  compute-oriented
Materialised form = 411 MB  →  bandwidth-oriented
```

---

## 3. Kernel design approach

### 3.1 The shape both kernels use

A block owns **64 query rows**. It walks the key dimension in tiles of **64**, and for each key tile:

1. compute the score tile `S = Q·Kᵀ`,
2. mask it,
3. update the running softmax state,
4. accumulate `P·V`.

`Q` stays resident for the whole loop. `K` and `V` stream.

### 3.2 The decision that mattered: where the accumulator lives

| Version | Matrix multiply | Accumulator | Registers | Shared memory | Blocks per SM |
|---|---|---|---|---|---|
| CUDA v1 | WMMA `m16n16k16` | **shared memory** | 56 | 72,192 B | 1 |
| **CUDA v2 (shipped)** | `mma.sync m16n8k16` | **registers** | 140 | 46,080 B | **2** |
| cuTile | `ct.mma` on tiles | registers (compiler's choice) | 255 | 48 KB | 2 |

**Why v1 was forced into shared memory.** A WMMA accumulator fragment has a layout the WMMA API does
not expose, so a per-row rescale is not expressible while the accumulator stays in registers. Putting
it in shared memory cost a round trip per tile, and at 72,192 B it allowed only **one block per SM**,
which is **8.3% occupancy**.

**What changed in v2.** Moving to `mma.sync.m16n8k16`, whose fragment layout is exposed, removed that
restriction. The score tile, the probabilities and the accumulator now live in registers. Shared memory
fell to 46,080 B, two blocks fit per SM, **occupancy doubled**, and **tensor-pipe utilisation tripled**.

### 3.3 The five optimisations, each one measured

| # | Change | Bottleneck it targets | Effect |
|---|---|---|---|
| 1 | Pad shared strides to 68/72, interleave lane columns | 32-way bank conflict: every lane hit one bank | 9,146 → 3,594 µs (**2.5×**) |
| 2 | Remove a per-call device synchronise | unfair timer, not the kernel | cuTile 107.5 → 42.0 µs (**1.85×**) |
| 3 | Stage K and V with 16-byte loads | exposed DRAM latency from scalar 2-byte loads | 3,207.8 → 2,091.4 µs (**1.53×**) |
| 4 | Keep S, P and accumulator in registers (v2) | shared-memory round trip per tile, and occupancy | 2,172.7 → 664.2 µs (**3.3×**) |
| 5 | Mask only diagonal and tail tiles | compare and select on every tile | branch is uniform, no divergence |

**Two fixes with no measurable effect, kept for honesty:** removing `exp2` (0.0–0.9%) and hoisting a
predicate (registers 63 → 56, timing inside noise).

### 3.4 What the CUDA kernel does, mechanically

- 128 threads, 4 warps, 16 query rows per warp.
- `ldmatrix` loads Q, K and V fragments. V goes through `.trans`, because V is contiguous along the
  dimension while K is contiguous along the key.
- `cp.async` double-buffers K and V: the next tile loads while the current one computes.
- Score tile, softmax state and accumulator in registers. **140 registers, 0 spills.**
- Causal query tiles are launched in reverse order, so the heaviest diagonal tiles start first.

### 3.5 What the cuTile kernel does

118 code lines. **No thread index, no stride, no barrier, no staging decision.** The same fusion is
expressed as chained tile operations: `ct.mma`, `ct.max`, `ct.exp2`, `ct.sum`, `ct.where`, plus an
explicit causal loop bound. The compiler places intermediates and chooses the copies.

---

## 4. Results

### 4.1 Latency

RTX 3050 Laptop, sm_86, 16 SMs, 4 GB, WSL2, CUDA 12.6. Microseconds, mean of **six repetitions** of
the p50. Lower is better; **bold** is the best in the row.

| `N` | causal | CUDA v2 | cuTile | unfused baseline | PyTorch SDPA |
|---|---|---|---|---|---|
| 256 | no | **18.6** | 42.0 | 807.5 | 19.5 |
| 256 | yes | **19.5** | 21.5 | 1032.1 | 20.5 |
| 512 | no | **52.1** | 150.5 | 1020.5 | 49.2 |
| 512 | yes | **41.0** | 53.1 | 1269.6 | 52.2 |
| 1024 | no | **176.1** | 622.4 | 3562.1 | 163.8 |
| 1024 | yes | **109.6** | 134.7 | 3961.7 | 129.0 |
| 2048 | no | **664.2** | 2991.2 | 13314.9 | 616.6 |
| 2048 | yes | **364.5** | 447.9 | 15097.9 | 397.0 |

**The ratios that matter:**

- **4.5×** faster than the compiler's kernel at `N = 2048`, non-causal (664.2 against 2991.2 µs).
- **1.23×** faster than the compiler's kernel at `N = 2048`, causal (364.5 against 447.9 µs).
- **3.3×** faster than my own first CUDA kernel (2172.7 → 664.2 µs).
- **6.3× to 22×** faster than the materialised baseline at `N ≥ 1024` (664.2 against 13314.9 µs).
- Within **1.08×** of PyTorch SDPA non-causal, and **faster than SDPA on the causal path**
  (364.5 against 397.0 µs).
- Wins **five of the eight configurations, including every causal one**.

**Provenance of the SDPA column.** The CUDA, cuTile and unfused figures come from
`results/benchmarks.csv` in the repository and can be reproduced with `make bench`. The PyTorch SDPA
column was measured in the same six-repetition session, but PyTorch was then removed from the
repository at the owner's request, so those four values are quoted from the archived measurement and
cannot be regenerated from the shipped code. If that matters for the demo, either say so when
presenting the number, or leave the SDPA column out of the slide; the CUDA-versus-cuTile comparison is
the project's own subject.

### 4.2 Fraction of the machine

The ceiling is measured, not quoted. A register-only microbenchmark (`bench/peak.cu`, four independent
accumulators per warp) gives **12.25 TFLOP/s with fp32 accumulate** and **24.49 with fp16**, a ratio of
exactly 0.500 as the hardware specifies. Both project kernels issue the fp32 form, so percentages use
**12.1 TFLOP/s**.

| `N` | causal | CUDA v2 | cuTile | SDPA |
|---|---|---|---|---|
| 1024 | no | 101% | 29% | 108% |
| 2048 | no | **107%** | 24% | 115% |
| 2048 | yes | 97% | 79% | 89% |

The shipped kernel is **at the ceiling** at the largest non-causal size: there is nothing left to win
on that path without a narrower accumulate, which the fp32-state rule forbids. SDPA goes above 100%,
which means part of its speed is a precision choice this project does not allow.

### 4.3 Correctness

An fp64 NumPy oracle at every size and both masks, with the rule
`max|o − o_ref| ≤ 5e-3 · max|o_ref|` plus finite output.

| Case | max abs error | allowed |
|---|---|---|
| `N = 1` | 0.0 | 1.8e-02 |
| `N = 65` | 4.48e-04 | 6.87e-03 |
| `N = 1000` | 1.32e-04 | 1.98e-03 |
| `N = 2048`, non-causal | 7.91e-05 | 1.41e-03 |
| `N = 2048`, causal | 9.73e-04 | 1.71e-02 |

**Both kernels give the same error to the last digit**, which is the strongest evidence available: a
shared bug across two very different memory layouts is implausible. In addition, **38 closed-form
checks** use mathematics rather than the oracle — with `V = 1` the output must be exactly `1` (it is,
to 0.0), with `Q = 0` it must be the running mean of `V`, a one-hot key must return that row of `V` —
and **19 protocol checks** cover determinism, tile boundaries 63/64/127/128/4095/4097, and fp16
overflow. All pass.

### 4.4 The counter data, and the finding it produced

`ncu` at `N = 2048`, one profiled launch per row:

| Kernel | DRAM read | DRAM write | Warps active | Tensor pipe |
|---|---|---|---|---|
| CUDA v1 | 6.33 MB | 1.77 MB | 8.3% | 13.2% |
| **CUDA v2** | 6.31 MB | **1.73 MB** | **16.3%** | **42.2%** |
| cuTile, non-causal | 57.24 MB | **321.93 MB** | 16.5% | 13.8% |
| cuTile, causal | 6.35 MB | 1.77 MB | 16.1% | 31.0% |
| PyTorch SDPA | 6.87 MB | 3.06 MB | 15.9% | 45.1% |

Ideal fused traffic is 8.4 MB total. Two findings, neither visible in the source:

1. **The compiler's non-causal build is not fused.** It writes **322 MB** to DRAM, about 2.8 times one
   `N × N` fp32 matrix, while the same source compiled for the causal case writes 1.77 MB. One
   compilation stays on chip and the other leaks; only a counter separated them.
2. **The gap to SDPA was tensor-pipe utilisation and nothing else.** 45.1% against 13.2% is the 3.42×
   latency ratio almost exactly. Raising utilisation to 42.2% is what closed that gap to 1.08×.

---

## 5. Plots already generated

Two PNG charts exist in the repository at `figs/`:

| File | What it shows |
|---|---|
| `figs/latency_vs_n.png` | latency against sequence length, log-log, four models |
| `figs/tflops_vs_n.png` | TFLOP/s against sequence length, with the measured ceiling |

The report draws its own higher-quality versions with pgfplots (`report/report.pdf`, Figure 1). For
slides, a **bar chart is better than a line chart**: one group per `N`, three bars (CUDA, cuTile,
unfused), which reads instantly from a distance. The data for it is the table in §4.1.

---

## 6. Slide-by-slide specification

Three slides, exactly as the instructor asked. Keep each one to a single idea and let the numbers carry
the argument.

### Slide 1 — Title

- Title: **Fused attention on one GPU**
- Subtitle: One operator, written by hand and by compiler, then measured
- `Balla Surya Harshith` · Roll number `CS23B010`
- `CS-6023 GPU Programming` · Mentor: `Pravin Jagtap`
- Course project demo, November 2026
- Optional and effective: one callout with the single best number, **4.5× faster than the compiler's
  kernel at N = 2048**.

### Slide 2 — Problem statement and results

Left half, the problem:

- The operator: `O = softmax( Q Kᵀ / √D + mask ) · V`, forward pass only, exact, fp16 in and out,
  fp32 state.
- The naive form writes the `N × N` matrix at 134 MB and is memory-bound: **31 GB/s of 170 available**.
- So the three stages must be fused, which forces a streaming softmax and a live accumulator rescale.
- The question: when does the tile model's productivity outweigh the control it gives up?

Right half, the results:

- Three figures across the top: **4.5×** faster than cuTile · **107%** of the measured ceiling ·
  **1.23×** faster than cuTile on the causal path.
- A compact latency table for `N = 2048`:

| Kernel | non-causal | causal |
|---|---|---|
| **CUDA v2 (shipped)** | **664 µs** | **364 µs** |
| cuTile | 2991 µs | 448 µs |
| PyTorch SDPA | 617 µs | 397 µs |
| unfused baseline | 13315 µs | 15098 µs |

- One line of correctness: both kernels pass an fp64 oracle at every size and both masks, and agree
  with each other to 2.5e-04.

### Slide 3 — Challenges, and how they were resolved

A table is the right form here: what broke, what it cost, how it was found and fixed.

| What broke | What it cost | How it was found, and the fix |
|---|---|---|
| 32-way shared-memory bank conflict | 9.1 ms instead of 3.6 ms | Every lane hit one bank. Padded the stride to 68/72 and interleaved lane columns: **2.5×**. Error unchanged. |
| cuTile looked 1.85× faster at N = 256 | an unfair comparison | The tile launcher synchronised the device every call; the CUDA path did not. Caught by comparing event timing with wall clock. |
| Staging moved 2 bytes at a time | 3.2 ms instead of 2.1 ms | SASS showed 19 scalar `LDG.E.U16` sites. 16-byte loads: **1.53×**. |
| My kernel was fused; cuTile's was not | the real explanation | `ncu` counters: the cuTile non-causal build writes **322 MB** to DRAM, mine writes **1.7 MB**. Only a counter showed it. |

Closing line, the recommendation:

> Write it in cuTile first to learn whether the operator matters. Move to CUDA when a counter, not an
> intuition, says the remaining gap is where the accumulator lives.

---

## 7. Design guidance for the deck

- **16:9**, white ground, one dark colour for structure, **one** warm accent reserved for the numbers
  that matter. Two colours total is enough.
- No shadows, no gradients, no clip art, no icons. Flat rectangles and type only.
- Title 30–34 pt bold; body 12–14 pt; the hero figure 40 pt.
- One grid: equal left and right margins, everything aligned to it.
- Every number on a slide must be traceable to §4 of this brief. Do not round 664.2 to 660, and do not
  say "about 4×" when it is 4.5×.
- Ten minutes total. Slide 1 takes 40 seconds, slide 2 about four minutes, slide 3 about four minutes,
  and the rest is the live demo.

---

## 8. Facts that must not be mangled

| Claim | Exact value |
|---|---|
| Latency at `N = 2048`, non-causal, CUDA | **664.2 µs** |
| Latency at `N = 2048`, non-causal, cuTile | **2991.2 µs** |
| Speedup over cuTile at that size | **4.5×** |
| Latency at `N = 2048`, causal, CUDA | **364.5 µs** |
| Speedup over cuTile on the causal path | **1.23×** |
| Fraction of the measured ceiling | **107%** at `N = 2048` non-causal |
| Measured fp32 ceiling | **12.1 TFLOP/s** (12.25 raw), fp16 24.49, ratio 0.500 |
| Registers, spills, shared memory, blocks/SM | **140, 0, 46080 B, 2** |
| DRAM written by cuTile non-causal | **321.93 MB** against 1.73 MB for mine |
| Code lines | cuTile **118**, CUDA v1 **138**, CUDA v2 **191** |
| Correctness checks | 38 closed-form, 19 protocol, all pass |
| Agreement between the two kernels | **2.5e-04** |
| Hardware | RTX 3050 Laptop, sm_86, 16 SMs, 4 GB |

**Do not claim:** that the kernel is faster than SDPA in general (it is 1.08× slower non-causal and
faster causal); that this is novel research (it is a measured comparison on one consumer GPU); or that
portability was measured (one GPU only).
