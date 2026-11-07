# Fused attention on one GPU: CUDA C++ and cuTile

One operator, the fused attention forward pass, written twice and measured against a materialised
reference.

## Project structure

| Path | What it holds |
|---|---|
| `kernels/cuda-fused-attn/flash_v2.cu` | the CUDA kernel: `mma.sync`, `ldmatrix`, `cp.async`, score tile and accumulator in registers |
| `kernels/cutile-fused-attn/flash.py` | the same fusion in tile vocabulary: no thread index, no stride, no barrier |
| `baseline/attn/unfused.py` | the materialised reference: cuBLAS, softmax, cuBLAS, with the score matrix in global memory |
| `bench/bench.py` | the harness: fp64 oracle, three models, timing protocol, CSV writer |
| `bench/peak.cu` | the tensor-core ceiling benchmark, fp32 and fp16 accumulate |
| `scripts/` | setup, the test ladder, the benchmarks, the checks, the profiler |
| `results/benchmarks.csv` | the measured table this README quotes |
| `profiles/*.csv` | raw `ncu` counters: DRAM bytes, occupancy, tensor-pipe cycles |
| `figs/*.png` | the two charts |
| `report/report.pdf` | the four-page written comparison, with `report.tex` beside it |
| `slides/` | the three-slide demo deck, its PDF, and the script that builds it |
| `requirements.txt` | the Python packages |
| `Makefile` | every command below, so you do not have to retype them |

## What this installs, and what it uses

Two groups of software. **System packages** come from your OS or from NVIDIA. **Python packages** come
from `pip` into a local virtual environment, so nothing system-wide is touched.

**System, you install these first**

| What | Version used | Why it is needed |
|---|---|---|
| NVIDIA driver | 617.42 | runs the kernels |
| CUDA toolkit | 12.6 | `nvcc` compiles the CUDA kernel and `bench/peak.cu`; `cuobjdump` reads the SASS |
| Linux or WSL2 on Ubuntu 24.04 | 24.04 | the project targets Linux. Windows alone cannot build the cuTile half |
| Python | 3.12 | the harness and every script |
| GNU Make | 3.81+ | drives the targets |
| Git | any | to clone |

A GPU of compute capability **8.0 or higher** is required for the cuTile kernel. `make check-gpu`
prints the capability before you run anything.

**Python, installed by the setup script**

| Package | Why it is needed | Required? |
|---|---|---|
| `cuda-tile[tileiras]` | the tile-level kernel, and the TileIR compiler `tileiras` that ships inside the wheel | yes |
| `cuda-toolkit[nvrtc]` | `tileiras` does not ship NVRTC, and the CUDA kernel is compiled at run time instead of at build time | yes |
| `numpy` | input generation and the fp64 oracle | yes |
| `cupy-cuda12x` | GPU arrays, the NVRTC launch path, and cuBLAS inside the baseline | yes |
| `pillow` | the two PNG charts in `figs/` | optional, only `make figs` needs it |

There is no PyTorch, no Triton and no CUTLASS. The harness compiles and launches the kernels directly.

## Install from scratch

```bash
# 1. the toolkit, if you do not have it (Ubuntu 24.04, WSL2 or native)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update && sudo apt-get install -y cuda-toolkit-12-6

# 2. the project itself
git clone https://github.com/harshith0518/Attention-Operator-in-CUDA-cuTile-with-fused-computation
cd Attention-Operator-in-CUDA-cuTile-with-fused-computation

# 3. the Python side: makes .venv, installs requirements.txt, then runs the toolchain gate
bash scripts/setup_linux.sh

# 4. confirm the machine can run both kernels
make check-gpu
```

Step 3 takes a few minutes and prints the gate result. The gate checks three things: cuPy sees the
device, `ct.mma` lowers a small matmul correctly, and NVRTC compiles the CUDA kernel. If it fails, stop
there; the later targets assume it passed.

If your toolkit is not at `/usr/local/cuda-12.6`, pass your own path: `make CUDA=/usr/local/cuda-13.0`.

## Run

```bash
make test      # five stages, stops at the first failure
make bench     # writes results/results.csv
make figs      # writes the two charts into figs/
make sass      # registers, spills, and the SASS instruction mix
make peak      # the fp32 and fp16 ceiling on this GPU
make profile   # ncu counters into profiles/ (needs the counter permission)
make report    # rebuilds report/report.pdf
make slides    # rebuilds the three-slide deck
make           # lists every target
```

`make test` is the gate, and it is the one to trust. It runs the toolchain check, the cuTile kernel
self-check, 38 closed-form checks whose answers follow from the mathematics, 19 protocol checks, and
then the oracle matrix across both kernels. A kernel that fails the oracle is never timed, and the
harness enforces that rather than trusting the person running it.

To reproduce the averaged table:

```bash
.venv/bin/python scripts/repeat_bench.py 6 256,512,1024,2048 cuda,cutile,unfused
```

## Expected results

Measured on an RTX 3050 Laptop: sm_86, 16 SMs, 4 GB, WSL2, CUDA 12.6. Latency in microseconds, mean of
six repetitions of the p50.

| `N` | causal | CUDA | cuTile | unfused |
|---|---|---|---|---|
| 256 | no | **18.6** | 42.0 | 807.5 |
| 256 | yes | **19.5** | 21.5 | 1032.1 |
| 512 | no | **52.1** | 150.5 | 1020.5 |
| 512 | yes | **41.0** | 53.1 | 1269.6 |
| 1024 | no | **176.1** | 622.4 | 3562.1 |
| 1024 | yes | **109.6** | 134.7 | 3961.7 |
| 2048 | no | **664.2** | 2991.2 | 13314.9 |
| 2048 | yes | **364.5** | 447.9 | 15097.9 |

**Calibrate before you compare.** Run `make peak`. You should see about 12.3 TFLOP/s with fp32
accumulate and 24.4 with fp16, a ratio of 0.5. The harness also measures about 170 GB/s of copy
bandwidth, below the 192 GB/s in the spec sheet. If your numbers are far from these, your clocks or
your memory are behaving differently, and the table above will not match either.

## How the numbers were obtained

One protocol for every model.

1. Fill Q, K and V from a fixed seed, in fp16.
2. Check the kernel against an fp64 NumPy oracle. If it fails, stop.
3. Run 50 warm-up iterations.
4. Loop until the GPU has done at least one second of work. Flush L2 before each timed region, outside
   the event window.
5. Record the p50 and p90 of each pass. Repeat the whole measurement N times, then average the p50s.

The pass rule is `max|o - o_ref| <= 5e-3 * max|o_ref|`, plus finite output. Both kernels give the same
error to the last digit. That is the strongest evidence available here, because a shared bug across two
very different memory layouts is implausible.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `make check-gpu` says cuTile cannot run | the GPU is older than compute capability 8.0. The CUDA kernel still works |
| `tileiras` fails inside `cuda_fp4.hpp` | two CUDA versions are mixed. Delete `.venv` and rerun `scripts/setup_linux.sh` |
| `ncu` returns `ERR_NVGPUCTRPERM` | counters are a permission, not a fault. On Windows: NVIDIA Control Panel, Desktop menu, Manage GPU Performance Counters, allow. It then works under WSL2 |
| `compute-sanitizer` cannot start | unsupported under WSL2. `scripts/extra_checks.py` covers what it would have reported |
| Timings are 20% off the table | laptop clocks move. Lock them with `sudo nvidia-smi -lgc 2100`, or compare ratios only |
| `make figs` complains about a missing CSV | run `make bench` first |

## What this does not do

Forward pass only. One GPU, so portability across GPU generations is argued rather than measured. The
baseline casts its GEMM operands to fp32 to avoid fp16 overflow, which makes the fused-versus-unfused
ratio an upper bound rather than a floor.
