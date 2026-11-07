"""Smoke test: does cuTile run on this GPU?

Three checks, cheapest first:
  1. import cuda.tile + report versions
  2. a 1-D tile kernel (vector add)          -- proves load/store/launch work
  3. a tiled matmul with ct.mma              -- proves the tensor-core path works

Check 3 is NOT the project's GEMM. The project is OP2 (attention) only. The matmul
here is a 10-line capability probe: ct.mma is the operation the attention kernel's
QK^T and PV products are built on, so if `ct.mma` does not lower correctly on this GPU
there is no project. That is why it is the gate.

Exit code 0 only if all three pass. This is the gate in PLAN.md section 3.2.

Run:  .venv\\Scripts\\python.exe scripts\\smoke_cutile.py
"""

import sys
import os
import traceback
from pathlib import Path

# CuPy and cuTile each want a cache directory. Their defaults live under the user
# profile (~/.cupy, ~/.cache/cuda_tile), which this machine refuses to create
# ("Access is denied: C:\Users\ASUS\.cupy"), so pin both inside the project
# before either library is imported. setdefault leaves an explicit setting alone.
_CACHE_ROOT = Path(__file__).resolve().parent.parent / "tmp"
for _env, _sub in (("CUPY_CACHE_DIR", "cupy"), ("CUDA_TILE_CACHE_DIR", "cuda_tile")):
    _dir = _CACHE_ROOT / _sub
    _dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(_env, str(_dir))

# tileiras (cuTile's compiler, a separate process) needs a writable TEMP or it exits 5
# with "Failed to create temporary file: permission denied" -- the user temp dir is
# locked down on this machine. Forced, not setdefault: inheriting the broken one is the
# whole failure. Must be set before the first kernel compile.
_TEMP = _CACHE_ROOT / "temp"
_TEMP.mkdir(parents=True, exist_ok=True)
for _env in ("TEMP", "TMP"):
    os.environ[_env] = str(_TEMP)

import numpy as np  # noqa: E402  (deliberately after the env pinning)


def hr(title):
    print("\n" + "=" * 62)
    print(title)
    print("=" * 62)


def main() -> int:
    hr("1. imports and versions")
    try:
        import cuda.tile as ct
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: 'import cuda.tile' raised {type(exc).__name__}: {exc}")
        print("\n  -> the cuda-tile package is not installed in this interpreter,")
        print("     or it cannot find the TileIR compiler (tileiras).")
        print("     Try: pip install --upgrade \"cuda-tile[tileiras]\"")
        return 1

    try:
        from importlib.metadata import version as pkg_version

        print(f"  cuda-tile   : {pkg_version('cuda-tile')}")
        for p in ("nvidia-cuda-tileiras", "nvidia-cuda-nvcc", "nvidia-nvvm"):
            try:
                print(f"  {p:<12}: {pkg_version(p)}")
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        print("  (could not read package versions)")

    try:
        import cupy as cp
        print(f"  cupy        : {cp.__version__}")
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: 'import cupy' raised {type(exc).__name__}: {exc}")
        print("\n  -> install cupy for CUDA 13:  pip install cupy-cuda13x")
        return 1

    try:
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props["name"]
        if isinstance(name, bytes):
            name = name.decode()
        cc = (props["major"], props["minor"])
        vram = props["totalGlobalMem"] / (1024**3)
        drv = cp.cuda.runtime.driverGetVersion()
        print(f"  GPU         : {name}")
        print(f"  compute cap : {cc[0]}.{cc[1]}")
        print(f"  VRAM        : {vram:.1f} GiB")
        print(f"  driver API  : {drv // 1000}.{(drv % 1000) // 10}")
        if cc[0] < 8:
            print("  WARNING: compute capability < 8.x is unsupported by cuTile")
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: could not query the CUDA device: {type(exc).__name__}: {exc}")
        return 1

    # ------------------------------------------------------------------ 2
    hr("2. vector add (1-D tile kernel)")

    @ct.kernel
    def vadd(a, b, c, tile: ct.Constant[int]):
        pid = ct.bid(0)
        a_t = ct.load(a, index=(pid,), shape=(tile,))
        b_t = ct.load(b, index=(pid,), shape=(tile,))
        ct.store(c, index=(pid,), tile=a_t + b_t)

    try:
        n, tile = 4096, 256
        rng = cp.random.default_rng(0)
        a = rng.random(n, dtype=cp.float32)
        b = rng.random(n, dtype=cp.float32)
        c = cp.zeros_like(a)
        cp.cuda.Device(0).synchronize()
        ct.launch(cp.cuda.get_current_stream(),
                  (ct.cdiv(n, tile), 1, 1),
                  vadd, (a, b, c, tile))
        cp.cuda.Device(0).synchronize()
        err = float(cp.max(cp.abs(c - (a + b))))
        assert err < 1e-5, f"max abs err {err:.3e}"
        print(f"  PASS  (max abs err {err:.2e})")
    except Exception:  # noqa: BLE001
        print("  FAIL")
        traceback.print_exc()
        return 1

    # ------------------------------------------------------------------ 3
    hr("3. tiled matmul via ct.mma (the tensor-core path)")

    @ct.kernel
    def matmul(A, B, C,
               tm: ct.Constant[int], tn: ct.Constant[int], tk: ct.Constant[int]):
        i, j = ct.bid(0), ct.bid(1)
        a_view = A.tiled_view((tm, tk), padding_mode=ct.PaddingMode.ZERO)
        b_view = B.tiled_view((tk, tn), padding_mode=ct.PaddingMode.ZERO)
        acc = ct.zeros((tm, tn), ct.float32)
        for k in range(a_view.num_tiles(1)):
            acc = ct.mma(a_view.load((i, k)), b_view.load((k, j)), acc)
        ct.store(C, index=(i, j), tile=acc.astype(C.dtype))

    try:
        M = N = K = 512
        tm = tn = 128
        tk = 64
        rng = cp.random.default_rng(1)
        A = rng.random((M, K), dtype=cp.float16)
        B = rng.random((K, N), dtype=cp.float16)
        C = cp.zeros((M, N), dtype=cp.float32)

        grid = (ct.cdiv(M, tm), ct.cdiv(N, tn), 1)
        cp.cuda.Device(0).synchronize()
        ct.launch(cp.cuda.get_current_stream(), grid,
                  matmul, (A, B, C, tm, tn, tk))
        cp.cuda.Device(0).synchronize()

        # Reference on the CPU in fp32. CuPy's matmul needs cuBLAS, which the pip
        # cuda-toolkit[nvrtc] extra does not ship ("Failure finding known cublasLt DLL"),
        # and 512^3 is 134 MFLOP -- numpy is already a dependency, so do not drag in
        # another 700 MB wheel just to check a kernel.
        ref = np.asarray(A.get(), dtype=np.float32) @ np.asarray(B.get(), dtype=np.float32)
        scale = float(np.max(np.abs(ref))) or 1.0
        rel = float(np.max(np.abs(np.asarray(C.get(), dtype=np.float32) - ref))) / scale
        flops = 2.0 * M * N * K

        # a rough timing, just so you have a first number to sanity-check
        start, end = cp.cuda.Event(), cp.cuda.Event()
        iters = 50
        start.record()
        for _ in range(iters):
            ct.launch(cp.cuda.get_current_stream(), grid,
                      matmul, (A, B, C, tm, tn, tk))
        end.record()
        end.synchronize()
        ms = cp.cuda.get_elapsed_time(start, end) / iters

        print(f"  PASS  ({M}x{N}x{K} fp16->fp32, max rel err {rel:.2e})")
        print(f"  first number: {ms:.3f} ms  =>  {flops / (ms * 1e-3) / 1e9:.1f} GFLOP/s")
        print("  (unwarmed, untuned, single run -- sanity check only, NOT a benchmark)")
        assert rel < 5e-3, f"relative error {rel:.3e} too large for fp16 inputs"
    except Exception:  # noqa: BLE001
        print("  FAIL")
        traceback.print_exc()
        return 1

    hr("ALL THREE CHECKS PASSED")
    print("  cuTile is alive on this GPU. Save the versions with")
    print("      .\\.venv\\Scripts\\python.exe -m pip freeze > docs\\requirements.lock.txt")
    print("  then tick the gate in AGENTS.md section 3 and start PLAN.md day 2.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
