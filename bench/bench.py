"""Attention harness: correctness against fp64, timing, CSV, figures.

This file owns every number in the project. A kernel never certifies itself: this harness builds
the inputs, computes the fp64 reference, applies the pass rule, times the kernel and appends a CSV
row. Figures are generated from that CSV only.

Pass rule: all values finite, max|out-ref| <= 5e-3 * max|ref|, each model
checked against the reference on its own, and the two models checked against each other.

Timing protocol: 50 warm-up iterations, repeat until the GPU has done at
least one second of work, report p50 and p90, flush L2 between iterations outside the event
window, and log SM clock, power and temperature with every row.

Run:
    .\\.venv\\Scripts\\python.exe .\\bench\\bench.py                    # full default run
    .\\.venv\\Scripts\\python.exe .\\bench\\bench.py --quick            # small, for iteration
    .\\.venv\\Scripts\\python.exe .\\bench\\bench.py --figs-only        # figures from the CSV
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _env, _sub in (("CUPY_CACHE_DIR", "cupy"), ("CUDA_TILE_CACHE_DIR", "cuda_tile")):
    _d = ROOT / "tmp" / _sub
    _d.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(_env, str(_d))
_TEMP = ROOT / "tmp" / "temp"
_TEMP.mkdir(parents=True, exist_ok=True)
for _env in ("TEMP", "TMP"):
    os.environ[_env] = str(_TEMP)

import numpy as np                                        # noqa: E402

try:                                                      # figures-only mode needs no GPU
    import cupy as cp                                     # noqa: E402
except ImportError:                                       # pragma: no cover
    cp = None

RESULTS = ROOT / "results" / "results.csv"
FIGS = ROOT / "figs"
SMEM = 46080                                              # kernels/cuda-fused-attn/flash_v2.cu dynamic smem
D = 64
B, H = 1, 8
TOL = 5e-3
WARMUP = 50
MIN_SECONDS = 1.0

COLUMNS = ["model", "causal", "dtype", "B", "H", "N", "D", "BQ", "BK", "p50_us", "p90_us",
           "tflops", "gbytes_s", "reference_name", "reference_p50_us", "pct_of_reference",
           "max_abs_err", "max_rel_err", "nan_inf", "clock_sm_mhz", "power_w", "temp_c",
           "git_commit", "notes", "min_us", "max_us"]


# ------------------------------------------------------------------ reference
def reference(q, k, v, causal, chunk=512):
    """fp64 attention on the CPU, chunked over query rows. Direct implementation: it does not
    share the online-softmax code, so the two cannot share a bug."""
    q = q.get() if hasattr(q, "get") else q
    k = k.get() if hasattr(k, "get") else k
    v = v.get() if hasattr(v, "get") else v
    q = np.asarray(q, np.float64)
    k = np.asarray(k, np.float64)
    v = np.asarray(v, np.float64)
    b, h, n, d = q.shape
    out = np.zeros((b, h, n, d), np.float64)
    scale = 1.0 / np.sqrt(d)
    for bi in range(b):
        for hi in range(h):
            for s in range(0, n, chunk):
                e = min(n, s + chunk)
                sc = (q[bi, hi, s:e] @ k[bi, hi].T) * scale
                if causal:
                    rows = np.arange(s, e)[:, None]
                    cols = np.arange(n)[None, :]
                    sc = np.where(cols <= rows, sc, -np.inf)
                sc -= sc.max(axis=1, keepdims=True)
                p = np.exp(sc)
                out[bi, hi, s:e] = (p / p.sum(axis=1, keepdims=True)) @ v[bi, hi]
    return out


# ------------------------------------------------------------------ models
class CudaModel:
    name = "cuda"

    def __init__(self, source="flash_v2.cu", smem=None, name="cuda"):
        self.name = name
        self.smem = SMEM if smem is None else smem
        src = (ROOT / "kernels" / "cuda-fused-attn" / source).read_text()
        self.mod = cp.RawModule(code=src, options=("--gpu-architecture=sm_86", "-lineinfo",
                                                   "--std=c++17"), backend="nvrtc")
        self.fn = self.mod.get_function("attn_fwd")
        self.fn.max_dynamic_shared_size_bytes = self.smem
        self.scale = np.float32(np.log2(np.e) / np.sqrt(D))
        self._buf = {}

    def __call__(self, q, k, v, causal, out=None):
        b, h, n, _ = q.shape
        # Reuse the output buffer. A cp.zeros here launches a fill kernel inside the timed window,
        # which the cuTile path does not do, so the comparison would be unfair.
        if out is None:
            key = (b * h, n, D)
            out = self._buf.get(key)
            if out is None:
                out = cp.zeros(key, np.float16)
                self._buf[key] = out
        # Same contiguity treatment as attn(), so neither path pays a hidden copy.
        qf = cp.ascontiguousarray(q.reshape(b * h, n, D))
        kf = cp.ascontiguousarray(k.reshape(b * h, n, D))
        vf = cp.ascontiguousarray(v.reshape(b * h, n, D))
        grid = ((n + 63) // 64, b * h, 1)
        self.fn(grid, (128, 1, 1),
                (qf, kf, vf, out, np.int32(n), self.scale, np.int32(1 if causal else 0)),
                shared_mem=self.smem)
        return out.reshape(b, h, n, D)

    def regs(self):
        return self.fn.num_regs


class CutileModel:
    name = "cutile"

    def __init__(self):
        sys.path.insert(0, str(ROOT / "kernels" / "cutile-fused-attn"))
        import flash                                            # noqa: E402
        self._attn = flash.attn
        self._buf = {}

    def __call__(self, q, k, v, causal):
        b, h, n, d = q.shape
        key = (b * h, n, d)
        buf = self._buf.get(key)
        if buf is None:
            buf = cp.zeros(key, np.float16)
            self._buf[key] = buf
        return self._attn(q, k, v, causal=causal, out=buf)

    def regs(self):
        return ""


class UnfusedModel:
    """Materialised attention: S and P go through global memory. The baseline for the fusion claim."""

    name = "unfused"

    def __init__(self):
        sys.path.insert(0, str(ROOT / "baseline" / "attn"))
        import unfused                                              # noqa: E402
        self._f = unfused.unfused_attn
        self._traffic = unfused.traffic_bytes

    def bytes_moved(self, n):
        return self._traffic(B, H, n, D)

    def __call__(self, q, k, v, causal, out=None):
        return self._f(q, k, v, causal=causal, out=out)


def make_models(which):
    models = []
    if "cuda" in which:
        # the shipped kernel: mma.sync + ldmatrix + cp.async, 46080 B of shared memory
        models.append(CudaModel("flash_v2.cu", 46080, "cuda"))
    if "cutile" in which:
        models.append(CutileModel())
    if "unfused" in which:
        models.append(UnfusedModel())
    return models


# ------------------------------------------------------------------ data
def inputs(n, seed=17, scale=1.0):
    rng = np.random.default_rng(seed)
    shape = (B, H, n, D)
    q = cp.asarray((rng.standard_normal(shape) * scale).astype(np.float16))
    k = cp.asarray((rng.standard_normal(shape) * scale).astype(np.float16))
    v = cp.asarray((rng.standard_normal(shape) * scale).astype(np.float16))
    return q, k, v


# ------------------------------------------------------------------ telemetry
def telemetry():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,power.draw,temperature.gpu",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        clock, power, temp = [x.strip() for x in out.split(",")]
        return clock, power, temp
    except Exception:
        return "", "", ""


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


# ------------------------------------------------------------------ timing
def flush_buffer():
    return cp.zeros(8 * 1024 * 1024, dtype=cp.uint32)          # 32 MB


def timed(fn, q, k, v, causal, flush, min_seconds=MIN_SECONDS):
    """50 warm-ups, then per-iteration events until the GPU has done min_seconds of work.
    The L2 flush happens before start.record(), so it is outside the event window."""
    for _ in range(WARMUP):
        fn(q, k, v, causal)
    cp.cuda.Device(0).synchronize()

    samples, total, iters = [], 0.0, 0
    while total < min_seconds and iters < 20000:
        flush.fill(1)                       # dirty the L2 with 32 MB, outside the window
        s, e = cp.cuda.Event(), cp.cuda.Event()
        s.record()
        fn(q, k, v, causal)
        e.record()
        e.synchronize()
        ms = cp.cuda.get_elapsed_time(s, e)
        samples.append(ms * 1000.0)          # us
        total += ms / 1000.0
        iters += 1
    arr = np.asarray(samples)
    return (float(np.percentile(arr, 50)), float(np.percentile(arr, 90)),
            float(arr.min()), float(arr.max()), iters)


# ------------------------------------------------------------------ CSV
def append_row(row):
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    new = not RESULTS.exists() or RESULTS.stat().st_size == 0
    with RESULTS.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in COLUMNS})


# ------------------------------------------------------------------ test matrix
def correctness_matrix(quick):
    if quick:
        return [(1, False), (65, False), (256, False), (256, True), (1000, False), (1000, True)]
    return [(1, False), (65, False), (130, False), (256, False), (256, True),
            (1000, False), (1000, True), (1024, False), (2048, False), (2048, True)]


def run_tests(models, quick):
    print("\n=== correctness ===")
    ok = True
    for n, causal in correctness_matrix(quick):
        q, k, v = inputs(n)
        ref = reference(q, k, v, causal)
        allowed = TOL * float(np.max(np.abs(ref)))
        outs = {}
        for m in models:
            out = m(q, k, v, causal)
            fin = bool(cp.isfinite(out).all())
            o = out.astype(np.float32).get()
            err = float(np.max(np.abs(o - ref)))
            denom = np.maximum(np.abs(ref), 1e-6)
            rel = float(np.max(np.abs(o - ref) / denom))
            outs[m.name] = o
            good = fin and err <= allowed
            ok &= good
            print(f"  {m.name:<7} N={n:<5} causal={int(causal)}  max_abs={err:.3e}  "
                  f"allowed={allowed:.3e}  finite={fin}  {'PASS' if good else 'FAIL'}")
        if len(outs) == 2:
            agree = float(np.max(np.abs(outs["cuda"] - outs["cutile"])))
            good = agree <= allowed
            ok &= good
            print(f"  cross-model agreement  max_abs={agree:.3e}  "
                  f"{'PASS' if good else 'FAIL'}")
    # stress: inputs scaled by 100, where a missing max-subtraction dies
    print("  stress x100:")
    q, k, v = inputs(256, scale=100.0)
    ref = reference(q, k, v, False)
    allowed = TOL * float(np.max(np.abs(ref)))
    for m in models:
        out = m(q, k, v, False).astype(np.float32).get()
        fin = bool(np.isfinite(out).all())
        err = float(np.max(np.abs(out - ref)))
        good = fin and err <= allowed
        ok &= good
        print(f"    {m.name:<7} N=256 max_abs={err:.3e}  finite={fin}  "
              f"{'PASS' if good else 'FAIL'}")
    print(f"=== {'ALL TESTS PASS' if ok else 'TEST FAILURES'} ===")
    return ok


# ------------------------------------------------------------------ timing run
def run_bench(models, sizes, quick):
    print("\n=== timing ===")
    print(f"{'model':<8}{'N':>6}{'causal':>7}{'p50_us':>10}{'p90_us':>10}{'TFLOP/s':>10}"
          f"{'%ref':>8}")
    flush = flush_buffer()
    for n in sizes:
        for causal in (False, True):
            q, k, v = inputs(n)
            t0 = time.perf_counter()
            ref = reference(q, k, v, causal)
            ref_us = (time.perf_counter() - t0) * 1e6
            allowed = TOL * float(np.max(np.abs(ref)))
            for m in models:
                # Verify BEFORE timing. A fast wrong kernel must never reach the CSV: that is
                # how an implied-throughput-above-peak row got in once.
                probe = m(q, k, v, causal).astype(np.float32).get()
                err = float(np.max(np.abs(probe - ref)))
                if not np.isfinite(probe).all() or err > allowed:
                    print(f"{m.name:<8}{n:>6}{int(causal):>7}   REFUSED: "
                          f"max_abs={err:.3e} > allowed={allowed:.3e}")
                    continue
                p50, p90, v_min, v_max, iters = timed(m, q, k, v, causal, flush,
                                                      min_seconds=0.3 if quick else MIN_SECONDS)
                # Executed FLOPs, one convention in one place: halve only when the kernel really
                # skips triangular work. BOTH causal paths now bound their K loop, so both take
                # the half convention. The cuTile causal build additionally removes work at compile
                # time (E2: the mask alone is worth 4.5x), so its half count is an upper bound on
                # the work done and its percentage of ceiling is an upper bound too.
                skips = causal
                flops = 4.0 * B * H * n * n * D * (0.5 if skips else 1.0)
                tflops = flops / (p50 * 1e-6) / 1e12
                clock, power, temp = telemetry()
                row = dict(model=m.name, causal=int(causal), dtype="fp16", B=B, H=H, N=n, D=D,
                           BQ=64, BK=64, p50_us=round(p50, 1), p90_us=round(p90, 1),
                           tflops=round(tflops, 2), gbytes_s="",
                           reference_name="numpy-fp64-cpu", reference_p50_us=round(ref_us, 1),
                           pct_of_reference="",   # CPU oracle: not a performance baseline
                           max_abs_err=f"{err:.3e}", max_rel_err="", nan_inf="False",
                           clock_sm_mhz=clock, power_w=power, temp_c=temp,
                           git_commit=git_commit(),
                           min_us=round(v_min, 1), max_us=round(v_max, 1),
                           notes=f"iters={iters}; l2 32MB flush outside window; "
                                 f"ref is a CPU oracle, not a baseline; "
                                 f"exec={'half (skips)' if skips else 'full'}")
                append_row(row)
                print(f"{m.name:<8}{n:>6}{int(causal):>7}{p50:>10.1f}{p90:>10.1f}"
                      f"{tflops:>10.2f}{row['pct_of_reference']:>8}")
    print(f"rows appended to {RESULTS.relative_to(ROOT)}")


# ------------------------------------------------------------------ figures
_FONT_CACHE = {}


def _pil_font(size, bold=False):
    """Find a usable TrueType font on Linux, WSL or Windows; fall back to Pillow's bitmap font.

    The figures used to hard-code C:/Windows/Fonts, which made them unbuildable anywhere else.
    """
    key = (size, bold)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]

    from PIL import ImageFont
    names = (("DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "NotoSans-Bold.ttf",
              "arialbd.ttf", "calibrib.ttf") if bold else
             ("DejaVuSans.ttf", "LiberationSans-Regular.ttf", "NotoSans-Regular.ttf",
              "arial.ttf", "calibri.ttf"))
    roots = [Path("/usr/share/fonts"), Path("/usr/local/share/fonts"),
             Path.home() / ".fonts", Path.home() / ".local/share/fonts",
             Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"]

    font = None
    for name in names:
        for root in roots:
            try:
                if not root.exists():
                    continue
                hit = next(root.rglob(name), None)
                if hit is not None:
                    font = ImageFont.truetype(str(hit), size)
                    break
            except OSError:
                continue
        if font is not None:
            break
    if font is None:
        font = ImageFont.load_default()
    _FONT_CACHE[key] = font
    return font


def _line_chart(path, panels, width, height, title, ylabel, xlabel):
    """Small native chart: log2 x, log10 y, grid, tick numbers, polyline series, legend.
    Written directly on Pillow so the figures do not depend on a plotting stack."""
    import math
    from PIL import Image, ImageDraw

    ink, muted, hair = (27, 31, 36), (89, 99, 110), (213, 217, 222)
    img = Image.new("RGB", (width, height), (255, 255, 255))
    dr = ImageDraw.Draw(img)
    f_small, f_bold = _pil_font(12), _pil_font(13, bold=True)
    dr.text((16, 10), title, font=f_bold, fill=ink)

    n_panels = len(panels)
    pw = (width - 40) // n_panels
    for pi, panel in enumerate(panels):
        x0, x1 = 40 + pi * pw + 46, 40 + pi * pw + pw - 14
        y0, y1 = height - 46, 52
        dr.rectangle([x0, y1, x1, y0], outline=ink, width=1)
        if panel.get("title"):
            dr.text((x0 + 2, y1 - 19), panel["title"], font=f_small, fill=muted)

        series = panel["series"]
        xs = sorted({p[0] for _, _, _, pts in series for p in pts})
        vs = [p[1] for _, _, _, pts in series for p in pts if p[1] > 0]
        if not xs or not vs:
            continue
        lx = (math.log2(min(xs)), math.log2(max(xs))) if len(xs) > 1 else (0.0, 1.0)
        v_min, v_max = min(vs) * 0.7, max(vs) * 1.4
        ly = (math.log10(v_min), math.log10(v_max))

        def px(n):
            return x0 if lx[1] == lx[0] else x0 + (math.log2(n) - lx[0]) / (lx[1] - lx[0]) * (x1 - x0)

        def py(v):
            v = max(v, v_min)
            return y0 - (math.log10(v) - ly[0]) / (ly[1] - ly[0]) * (y0 - y1)

        # grid + tick labels
        last_lab = -999
        for n in xs:
            dr.line([px(n), y1, px(n), y0], fill=hair)
            if px(n) - last_lab > 30:
                dr.text((px(n) - 10, y0 + 5), str(n), font=f_small, fill=muted)
                last_lab = px(n)
        decade = 10 ** math.floor(ly[0])
        while decade < 10 ** ly[1]:
            yy = py(decade)
            dr.line([x0, yy, x1, yy], fill=hair)
            label = (f"{decade:.1f}" if decade < 1 else f"{decade:g}" if decade < 1000 else f"{decade/1000:g}k")
            dr.text((x0 - len(label) * 7 - 6, yy - 7), label, font=f_small, fill=muted)
            decade *= 10
        dr.text((x1 - 30, y0 + 18), xlabel, font=f_small, fill=muted)

        for label, color, dashed, pts in series:
            pts = sorted(pts)
            coords = [(px(n), py(v)) for n, v in pts]
            if dashed:
                for i in range(len(coords) - 1):
                    (ax, ay), (bx, by) = coords[i], coords[i + 1]
                    steps = max(2, int(math.dist((ax, ay), (bx, by)) // 7))
                    for s in range(0, steps, 2):
                        t0, t1 = s / steps, min(1.0, (s + 1) / steps)
                        dr.line([ax + (bx - ax) * t0, ay + (by - ay) * t0,
                                 ax + (bx - ax) * t1, ay + (by - ay) * t1], fill=color, width=2)
            else:
                dr.line(coords, fill=color, width=2)
            for cx, cy in coords:
                dr.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=color)

        # legend inside the panel
        lx0, ly0 = x0 + 8, y1 + 8
        for label, color, dashed, _ in series:
            dr.line([lx0, ly0 + 6, lx0 + 18, ly0 + 6], fill=color, width=2)
            dr.text((lx0 + 24, ly0), label, font=f_small, fill=ink)
            ly0 += 16

    img.save(path)


def figures():
    if not RESULTS.exists():
        print("no CSV yet")
        return
    rows = list(csv.DictReader(RESULTS.open(encoding="utf-8")))
    if not rows:
        print("CSV empty")
        return
    FIGS.mkdir(parents=True, exist_ok=True)
    colors = {"cuda": (180, 81, 31), "cutile": (14, 107, 107)}
    models = sorted({r["model"] for r in rows})

    def series(metric, causal, dashed=False, tag=""):
        out = []
        for m in models:
            pts = [(int(r["N"]), float(r[metric])) for r in rows
                   if r["model"] == m and r["causal"] == str(causal)
                   and r.get(metric) not in ("", None) and float(r[metric]) > 0]
            if pts:
                out.append((f"{m} {tag}".strip(), colors.get(m, (60, 60, 60)), dashed, pts))
        return out

    try:
        _line_chart(FIGS / "latency_vs_n.png",
                    [{"title": "causal = 0", "series": series("p50_us", 0)},
                     {"title": "causal = 1", "series": series("p50_us", 1)}],
                    1100, 430,
                    f"Fused attention latency, B={B} H={H} D={D} fp16, p50 with L2 flushed",
                    "p50 (us)", "N")
        _line_chart(FIGS / "tflops_vs_n.png",
                    [{"title": "", "series": series("tflops", 0, tag="full") + series("tflops", 1, True, tag="causal")}],
                    760, 430,
                    "Achieved throughput (CUDA causal skips tiles; cuTile does not)",
                    "TFLOP/s", "N")
    except ImportError:
        print("figures need Pillow; install pillow in this venv, or run --figs-only "
              "with an interpreter that has it")
        return
    print(f"figures written to {FIGS.relative_to(ROOT)}")

# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="cuda,cutile,unfused")
    ap.add_argument("--sizes", default="256,512,1000,1024,2048")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tests-only", action="store_true")
    ap.add_argument("--figs-only", action="store_true")
    args = ap.parse_args()

    if args.figs_only:
        figures()
        return 0

    which = [s.strip() for s in args.models.split(",") if s.strip()]
    models = make_models(which)
    for m in models:
        regs = getattr(m, "regs", lambda: None)()
        extra = f" regs={regs}" if regs else ""
        print(f"model {m.name} ready{extra}")

    ok = run_tests(models, args.quick)
    if not ok:
        print("\ncorrectness failed: no timing recorded ")
        return 1
    if not args.tests_only:
        sizes = ([256, 512, 1000] if args.quick
                 else [int(s) for s in args.sizes.split(",") if s.strip()])
        run_bench(models, sizes, args.quick)
        figures()
    return 0


if __name__ == "__main__":
    sys.exit(main())
