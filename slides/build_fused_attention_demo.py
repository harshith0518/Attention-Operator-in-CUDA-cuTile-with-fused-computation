"""Build fused_attention_demo.pptx: three slides, exactly as the build spec defines them.

Layout, colours, type sizes, chart data and speaker notes come from the spec. Nothing is rounded and no
slide is added: the instructor stops the talk after slide 3 and a live demo follows.

Slide 1 carries a workflow strip drawn with shapes: five stages of attention, the middle three enclosed
as one fused kernel. It states the project's thesis before slide 2 proves it with numbers.

Run:  .venv/bin/python slides/build_fused_attention_demo.py
"""

import sys
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION, XL_TICK_MARK
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "slides" / "fused_attention_demo.pptx"

# ------------------------------------------------------------------ tokens
NAVY = RGBColor(0x1B, 0x2A, 0x41)
ACCENT = RGBColor(0xE8, 0x59, 0x0C)
INK = RGBColor(0x2B, 0x2F, 0x36)
GREY = RGBColor(0x5C, 0x66, 0x70)
PANEL = RGBColor(0xF2, 0xF4, 0xF7)
RULE = RGBColor(0xD5, 0xDA, 0xE1)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

W, H = 13.333, 7.5
M = 0.6
CW = 12.133


# ------------------------------------------------------------------ helpers
def _flatten(shape):
    """Remove the theme style reference, which is where LibreOffice takes its shadow from."""
    style = shape._element.find(qn("p:style"))
    if style is not None:
        shape._element.remove(style)
    spPr = shape._element.spPr
    for e in spPr.findall(qn("a:effectLst")):
        spPr.remove(e)
    spPr.append(spPr.makeelement(qn("a:effectLst"), {}))
    return shape


def box(slide, x, y, w, h, fill=None, line=None, line_w=0.75):
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    if fill is None:
        shp.fill.background()
    else:
        shp.fill.solid()
        shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(line_w)
    _flatten(shp)
    shp.text_frame.text = ""
    return shp


def text(slide, x, y, w, h, runs, size, color, bold=False, font="Calibri",
         align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, spacing=1.0, space_after=0):
    """`runs` is a string, a list of paragraphs, or one paragraph built from (text, overrides) tuples."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    if isinstance(runs, (str, tuple)):
        runs = [runs]
    elif runs and isinstance(runs[0], tuple):
        runs = [runs]            # a single paragraph made of several styled runs
    for i, para in enumerate(runs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        if space_after:
            p.space_after = Pt(space_after)
        pieces = [(para, {})] if isinstance(para, str) else para
        for piece, over in pieces:
            run = p.add_run()
            run.text = piece
            run.font.size = Pt(over.get("size", size))
            run.font.bold = over.get("bold", bold)
            run.font.name = over.get("font", font)
            run.font.color.rgb = over.get("color", color)
    return tb


def _border(cell, side, color=None, width=0.75):
    tag = qn(f"a:ln{side}")
    tcPr = cell._tc.get_or_add_tcPr()
    for old in tcPr.findall(tag):
        tcPr.remove(old)
    ln = etree.SubElement(tcPr, tag)
    ln.set("w", str(int(width * 12700)))
    if color is None:
        etree.SubElement(ln, qn("a:noFill"))
    else:
        fill = etree.SubElement(ln, qn("a:solidFill"))
        clr = etree.SubElement(fill, qn("a:srgbClr"))
        clr.set("val", str(color))
    tcPr.remove(ln)
    tcPr.insert(0, ln)           # ln* must precede the fill inside tcPr


def table(slide, x, y, col_w, row_h, data, header=True, font_size=12,
          col_styles=None, align=None, pad=0.08):
    """Every cell styled by hand: no banding, no theme style, thin bottom rules only."""
    nrows, ncols = len(data), len(data[0])
    gf = slide.shapes.add_table(nrows, ncols, Inches(x), Inches(y),
                                Inches(sum(col_w)), Inches(sum(row_h)))
    tbl = gf.table
    tbl.first_row = False
    tbl.horz_banding = False
    tbl.vert_banding = False
    for j, wv in enumerate(col_w):
        tbl.columns[j].width = Inches(wv)
    for i, hv in enumerate(row_h):
        tbl.rows[i].height = Inches(hv)

    for i, row in enumerate(data):
        for j, value in enumerate(row):
            cell = tbl.cell(i, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = NAVY if (header and i == 0) else WHITE
            cell.margin_left = cell.margin_right = Inches(pad)
            cell.margin_top = cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            for side in ("L", "R", "T"):
                _border(cell, side, None)
            _border(cell, "B", "1B2A41" if (header and i == 0) else "D5DAE1")

            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = (align[j] if align else PP_ALIGN.LEFT)
            pieces = [(value, {})] if isinstance(value, str) else value
            style = (col_styles or {}).get(j, {})
            for piece, over in pieces:
                run = p.add_run()
                run.text = piece
                run.font.size = Pt(over.get("size", font_size))
                run.font.bold = over.get("bold", style.get("bold", False))
                run.font.name = over.get("font", style.get("font", "Calibri"))
                run.font.color.rgb = (WHITE if (header and i == 0)
                                      else over.get("color", style.get("color", INK)))
    return tbl


def notes(slide, body):
    slide.notes_slide.notes_text_frame.text = body


def slide_number(slide, label_txt, y=7.1, h=0.25):
    # wide enough that the label stays on one line; right edge stays on the 12.733 grid
    text(slide, 11.4, y, 1.333, h, label_txt, 10, GREY, align=PP_ALIGN.RIGHT)


def new_deck():
    prs = Presentation()
    prs.slide_width = Inches(W)
    prs.slide_height = Inches(H)
    return prs


def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def label(slide, x, y, w, body_txt):
    return text(slide, x, y, w, 0.3, body_txt, 12, ACCENT, bold=True)


# ------------------------------------------------------------------ slide 1
def slide_one(prs):
    s = blank(prs)
    box(s, 0, 0, 0.35, H, NAVY)

    text(s, 0.9, 0.7, 7.5, 0.35, "CS-6023 GPU Programming · Course project demo", 14, GREY)
    text(s, 0.9, 1.55, 7.6, 1.6, "Fused attention on one GPU", 44, NAVY, bold=True, spacing=0.95)
    box(s, 0.9, 3.15, 1.2, 0.07, ACCENT)
    text(s, 0.9, 3.35, 7.6, 0.9,
         "One operator, written by hand and by compiler, then measured", 20, INK)

    text(s, 0.9, 4.25, 7.6, 0.4, "Balla Surya Harshith", 20, NAVY, bold=True)
    text(s, 0.9, 4.62, 7.6, 0.33, "Roll number CS23B010", 16, INK)
    text(s, 0.9, 4.96, 7.6, 0.33, "Mentor: Pravin Jagtap", 16, INK)

    box(s, 8.9, 1.55, 3.83, 3.1, PANEL)
    text(s, 9.1, 1.8, 3.43, 1.5, "4.5×", 80, ACCENT, bold=True)
    text(s, 9.1, 3.4, 3.43, 0.8, ["faster than the compiler's kernel", "at N = 2048"],
         16, INK, spacing=1.0)
    text(s, 9.1, 4.18, 3.43, 0.4, "664.2 µs against 2991.2 µs", 14, GREY)

    # ---- the workflow: five stages, the middle three fused into one kernel
    chips = [("Q, K, V", "io"), ("S = Q·Kᵀ", "fused"), ("softmax", "fused"),
             ("P·V", "fused"), ("O", "io")]
    cw, step, y_chip, h_chip = 1.55, 2.35, 5.62, 0.46
    for i, (chip_txt, kind) in enumerate(chips):
        cx = 0.9 + i * step
        if kind == "fused":
            box(s, cx, y_chip, cw, h_chip, NAVY)
            text(s, cx, y_chip, cw, h_chip, chip_txt, 12, WHITE, bold=True,
                 align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        else:
            box(s, cx, y_chip, cw, h_chip, PANEL)
            text(s, cx, y_chip, cw, h_chip, chip_txt, 12, NAVY, bold=True,
                 align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        if i < len(chips) - 1:
            text(s, cx + cw, y_chip + 0.06, step - cw, 0.3, "▶", 12, GREY, align=PP_ALIGN.CENTER)

    # bracket under the fused stages
    bx0 = 0.9 + step - 0.12
    bx1 = 0.9 + 3 * step + cw + 0.12
    box(s, bx0, 6.2, bx1 - bx0, 0.025, NAVY)
    box(s, bx0, 6.2, 0.025, 0.1, NAVY)
    box(s, bx1 - 0.025, 6.2, 0.025, 0.1, NAVY)
    text(s, bx0, 6.3, bx1 - bx0, 0.28,
         "ONE FUSED KERNEL · S, P AND ACC STAY IN REGISTERS", 12, ACCENT, bold=True,
         align=PP_ALIGN.CENTER)
    text(s, 0.9, 6.58, 11.8, 0.3,
         "8.4 MB of traffic instead of 134 MB: the N\u00d7N score matrix never leaves the chip.",
         12, GREY)

    text(s, 0.9, 7.02, 11.8, 0.3,
         "Today: the problem · the math · the kernel idea · the results · what broke and how we fixed it",
         12, GREY)

    notes(s, "Good morning. I am Harshith, roll number CS23B010, mentor Pravin Jagtap. I built one "
             "operator twice: fused attention, written by hand in CUDA and by compiler in cuTile. Then I "
             "measured both on the same GPU. The headline: my CUDA kernel is 4.5 times faster than the "
             "compiler's kernel at sequence length 2048. In the next ten minutes I show the problem, the "
             "math, the kernel idea, the results, and what broke. The strip at the bottom is the whole "
             "idea in one line: five stages, but the middle three run inside a single kernel, so the "
             "N-by-N score matrix never reaches memory. That is 8.4 megabytes of traffic instead of 134.")


# ------------------------------------------------------------------ slide 2
def slide_two(prs):
    s = blank(prs)
    text(s, M, 0.35, CW, 0.7,
         "Attention is memory-bound. Fusing it keeps the N×N matrix on the chip.", 28, NAVY, bold=True)
    box(s, M, 1.1, 0.9, 0.06, ACCENT)

    LX, LW = M, 5.9
    RX, RW = 7.0, 5.73

    label(s, LX, 1.35, LW, "THE PROBLEM")
    box(s, LX, 1.7, LW, 0.45, PANEL)
    text(s, 0.75, 1.74, 5.6, 0.4, "O = softmax( Q·Kᵀ / √D + mask ) · V", 16, NAVY,
         font="Consolas", anchor=MSO_ANCHOR.MIDDLE)
    bullets = [
        "Shape [B,H,N,D] = [1, 8, N, 64]. N runs 256 to 2048.",
        "The naive form writes an N×N matrix: 134 MB at N = 2048.",
        "It uses 31 of 170 GB/s. The tensor cores sit idle.",
        "Goal: fuse all three stages into one kernel.",
    ]
    text(s, LX, 2.25, LW, 1.15, ["–  " + b for b in bullets], 13, INK, space_after=2.2)

    label(s, LX, 3.5, LW, "THE MATH THAT MAKES FUSION POSSIBLE")
    box(s, LX, 3.85, LW, 1.0, PANEL)
    text(s, 0.75, 3.9, 5.6, 0.9, [
        [("α", {"color": ACCENT}), ("   = exp2(m_old − m_new)", {})],
        [("ℓ", {}), ("  ← α·ℓ + Σ exp2(s − m_new)", {})],
        [("acc", {}), (" ← α·acc + P·V", {})],
    ], 14, NAVY, font="Consolas")
    text(s, LX, 4.9, LW, 0.3,
         "Online softmax, per key tile. Exact. One pass. fp32 state.", 12, GREY)

    label(s, LX, 5.25, LW, "THE KERNEL IDEA")
    xs = [0.6, 1.825, 3.05, 4.275, 5.5]
    flow = ["load K, V", "S = Q·Kᵀ", "mask", "softmax", "acc += P·V"]
    for fx, ftxt in zip(xs, flow):
        box(s, fx, 5.6, 1.0, 0.5, NAVY)
        text(s, fx, 5.6, 1.0, 0.5, ftxt, 12, WHITE, bold=True,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    for ax in (1.6, 2.825, 4.05, 5.275):
        text(s, ax, 5.68, 0.225, 0.3, "▶", 12, GREY, align=PP_ALIGN.CENTER)
    box(s, LX, 6.2, LW, 0.32, ACCENT)
    text(s, LX, 6.2, LW, 0.32, "S, P and acc stay in registers. No N×N matrix reaches DRAM.",
         12, WHITE, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(s, LX, 6.58, LW, 0.3,
         "One block = 64 query rows, 128 threads. It loops over key tiles of 64.", 12, GREY)

    kpis = [("4.5×", "faster than cuTile, N = 2048"),
            ("107%", "of the measured fp32 ceiling"),
            ("1.23×", "faster than cuTile, causal path")]
    for i, (num, cap) in enumerate(kpis):
        cx = RX + i * 1.94
        text(s, cx, 1.35, 1.84, 0.6, num, 40, ACCENT, bold=True)
        text(s, cx, 1.95, 1.84, 0.45, cap, 12, INK)
    box(s, RX, 2.42, RW, 0.02, RULE)

    text(s, RX, 2.5, RW, 0.3,
         "Latency in µs, non-causal, log scale. Lower is better.", 12, NAVY, bold=True)

    data = CategoryChartData()
    data.categories = ["256", "512", "1024", "2048"]
    data.add_series("CUDA v2", (18.6, 52.1, 176.1, 664.2))
    data.add_series("cuTile", (42.0, 150.5, 622.4, 2991.2))
    data.add_series("unfused baseline", (807.5, 1020.5, 3562.1, 13314.9))
    ch = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(RX), Inches(2.8),
                            Inches(RW), Inches(2.3), data).chart
    ch.has_title = False
    ch.has_legend = True
    ch.legend.position = XL_LEGEND_POSITION.BOTTOM
    ch.legend.include_in_layout = False
    ch.legend.font.size = Pt(11)
    ch.legend.font.color.rgb = INK
    ch.legend.font.name = "Calibri"

    plot = ch.plots[0]
    plot.gap_width = 40
    plot.overlap = 0
    for ser, col in zip(plot.series, (ACCENT, NAVY, RULE)):
        ser.format.fill.solid()
        ser.format.fill.fore_color.rgb = col
        ser.format.line.fill.background()

    # Series-level labels collided at the smaller sizes, so the spec's fallback applies: label the
    # 2048 group only.
    for ser, col, shown in ((plot.series[0], ACCENT, "664.2"),
                            (plot.series[1], NAVY, "2,991.2")):
        dl = ser.points[3].data_label
        dl.has_text_frame = True
        dl.text_frame.text = shown
        for para in dl.text_frame.paragraphs:
            for run in para.runs:
                run.font.size = Pt(10)
                run.font.name = "Calibri"
                run.font.bold = False
                run.font.color.rgb = col
        try:
            dl.position = XL_LABEL_POSITION.OUTSIDE_END
        except Exception:
            pass

    va = ch.value_axis
    va.minimum_scale = 10
    va.maximum_scale = 100000
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = RULE
    va.major_gridlines.format.line.width = Pt(0.5)
    va.tick_labels.font.size = Pt(11)
    va.tick_labels.font.color.rgb = GREY
    va.tick_labels.number_format = "#,##0"
    va.tick_labels.number_format_is_linked = False
    va.format.line.fill.background()
    va.major_tick_mark = XL_TICK_MARK.NONE

    # log scale: no python-pptx API, so write c:logBase as the first child of c:scaling
    scaling = va._element.find(qn("c:scaling"))
    for old in scaling.findall(qn("c:logBase")):
        scaling.remove(old)
    log_base = etree.SubElement(scaling, qn("c:logBase"))
    log_base.set("val", "10")
    scaling.remove(log_base)
    scaling.insert(0, log_base)

    ca = ch.category_axis
    ca.tick_labels.font.size = Pt(12)
    ca.tick_labels.font.color.rgb = INK
    ca.has_title = True
    ca.axis_title.text_frame.text = "Sequence length N"
    run = ca.axis_title.text_frame.paragraphs[0].runs[0]
    run.font.size = Pt(11)
    run.font.color.rgb = GREY
    run.font.name = "Calibri"
    ca.format.line.color.rgb = RULE
    ca.major_tick_mark = XL_TICK_MARK.NONE

    bold_acc = {"bold": True, "color": ACCENT}
    navy_bold = {"bold": True, "color": NAVY}
    rows = [
        ["N = 2048", "non-causal", "causal"],
        [[("CUDA v2 (shipped)", navy_bold)], [("664.2 µs", bold_acc)], [("364.5 µs", bold_acc)]],
        ["cuTile", "2991.2 µs", "447.9 µs"],
        ["unfused baseline", "13314.9 µs", "15097.9 µs"],
    ]
    table(s, RX, 5.2, [2.5, 1.6, 1.63], [0.3, 0.3, 0.3, 0.3], rows,
          align=[PP_ALIGN.LEFT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT])

    text(s, RX, 6.5, RW, 0.45,
         [[("Correct:", {"bold": True}),
           (" both kernels pass an fp64 oracle at every size and mask. They agree to 2.5e-04.", {})]],
         12, INK)

    slide_number(s, "CS23B010 · 2 / 3", y=7.1, h=0.25)

    notes(s, "The operator is attention: O equals softmax of Q K-transpose over root D, plus a mask, "
             "times V. Eight heads, head width 64.\n"
             "The naive way builds the full N-by-N score matrix in global memory. At N of 2048 that is "
             "134 megabytes. It is written, read back, and written again. The kernel moves 31 gigabytes "
             "per second out of 170 available. The tensor cores wait for memory. So the job is to fuse "
             "all three stages and never write that matrix.\n\n"
             "Fusion has a cost. The softmax cannot see a full row at once. So it runs in streaming form, "
             "called online softmax. The kernel keeps a running maximum m, a running sum l, and a running "
             "output acc. When a new key tile arrives, it computes alpha, the exponential of the old "
             "maximum minus the new maximum. It scales the sum and the accumulator by alpha once, then "
             "adds the new tile. The answer is exact, in one pass. The state is fp32. Only the two "
             "matrix-multiply inputs are fp16.\n\n"
             "The kernel idea: one block owns 64 query rows. Q stays resident. K and V stream in tiles of "
             "64. For each tile: compute scores, mask, update the softmax state, accumulate P times V. "
             "Scores, probabilities, and accumulator stay in registers. No N-by-N matrix reaches DRAM.\n\n"
             "Results. At N of 2048, non-causal, CUDA takes 664 microseconds. cuTile takes 2991. That is "
             "4.5 times. On the causal path it is 364 against 448, which is 1.23 times. The unfused "
             "baseline takes 13 milliseconds. The chart is log scale, so each gridline is ten times "
             "larger. CUDA is the fastest of the three at all sizes. The 107 percent means the kernel "
             "reaches the tensor-core ceiling I measured with a separate microbenchmark. Say \"at the "
             "measured ceiling\". If asked why it is above 100 percent, say the ceiling is a measured "
             "microbenchmark, not a hardware specification, and that I have not isolated the cause. "
             "Correctness: both kernels pass an fp64 oracle at every size and mask, and they agree to "
             "2.5e-04.\n\n"
             "If asked about the hardware: RTX 3050 Laptop, sm_86, 16 SMs, 4 GB, WSL2, CUDA 12.6. Every "
             "number is the mean of six repetitions of the p50.")


# ------------------------------------------------------------------ slide 3
def slide_three(prs):
    s = blank(prs)
    text(s, M, 0.35, CW, 0.7,
         "Five problems. Each one was found by measuring, not guessing.", 30, NAVY, bold=True)
    box(s, M, 1.1, 0.9, 0.06, ACCENT)

    mono = {"font": "Consolas"}
    rows = [
        ["What broke", "What it cost", "How it was found and fixed"],
        ["32-way shared-memory bank conflict",
         "9,146 µs instead of 3,594 µs",
         [("Every lane hit one bank. Padded strides to 68/72 and interleaved lane columns. ", {}),
          ("2.5× faster", {"bold": True}), (", same error.", {})]],
        ["Staging loaded 2 bytes at a time",
         "3,207.8 µs instead of 2,091.4 µs",
         [("GPU assembly (SASS) showed 19 scalar 2-byte loads (", {}),
          ("LDG.E.U16", mono), ("). Switched to 16-byte loads. ", {}),
          ("1.53× faster.", {"bold": True})]],
        ["WMMA hides the accumulator layout",
         "2,172.7 µs. 1 block per SM, 8.3% occupancy.",
         [("Rewrote with ", {}), ("mma.sync m16n8k16", mono),
          (", whose layout is exposed. Accumulator now in registers: ", {}),
          ("664.2 µs, 3.3× faster.", {"bold": True})]],
        ["cuTile timing was unfair",
         "cuTile read 107.5 µs. The true value was 42.0 µs.",
         "The tile launcher synchronised the device on every call. The CUDA path did not. Found by "
         "comparing event timing with wall clock. Removed it."],
        ["cuTile non-causal kernel was not fused",
         "322 MB written to DRAM. Mine writes 1.7 MB.",
         [("The source looked fused. Only Nsight Compute (", {}), ("ncu", mono),
          (") counters showed the leak. The same source compiled for the causal case writes 1.77 MB.",
           {})]],
    ]
    table(s, M, 1.35, [3.0, 2.7, 6.433], [0.35, 0.6, 0.6, 0.6, 0.6, 0.6], rows,
          font_size=12,
          col_styles={0: {"bold": True, "color": NAVY}, 1: {"bold": True, "color": ACCENT}},
          align=[PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.LEFT])

    label(s, M, 4.85, CW,
          "THE DECISION THAT MATTERED: WHERE THE ACCUMULATOR LIVES.  CUDA v1 → v2")
    stats = [("Registers per thread", "56", "140"),
             ("Shared memory per block", "72,192", "46,080 B"),
             ("Blocks per SM", "1", "2"),
             ("Tensor-pipe utilisation", "13.2%", "42.2%")]
    for i, (lab, before, after) in enumerate(stats):
        cx = M + i * 3.08
        box(s, cx, 5.2, 2.9, 0.7, PANEL)
        text(s, cx + 0.1, 5.25, 2.7, 0.25, lab, 12, GREY)
        text(s, cx + 0.1, 5.5, 2.7, 0.3,
             [(before + " → ", {}), (after, {"color": ACCENT, "bold": True})],
             18, NAVY, bold=True)

    box(s, M, 6.0, CW, 0.6, NAVY)
    text(s, 0.8, 6.03, 11.73, 0.54,
         [[("Conclusion:", {"bold": True}),
           (" write it in cuTile first, to learn whether the operator matters. Move to CUDA when a "
            "counter, not an intuition, says the remaining gap is where the accumulator lives.", {})]],
         14, WHITE, anchor=MSO_ANCHOR.MIDDLE)

    def footer(x, w, head, lines, body_y=6.9, body_h=0.45, lead=0.95):
        text(s, x, 6.7, w, 0.22, head, 10, ACCENT, bold=True)
        return text(s, x, body_y, w, body_h, lines, 10, GREY, spacing=lead)

    footer(0.6, 3.6, "TOOLS",
           ["CUDA 12.6: mma.sync, ldmatrix, cp.async",
            "cuTile (Python) · cuBLAS",
            "Nsight Compute · NumPy fp64 oracle"])
    footer(4.4, 3.7, "REPO",
           ["github.com/harshith0518/",
            "Attention-Operator-in-CUDA-cuTile-with-fused-computation"])

    url = "https://github.com/harshith0518/Attention-Operator-in-CUDA-cuTile-with-fused-computation"
    for para in s.shapes[-2].text_frame.paragraphs:
        for run in para.runs:
            run.hyperlink.address = url

    footer(8.3, 3.4, "REFERENCES",
           ["Dao et al., FlashAttention, NeurIPS 2022",
            "Dao, FlashAttention-2, 2023",
            "Milakov & Gimelshein, Online softmax, 2018",
            "NVIDIA: PTX ISA, cuTile docs"],
           body_y=6.87, body_h=0.3, lead=0.9)

    slide_number(s, "CS23B010 · 3 / 3", y=7.2, h=0.2)

    notes(s, "Five things went wrong. Each one was found with a measurement.\n"
             "One: a shared-memory bank conflict. Shared memory has 32 banks. Every lane in a warp hit "
             "the same bank, so the accesses ran one after another. I padded the row stride to 68 and 72 "
             "elements and interleaved the lane columns. Time fell from 9.1 to 3.6 milliseconds. Error "
             "did not change.\n"
             "Two: loading K and V two bytes at a time. The assembly showed 19 scalar load sites. I "
             "switched to 16-byte loads. 3.2 to 2.1 milliseconds.\n"
             "Three, the big one: where the accumulator lives. WMMA does not expose the register layout "
             "of its accumulator, so a per-row rescale is not possible there. Version 1 therefore kept "
             "the accumulator in shared memory. That cost a round trip per tile and used 72 kilobytes, "
             "so only one block fit on each SM. Version 2 uses mma.sync, whose layout is exposed. The "
             "accumulator moved to registers. Shared memory fell to 46 kilobytes, two blocks fit, and "
             "tensor-pipe use rose from 13 to 42 percent. Time fell from 2,173 to 664 microseconds.\n"
             "Four: cuTile timing looked unfair. The tile launcher synchronised the whole device on every "
             "call and my CUDA path did not. I found it by comparing CUDA-event time with wall-clock "
             "time. After the fix, cuTile at N of 256 reads 42 microseconds, not 107.\n"
             "Five: the compiler's non-causal kernel was not fused. Reading the source did not show it. "
             "Nsight Compute counters did: 322 megabytes written to DRAM, against 1.7 for mine. The same "
             "source compiled for the causal case stays on chip.\n"
             "Conclusion: write it in cuTile first, to learn whether the operator matters. Move to CUDA "
             "when a counter says the gap is where the accumulator lives. Then hand over to the demo.\n\n"
             "Terms, if asked. SM: streaming multiprocessor, one of 16 compute units on this GPU. "
             "Occupancy: how many warps are active compared with the maximum. Tensor pipe: the hardware "
             "unit that runs matrix multiply. Fused: several steps run in one kernel, so intermediate "
             "data stays on chip.")


def main() -> int:
    prs = new_deck()
    slide_one(prs)
    slide_two(prs)
    slide_three(prs)
    prs.save(OUT)
    print(f"saved {OUT.name}: {len(prs.slides._sldIdLst)} slides")
    return 0


if __name__ == "__main__":
    sys.exit(main())
