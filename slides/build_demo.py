"""Build the three-slide demo deck.

The brief for the demo is specific: three slides only, covering the title block with names and roll
numbers, the problem statement with results, and the challenges with how they were resolved. Each
team gets ten minutes, then a live demo.

Design: 16:9, white ground, one slate navy for structure, one warm accent reserved for the numbers
that matter. Nothing decorative. Big figures carry the argument, and the prose is short enough to say
out loud in the time available.

Run:  python slides/build_demo.py
"""

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Emu, Inches, Pt

OUT = Path(__file__).parent / "Harshith_GPU_Demo_3slides.pptx"

# geometry: 13.333 x 7.5 inches
W, H = 13.333, 7.5
M = 0.72                      # left and right margin
CW = W - 2 * M                # content width

NAVY = RGBColor(0x14, 0x2B, 0x4F)
INK = RGBColor(0x1E, 0x1E, 0x1E)
GREY = RGBColor(0x5A, 0x5F, 0x66)
LIGHT = RGBColor(0xEC, 0xEE, 0xF1)
ACCENT = RGBColor(0xB4, 0x3C, 0x2A)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

TITLE = "Fused attention on one GPU"
SUB = "One operator, written by hand and by compiler, then measured"


def new_deck() -> Presentation:
    prs = Presentation()
    prs.slide_width = Inches(W)
    prs.slide_height = Inches(H)
    return prs


def blank(prs: Presentation):
    return prs.slides.add_slide(prs.slide_layouts[6])


def box(slide, x, y, w, h, lines, size=16, color=INK, bold=False, align=PP_ALIGN.LEFT,
        anchor=MSO_ANCHOR.TOP, spacing=1.0, italic=False):
    """A text frame. `lines` is a string or a list of strings, one paragraph each."""
    tf = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)).text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    if isinstance(lines, str):
        lines = [lines]
    for i, text in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        run = p.add_run()
        run.text = text
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.italic = italic
        run.font.color.rgb = color
        run.font.name = "Calibri"
    return tf


def rect(slide, x, y, w, h, fill, line=None):
    from pptx.enum.shapes import MSO_SHAPE
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(0.75)
    # A shape carries a theme style reference, and that is where the shadow comes from. Removing the
    # reference and writing an empty effect list gives a genuinely flat rectangle.
    from pptx.oxml.ns import qn
    style = shp._element.find(qn("p:style"))
    if style is not None:
        shp._element.remove(style)
    spPr = shp._element.spPr
    for e in spPr.findall(qn("a:effectLst")):
        spPr.remove(e)
    spPr.append(spPr.makeelement(qn("a:effectLst"), {}))
    return shp


def footer(slide, n):
    rect(slide, M, H - 0.62, CW, 0.012, LIGHT)
    box(slide, M, H - 0.52, 8.0, 0.3, "Attention on the GPU: CUDA C++ vs cuTile", 10, GREY)
    box(slide, W - M - 1.2, H - 0.52, 1.2, 0.3, f"{n} / 3", 10, GREY, align=PP_ALIGN.RIGHT)


def table(slide, x, y, w, rows, col_w, size=12, head=True, row_h=0.34, aligns=None):
    """Rows is a list of tuples. The first row is the header when head is True."""
    nrows, ncols = len(rows), len(rows[0])
    shape = slide.shapes.add_table(nrows, ncols, Inches(x), Inches(y), Inches(w),
                                   Inches(row_h * nrows)).table
    for j, cw in enumerate(col_w):
        shape.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = shape.cell(i, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = NAVY if (head and i == 0) else (WHITE if i % 2 else LIGHT)
            cell.margin_left = cell.margin_right = Inches(0.07)
            cell.margin_top = cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = aligns[j] if aligns else (PP_ALIGN.LEFT if j == 0 else PP_ALIGN.RIGHT)
            run = p.add_run()
            text = str(val)
            emph = text.startswith("**")
            run.text = text[2:] if emph else text
            run.font.size = Pt(size)
            run.font.name = "Calibri"
            run.font.bold = (head and i == 0) or emph
            run.font.color.rgb = WHITE if (head and i == 0) else INK
    return shape


# --------------------------------------------------------------------------- slide 1
def slide_title(prs):
    s = blank(prs)
    rect(s, 0, 0, W, 0.09, NAVY)
    rect(s, M, 1.62, 1.05, 0.055, ACCENT)

    box(s, M, 1.95, CW - 0.4, 1.1, TITLE, 44, NAVY, bold=True)
    box(s, M, 3.02, CW - 1.2, 0.9, SUB, 20, GREY)

    box(s, M, 4.35, 6.6, 0.36, "Balla Surya Harshith   ·   Roll number CS23B010", 20, INK, bold=True)
    box(s, M, 4.78, 6.8, 0.34, "CS-6023 GPU Programming   ·   Mentor: Pravin Jagtap", 15, GREY)
    box(s, M, 5.14, 6.8, 0.34, "Course project demo   ·   November 2026", 15, GREY)

    # the one result worth reading from the back of the room
    rect(s, W - M - 3.5, 4.28, 3.5, 1.42, LIGHT)
    box(s, W - M - 3.34, 4.40, 3.2, 0.3, "sequence length 2048, non-causal", 11, GREY)
    box(s, W - M - 3.34, 4.68, 3.2, 0.62, "4.5\u00d7", 40, ACCENT, bold=True)
    box(s, W - M - 3.34, 5.30, 3.2, 0.3, "faster than the compiler's kernel", 12, INK)
    footer(s, 1)


# --------------------------------------------------------------------------- slide 2
def slide_problem_and_results(prs):
    s = blank(prs)
    rect(s, 0, 0, W, 0.09, NAVY)
    box(s, M, 0.40, CW, 0.5, "The problem, and what the two kernels achieve", 30, NAVY, bold=True)

    # ---- left: the problem
    col = 5.55
    box(s, M, 1.18, col, 0.3, "THE OPERATOR", 11, ACCENT, bold=True)
    rect(s, M, 1.50, col, 0.95, LIGHT)
    box(s, M + 0.16, 1.62, col - 0.32, 0.72,
        ["O = softmax( Q K\u1d40 / \u221aD + mask ) \u00b7 V",
         "forward pass only, exact, fp16 in and out, fp32 state"],
        13, INK, spacing=1.15)

    box(s, M, 2.62, col, 1.5, [
        "\u2022  The naive form writes the N\u00d7N score matrix to memory. At N=2048 that is 134 MB, "
        "and it is read again for the second product.",
        "\u2022  So the operator becomes memory-bound: measured at 31 GB/s against 170 GB/s available.",
        "\u2022  Fixing it means fusing all three stages into one kernel, which forces the softmax to "
        "stream and forces a rescale of a live accumulator.",
    ], 13, INK, spacing=1.12)

    box(s, M, 4.30, col, 0.3, "THE QUESTION", 11, ACCENT, bold=True)
    box(s, M, 4.60, col, 0.85,
        "When does the tile model's productivity outweigh the control it gives up? "
        "Answered by measuring, not by opinion.", 13, GREY, italic=True)

    # ---- right: the numbers
    x = M + col + 0.45
    rw = W - M - x
    box(s, x, 1.18, rw, 0.3, "MEASURED, N = 2048, SIX REPETITIONS AVERAGED", 11, ACCENT, bold=True)

    kpis = [("4.5\u00d7", "faster than cuTile", "664 against 2991 \u00b5s"),
            ("107%", "of the measured ceiling", "12.1 TFLOP/s, fp32 accumulate"),
            ("1.23\u00d7", "faster than cuTile, causal", "364 against 448 \u00b5s")]
    kw = (rw - 0.3) / 3
    for i, (big, mid, small) in enumerate(kpis):
        cx = x + i * (kw + 0.15)
        rect(s, cx, 1.50, kw, 1.30, LIGHT)
        box(s, cx + 0.10, 1.58, kw - 0.2, 0.55, big, 26, ACCENT, bold=True)
        box(s, cx + 0.10, 2.14, kw - 0.2, 0.32, mid, 10, INK, bold=True)
        box(s, cx + 0.10, 2.44, kw - 0.2, 0.3, small, 9, GREY)

    box(s, x, 3.02, rw, 0.3, "LATENCY, \u00b5s (LOWER IS BETTER)", 11, ACCENT, bold=True)
    rows = [("kernel", "non-causal", "causal"),
            ("CUDA v2 (shipped)", "**664", "**364"),
            ("cuTile", "2991", "448"),
            ("PyTorch SDPA", "**617", "397"),
            ("unfused baseline", "13315", "15098")]
    table(s, x, 3.32, rw, rows, [2.15, rw - 3.35, 1.2], size=11, row_h=0.30)

    box(s, M, 5.42, col, 0.7,
        "Correctness: both kernels pass an fp64 oracle at every size and both masks, and they agree "
        "with each other to 2.5e-04.", 12, GREY, spacing=1.1)
    footer(s, 2)


# --------------------------------------------------------------------------- slide 3
def slide_challenges(prs):
    s = blank(prs)
    rect(s, 0, 0, W, 0.09, NAVY)
    box(s, M, 0.40, CW, 0.5, "What went wrong, and how each one was found", 30, NAVY, bold=True)
    box(s, M, 1.00, CW, 0.34,
        "Two of these were errors in my own numbers. A wrong measurement is more dangerous than a "
        "slow kernel, because it points the work in the wrong direction.", 12, GREY)

    rows = [
        ("What broke", "What it cost", "How it was found, and the fix"),
        ("32-way shared-memory bank conflict",
         "9.1 ms instead of 3.6 ms",
         "One lane per bank, every access serialised. Padded the stride to 68/72 and interleaved the "
         "lane columns: 2.5\u00d7. Error unchanged to the last digit."),
        ("cuTile looked 1.85\u00d7 faster at N=256",
         "an unfair comparison",
         "The tile launcher synchronised the device on every call and the CUDA path did not. Caught by "
         "comparing event timing with wall clock; the sync was removed."),
        ("Staging moved 2 bytes at a time",
         "3.2 ms instead of 2.1 ms",
         "SASS showed 19 scalar LDG.E.U16 sites. Replaced with 16-byte loads; SASS now shows 8 "
         "LDG.E.128. 1.53\u00d7."),
        ("My kernel was fused. cuTile's was not.",
         "the real explanation",
         "ncu counters: the cuTile non-causal build writes 322 MB to DRAM, mine writes 1.7 MB. Only a "
         "counter separated them; no amount of reading the source showed it."),
    ]
    table(s, M, 1.44, CW, rows, [3.05, 2.00, CW - 5.05], size=10.5, row_h=0.74,
          aligns=[PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.LEFT])

    y = 5.16
    rect(s, M, y, CW, 0.92, LIGHT)
    box(s, M + 0.18, y + 0.12, CW - 0.36, 0.30, "WHAT I WOULD TELL SOMEONE STARTING THIS", 10, ACCENT,
        bold=True)
    box(s, M + 0.18, y + 0.42, CW - 0.36, 0.44,
        "Write it in cuTile first to learn whether the operator matters. Move to CUDA when a counter, "
        "not an intuition, says the remaining gap is where the accumulator lives.", 13, INK)
    footer(s, 3)


def main() -> int:
    prs = new_deck()
    slide_title(prs)
    slide_problem_and_results(prs)
    slide_challenges(prs)
    prs.save(OUT)
    print(f"saved {OUT.name}: {len(prs.slides.__iter__.__self__._sldIdLst)} slides")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
