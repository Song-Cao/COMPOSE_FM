"""Figure mechanics for the paper's deliverable figures. Import, don't reinvent.

This is a repo-local module so that `make_fig*.py` runs standalone (`PYTHONPATH=src
python experiments/make_fig1.py`) and reproduces byte-identical output later. It sets
MECHANICS only -- a role-mapped three-step font ladder, an open frame, outward ticks,
frameless legends, embedded Type-42 fonts and a 200 dpi save default. It imposes no
palette: colour is chosen per figure so that a focal series can be made visually dominant
and every panel stays readable in greyscale.

The rules it encodes (font ladder mapped to role rather than to available space, panel
letters bold and outside the axes, a geometric overlap check before saving, and crop boxes
for a per-panel perceptual check) come from the project's figure-style checklist.
"""
from __future__ import annotations

import matplotlib as mpl
import numpy as np

# ---- shared semantic colours ----------------------------------------------------
# Chosen for greyscale separability (dark navy ~30% luminance, rust ~45%, mid grey ~55%)
# and for deuteranopia safety: navy/rust is a blue-orange contrast, never red/green.
FOCAL = "#1f4e79"        # the method (IHC-FM)
FOCAL_LIGHT = "#5a7fa6"  # the method, secondary arm (e.g. the ablation)
ALARM = "#b8471f"        # the arm that BREAKS a property / the loss
META = "#888888"         # reference lines, thresholds, annotations
NEUTRAL = "#444444"
BASELINE_GREYS = ("#2b2b2b", "#555555", "#7a7a7a", "#9e9e9e", "#bdbdbd")


def apply_figure_style(frame: str = "open", sizes=(8, 7, 6), dpi: int = 200) -> None:
    """Set rcParams once before plotting. `sizes` = (base, secondary, tick).

    base      titles, axis labels, series identity
    secondary legend and annotation text
    tick      tick labels
    Three sizes only: if a label does not fit at its role's size, the layout is wrong.
    """
    base, second, tick = sizes
    mpl.rcParams.update({
        "font.size": float(base),
        "axes.titlesize": float(base),
        "axes.labelsize": float(base),
        "legend.fontsize": float(second),
        "xtick.labelsize": float(tick),
        "ytick.labelsize": float(tick),
        "axes.titlelocation": "left",
        "axes.linewidth": 0.6,
        "axes.spines.top": frame not in ("open", "none"),
        "axes.spines.right": frame not in ("open", "none"),
        "axes.spines.left": frame != "none",
        "axes.spines.bottom": frame != "none",
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3.0, "ytick.major.size": 3.0,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "lines.linewidth": 1.2, "patch.linewidth": 0.6,
        "legend.frameon": False,
        "figure.dpi": float(dpi), "savefig.dpi": float(dpi),
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": None,          # explicit figure geometry, never auto-cropped
    })


def panel_letter(ax, letter: str, dx: float = -0.18, dy: float = 1.02,
                 fontsize: float = 10.0) -> None:
    """Bold panel letter outside the top-left of the axes box."""
    ax.text(dx, dy, letter, transform=ax.transAxes, fontweight="bold",
            fontsize=fontsize, ha="left", va="bottom")


def goodness_cue(ax, text: str = "lower = better", x: float = 0.985,
                 y: float = 0.02, fontsize: float = 6.0) -> None:
    """Upright direction-of-goodness cue in the panel margin. Once per row of panels."""
    ax.text(x, y, text, transform=ax.transAxes, fontsize=fontsize, ha="right",
            va="bottom", color=NEUTRAL)


def overlaps(fig) -> list:
    """Geometric check: visible text boxes that collide, or fall outside the canvas.

    A tick label sitting on its own spine is not a finding. Returns a list of
    (text, other) pairs; an empty list is the gate for saving.
    """
    r = fig.canvas.get_renderer()
    texts = [(t, t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text)
             if t.get_text().strip() and t.get_visible()]
    spines = [(s, s.get_window_extent(r)) for ax in fig.axes
              for s in ax.spines.values() if s.get_visible()]
    own = {ax: set(ax.get_xticklabels(which="both")
                   + ax.get_yticklabels(which="both")) for ax in fig.axes}
    out = [(a.get_text(), b.get_text())
           for i, (a, ba) in enumerate(texts) for b, bb in texts[i + 1:]
           if ba.overlaps(bb)]
    out += [(t.get_text(), "spine") for t, bt in texts for s, bs in spines
            if bt.overlaps(bs) and t not in own.get(s.axes, ())]
    out += [(t.get_text(), "outside canvas") for t, bt in texts
            if not (bt.x0 >= -1 and bt.y0 >= -1
                    and bt.x1 <= fig.bbox.x1 + 1 and bt.y1 <= fig.bbox.y1 + 1)]
    return out


def panel_crops(fig, pad_px: int = 8) -> dict:
    """Pixel crop boxes per lettered panel in the SAVED png, for the perceptual check."""
    r = fig.canvas.get_renderer()
    H = fig.bbox.y1
    boxes = {}
    for ax in fig.axes:
        letter = None
        for t in ax.texts:
            s = t.get_text().strip()
            if len(s) == 1 and s.isalpha() and t.get_fontweight() == "bold":
                letter = s
                break
        if letter is None:
            continue
        bb = ax.get_tightbbox(r)
        boxes[letter] = (max(0, int(bb.x0) - pad_px),
                         max(0, int(H - bb.y1) - pad_px),
                         int(bb.x1) + pad_px,
                         int(H - bb.y0) + pad_px)
    return boxes


def greyscale_check(path: str) -> dict:
    """Luminance spread of the saved png -- a coarse proxy for greyscale readability.

    Returns `ink_fraction` (share of pixels darker than the near-white background),
    `luminance_quantiles` (the 2/25/50/75/98th percentiles of those ink pixels) and
    `span` (the 2nd-to-98th percentile range). A wide span with well-separated interior
    quantiles means the figure's marks are not all landing on one grey level.

    It does NOT identify which series owns which luminance, count distinct levels, or
    measure the gap between the specific levels a reader must tell apart -- so it is a
    smoke test that sits on top of the per-panel visual check, never a replacement for
    it. A blank image returns `n_levels=0` and nothing else.
    """
    import matplotlib.image as mpimg
    im = mpimg.imread(path)
    if im.ndim == 3:
        lum = 0.2126 * im[..., 0] + 0.7152 * im[..., 1] + 0.0722 * im[..., 2]
    else:
        lum = im
    ink = lum[lum < 0.97]
    if ink.size == 0:
        return dict(n_levels=0, note="blank")
    q = np.quantile(ink, [0.02, 0.25, 0.5, 0.75, 0.98])
    return dict(ink_fraction=float(ink.size / lum.size),
                luminance_quantiles=[float(x) for x in q],
                span=float(q[-1] - q[0]))
