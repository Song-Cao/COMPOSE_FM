"""Figure 1 -- the geometric claim, drawn entirely from ALREADY-MEASURED results.

Every number is read back from results/*.json. Nothing here recomputes a model, so the
figure cannot drift from the verified numbers in docs/PROPOSITIONS.md.

THE ONE SENTENCE THIS FIGURE MAKES TRUE (figure-style 7.2): the method's velocity field is
chart-covariant and structurally exact by CONSTRUCTION, and each architectural component is
load-bearing -- the alternatives that look simpler measurably break the property.

Panel a  chart covariance across dtypes, charts and component choices, against the two
         alternatives that BREAK it (coordinate gating, latent eps*I). Log axis, because
         the contrast spans 13 orders of magnitude.
Panel b  the bit-exact structural guarantees (singleton vanishing, zero-exposure,
         zero-dose flow identity, permutation invariance). Exact zeros are drawn at the
         float64 epsilon floor with an explicit "= 0" annotation, because a zero cannot
         be plotted on a log axis and pretending otherwise would misrepresent it.
Panel c  the O(kr) vs O(k^2) interaction-moment cost curve.

USAGE
    PYTHONPATH=src python experiments/make_fig1.py
"""
from __future__ import annotations

import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import (ALARM, FOCAL, FOCAL_LIGHT, META, NEUTRAL,  # noqa: E402
                      apply_figure_style, goodness_cue, greyscale_check, overlaps,
                      panel_crops, panel_letter)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "results")
OUT_DIRS = (os.path.join(RESULTS, "figures"), os.path.join(HERE, "paper", "figures"))

# float64 machine epsilon -- where an EXACT zero is drawn on a log axis, always labelled
EPS64 = float(np.finfo(np.float64).eps)
FLOOR = 1e-17          # below eps64, so an exact zero is visibly distinct from ~1e-16


def load() -> dict:
    with open(os.path.join(RESULTS, "ihcfm_geometry_selftest.json")) as fh:
        geo = json.load(fh)
    with open(os.path.join(RESULTS, "ihcfm_hierarchy_selftest.json")) as fh:
        hier = json.load(fh)
    with open(os.path.join(RESULTS, "propositions.json")) as fh:
        props = json.load(fh)
    return dict(geo=geo, hier=hier, props=props)


def panel_a(ax, D) -> None:
    """Chart covariance: the implemented field vs the arms that break covariance."""
    geo, props = D["geo"], D["props"]
    t1 = geo["test1_whole_field_covariance"]
    t5 = geo["test5_metric_radial_saturation"]
    stress = t5["stress_bound_binding"]

    rows = [
        ("Implemented field, float64", t1["rel_err"], "ok"),
        ("  same, stronger chart", geo["test1_stronger_chart"]["rel_err"], "ok"),
        ("  same, 50x drive", stress["b_scale_50"]["covariance_rel_err"], "ok"),
        ("  same, 1000x drive", stress["b_scale_1000"]["covariance_rel_err"], "ok"),
        ("Random nonlinear decoder + chart",
         props["prop3"]["P3_wholefield_covariance_relerr"], "ok"),
        ("Implemented field, float32", geo["test1_float32_for_contrast"]["rel_err"],
         "dtype"),
        ("Coordinate scalar gate (ablation)",
         t5["coordinate_arm_worst_case_rel_err"]["coord_scalar"], "bad"),
        ("Coordinate norm gate (ablation)",
         t5["coordinate_arm_worst_case_rel_err"]["coord_norm"], "bad"),
        (r"Latent $\epsilon I$, $\epsilon=10^{-2}$ (ablation)",
         geo["test2_latent_eps_ablation_rel_err"]["latent_eps_0.01"], "bad"),
    ]
    colours = dict(ok="#1f4e79", dtype="#888888", bad="#b8471f")
    markers = dict(ok="o", dtype="s", bad="X")
    y = np.arange(len(rows))[::-1]
    for yy, (lab, val, kind) in zip(y, rows):
        ax.plot([1e-17, val], [yy, yy], color=colours[kind], lw=0.7, alpha=0.5, zorder=1)
        ax.plot([val], [yy], marker=markers[kind], ms=5.5, color=colours[kind],
                mec="white", mew=0.5, zorder=3)
    ax.axvline(EPS64, color="#444444", lw=0.8, ls=":", zorder=0)
    ax.text(EPS64 * 1.5, len(rows) - 0.55, r"float64 $\epsilon$", fontsize=6,
            color="#444444", ha="left", va="center")
    ax.axvline(1e-10, color="#444444", lw=0.8, ls="--", zorder=0)
    ax.text(1e-10 / 1.8, -0.42, "pass threshold", fontsize=6, color="#444444",
            ha="right", va="center")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xscale("log")
    ax.set_xlim(1e-17, 3e-1)
    ax.set_xticks([1e-16, 1e-12, 1e-8, 1e-4, 1e-1])
    ax.set_xlabel("Relative chart-covariance error")
    ax.set_ylim(-1.0, len(rows) - 0.2)
    ax.set_title("Covariance holds to machine precision;\nthe simpler gates break it",
                 loc="left")
    # the headline number and the contrast it beats (2.5: only the quotable ones)
    ax.annotate(f"{rows[0][1]:.2e}", (rows[0][1], y[0]), textcoords="offset points",
                xytext=(9, 0), fontsize=6, va="center", color=colours["ok"])
    # the worst coordinate-gate arm: the contrast that justifies the metric-radial
    # saturator, and the number carried in the project record (2.20e-02)
    ax.annotate(f"{rows[6][1]:.2e}", (rows[6][1], y[6]), textcoords="offset points",
                xytext=(9, 0), fontsize=6, va="center", color=colours["bad"])
    ax.text(0.985, 0.03, "lower = better", transform=ax.transAxes, fontsize=6,
            ha="right", va="bottom", color="#444444")


def panel_b(ax, D) -> None:
    """The bit-exact structural guarantees. Exact zeros are labelled, never implied."""
    geo, hier = D["geo"], D["hier"]
    sv = hier["singleton_vanishing"]
    zd = hier["zero_dose_identity"]
    pi = hier["permutation_invariance"]
    t4 = geo["test4_hill_dose"]

    rows = [
        ("Interaction on singletons\n(15/15 cases)", sv["max_abs_interaction_on_singletons"]),
        ("Second moment on singletons", sv["max_abs_m2_on_singletons"]),
        ("Zero-exposure embedding\n(5/5 cases)",
         sv["max_dev_perturbing_zero_exposure_embedding"]),
        ("Dropped-member contribution", sv["max_abs_dropped_member_contribution"]),
        ("Zero-dose flow identity", zd["max_abs_flow_identity_defect"]),
        ("Hill dose at $\\tau=0$", t4["max_abs_at_tau0"]),
        ("Permutation invariance\nof the one-form", pi["max_abs_dev_b"]),
        ("Saturator direction\npreservation", geo["test5_metric_radial_saturation"][
            "direction"]["max_abs_cosine_deviation"]),
    ]
    y = np.arange(len(rows))[::-1]
    for yy, (lab, val) in zip(y, rows):
        exact = (val == 0.0)
        x = FLOOR if exact else val
        col = "#1f4e79" if exact else "#5a7fa6"
        ax.plot([FLOOR, x], [yy, yy], color=col, lw=0.7, alpha=0.5, zorder=1)
        ax.plot([x], [yy], marker=("D" if exact else "o"), ms=(5.0 if exact else 5.5),
                color=col, mec="white", mew=0.5, zorder=3)
        ax.annotate("exactly 0" if exact else f"{val:.1e}", (x, yy),
                    textcoords="offset points", xytext=(9, 0), fontsize=6,
                    va="center", color=col)
    ax.axvline(EPS64, color="#444444", lw=0.8, ls=":", zorder=0)
    ax.text(EPS64 * 1.6, len(rows) - 0.55, r"float64 $\epsilon$", fontsize=6,
            color="#444444", ha="left", va="center")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xscale("log")
    ax.set_xlim(FLOOR * 0.6, 3e-13)
    ax.set_xticks([1e-16, 1e-15, 1e-14])
    ax.set_xlabel("Maximum absolute defect")
    # headroom below the lowest row so the key sits in whitespace instead of colliding
    # with the bottom row's value annotation
    ax.set_ylim(-2.0, len(rows) - 0.2)
    # 1.4: the title is tested against every plotted row. Six of the eight rows are
    # bit-exact 0.0; permutation invariance and saturator direction land at the float64
    # rounding level instead, so the title says "6 of 8" rather than implying all eight.
    n_exact = sum(1 for _, v in rows if v == 0.0)
    ax.set_title(f"Structural guarantees are bit-exact\non {n_exact} of {len(rows)} "
                 f"checks; the rest are rounding-level", loc="left")
    ax.plot([], [], marker="D", ms=5, color="#1f4e79", ls="none",
            label="exact zero (drawn at axis floor)")
    ax.plot([], [], marker="o", ms=5, color="#5a7fa6", ls="none",
            label="rounding-level, not exact")
    ax.legend(loc="lower right", fontsize=6, handletextpad=0.4, borderpad=0.3)


def panel_c(ax, D) -> None:
    """Interaction-moment cost: O(kr) against the O(k^2) pairwise construction."""
    mc = D["hier"]["moment_cost"]
    per = mc["per_k"]
    ks = np.array([p["k"] for p in per], dtype=float)
    fast = np.array([p["t_okr_s"] for p in per]) * 1e3          # ms
    slow = np.array([p["t_ok2_s"] for p in per]) * 1e3
    ax.plot(ks, slow, marker="s", ms=5, color="#b8471f", lw=1.4,
            label=r"explicit pairwise, $O(k^2)$")
    ax.plot(ks, fast, marker="o", ms=5, color="#1f4e79", lw=1.4,
            label=r"factored moments, $O(kr)$")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([f"{int(k)}" for k in ks])
    # Pin the decade ticks to the data range. Left to itself the log locator emits
    # 10^-2 and 10^2 outside the view, whose labels render off-canvas.
    ax.set_ylim(0.7 * fast.min(), 1.6 * slow.max())
    ax.set_yticks([0.05, 0.1, 0.5, 1.0, 5.0, 10.0])
    ax.set_yticklabels(["0.05", "0.1", "0.5", "1", "5", "10"])
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_xlabel("Number of interventions $k$")
    ax.set_ylabel("Moment cost per batch (ms)")
    ax.set_title("Interaction cost grows with $kr$,\nnot $k^2$", loc="left")
    # The headline number, placed in the gap between the two curves at the right edge
    # (5.2: the one value a reader would quote; everything else is read off the axis).
    ax.annotate(f"{mc['ratio_last']:.0f}$\\times$ faster\nat $k$ = {int(ks[-1])}",
                (ks[-1], np.sqrt(fast[-1] * slow[-1])), textcoords="offset points",
                xytext=(-4, -14), ha="right", va="center", fontsize=6.5,
                color="#333333")
    ax.text(0.98, 0.035,
            f"log-log slopes: {mc['loglog_slope_ok2']:.2f} vs "
            f"{mc['loglog_slope_okr']:.2f}",
            transform=ax.transAxes, fontsize=6, ha="right", va="bottom",
            color=NEUTRAL)
    # The upper-left of this panel is empty (both curves start low and rise to the
    # right), so the key goes there. Direct end-of-line labels were tried first and sat
    # on top of the curves at every offset that stayed inside the axes.
    ax.legend(loc="upper left", fontsize=6.5, handletextpad=0.5, borderpad=0.2,
              handlelength=1.6, labelspacing=0.35)
    # No direction-of-goodness cue here: one cue serves the row (panel a), and "cost"
    # already carries its direction.


def build(D) -> mpl.figure.Figure:
    fig = plt.figure(figsize=(13.8, 4.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.26, 1.12, 1.02],
                          wspace=0.60, left=0.150, right=0.985, top=0.845, bottom=0.135)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    panel_a(ax_a, D)
    panel_b(ax_b, D)
    panel_c(ax_c, D)
    for ax, L, dx in ((ax_a, "a", -0.52), (ax_b, "b", -0.40), (ax_c, "c", -0.235)):
        panel_letter(ax, L, dx=dx, dy=1.045)
    return fig


def main() -> dict:
    apply_figure_style(frame="open", sizes=(8, 7, 6))
    D = load()
    fig = build(D)
    for d in OUT_DIRS:
        os.makedirs(d, exist_ok=True)
    paths = [os.path.join(d, "fig1_theory.png") for d in OUT_DIRS]
    fig.savefig(paths[0], dpi=200)
    issues = overlaps(fig)
    for p in paths[1:]:
        fig.savefig(p, dpi=200)
    print(f"overlaps: {len(issues)}")
    for a, b in issues[:14]:
        print(f"   {a!r:<46} <-> {b!r}")
    print("greyscale:", greyscale_check(paths[0]))
    for p in paths:
        print("wrote", p)
    return dict(paths=paths, issues=issues, fig=fig, crops=panel_crops(fig))


if __name__ == "__main__":
    main()
