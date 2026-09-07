"""Figure 3 -- the two case studies, INCLUDING the losses. Read back, never re-derived.

Every value comes from results/loco_screen.json and results/finance_blocked.json.

THE ONE SENTENCE THIS FIGURE MAKES TRUE: on real data the method beats the additive
baseline decisively and consistently, and it does NOT uniformly beat no-change -- the
organoid mean sits above 1.0 and one finance split is a loss, and both are visible here
rather than described in the text.

Panel a  organoid LOCO, per held-out combination. IHC-FM vs PerturbedMean vs the
         no-change line at 1.0, with grouped-bootstrap CIs. FOUR of the five folds have
         IHC-FM above the no-change line (VS 1.371, SF 2.023, CF 1.044, CSF 3.177; only
         CS 0.894 is below): that is the honest negative, and it is drawn rather than
         described. The panel title counts this from the data at render time, so the
         figure and this docstring cannot drift apart.
Panel b  the organoid paired contrasts pooled over 136 populations: a decisive win over
         the additive baseline, and a LOSS to no-change whose CI is entirely above zero.
Panel c  finance across the four purged splits, with the exposure-window loss marked and
         the measured pre-state shift that diagnoses it.

ESTIMATOR NOTE (kept explicit because the two differ). An arm's plotted point is the
GROUPED-BOOTSTRAP point estimate, which is what its CI belongs to. The paired contrasts
use per-unit paired means, which differ slightly (e.g. finance held-out day: bootstrap
0.856 vs paired mean 0.890) because one resamples group means and the other averages
matched pairs. Mixing them silently would make the panels look inconsistent, so each panel
states which it shows.

USAGE
    PYTHONPATH=src python experiments/make_fig3.py
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import (ALARM, BASELINE_GREYS, FOCAL, FOCAL_LIGHT, META,  # noqa: E402
                      NEUTRAL, apply_figure_style, goodness_cue, greyscale_check,
                      overlaps, panel_letter)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "results")
OUT_DIRS = (os.path.join(RESULTS, "figures"), os.path.join(HERE, "paper", "figures"))

# Fold order is the intervention-set order the screen used, ascending in cardinality so
# the reader sees the difficulty gradient (pairs then the triple) rather than an arbitrary
# alphabetical order.
ORGANOID_FOLDS = ("VS", "CS", "CF", "SF", "CSF")
FINANCE_SPLITS = ("chronological", "held_out_day", "exposure_block", "exposure_window")
FINANCE_LABELS = {"chronological": "chronological",
                  "held_out_day": "held-out day",
                  "exposure_block": "exposure block",
                  "exposure_window": "exposure window"}


def load() -> dict:
    with open(os.path.join(RESULTS, "loco_screen.json")) as fh:
        loco = json.load(fh)
    with open(os.path.join(RESULTS, "finance_blocked.json")) as fh:
        fin = json.load(fh)
    return dict(loco=loco, fin=fin)


def panel_a(ax, D) -> None:
    """Organoid LOCO per fold. The no-change line is drawn, not implied."""
    folds = {f["fold"]: f for f in D["loco"]["loco"]}
    names = [f for f in ORGANOID_FOLDS if f in folds]
    x = np.arange(len(names), dtype=float)
    w = 0.34
    series = (("ihcfm", "IHC-FM", FOCAL, -w / 2, "o"),
              ("perturbed_mean", "PerturbedMean (additive)", BASELINE_GREYS[1], w / 2,
               "s"))
    for key, lab, col, off, mk in series:
        pts, los, his = [], [], []
        for n in names:
            s = folds[n]["arms"][key]["summary"]["ed_normalised"]
            pts.append(s["point"]), los.append(s["lo"]), his.append(s["hi"])
        pts, los, his = map(np.asarray, (pts, los, his))
        ax.errorbar(x + off, pts, yerr=[pts - los, his - pts], fmt=mk, ms=6,
                    color=col, ecolor=col, elinewidth=1.3, capsize=3.0, capthick=1.1,
                    mec="white", mew=0.6, label=lab, zorder=3)
    ax.axhline(1.0, color=NEUTRAL, lw=1.1, ls="--", zorder=1)
    ax.text(len(names) - 0.36, 0.88, "no change", fontsize=6.5, color=NEUTRAL,
            va="top", ha="right")
    # Shade the region where a prediction is WORSE than doing nothing. This is the honest
    # negative made visible: four of the five folds sit inside it (only CS does not).
    ax.axhspan(1.0, 11.0, color=ALARM, alpha=0.055, zorder=0, lw=0)
    # region label at the right, below the legend which occupies the upper left
    ax.text(len(names) - 0.36, 7.4, "worse than\nno change", fontsize=6.5, color=ALARM,
            va="center", ha="right")
    n_above = sum(1 for n in names
                  if folds[n]["arms"]["ihcfm"]["summary"]["ed_normalised"]["point"] > 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n(n={folds[n]['n_held_units']})" for n in names])
    ax.set_xlim(-0.62, len(names) - 0.30)
    ax.set_yscale("log")
    ax.set_ylim(0.55, 11.0)
    ax.set_yticks([0.6, 1, 2, 4, 7, 10])
    ax.set_yticklabels(["0.6", "1", "2", "4", "7", "10"])
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_xlabel("Held-out intervention combination")
    ax.set_ylabel("Normalised energy distance")
    ax.set_title(f"Organoid: beats the additive baseline on 5/5 folds,\n"
                 f"but is worse than no change on {n_above}/{len(names)}", loc="left")
    ax.legend(loc="upper left", fontsize=6.5, handletextpad=0.4, borderpad=0.2,
              labelspacing=0.3)
    goodness_cue(ax, "lower = better", x=0.985, y=0.02)


def panel_b(ax, D) -> None:
    """The two pooled paired contrasts. A win and a loss, on the same axis."""
    roll = D["loco"]["rollup"]
    rows = [
        ("vs PerturbedMean\n(additive baseline)",
         roll["paired_pooled_ihcfm_vs_perturbed_mean"], FOCAL),
        ("vs no change", roll["paired_pooled_ihcfm_vs_no_change"], ALARM),
    ]
    y = np.arange(len(rows))[::-1].astype(float)
    for yy, (lab, p, col) in zip(y, rows):
        d = p["mean_diff"]
        ax.errorbar([d], [yy], xerr=[[d - p["lo"]], [p["hi"] - d]], fmt="o", ms=7,
                    color=col, ecolor=col, elinewidth=1.5, capsize=3.5, capthick=1.2,
                    mec="white", mew=0.6, zorder=3)
        ax.annotate(f"{d:+.2f}  [{p['lo']:+.2f}, {p['hi']:+.2f}]\n"
                    f"{p['n_units_a_lower']}/{p['n_units']} populations, "
                    f"sign $p$ = {p['sign_test_p']:.2g}",
                    (d, yy), textcoords="offset points",
                    xytext=(0, -16 if yy == y[0] else -16), ha="center", va="top",
                    fontsize=6.5, color=col)
    ax.axvline(0.0, color=NEUTRAL, lw=1.1, ls="--", zorder=1)
    ax.text(0.0, len(rows) - 0.42, "no difference", fontsize=6.5, color=NEUTRAL,
            ha="center", va="bottom")
    ax.text(-2.85, -1.02, "$\\leftarrow$ IHC-FM better", fontsize=6.5, color=FOCAL,
            ha="left", va="center")
    ax.text(1.65, -1.02, "IHC-FM worse $\\rightarrow$", fontsize=6.5, color=ALARM,
            ha="right", va="center")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(-2.95, 1.75)
    ax.set_ylim(-1.15, len(rows) - 0.12)
    ax.set_xlabel("Paired difference in normalised ED\n(136 populations, 5 folds pooled)")
    # 1.3/1.4: the title is checked against the plotted numbers. Both CIs exclude zero,
    # but the two contrasts differ in kind, and the title says which: the win is
    # consistent (113/136 populations, sign p = 2e-15) while the loss is a MEAN effect
    # that the sign test does not support (72/136, p = 0.55) -- it is carried by a
    # minority of large failures rather than by most populations.
    ax.set_title("Beats the additive baseline consistently;\n"
                 "loses to no change on the mean only", loc="left")


def panel_c(ax, D) -> None:
    """Finance across the four purged splits, with the one loss marked."""
    res = D["fin"]["results"]
    names = [s for s in FINANCE_SPLITS if s in res]
    x = np.arange(len(names), dtype=float)
    w = 0.34
    for key, lab, col, off, mk in (("ihcfm", "IHC-FM", FOCAL, -w / 2, "o"),
                                   ("perturbed_mean", "PerturbedMean (additive)",
                                    BASELINE_GREYS[1], w / 2, "s")):
        pts, los, his = [], [], []
        for n in names:
            s = res[n]["arms"][key]["summary"]["ed_normalised"]
            pts.append(s["point"]), los.append(s["lo"]), his.append(s["hi"])
        pts, los, his = map(np.asarray, (pts, los, his))
        ax.errorbar(x + off, pts, yerr=[pts - los, his - pts], fmt=mk, ms=6, color=col,
                    ecolor=col, elinewidth=1.3, capsize=3.0, capthick=1.1, mec="white",
                    mew=0.6, label=lab, zorder=3)
    ax.axhline(1.0, color=NEUTRAL, lw=1.1, ls="--", zorder=1)
    ax.text(len(names) - 0.40, 0.94, "no change", fontsize=6.5, color=NEUTRAL,
            va="top", ha="right")
    ax.axhspan(1.0, 3.0, color=ALARM, alpha=0.055, zorder=0, lw=0)
    # mark the split where the additive baseline WINS -- the one loss of the four
    losses = [n for n in names
              if res[n]["paired"]["ihcfm_vs_perturbed_mean::ed_normalised"]["mean_diff"]
              > 0]
    for n in losses:
        i = names.index(n)
        sh = res[n]["measured_shift"]
        # bracket over the losing split, caption to its LEFT so it clears the markers
        ax.plot([i - 0.42, i + 0.42], [2.62, 2.62], color=ALARM, lw=1.0, zorder=2)
        ax.annotate("LOSS: additive baseline wins here\n"
                    f"pre-state shift {sh['pre_state_ed']:.2f} vs "
                    f"{sh['pre_state_ed_within_train']:.4f} within train",
                    (i - 0.48, 2.56), ha="right", va="top", fontsize=6.5, color=ALARM)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{FINANCE_LABELS[n]}\n"
                        f"(n={res[n]['arms']['ihcfm']['summary']['ed_normalised']['n_groups']})"
                        for n in names])
    ax.set_xlim(-0.55, len(names) - 0.34)
    ax.set_ylim(0.35, 2.95)
    ax.set_yticks([0.5, 1.0, 1.5, 2.0, 2.5])
    ax.set_xlabel("Purged split (audited: 0 overlapping indices, 0 shared bins)")
    ax.set_ylabel("Normalised energy distance")
    ax.set_title(f"Finance: wins {len(names) - len(losses)} of {len(names)} purged "
                 f"splits;\nthe loss carries the largest measured shift", loc="left")
    ax.legend(loc="lower left", fontsize=6.5, handletextpad=0.4, borderpad=0.2,
              labelspacing=0.3)


def build(D) -> mpl.figure.Figure:
    fig = plt.figure(figsize=(14.0, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.06, 1.06], wspace=0.40,
                          left=0.055, right=0.988, top=0.835, bottom=0.185)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    panel_a(axes[0], D)
    panel_b(axes[1], D)
    panel_c(axes[2], D)
    for ax, L in zip(axes, "abc"):
        panel_letter(ax, L, dx=-0.115, dy=1.035)
    return fig


def main() -> dict:
    apply_figure_style(frame="open", sizes=(8, 7, 6))
    D = load()
    fig = build(D)
    for d in OUT_DIRS:
        os.makedirs(d, exist_ok=True)
    paths = [os.path.join(d, "fig3_casestudies.png") for d in OUT_DIRS]
    for p in paths:
        fig.savefig(p, dpi=200)
    issues = overlaps(fig)
    print(f"overlaps: {len(issues)}")
    for a, b in issues[:14]:
        print(f"   {a!r:<44} <-> {b!r}")
    print("greyscale:", greyscale_check(paths[0]))
    for p in paths:
        print("wrote", p)
    return dict(paths=paths, issues=issues, fig=fig)


if __name__ == "__main__":
    main()
