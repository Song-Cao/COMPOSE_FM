"""Figure 2 -- Table 1's comparison, with the separation structure and the ablation null.

Read back from results/table1.json. Nothing is refitted.

THE ONE SENTENCE THIS FIGURE MAKES TRUE: all six model arms beat no-change, no adjacent
pair AMONG THOSE SIX is separated, and the interaction hierarchy the paper proposes does
not measurably pay for itself.

Two scope limits on the compact letter display, both easy to overstate.

Adjacent, not all-pairs. The six model arms sharing letter (a) means no ADJACENT pair
among them was separated -- it does NOT mean no pair at all was separated. IHC-FM (full)
vs FactoredAdditiveCFM, two rows apart, IS separated in its paired test (CI
[-0.253, -0.023], excludes zero); it simply fails the project's 0.25 nED claim margin, so
it is reported as a small precisely bounded difference rather than a win. Panel b shows
that contrast directly for this reason.

Within the model block, not the whole ranking. Exactly one adjacent pair in the full
nine-arm ranking IS separated: DeepSetsEndpoint vs NoChange (CI [-0.241, -0.004]). That
is the model/control boundary, and it is precisely the test that creates the a/b letter
break this panel draws. The panel title scopes its claim to the model block and computes
the exception count from the data, so the sentence cannot contradict the letters beside it.

Panel a  the nine arms with grouped-bootstrap CIs and the no-change line at 1.0 drawn
         explicitly. Arms are grouped by the compact letter display, so the reader sees
         that the entire top block shares one letter and must NOT be read as ranked.
Panel b  the two contrasts the paper turns on, as paired differences with CIs: IHC-FM
         full vs FactoredAdditiveCFM (the competitor) and full vs main-effects-only (the
         better-powered within-model ablation). The claim-margin threshold is drawn.
Panel c  why the ablation null is not a floor effect: the achievable floor, the
         interaction-free oracle, and where the two IHC-FM arms actually sit (21 folds x
         3 seeds). The gap between the oracle bars is the interaction signal that EXISTS,
         so a null cannot be blamed on there being nothing to find.
Panel d  what the null actually was -- undertraining. The step-budget sweep on its OWN
         fold set (4 folds x 2 seeds), both arms at every budget, paired differences
         annotated and starred where the CI excludes zero. The two arms DIVERGE with
         budget: the full arm's gap to the interaction-free oracle closes ~3x while the
         main-effects gap barely moves, so the gain is specific to the interaction
         pathway rather than shared. Panel c's bars are a different set and are NOT
         comparable to these.

USAGE
    PYTHONPATH=src python experiments/make_fig2.py
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

IHCFM_ARMS = ("ihcfm_full", "ihcfm_main_only")


def load() -> dict:
    with open(os.path.join(RESULTS, "table1.json")) as fh:
        return json.load(fh)


def panel_a(ax, T) -> None:
    """Every arm with its CI, ordered by measured score, coloured by separation group."""
    rows, order = T["table"]["rows"], T["table"]["ranking"]
    y = np.arange(len(order))[::-1].astype(float)
    for yy, a in zip(y, order):
        rr = rows[a]
        if a in IHCFM_ARMS:
            col = FOCAL if a == "ihcfm_full" else FOCAL_LIGHT
        elif rr["mean_normalised_ed"] > 1.0 or a == "no_change":
            col = BASELINE_GREYS[2]
        else:
            col = BASELINE_GREYS[1]
        m, lo, hi = rr["mean_normalised_ed"], rr["ci_lo"], rr["ci_hi"]
        mk = "o" if a in IHCFM_ARMS else ("D" if a == "no_change" else "s")
        ax.errorbar([m], [yy], xerr=[[m - lo], [hi - m]], fmt=mk,
                    ms=6.5 if a in IHCFM_ARMS else 5.5, color=col, ecolor=col,
                    elinewidth=1.3, capsize=3.0, capthick=1.1, mec="white", mew=0.6,
                    zorder=3)
    ax.axvline(1.0, color=NEUTRAL, lw=1.1, ls="--", zorder=1)
    ax.text(1.0, len(order) - 0.35, "no change", fontsize=6.5, color=NEUTRAL,
            ha="center", va="bottom")
    labels = [f"{rows[a]['label']}  ({rows[a]['separation_group']})" for a in order]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    for t, a in zip(ax.get_yticklabels(), order):
        if a in IHCFM_ARMS:
            t.set_color(FOCAL if a == "ihcfm_full" else FOCAL_LIGHT)
    ax.set_xlim(0.40, 1.20)
    ax.set_ylim(-0.7, len(order) - 0.15)
    ax.set_xlabel("Normalised energy distance  (21 folds $\\times$ 3 seeds)")
    # Count the separated adjacent pairs that lie WITHIN the model block. The one
    # separated adjacent pair in the ranking is DeepSetsEndpoint vs NoChange -- the
    # model/control boundary, which is what creates the a/b letter break this panel
    # shows. Claiming "none is separated from its neighbour" unscoped would contradict
    # the figure's own letters, so the scope is computed rather than asserted.
    n_a = sum(1 for a in order if rows[a]["separation_group"] == "a")
    block = [a for a in order if rows[a]["separation_group"] == "a"]
    sep_within = sum(1 for d in T["table"]["adjacent_separation"]
                     if d["better"] in block and d["worse"] in block and d["separated"])
    ax.set_title(f"All {n_a} model arms share separation group (a):\n"
                 f"no adjacent pair among them is separated"
                 + ("" if sep_within == 0 else f" ({sep_within} exception)"), loc="left")
    # Spell out what the shared letter means, since it is the panel's whole point. It
    # goes BELOW the axes: the lower-right interior is occupied by the three control
    # arms' markers, which the geometric text check cannot see.
    ax.text(0.0, -0.235, "(letter) = separation group; shared letter = NOT separated"
                         "        lower = better",
            transform=ax.transAxes, fontsize=6, ha="left", va="top", color=NEUTRAL)


def panel_b(ax, T) -> None:
    """The two decisive paired contrasts, with the claim-margin threshold drawn."""
    ct = T["table"]["contrasts"]
    hd = T["table"]["headline"]
    thr = hd["margin_threshold"]
    items = [
        ("IHC-FM (full)\n$-$ FactoredAdditiveCFM",
         ct["ihcfm_full_vs_factored_additive"], FOCAL, "cross-model"),
        ("IHC-FM (full)\n$-$ IHC-FM (main effects only)",
         ct["ihcfm_full_vs_main_only"], ALARM, "within-model"),
    ]
    y = np.arange(len(items))[::-1].astype(float)
    for yy, (lab, p, col, kind) in zip(y, items):
        d = p["mean_diff"]
        ax.errorbar([d], [yy], xerr=[[d - p["lo"]], [p["hi"] - d]], fmt="o", ms=7,
                    color=col, ecolor=col, elinewidth=1.5, capsize=3.5, capthick=1.2,
                    mec="white", mew=0.6, zorder=3)
        ax.annotate(f"{d:+.3f}  [{p['lo']:+.3f}, {p['hi']:+.3f}]\n"
                    f"{p['n_units_a_lower']}/{p['n_units']} folds, "
                    f"sign $p$ = {p['sign_test_p']:.2g}"
                    + ("" if p["excludes_zero"] else "   (CI includes zero)"),
                    (d, yy), textcoords="offset points", xytext=(0, -15),
                    ha="center", va="top", fontsize=6.5, color=col)
    ax.axvline(0.0, color=NEUTRAL, lw=1.1, ls="--", zorder=1)
    ax.text(0.0, len(items) - 0.30, "no difference", fontsize=6.5, color=NEUTRAL,
            ha="center", va="bottom")
    # The margin the project requires before a gap may be called a win.
    ax.axvline(-thr, color=META, lw=1.0, ls=":", zorder=1)
    ax.text(-thr, len(items) - 0.30, f"claim margin\n$-${thr:g}", fontsize=6,
            color=META, ha="center", va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels([it[0] for it in items])
    for t, it in zip(ax.get_yticklabels(), items):
        t.set_color(it[2])
    ax.set_xlim(-0.42, 0.20)
    ax.set_ylim(-1.05, len(items) + 0.22)
    ax.set_xlabel("Paired difference in normalised ED\n(paired on identical folds "
                  "and seeds)")
    ax.text(-0.40, -0.92, "$\\leftarrow$ IHC-FM better", fontsize=6.5, color=FOCAL,
            ha="left", va="center")
    ax.text(0.18, -0.92, "IHC-FM worse $\\rightarrow$", fontsize=6.5, color=ALARM,
            ha="right", va="center")
    ax.set_title("Neither contrast supports a claim:\none below margin, one null",
                 loc="left")


def panel_c(ax, T) -> None:
    """Why the null is informative: available signal vs where the models actually sit."""
    orc, rows = T["oracle"], T["table"]["rows"]
    bars = [
        ("oracle:\ntrue field", orc["oracle_full_normalised_ed"]["mean"],
         BASELINE_GREYS[0], None),
        ("oracle:\nno interaction", orc["oracle_additive_normalised_ed"]["mean"],
         BASELINE_GREYS[2], None),
        ("IHC-FM\nmain only", rows["ihcfm_main_only"]["mean_normalised_ed"],
         FOCAL_LIGHT, (rows["ihcfm_main_only"]["ci_lo"],
                       rows["ihcfm_main_only"]["ci_hi"])),
        ("IHC-FM\nfull", rows["ihcfm_full"]["mean_normalised_ed"], FOCAL,
         (rows["ihcfm_full"]["ci_lo"], rows["ihcfm_full"]["ci_hi"])),
    ]
    # This panel is Table 1's oracle diagnosis ONLY (21 folds x 3 seeds). The step-budget
    # sweep that resolves the null is a different fold set (4 folds x 2 seeds) and lives on
    # its own axis -- see panel_d.
    x = np.arange(len(bars), dtype=float)
    for xx, (lab, v, col, ci) in zip(x, bars):
        ax.bar([xx], [v], width=0.62, color=col, edgecolor="white", linewidth=0.6,
               zorder=2)
        if ci is not None:
            ax.errorbar([xx], [v], yerr=[[v - ci[0]], [ci[1] - v]], fmt="none",
                        ecolor=NEUTRAL, elinewidth=1.2, capsize=3.0, capthick=1.0,
                        zorder=4)
        # Offset the value label sideways on bars that carry an error bar, so the text
        # does not sit behind the whisker.
        ax.annotate(f"{v:.3f}", (xx, v), textcoords="offset points",
                    xytext=((0, 26) if ci is not None else (0, 4)),
                    ha="center",
                    va="bottom",
                    fontsize=6.5, color=NEUTRAL)
    # The interaction signal that EXISTS on this benchmark: the gap between the two
    # oracles. The models sit above both, which is the whole diagnosis.
    lo, hi = bars[0][1], bars[1][1]
    ax.annotate("", xy=(-0.34, hi), xytext=(-0.34, lo),
                arrowprops=dict(arrowstyle="<->", color=ALARM, lw=1.1))
    # Label placed in the empty band above the two oracle bars, clear of both the
    # interaction-free reference line and the bar value annotations.
    ax.text(-0.42, 0.86,
            f"interaction signal\navailable: "
            f"{orc['interaction_worth_normalised_ed']:.3f}",
            fontsize=6.5, color=ALARM, ha="left", va="bottom")
    ax.axhline(bars[1][1], color=BASELINE_GREYS[2], lw=0.9, ls=":", zorder=1)
    ax.axhline(1.0, color=NEUTRAL, lw=1.1, ls="--", zorder=1)
    ax.text(len(bars) - 0.52, 1.0, "no change", fontsize=6.5, color=NEUTRAL,
            ha="right", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in bars])
    ax.set_xlim(-0.62, len(bars) - 0.30)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Normalised energy distance")
    ax.set_title("At the matched budget both arms sit above the\n"
                 "interaction-free oracle (21 folds x 3 seeds)", loc="left")


def panel_d(ax, T) -> None:
    """The step-budget sweep, on its OWN axis because it is a different fold set.

    4 folds x 2 seeds, architecture/splits/coupling/r fixed, only the budget varying. Both
    arms are drawn at every budget so the effect is read within this set: the Table 1 bars
    in panel c come from 21 folds x 3 seeds and are NOT comparable to these.
    """
    with open(os.path.join(HERE, "results", "step_sweep_summary.json")) as f:
        ss = json.load(f)
    budgets = sorted(ss, key=int)
    xs = np.arange(len(budgets), dtype=float)
    mo = [ss[b]["main_only"] for b in budgets]
    fu = [ss[b]["full"] for b in budgets]
    ax.plot(xs, mo, marker="s", ms=4.5, lw=1.4, color=FOCAL_LIGHT,
            label="main effects only", zorder=3)
    ax.plot(xs, fu, marker="o", ms=4.5, lw=1.4, color=FOCAL,
            label="full (interaction on)", zorder=3)
    # The 0.310 interaction-free oracle comes from table1.json's 21x3 set, NOT from this
    # 4x2 sweep, so it is drawn as a cross-set reference and labelled as one. Without that
    # label a reader would take it as this panel's own oracle.
    ax.axhline(0.310, color=BASELINE_GREYS[2], lw=0.9, ls=":", zorder=1)
    ax.text(len(budgets) - 1.04, 0.318,
            "interaction-free oracle (panel c set, 21x3)", fontsize=5.8,
            color=NEUTRAL, ha="right", va="bottom")
    # Annotate the paired effect, which is the actual statistic (not the bar difference).
    for i, b in enumerate(budgets):
        d = ss[b]["diff"]
        sep = ss[b]["ci"][1] < 0
        ax.annotate(f"{d:+.3f}" + ("*" if sep else ""),
                    (xs[i], min(mo[i], fu[i])), textcoords="offset points",
                    xytext=(0, -14), ha="center", va="top", fontsize=6.2,
                    color=(ALARM if sep else NEUTRAL))
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{b}\nsteps" for b in budgets])
    ax.set_xlim(-0.45, len(budgets) - 0.55)
    ax.set_ylim(0.24, 0.80)
    ax.set_ylabel("Normalised energy distance")
    ax.legend(loc="lower left", frameon=False, fontsize=6.4, handlelength=1.5,
              borderaxespad=0.15)
    ax.set_title("Paired effect is monotone in the budget;\n"
                 "* = CI excludes zero (8/8 cells)", loc="left")


def build(T) -> mpl.figure.Figure:
    fig = plt.figure(figsize=(17.6, 4.8))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.05, 0.92, 1.10, 0.74], wspace=0.46,
                          left=0.110, right=0.990, top=0.830, bottom=0.225)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    panel_a(axes[0], T)
    panel_b(axes[1], T)
    panel_c(axes[2], T)
    panel_d(axes[3], T)
    for ax, L, dx in zip(axes, "abcd", (-0.42, -0.36, -0.18, -0.26)):
        panel_letter(ax, L, dx=dx, dy=1.035)
    return fig


def main() -> dict:
    apply_figure_style(frame="open", sizes=(8, 7, 6))
    T = load()
    fig = build(T)
    for d in OUT_DIRS:
        os.makedirs(d, exist_ok=True)
    paths = [os.path.join(d, "fig2_table1.png") for d in OUT_DIRS]
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
