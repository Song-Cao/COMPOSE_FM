"""Figure 2: the two case studies, one ablation ladder each.

Reads results/finance_case_study.json and results/organoid_case_study.json. Every number
plotted comes from those files -- nothing is hardcoded here.
"""
from __future__ import annotations
import json, pathlib, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"

C_DISP = "#B25A00"      # displacement baseline
C_FIELD = "#4C6FA5"     # plain field composition
C_CONST = "#9E86C8"     # global constant
C_GATE = "#1B6B4A"      # saturating gate
C_FULL = "#4C2A85"      # full operator
GREY = "#6E6E6E"

LABEL = {"D1_displacement": "displacement\n(CPA-class)",
         "C1_fields": "field\ncomposition",
         "C2_const": "+ global\nconstant",
         "C3_gate_nosat": "+ state gate\n(no saturation)",
         "C3_gate": "+ saturating\ngate",
         "C4_full": "+ covariant\ncoupling"}
COL = {"D1_displacement": C_DISP, "C1_fields": C_FIELD, "C2_const": C_CONST,
       "C3_gate_nosat": GREY, "C3_gate": C_GATE, "C4_full": C_FULL}
ORDER = ["D1_displacement", "C1_fields", "C2_const", "C3_gate_nosat", "C3_gate", "C4_full"]


def panel(ax, res, split, title, ylabel=None, logy=False, annotate_best=True):
    names = [n for n in ORDER if n in res and np.isfinite(res[n].get(split, np.nan))]
    vals = [res[n][split] for n in names]
    xs = np.arange(len(names))
    bars = ax.bar(xs, vals, color=[COL[n] for n in names], width=0.68,
                  edgecolor="white", linewidth=0.8)
    ax.axhline(1.0, color=GREY, lw=1.1, ls="--", zorder=1)
    ax.text(len(names) - 0.45, 1.0, " no-change\n baseline", va="center", ha="left",
            fontsize=7.5, color=GREY)
    if logy:
        ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([LABEL[n] for n in names], fontsize=7.6)
    ax.set_title(title, fontsize=9.5, pad=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8.8)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.3f}" if v < 100 else f"{v:.0f}",
                    (b.get_x() + b.get_width() / 2, b.get_height()),
                    textcoords="offset points", xytext=(0, 2.5),
                    ha="center", fontsize=7.3)
    if annotate_best:
        best = int(np.argmin(vals))
        bars[best].set_edgecolor("black"); bars[best].set_linewidth(1.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(y=0.20)
    return names, vals


def main():
    fin = json.load(open(RES / "finance_case_study.json"))
    org = json.load(open(RES / "organoid_case_study.json"))
    fr, orr = fin["results"], org["results"]

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.1))

    panel(axes[0], fr, "exposure",
          f"a   Finance: exposure extrapolation\n(train $\\tau\\leq${fin['config']['tau_train_max']:.0f},"
          f" test $\\tau\\geq${fin['config']['tau_test_min']:.0f}; {fin['splits']['exposure']} windows)",
          ylabel="normalised energy distance  (lower better)", logy=True)
    panel(axes[1], orr, "triple",
          f"b   Biology: held-out triple C+S+F\n({org['splits']['triple']} populations, never seen jointly)",
          logy=False)
    panel(axes[2], orr, "top_dose",
          f"c   Biology: held-out top dose\n({org['splits']['top_dose']} populations)",
          logy=False)

    fig.suptitle("COMPOSE-FM: composing generators generalises where adding displacements does not",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    out = RES / "figures/fig2_case_studies.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", out)

    # text summary for the paper
    lines = []
    for nm, res, splits in (("FINANCE", fr, ["train", "exposure", "composition", "both"]),
                            ("ORGANOID", orr, ["train", "triple", "top_dose"])):
        lines.append(f"== {nm} ==")
        for n in ORDER:
            if n in res:
                row = "  ".join(f"{s} {res[n][s]:.4f}" for s in splits
                                if np.isfinite(res[n].get(s, np.nan)))
                nu = res[n].get("nu")
                lines.append(f"  {n:16s} {row}" + (f"   nu={nu:.3f}" if nu else ""))
    (RES / "tables/case_study_summary.txt").parent.mkdir(parents=True, exist_ok=True)
    open(RES / "tables/case_study_summary.txt", "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
