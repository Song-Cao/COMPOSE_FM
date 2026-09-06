"""Multi-seed replication of the two headline comparisons.

A single seed is an anecdote. This re-runs the decisive contrast in each domain across
seeds and reports mean +/- sd, plus the per-seed win/loss record against the displacement
baseline. Only the models that matter for the claim are run, to stay inside budget:
    D1_displacement, C1_fields, C3_gate_nosat, C4_full
"""
from __future__ import annotations
import json, pathlib, sys, time
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from composefm.compose import ComposeFM, DisplacementBaseline

SEEDS = (0, 1, 2)
SPECS = [("D1_displacement", None, True),
         ("C1_fields", "additive", False),
         ("C3_gate_nosat", "gate", False),
         ("C4_full", "full", False)]


def run_finance(steps=1500):
    import finance_case_study as fcs
    import composefm.data_finance as dfin
    z0r, z1r, tau, day, files = fcs.load_all()
    z0, z1, mu, sd = dfin.standardise(z0r, z1r)
    k = (tau > 0).sum(1); tmax = tau.max(1)
    tr = np.nonzero((tmax <= fcs.TAU_TRAIN_MAX) & (k <= fcs.K_TRAIN_MAX))[0]
    te = np.nonzero((tmax <= fcs.TAU_TRAIN_MAX) & (k > fcs.K_TRAIN_MAX))[0]
    out = {}
    for name, mode, is_disp in SPECS:
        vals = []
        for s in SEEDS:
            m = (DisplacementBaseline(6, dfin.N_CHANNELS, hidden=fcs.HIDDEN, depth=fcs.DEPTH)
                 if is_disp else
                 ComposeFM(6, dfin.N_CHANNELS, mode=mode, hidden=fcs.HIDDEN,
                           depth=fcs.DEPTH, saturate=(name != "C3_gate_nosat")))
            m = fcs.train_model(m, z0, z1, tau, tr, steps=steps, is_disp=is_disp, seed=s)
            m.eval()
            v, _ = fcs.evaluate(m, z0, z1, tau, te, is_disp=is_disp)
            vals.append(v)
            print(f"    finance {name} seed {s}: {v:.4f}", flush=True)
        out[name] = vals
    return out


def run_organoid(steps=2500):
    import organoid_case_study as ocs
    pops, meta = ocs.prepare()
    treats = np.array([p["treat"] for p in pops]); doses = np.array([p["dose"] for p in pops])
    tri = treats == "CSF"; hi = (doses >= 4) & np.isin(treats, ocs.SINGLES)
    tr = np.nonzero(~tri & ~hi)[0]; te = np.nonzero(tri)[0]
    out = {}
    for name, mode, is_disp in SPECS:
        vals = []
        for s in SEEDS:
            m = (DisplacementBaseline(ocs.LATENT_D, ocs.N_PERT, hidden=96, depth=2)
                 if is_disp else
                 ComposeFM(ocs.LATENT_D, ocs.N_PERT, mode=mode, hidden=96, depth=2,
                           saturate=(name != "C3_gate_nosat"), code=ocs.POP_CODE))
            m = ocs.train_model(m, pops, tr, steps=steps, is_disp=is_disp, seed=s)
            m.eval()
            v = float(np.mean(list(ocs.evaluate(m, pops, te, is_disp=is_disp).values())))
            vals.append(v)
            print(f"    organoid {name} seed {s}: {v:.4f}", flush=True)
        out[name] = vals
    return out


if __name__ == "__main__":
    t0 = time.time()
    res = {"finance": run_finance(), "organoid": run_organoid()}
    summary = {}
    for dom, d in res.items():
        base = np.array(d["D1_displacement"])
        summary[dom] = {}
        for k, v in d.items():
            v = np.array(v)
            summary[dom][k] = dict(mean=float(v.mean()), sd=float(v.std(ddof=1)),
                                   vals=[float(x) for x in v],
                                   wins_vs_disp=int((v < base).sum()), n=len(v))
    json.dump(dict(summary=summary, raw=res, seeds=list(SEEDS),
                   wall_s=time.time() - t0),
              open(ROOT / "results/seed_robustness.json", "w"), indent=2)
    print("\n=== SUMMARY (normalised energy distance, lower better) ===")
    for dom, d in summary.items():
        print(f"-- {dom}")
        for k, v in d.items():
            print(f"   {k:16s} {v['mean']:.4f} +/- {v['sd']:.4f}   "
                  f"beats displacement in {v['wins_vs_disp']}/{v['n']} seeds")
