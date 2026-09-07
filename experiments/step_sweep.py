"""Step-budget sweep: does the main-effects fit approach the interaction-free oracle
(0.310 nED)? If it does, the interaction ablation becomes informative. Diagnostic only."""
import sys, os, json, time
sys.path.insert(0,'src'); sys.path.insert(0,'experiments')
import numpy as np, torch
torch.set_num_threads(2)
import table1 as T

BUDGETS = [(300,300,200), (900,900,600), (2400,2400,1600)]
FOLDS   = T.fold_list(T.K_PERT)[:4]
SEEDS   = (0,1)
R       = 2

import composefm.metrics as M
system, pops = T.benchmark(k=T.K_PERT, n_cells=T.N_CELLS)

# Replicate the harness's fold construction exactly: hold out one pair, score only
# order>=2 populations, and use the training-split median no-change ED as the reference.
SPLITS={}
for fold in FOLDS:
    tr,te = system.loco_split(pops,[fold])
    te=[i for i in te if len(pops[i].P)>=2]
    ref=float(np.median([M.energy_distance(np.asarray(pops[i].pre,dtype=np.float64),
                                           np.asarray(pops[i].post,dtype=np.float64),
                                           max_n=None)
                         for i in tr if len(pops[i].P)>=2]))
    SPLITS[fold]=(tr,te,ref)

rows=[]
for steps in BUDGETS:
    tot=sum(steps)
    for arm in ("ihcfm_main_only","ihcfm_full"):
        for fi,fold in enumerate(FOLDS):
            tr,te,ref = SPLITS[fold]
            for sd in SEEDS:
                t0=time.time()
                res = T.run_arm(arm, pops, system, tr, te, sd, R,
                                steps_ihcfm=steps, steps_flat=tot,
                                denominator_reference=ref)
                per = res.get("per_population") or res.get("populations") or []
                vals=[p["ed_normalised"] for p in per if np.isfinite(p.get("ed_normalised",np.nan))]
                rows.append(dict(steps=tot, arm=arm, fold=str(fold), seed=sd,
                                 ned=float(np.mean(vals)) if vals else float('nan'),
                                 n=len(vals), wall=time.time()-t0))
                print(f"{tot:5d} {arm:16s} {str(fold):10s} s{sd} nED {rows[-1]['ned']:.4f} ({rows[-1]['wall']:.0f}s)", flush=True)
json.dump(rows, open("results/step_sweep.json","w"), indent=1)
print("\n=== SUMMARY (paired within fold+seed) ===")
for steps in BUDGETS:
    tot=sum(steps)
    mo=[r["ned"] for r in rows if r["steps"]==tot and r["arm"]=="ihcfm_main_only"]
    fu=[r["ned"] for r in rows if r["steps"]==tot and r["arm"]=="ihcfm_full"]
    d=[f-m for m,f in zip(mo,fu)]
    print(f"steps {tot:5d}: main_only {np.mean(mo):.4f}  full {np.mean(fu):.4f}  "
          f"paired diff {np.mean(d):+.4f} (sd {np.std(d,ddof=1):.4f}, n={len(d)})  "
          f"gap to additive oracle 0.310: {np.mean(mo)-0.310:+.4f}")
