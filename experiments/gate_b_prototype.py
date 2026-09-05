"""GATE B -- prototype: does a TRAINED COMPOSE-FM beat trained baselines?

Gate A2 established that generator-composition is covariant while displacement-addition is
not, and located the regime where that matters (higher k, higher exposure). Those were
analytic properties of the operator. This gate trains actual models and asks whether the
architecture wins on held-out prediction, which is what the paper must claim.

Pre-registered comparisons (all trained on the SAME data, SAME budget):
  M1 DisplacementCPA      -- additive latent displacement (CPA/GEARS class)
  M2 ComposeFM mode=additive -- generator composition, no gate (isolates "compose fields")
  M3 ComposeFM mode=const    -- generator composition + ONE global learned scalar
  M4 ComposeFM mode=full     -- state-dependent gate (+ optional coupling term)

The critical comparison is M4 vs M3: on real Norman a global constant captured essentially
all composition signal (learned gate beat constant by only +0.0085 R^2), so the gate must
earn its place. The second critical comparison is M2 vs M1: that isolates the covariance
claim under training rather than analytically.

Pass condition (registered before running):
  (1) ZERO-SHOT combination energy distance: M4 < M3 < M1, with M4 beating M1 by >= 20%;
  (2) EXPOSURE EXTRAPOLATION (train tau in {0.5,1}, test tau=2): M4 beats M1 by >= 30%
      (this is where semigroup-by-construction should pay off);
  (3) the k=4 regime shows a larger M2-vs-M1 margin than the k=2 regime (consistent with
      Gate A2's scaling finding).
"""
import json, sys, pathlib, time
import numpy as np
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from composefm.toy import ToySystem
from composefm.models import (ComposeFM, DisplacementCPA, ot_pairs,
                              energy_distance, energy_distance_np)

OUT = pathlib.Path(__file__).resolve().parents[1] / "results"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def make_dataset(sysm, singles, combos, n_cells=256, taus=(0.5, 1.0), seed=0, noise=0.05):
    """Training set: single perturbations at several exposures + a subset of combinations."""
    data = []
    for p in singles:
        for tau in taus:
            Z0 = sysm.control(n_cells, seed=seed)
            Z1 = sysm.observe([p], n_cells, tau=tau, seed=seed, noise=noise)
            data.append(([p], tau, Z0, Z1))
    for P in combos:
        for tau in taus:
            Z0 = sysm.control(n_cells, seed=seed)
            Z1 = sysm.observe(list(P), n_cells, tau=tau, seed=seed, noise=noise)
            data.append((list(P), tau, Z0, Z1))
    return data


def train(model, data, steps=600, lr=3e-3, is_cpa=False, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    idx = np.arange(len(data))
    rng = np.random.default_rng(seed)
    for s in range(steps):
        b = data[int(rng.choice(idx))]
        P, tau, Z0, Z1 = b
        # OT coupling gives each control cell a target
        tgt = ot_pairs(Z0, Z1, eps=0.05)
        z0 = torch.tensor(Z0, dtype=torch.float32, device=DEV)
        z1 = torch.tensor(tgt, dtype=torch.float32, device=DEV)
        if is_cpa:
            pred = model.predict(z0, P, tau=tau)
        else:
            pred = model.flow(z0, P, tau=tau, n_steps=6)
        loss = ((pred - z1) ** 2).sum(-1).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
    return model


@torch.no_grad()
def evaluate(model, sysm, P, tau, n=256, is_cpa=False, seed=99):
    """NORMALISED energy distance: ED(pred, true) / ED(control, true).

    The raw energy-distance U-statistic goes slightly NEGATIVE when a fit is near-perfect
    (it is unbiased, not non-negative), which makes relative margins meaningless -- the
    first run of this gate produced a margin of 8.5e6 by dividing by ~0. Normalising by the
    null distance (how far the unperturbed control already is from the target) gives a
    scale-free, interpretable quantity: 1.0 = no better than predicting "no change",
    0.0 = perfect. Margins are then compared as ABSOLUTE differences in this ratio.
    """
    Z0 = sysm.control(n, seed=seed)
    true = sysm.observe(list(P), n, tau=tau, seed=seed, noise=0.05)
    z0 = torch.tensor(Z0, dtype=torch.float32, device=DEV)
    pred = (model.predict(z0, list(P), tau=tau) if is_cpa
            else model.flow(z0, list(P), tau=tau, n_steps=8))
    ed = energy_distance_np(pred.cpu().numpy(), true)
    null = energy_distance_np(Z0, true)          # "predict no change"
    return max(ed, 0.0) / max(null, 1e-9)


def run(d=6, n_pert=6, seed=0):
    sysm = ToySystem(d=d, n_pert=n_pert, seed=seed)
    singles = list(range(n_pert))
    train_combos = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)]
    heldout_pairs = [(1, 2), (0, 3), (2, 3), (1, 4), (0, 4)]
    heldout_k4 = [(0, 1, 2, 3), (1, 2, 3, 4), (0, 2, 3, 4)]

    data = make_dataset(sysm, singles, train_combos, taus=(0.5, 1.0), seed=seed)

    models = {}
    models["M1_cpa"] = train(DisplacementCPA(d, n_pert).to(DEV), data, is_cpa=True, seed=seed)
    models["M2_fields"] = train(ComposeFM(d, n_pert, mode="additive").to(DEV), data, seed=seed)
    models["M3_const"] = train(ComposeFM(d, n_pert, mode="const").to(DEV), data, seed=seed)
    models["M4_full"] = train(ComposeFM(d, n_pert, mode="full", use_coupling=False).to(DEV),
                              data, seed=seed)

    res = {}
    for name, m in models.items():
        is_cpa = name == "M1_cpa"
        zs = float(np.mean([evaluate(m, sysm, P, 1.0, is_cpa=is_cpa) for P in heldout_pairs]))
        ex = float(np.mean([evaluate(m, sysm, P, 2.0, is_cpa=is_cpa) for P in heldout_pairs]))
        k4 = float(np.mean([evaluate(m, sysm, P, 1.0, is_cpa=is_cpa) for P in heldout_k4]))
        res[name] = dict(zeroshot_pair=zs, exposure_extrap=ex, k4=k4)
    return res


if __name__ == "__main__":
    t0 = time.time()
    all_res = [run(seed=s) for s in (0, 1, 2)]
    keys = ["zeroshot_pair", "exposure_extrap", "k4"]
    agg = {m: {k: float(np.mean([r[m][k] for r in all_res])) for k in keys}
           for m in all_res[0]}
    print(f"device {DEV} | wall {time.time()-t0:.1f}s | 3 seeds\n")
    print(f"{'model':12s} {'zeroshot k=2':>14s} {'exposure tau=2':>16s} {'k=4':>10s}")
    for m, v in agg.items():
        print(f"{m:12s} {v['zeroshot_pair']:14.4f} {v['exposure_extrap']:16.4f} {v['k4']:10.4f}")

    m1, m2, m3, m4 = (agg["M1_cpa"], agg["M2_fields"], agg["M3_const"], agg["M4_full"])
    c1 = (m4["zeroshot_pair"] <= m3["zeroshot_pair"] <= m1["zeroshot_pair"]
          and m4["zeroshot_pair"] <= 0.80 * m1["zeroshot_pair"])
    c2 = m4["exposure_extrap"] <= 0.70 * m1["exposure_extrap"]
    # ABSOLUTE margins in normalised-ED units (see evaluate() docstring)
    marg_k2 = m1["zeroshot_pair"] - m2["zeroshot_pair"]
    marg_k4 = m1["k4"] - m2["k4"]
    c3 = marg_k4 > marg_k2
    print(f"\n(1) M4<=M3<=M1 and M4 >=20% better than M1: {c1}")
    print(f"(2) exposure tau=2, M4 >=30% better       : {c2} "
          f"({m4['exposure_extrap']:.4f} vs {m1['exposure_extrap']:.4f})")
    print(f"(3) field-composition margin grows with k : {c3} "
          f"(abs margin k2 {marg_k2:+.4f} -> k4 {marg_k4:+.4f})")
    verdict = bool(c1 and c2 and c3)
    print("\nGATE B:", "PASS" if verdict else "FAIL")
    OUT.mkdir(exist_ok=True)
    json.dump(dict(agg=agg, per_seed=all_res, c1=c1, c2=c2, c3=c3,
                   margin_k2=marg_k2, margin_k4=marg_k4,
                   verdict="PASS" if verdict else "FAIL"),
              open(OUT / "gate_b.json", "w"), indent=2)
