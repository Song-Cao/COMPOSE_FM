"""CASE STUDY 1 (finance): the general operator applied to order-flow composition.

This script contains NO finance-specific modelling. It maps a domain onto the general
interface (state / interventions / exposure) and then runs the same operator and the same
ablation ladder used in every other case study.

    state z        6-d microstructure features of a 10 s window
    interventions  4 order-flow channels (side x size class)
    exposure tau   executed quantity in that channel, normalised to its median

Held-out generalisation axes, chosen to match the method's two structural claims:
  (A) COMPOSITION: train on windows with k <= 2 active channels, test on k >= 3.
  (B) EXPOSURE:    train on tau <= 2 (per active channel), test on tau > 4.
Both are extrapolation, not interpolation: the test regimes are absent from training.

Baseline is DisplacementBaseline (CPA/GEARS-class): identical trunk, identical budget,
differing only in the composition rule. Metric is normalised energy distance,
ED(pred,true)/ED(no-change,true), so 1.0 = no better than predicting no change.
"""
from __future__ import annotations
import glob, json, pathlib, sys, time
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from composefm import data_finance as dfin
from composefm.compose import ComposeFM, DisplacementBaseline

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = ROOT / "results"
# Split design. In crypto spot, all four channels are active in most 10 s windows
# (measured k distribution: k=2 3150, k=3 12644, k=4 18751 of 34545), so k is a WEAK
# composition axis here -- a k<=2 training set is only 2935 windows against 14206 test.
# Exposure is the informative axis, and it is the one the square-root law speaks about
# (measured bin-mean log-log slope 0.587, vs 1.0 for additive). We therefore make EXPOSURE
# the primary held-out axis and keep composition as a secondary split on k=4 vs k<=3.
#
# WARNING ON THE k=4 ("composition") SPLIT -- it is reported but NOT interpretable here.
# The normalising denominator ED(no-change, true) collapses on that split: measured 0.154
# versus 0.351 on train and 0.320 on the exposure split, because windows with all four
# channels active are the BALANCED ones where buy and sell flow offset and the net move is
# small. An oracle mean-shift predictor still scores 0.724 there (vs 0.232 / 0.295
# elsewhere), so the split has little headroom, and dividing by the small denominator
# inflates every model's score several-fold. This is the same normalisation trap that
# produced the absurd 8.5e6 relative margin in Gate B run 1. Treat k=4 numbers as
# diagnostic only; the finance claim rests on the EXPOSURE axis, and the composition claim
# is carried by the organoid case study, which has a genuine held-out triple.
TAU_TRAIN_MAX = 3.0          # train on typical exposures
TAU_TEST_MIN = 6.0           # test on 2x-and-beyond: strict extrapolation
K_TRAIN_MAX = 3              # secondary axis: hold out the fully-loaded k=4 windows
HIDDEN = 96                  # 545 ms/step for the full model at 4 flow steps
N_SLICES = 64                # fixed directions for the sliced-W2 objective
DEPTH = 2
FLOW_STEPS = 4


# ------------------------------------------------------------------ data
def load_all(window="10s"):
    files = sorted(glob.glob(str(ROOT / "data/raw/finance/*.zip")))
    assert files, "no finance data found"
    Z0, Z1, TAU, DAY = [], [], [], []
    for i, f in enumerate(files):
        d = dfin.assign_channels(dfin.load_agg_trades(f))
        W = dfin.impact_state(d, window=window)      # state contains price impact
        Z0.append(W["z0"]); Z1.append(W["z1"]); TAU.append(W["tau"])
        DAY.append(np.full(len(W["z0"]), i))
    z0 = np.concatenate(Z0); z1 = np.concatenate(Z1)
    tau = np.concatenate(TAU); day = np.concatenate(DAY)
    ok = np.isfinite(z0).all(1) & np.isfinite(z1).all(1) & np.isfinite(tau).all(1)
    return z0[ok], z1[ok], tau[ok], day[ok], files


def energy_distance(x, y):
    """Unbiased-ish energy distance between two point clouds (torch, no grad needed)."""
    x = torch.as_tensor(x, dtype=torch.float32); y = torch.as_tensor(y, dtype=torch.float32)
    def pd(a, b):
        return torch.cdist(a, b).mean()
    return float(2 * pd(x, y) - pd(x, x) - pd(y, y))


def norm_ed(pred, true, ctrl):
    num = max(energy_distance(pred, true), 0.0)
    den = max(energy_distance(ctrl, true), 1e-9)
    return num / den


# ------------------------------------------------------------------ training
# Fixed slicing directions, drawn once (redrawing per step makes the gradient too noisy).
_g = torch.Generator().manual_seed(12345)
SLICE_DIRS = torch.randn(6, N_SLICES, generator=_g)
SLICE_DIRS = (SLICE_DIRS / SLICE_DIRS.norm(dim=0, keepdim=True)).to(DEV)


def make_batches(z0, z1, tau, idx, batch=256, rng=None):
    """Group windows by their ACTIVE CHANNEL SET so a batch shares one intervention set."""
    rng = rng or np.random.default_rng(0)
    active = [tuple(np.nonzero(tau[i] > 0)[0].tolist()) for i in idx]
    groups = {}
    for j, a in zip(idx, active):
        groups.setdefault(a, []).append(j)
    keys = [k for k, v in groups.items() if len(v) >= 8 and len(k) > 0]
    return groups, keys


def train_model(model, z0, z1, tau, idx, steps=1500, lr=2e-3, is_disp=False,
                batch=256, seed=0, l0_weight=1e-3, n_steps_flow=FLOW_STEPS, log_every=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    groups, keys = make_batches(z0, z1, tau, idx, rng=rng)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    t0 = time.time()
    n_skipped = 0
    for s in range(steps):
        k = keys[rng.integers(len(keys))]
        pool = groups[k]
        sel = rng.choice(pool, size=min(batch, len(pool)), replace=False)
        zb0 = torch.tensor(z0[sel], dtype=torch.float32, device=DEV)
        zb1 = torch.tensor(z1[sel], dtype=torch.float32, device=DEV)
        tb = torch.tensor(tau[sel][:, list(k)], dtype=torch.float32, device=DEV)
        coll = {}
        if is_disp:
            pred = model.predict(zb0, list(k), tb)
        else:
            pred = model.flow(zb0, list(k), tb, n_steps=n_steps_flow, collect=coll)
        # DISTRIBUTIONAL objective, not per-sample MSE. This is a correctness fix, not a
        # tuning choice: a squared-error loss drives the prediction towards the conditional
        # MEAN, which collapses the predicted spread. Under the energy-distance metric that
        # collapse is punished harder than doing nothing -- measured on the training split,
        # a conditional-mean predictor scores 3.49 while "predict no change" scores 1.00,
        # and the MSE-trained models duly came out at ~1.52 in-distribution. Matching the
        # distribution (sliced W2 over fixed directions, plus an explicit mean term) scores
        # 0.23 for an oracle mean shift and 0.15 with the spread restored.
        a = (pred @ SLICE_DIRS).sort(0).values
        b = (zb1 @ SLICE_DIRS).sort(0).values
        loss = ((a - b) ** 2).mean() + ((pred.mean(0) - zb1.mean(0)) ** 2).sum()
        if "l0" in coll:
            loss = loss + l0_weight * coll["l0"]
        opt.zero_grad(); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        # Never step on a non-finite loss or gradient. Clipping alone does NOT protect
        # against this: a non-finite value passes through clip_grad_norm_ and the
        # subsequent Adam step writes NaN into every parameter, so one bad step turns the
        # whole run into NaN. Measured: 1 of 3 seeds on the organoid ladder with the
        # coupling active. Skipping the step recovers that seed; the counter is reported
        # so a run that skips many steps cannot look healthy.
        if torch.isfinite(loss) and torch.isfinite(gn):
            opt.step()
        else:
            n_skipped += 1
        sched.step()
        if log_every and (s + 1) % log_every == 0:
            print(f"    step {s+1}/{steps} loss {loss.item():.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    if n_skipped:
        print(f"    [warn] skipped {n_skipped}/{steps} non-finite steps", flush=True)
    model._n_skipped = n_skipped
    return model


@torch.no_grad()
def evaluate(model, z0, z1, tau, idx, is_disp=False, min_group=16, n_steps_flow=FLOW_STEPS*2):
    """Per-intervention-set normalised energy distance, averaged over sets."""
    groups, keys = make_batches(z0, z1, tau, idx)
    scores, ws = [], []
    for k in keys:
        pool = groups[k]
        if len(pool) < min_group:
            continue
        sel = np.array(pool)
        zb0 = torch.tensor(z0[sel], dtype=torch.float32, device=DEV)
        tb = torch.tensor(tau[sel][:, list(k)], dtype=torch.float32, device=DEV)
        pred = (model.predict(zb0, list(k), tb) if is_disp
                else model.flow(zb0, list(k), tb, n_steps=n_steps_flow))
        scores.append(norm_ed(pred.cpu().numpy(), z1[sel], z0[sel]))
        ws.append(len(pool))
    if not scores:
        return float("nan"), 0
    w = np.array(ws, dtype=float); w /= w.sum()
    return float(np.sum(np.array(scores) * w)), len(scores)


def main(steps=1500, d_state=6, seed=0, quick=False):
    z0r, z1r, tau, day, files = load_all()
    z0, z1, mu, sd = dfin.standardise(z0r, z1r)
    n = len(z0)
    k_act = (tau > 0).sum(1)
    tau_max = tau.max(1)

    # Split axes. PRIMARY = exposure extrapolation (train tau<=3, test tau>=6).
    # SECONDARY = composition (train k<=3, test k=4), evaluated within trained exposures.
    tr = np.nonzero((tau_max <= TAU_TRAIN_MAX) & (k_act <= K_TRAIN_MAX))[0]
    te_expo = np.nonzero((tau_max >= TAU_TEST_MIN) & (k_act <= K_TRAIN_MAX))[0]
    te_comp = np.nonzero((tau_max <= TAU_TRAIN_MAX) & (k_act > K_TRAIN_MAX))[0]
    te_both = np.nonzero((tau_max >= TAU_TEST_MIN) & (k_act > K_TRAIN_MAX))[0]
    print(f"windows {n} | train {len(tr)} | test-exposure {len(te_expo)} "
          f"| test-composition {len(te_comp)} | test-both {len(te_both)}", flush=True)
    assert len(tr) > 500 and len(te_expo) > 200, "degenerate split"
    if quick:
        steps = 200

    res = {}
    specs = [("D1_displacement", None, True),
             ("C1_fields",   "additive", False),
             ("C2_const",    "const",    False),
             ("C3_gate_nosat", "gate",   False),
             ("C3_gate",     "gate",     False),
             ("C4_full",     "full",     False)]
    for name, mode, is_disp in specs:
        t0 = time.time()
        if is_disp:
            m = DisplacementBaseline(d_state, dfin.N_CHANNELS,
                                     hidden=HIDDEN, depth=DEPTH).to(DEV)
        else:
            m = ComposeFM(d_state, dfin.N_CHANNELS, mode=mode,
                          hidden=HIDDEN, depth=DEPTH,
                          saturate=(name != "C3_gate_nosat")).to(DEV)
        m = train_model(m, z0, z1, tau, tr, steps=steps, is_disp=is_disp, seed=seed,
                        log_every=max(steps // 3, 1))
        m.eval()
        row = {}
        for split, idx in (("train", tr), ("composition", te_comp),
                           ("exposure", te_expo), ("both", te_both)):
            v, ns = evaluate(m, z0, z1, tau, idx, is_disp=is_disp)
            row[split] = v; row[split + "_nsets"] = ns
        row["params"] = sum(p.numel() for p in m.parameters())
        if getattr(m, "gate", None) is not None:
            row["nu"] = m.gate.exponent()
        row["wall_s"] = time.time() - t0
        res[name] = row
        print(f"  {name:16s} train {row['train']:.4f}  comp {row['composition']:.4f}  "
              f"expo {row['exposure']:.4f}  both {row['both']:.4f}  "
              f"({row['wall_s']:.0f}s)", flush=True)

    OUT.mkdir(exist_ok=True)
    json.dump(dict(results=res, n_windows=int(n), files=[pathlib.Path(f).name for f in files],
                   splits=dict(train=len(tr), composition=len(te_comp),
                               exposure=len(te_expo), both=len(te_both)),
                   config=dict(quick=bool(quick), run_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), steps=steps, d_state=d_state, seed=seed,
                               k_train_max=K_TRAIN_MAX, tau_train_max=TAU_TRAIN_MAX,
                               tau_test_min=TAU_TEST_MIN, hidden=HIDDEN, depth=DEPTH,
                               flow_steps=FLOW_STEPS, state=dfin.IMPACT_STATE_NAMES)),
              open(OUT / "finance_case_study.json", "w"), indent=2)
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    main(steps=a.steps, quick=a.quick)
