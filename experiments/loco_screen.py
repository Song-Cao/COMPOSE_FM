"""Leave-one-combination-out SCREEN on the organoid drug-combination ladder.

This is the evaluation-machinery version of the organoid case study: instead of one
hand-picked held-out split it screens EVERY feasible held-out combination, and it adds a
LEAVE-REPLICATE-OUT split that the previous study did not have. Nothing here defines a
model; it imports IHC-FM (`ihcfm_geometry` + `ihcfm_hierarchy`), the additive controls
(`baselines`), and the metric suite (`metrics`), and it writes predictions and scores.

DATA (provenance re-verified in-session against the files on disk, not quoted)
-----------------------------------------------------------------------------
Ramos Zapatero et al., Cell 186(25), 2023, DOI 10.1016/j.cell.2023.11.005, in the
preprocessed release of Atanackovic et al. (Meta Flow Matching, arXiv:2408.14608). Files:
`trellis_replicas_1_normalized.npy` (24,756,030 cells x 44 mass-cytometry markers,
8.7 GB, memory-mapped -- never loaded whole) and `data_splits_replicas_1.pickle`.

ENUMERATED FROM THE PICKLE (counts printed by this script, not assumed):
  252 split records collapsing to 87 records that carry a matched PDO control, giving
  666 treated populations over 11 treatments and 24 distinct (treatment, dose) cells:
      singles  S F C V L O          pairs  VS CS SF CF          triple  CSF
  Doses present per treatment are NOT uniform (enumerated, printed by this script):
      S [1,2,3,4]   F [1,2,3,4]   L [1,2,3,4]   O [1,2,3]   V [1]   C [2]
      VS [2,3,4]    CS [3]   SF [3]   CF [3]     CSF [4]

  *** CONFOUND, STATED UP FRONT, AND IT IS PER-COMBINATION ***
  FOUR of the five held-out combinations -- CS, SF, CF (all at dose 3) and CSF (dose 4) --
  occur at exactly ONE dose level. For those folds "held-out combination" and "held-out
  (combination, dose)" are the SAME split, so the exposure axis cannot be separated from
  the composition axis and the honest reading is "unseen combination at an exposure seen
  only for other conditions".
  VS is the EXCEPTION: it appears at doses 2, 3 and 4, so its fold holds out a combination
  across three exposures and is the one pair whose result is not dose-confounded. It is
  therefore the most informative pair in the screen, and `dose_confounded` is recorded
  per fold rather than asserted globally.
  This is a property of the experimental design, not of the split.

WHAT AN INDEPENDENT UNIT IS
---------------------------
The REPLICATE RECORD, not the cell and not the population. Every population inside one
record is pushed forward from the SAME matched control draw and comes from the same
patient-derived organoid line and plate, so two populations in one record are two reads of
one experiment. All confidence intervals group on the record id (`metrics.py` refuses to
bootstrap without a group label, for exactly this reason). The held-out pair VS, for
instance, is 86 populations but only 29 records; treating it as 86 units would shrink
every interval by ~1.7x, and treating it as ~460k cells by ~120x.

INTERACTION RANK r IS TUNED, NOT DERIVED FROM A FORMULA
-------------------------------------------------------
An earlier version of this script fixed r = k(k-1)/2 = 15 (k = 6 single compounds) and
aborted otherwise, on the argument that a smaller rank ALIASES distinct pairs onto one
interaction direction. The rank arithmetic is correct -- the pair basis saturates at
min(r, k(k-1)/2) -- but the requirement has been RETRACTED by the project lead because
performance does not follow it. On the benchmark under the correct coupling, recovery
measured +0.5930 (r=2), +0.4414 (r=4), +0.5121 (r=6), +0.6427 (r=8) over ALL
combinations, but on the HELD-OUT column r=2 leads (+0.5986) and r=8 trails (+0.5440).
The two columns disagree, so no single benchmark number settles r.

r is therefore treated as a capacity hyperparameter and, with `--tune-r`, SELECTED BY
HELD-OUT DISTRIBUTIONAL SCORE -- PER FOLD, never globally. Selecting globally is the
first thing one reaches for and it does not work here: a leak-free global pool has to
exclude every screened combination, and since all four pairs and the triple are screened,
that pool is singletons only. With no combination in either the fit or the validation
half the interaction branch is never exercised, every candidate r scores alike, and the
"selection" is noise. Inside a fold the training side still holds the OTHER combinations,
so r is chosen on real composition data while that fold's own held-out combination stays
unseen. `select_rank` REFUSES to run on a pool without combinations rather than return an
undiscriminating number; the fold then falls back to the default r=8 and records why in
its `rank_tuning`. The leave-replicate-out arm runs at a FIXED r on purpose, so the
code_dim 0-vs-16 contrast is not confounded with a capacity difference. Aliasing is
REPORTED per fold (`n_pairs_representable`) so a reader can see which pairs a given r can
separate; it is not an abort condition.

VERIFIED TREATMENT STRUCTURE (enumerated from the pickle, printed by this script)
--------------------------------------------------------------------------------
14 top-level keys = 11 treatments + 3 controls:
  6 singles C F L O S V | 4 pairs CF CS SF VS | 1 triple CSF | controls AH DMSO H2O
Per-record presence: S/VS/L/F 86, V/CS 85, C/SF 84, CF 83, CSF 82, O 80 (of 87 records
carrying a matched PDO control). This corrects a brief that said "4 singles" while
listing six labels.

THE LEAVE-REPLICATE-OUT ARM
---------------------------
Population conditioning (`PopulationEncoder`, code_dim > 0) can win a combination split by
recognising which replicate a control cloud came from rather than by modelling the
perturbation -- a previously measured failure on this dataset: code_dim=16 improved the
training score but made the held-out TRIPLE worse, 0.903 -> 1.283. A combination split
cannot detect this, because the held-out combination's replicates are still seen under
other treatments. The leave-replicate-out arm holds out WHOLE RECORDS, so no control cloud
from an evaluated record was ever seen in training, and runs code_dim = 0 and 16 as a
matched pair on the same folds. If conditioning is memorising replicate identity, it must
lose here.

Usage
    PYTHONPATH=src python experiments/loco_screen.py --smoke
    PYTHONPATH=src python experiments/loco_screen.py            # full screen
"""
from __future__ import annotations

import argparse
import json
import pathlib
import pickle
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from composefm import metrics as M
from composefm.baselines import NoChange, PerturbedMean
from composefm.ihcfm_geometry import (AmbientMetric, Decoder, HillDose,
                                      MetricRadialSat, PullbackField)
from composefm.ihcfm_hierarchy import (CFMObjective, ConditionalVelocity,
                                       integrate_velocity, train_hierarchical)

torch.set_num_threads(2)          # LOCAL CPU ONLY, 8 cores shared with other tracks

RAW = ROOT / "data/raw/organoid/organoid_data_preprocessed"
PROC = ROOT / "data/processed"
OUT = ROOT / "results"

# ---- design ---------------------------------------------------------------------
SINGLES = ["S", "F", "C", "V", "L", "O"]
PERT_ID = {p: i for i, p in enumerate(SINGLES)}
K_PERT = len(SINGLES)                       # k = 6
COMPOSITION = {
    "S": ["S"], "F": ["F"], "C": ["C"], "V": ["V"], "L": ["L"], "O": ["O"],
    "VS": ["V", "S"], "CS": ["C", "S"], "SF": ["S", "F"], "CF": ["C", "F"],
    "CSF": ["C", "S", "F"],
}
COMBINATIONS = ["VS", "CS", "SF", "CF", "CSF"]
CONTROLS = {"DMSO", "H2O", "AH"}
CULTURE = "PDO"

# ---- sizes (CPU / RAM constrained: d <= 8, hidden <= 96) -------------------------
LATENT_D = 8                                # data dimension the CFM acts on
# D of the method's chart. It MUST equal d: `ConditionalVelocity` validates
# one_form.d == d, while `PullbackField` computes alpha = J^T b with J of shape (B, D, d),
# so the one-form's output must be D-dimensional. Both constraints together force D = d,
# which is also the convention the synthetic benchmark uses ("the benchmark observes the
# same space it acts on"). The chart is still fully nonlinear -- Decoder's tanh branch,
# the ambient metric and the metric-radial saturator are all active; only the dimension
# is shared. Raising D would require editing ihcfm_hierarchy.py, which this track may not.
AMBIENT_D = LATENT_D
# r is a TUNABLE CAPACITY HYPERPARAMETER, not a correctness requirement. An earlier
# version of this script set r = k(k-1)/2 = 15 from a rank-saturation argument; that
# requirement was RETRACTED by the project lead. The rank arithmetic is still true (the
# pair basis saturates at min(r, k(k-1)/2), so at k=6 fewer than 15 factors alias some
# pairs onto a shared interaction direction), but performance does not follow it.
#
# The benchmark sweep it was retracted on has TWO columns and they DISAGREE, which matters
# because this script's declared selection standard is the held-out one:
#     r=2  all-combination +0.5930   HELD-OUT +0.5986   <- best on held-out
#     r=4  all-combination +0.4414   HELD-OUT (lower)
#     r=6  all-combination +0.5121   HELD-OUT (lower)
#     r=8  all-combination +0.6427   HELD-OUT +0.5440   <- best on all-combination
# So r=8 is best on the aggregate and r=2 is best on held-out. R_DEFAULT follows the
# instruction to default to 8, but the disagreement is why `--tune-r` exists and why r=2
# is in the candidate list: on THIS dataset the choice is made by held-out score here,
# not inherited from the benchmark aggregate.
R_DEFAULT = 8
R_CANDIDATES = (2, 8, 15)   # 2 = benchmark held-out winner, 8 = aggregate
                            # winner and default, 15 = fully unaliased at k=6
HIDDEN = 96
DEPTH = 2
MAX_CELLS = 400                             # per population, subsampled
MAX_DOSE = 4
STEPS = (300, 300, 200)
BATCH = 128
N_INT_STEPS = 12                            # RK4 steps for prediction
EVAL_MAX_N = 400                            # cloud size cap inside the metrics


# ==================================================================================
# populations
# ==================================================================================
@dataclass
class Pop:
    """Duck-typed for `ihcfm_hierarchy._pop_arrays` (.P, .tau, .pre, .post) plus provenance.

    `replicate` is the split-record index and is THE INDEPENDENT UNIT (see module
    docstring): every population sharing a `replicate` also shares its control draw.
    """
    label: str
    P: tuple
    tau: float
    pre: np.ndarray
    post: np.ndarray
    replicate: int = 0
    treat: str = ""
    dose: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def order(self) -> int:
        return len(self.P)


def dose_tau(dose: int) -> float:
    """Dose level -> exposure. Level 0 is control; 1..4 map to 0.25..1.0."""
    return max(int(dose), 1) / float(MAX_DOSE)


def enumerate_design(verbose: bool = True) -> tuple[list[dict], int]:
    """Walk the pickle and list (record, treatment, dose, control idx, treated idx).

    Cell INDICES only -- the 8.7 GB matrix is never touched here.
    """
    with open(RAW / "data_splits_replicas_1.pickle", "rb") as f:
        splits = pickle.load(f)
    recs = splits["train"] + splits["val"] + splits["test"]
    out, n_with_ctrl = [], 0
    for ri, rec in enumerate(recs):
        ctrl = []
        for t, dd in rec.items():
            if t not in CONTROLS:
                continue
            for _dose, cc in dd.items():
                ct = cc.get(CULTURE)
                if not ct:
                    continue
                a = ct.get("PDOs")
                if a is not None and np.asarray(a).size:
                    ctrl.append(np.asarray(a))
        if not ctrl:
            continue
        n_with_ctrl += 1
        ctrl_idx = np.unique(np.concatenate(ctrl))
        for t, dd in rec.items():
            if t in CONTROLS or t not in COMPOSITION:
                continue
            for dose, cc in dd.items():
                ct = cc.get(CULTURE)
                if not ct:
                    continue
                a = ct.get("PDOs")
                if a is None or np.asarray(a).size < 50:
                    continue
                out.append(dict(replicate=ri, treat=t, dose=int(dose),
                                ctrl=ctrl_idx, treated=np.asarray(a)))
    if verbose:
        print(f"  records with a matched {CULTURE} control: {n_with_ctrl}"
              f" | populations: {len(out)}", flush=True)
    return out, n_with_ctrl


def fit_control_pca(design: list[dict], A, n_fit: int = 60_000, seed: int = 0) -> dict:
    """PCA fitted on CONTROL cells ONLY, so the basis is not shaped by treatment effects.

    Two choices worth stating. (1) Only control cells enter the fit: if treated cells did,
    the leading directions would already encode the response the model is asked to predict
    and every held-out score would be optimistic. (2) The eigendecomposition is taken on
    the 44 x 44 covariance rather than by SVD of the (n x 44) matrix -- identical basis,
    but it keeps the peak allocation at a few hundred MB, which this machine needs.
    """
    rng = np.random.default_rng(seed)
    ctrl_all = np.unique(np.concatenate([d["ctrl"] for d in design]))
    sel = np.sort(rng.choice(ctrl_all, size=min(n_fit, ctrl_all.size), replace=False))
    X = np.asarray(A[sel], dtype=np.float64)
    mu = X.mean(0)
    Xc = X - mu
    C = (Xc.T @ Xc) / max(Xc.shape[0] - 1, 1)
    ev, V = np.linalg.eigh(C)                 # ascending
    order = np.argsort(-ev)
    ev, V = ev[order], V[:, order]
    W = V[:, :LATENT_D]
    scale = (Xc @ W).std(0) + 1e-9
    return dict(mu=mu, W=W, scale=scale,
                evr=(ev[:LATENT_D] / max(ev.sum(), 1e-300)),
                n_control_cells_available=int(ctrl_all.size),
                n_control_cells_fitted=int(sel.size))


def _project(A, idx: np.ndarray, proj: dict, rng, max_cells: int) -> np.ndarray:
    if idx.size > max_cells:
        idx = rng.choice(idx, size=max_cells, replace=False)
    idx = np.sort(idx)
    X = np.asarray(A[idx], dtype=np.float64)
    return ((X - proj["mu"]) @ proj["W"]) / proj["scale"]


def prepare(seed: int = 0, max_cells: int = MAX_CELLS, cache: bool = True,
            verbose: bool = True) -> tuple[list[Pop], dict]:
    """Build the population list (lazily, subsampled) and cache the LATENTS only.

    The cache carries `replicate`, which the earlier organoid cache did not, and which the
    leave-replicate-out split cannot be built without. It is written under a distinct
    filename so the previous case study's cache is left untouched.
    """
    PROC.mkdir(parents=True, exist_ok=True)
    cf = PROC / f"organoid_loco_d{LATENT_D}_{CULTURE}_n{max_cells}_s{seed}.npz"
    if cache and cf.exists():
        z = np.load(cf, allow_pickle=True)
        pops = [Pop(**dict(p)) for p in z["pops"]]
        if verbose:
            print(f"  loaded cache {cf.name}: {len(pops)} populations", flush=True)
        return pops, dict(z["meta"].item())
    t0 = time.time()
    design, n_rec = enumerate_design(verbose=verbose)
    A = np.load(RAW / "trellis_replicas_1_normalized.npy", mmap_mode="r")
    if verbose:
        print(f"  matrix {A.shape} memory-mapped ({A.dtype})", flush=True)
    proj = fit_control_pca(design, A, seed=seed)
    if verbose:
        print(f"  PCA on controls only: {proj['n_control_cells_fitted']} of "
              f"{proj['n_control_cells_available']} control cells | "
              f"evr {np.round(proj['evr'], 4)}", flush=True)
    rng = np.random.default_rng(seed)
    pops: list[Pop] = []
    for d in design:
        P = tuple(PERT_ID[c] for c in COMPOSITION[d["treat"]])
        pops.append(Pop(label=f"{d['treat']}@d{d['dose']}r{d['replicate']}",
                        P=P, tau=dose_tau(d["dose"]),
                        pre=_project(A, d["ctrl"], proj, rng, max_cells),
                        post=_project(A, d["treated"], proj, rng, max_cells),
                        replicate=int(d["replicate"]), treat=d["treat"],
                        dose=int(d["dose"]),
                        meta=dict(n_ctrl_avail=int(d["ctrl"].size),
                                  n_treated_avail=int(d["treated"].size))))
    meta = dict(n_records=int(n_rec), n_populations=len(pops),
                evr=[float(x) for x in proj["evr"]],
                evr_cumulative=float(np.sum(proj["evr"])),
                n_control_cells_fitted=proj["n_control_cells_fitted"],
                latent_d=LATENT_D, max_cells=int(max_cells),
                prepare_wall_clock_s=float(time.time() - t0))
    if cache:
        np.savez_compressed(cf, pops=np.array([p.__dict__ for p in pops], dtype=object),
                            meta=np.array(meta, dtype=object))
    if verbose:
        print(f"  prepared in {meta['prepare_wall_clock_s']:.0f}s", flush=True)
    return pops, meta


# ==================================================================================
# model
# ==================================================================================
class PullbackAdapter(torch.nn.Module):
    """Adapts `PullbackField(t, z, one_form=...)` to the `pullback(t, z, b)` contract.

    This is an nn.Module rather than a closure for a load-bearing reason: closures are
    invisible to `nn.Module` registration, so a `PullbackField` captured in one is NOT
    reached by `.to(dtype)`. `ConditionalVelocity.to_inference_dtype()` deep-copies and
    casts the model to float64 (project fact D: float32 floors chart Test A at ~1e-8),
    and with a closure the velocity net would be double while the decoder inside the
    pullback stayed single -- which raises `expected m1 and m2 to have the same dtype`.
    Registering it as a submodule makes the whole composed field cast together.

    The decoder is deliberately shared with `ConditionalVelocity.decoder`: it is the SAME
    chart on both sides of the pullback. `nn.Module.parameters()` deduplicates by identity
    so it is not double-counted, and one `deepcopy` preserves the sharing.
    """

    def __init__(self, field: PullbackField):
        super().__init__()
        self.field = field

    def forward(self, t, z, b):
        return self.field(t, z, one_form=lambda tt, xx, cc: b)


def make_ihcfm(d: int = LATENT_D, k: int = K_PERT, r: int = R_DEFAULT, code_dim: int = 0,
               hidden: int = HIDDEN, depth: int = DEPTH, seed: int = 0,
               interaction_enabled: bool = True) -> ConditionalVelocity:
    """The FULL model: Hill dose, one-form hierarchy, pullback field, radial saturation.

    Every architectural component is present. `interaction_enabled=False` and
    `code_dim` are ABLATION FLAGS whose defaults give the full model; nothing in this
    screen simplifies the architecture to make a number look better.
    """
    torch.manual_seed(int(seed))
    dec = Decoder(int(d), int(AMBIENT_D), hidden=32, s=0.5, seed=int(seed))
    met = AmbientMetric(int(AMBIENT_D), hidden=48, w_min=0.25, l_cap=1.5)
    sat = MetricRadialSat(x_dim=int(AMBIENT_D), latent_dim=int(d), hidden=32,
                          mode="metric_radial", kappa_init=2.0)
    pbf = PullbackField(dec, met, saturator=sat)      # saturator wraps the PULLBACK
    pullback = PullbackAdapter(pbf)
    dose = HillDose(int(k), amp_init=1.0, ec50_init=1.0, hill_init=1.0)
    return ConditionalVelocity(int(d), int(k), r=int(r), code_dim=int(code_dim),
                               hidden=int(hidden), depth=int(depth), dose=dose,
                               pullback=pullback, decoder=dec,
                               interaction_enabled=bool(interaction_enabled))


def objective(seed: int = 0) -> CFMObjective:
    """I-CFM with unbalanced-Sinkhorn coupling ("sinkhorn_uot" = this exact kwarg set).

    *** CFM LOSS MUST NOT BE USED TO COMPARE COUPLINGS. ***
    An earlier selection of this coupling cited a CFM-loss spread (independent 6.987 / ot
    0.860 / sinkhorn_uot 0.858) and that comparison is INVALID: CFM loss tracks the
    fraction of trivially identical source/target pairs a coupling produces (measured
    identity-pair fraction: paired 1.0, ot 0.795, sinkhorn 0.229, independent 0.0039 --
    the `paired` arm reaches loss 6.4e-05 for that reason alone), so it rewards couplings
    for making the regression target trivial rather than for making it correct. On the
    DISTRIBUTIONAL metric the ranking INVERTS: normalised ED independent 0.8094,
    sinkhorn_uot 0.8387, ot 1.1188 -- `ot` is WORSE THAN PREDICTING NO CHANGE.

    The coupling used here was re-selected on ground-truth interaction recovery, which is
    scale-free and comparable across couplings (3 seeds): sinkhorn_uot beats independent
    at every rank, +0.5930 vs +0.0799 (r=2), +0.4414 vs +0.2586 (r=4), +0.6427 vs +0.4361
    (r=8).

    Note there is no `coupling="sinkhorn_uot"` string -- it raises ValueError. The label
    means the four kwargs below.
    """
    return CFMObjective(sigma=0.0, coupling="sinkhorn", sinkhorn_eps=0.05,
                        sinkhorn_iters=25, unbalanced_tau=1.0, seed=int(seed),
                        max_coupling_n=128)


# ==================================================================================
# prediction records (what the metric suite consumes)
# ==================================================================================
def predict_records(pops: list[Pop], idx, predict_fn, tag: str) -> list[dict]:
    """Push each population's PRE cloud forward and package it for `metrics.score_records`.

    `unit` is the replicate record id -- the independent unit the bootstrap groups on.
    """
    recs = []
    for i in idx:
        p = pops[i]
        pred = predict_fn(p)
        recs.append(dict(unit=f"rep{p.replicate}", condition=p.treat,
                         pred=np.asarray(pred, dtype=np.float64),
                         true=p.post, ctrl=p.pre,
                         dose=int(p.dose), order=int(p.order), split=tag,
                         replicate=int(p.replicate), pop_index=int(i)))
    return recs


def ihcfm_predictor(model: ConditionalVelocity, n_steps: int = N_INT_STEPS):
    m64 = model.to_inference_dtype(torch.float64)

    def f(p: Pop) -> np.ndarray:
        return integrate_velocity(m64, p.pre, p.P, p.tau, K_PERT, n_steps=n_steps,
                                  mu0=p.pre)
    return f


def baseline_predictor(model):
    def f(p: Pop) -> np.ndarray:
        return model.predict(p.pre, p.P, p.tau)
    return f


def score_arm(recs: list[dict], n_boot: int, seed: int = 0,
              denominator_reference: float | None = None) -> dict:
    """Per-record metric rows + grouped-bootstrap summary + the residualised report.

    `denominator_reference` is the typical no-change denominator on the TRAINING split.
    Passing it lets `flag_denominator_collapse` say whether this held-out fold's
    normalised column is divided by a collapsed denominator -- the trap that inflated
    every finance k=4 score. Without it the ratio is still reported, but so is the
    absolute distance and the denominator itself, so the reader can always see the scale.
    """
    rows = M.score_records(recs, seed=seed, max_n=EVAL_MAX_N, n_slices=64)
    out = dict(per_population=rows,
               summary=M.summarise_per_unit(rows, n_boot=n_boot, seed=seed),
               denominator_check=M.flag_denominator_collapse(
                   rows, reference=denominator_reference))
    if len(recs) >= 2:
        out["residualised"] = M.residualised_report(recs, max_n=EVAL_MAX_N, seed=seed)
        out["residualised"].pop("distribution_level_full", None)
    return out


# ==================================================================================
# folds
# ==================================================================================
def loco_folds(pops: list[Pop], combos=tuple(COMBINATIONS)) -> list[dict]:
    """One fold per held-out combination, with a FEASIBILITY check on each.

    A combination is feasible only if every one of its constituent single compounds is
    still observed in the training set -- otherwise the fold asks the model to compose a
    primitive it has never seen, which is a different (and unidentifiable) question. All
    four pairs and the triple pass here; the check is kept because it is the assumption the
    fold rests on and a future subsample could break it.
    """
    treats = np.array([p.treat for p in pops])
    folds = []
    for cb in combos:
        held = np.nonzero(treats == cb)[0]
        train = np.nonzero(treats != cb)[0]
        if held.size == 0:
            continue
        tr_treats = set(treats[train])
        need = list(COMPOSITION[cb])
        missing = [c for c in need if c not in tr_treats]
        n_units = len({pops[i].replicate for i in held})
        doses = sorted({pops[i].dose for i in held})
        folds.append(dict(name=cb, held_out=[cb], order=len(need), doses=doses,
                          train_idx=train, held_idx=held,
                          n_train=int(train.size), n_held=int(held.size),
                          n_held_units=int(n_units),
                          # A single observed dose means composition and exposure are the
                          # same axis in this fold (see module docstring); VS spans three
                          # doses and is the one pair where they separate.
                          dose_confounded=bool(len(doses) == 1),
                          feasible=(len(missing) == 0), missing_singles=missing))
    return folds


def replicate_folds(pops: list[Pop], n_folds: int = 3, seed: int = 0) -> list[dict]:
    """Leave-REPLICATE-out: hold out whole records, so no evaluated control was ever seen.

    Records are partitioned (not sampled), so the folds are disjoint and every record is
    held out exactly once across the full set. A fold is kept only if the held-out side
    still contains at least one combination population -- the composition question is the
    one being asked, and a fold of singletons only cannot answer it.
    """
    reps = np.array(sorted({p.replicate for p in pops}))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(reps)
    chunks = np.array_split(perm, int(n_folds))
    pop_rep = np.array([p.replicate for p in pops])
    pop_ord = np.array([p.order for p in pops])
    out = []
    for j, ch in enumerate(chunks):
        held = np.nonzero(np.isin(pop_rep, ch))[0]
        train = np.nonzero(~np.isin(pop_rep, ch))[0]
        n_combo = int(np.sum(pop_ord[held] >= 2))
        out.append(dict(name=f"repfold{j}", held_records=[int(x) for x in ch],
                        train_idx=train, held_idx=held,
                        n_train=int(train.size), n_held=int(held.size),
                        n_held_units=int(ch.size), n_held_combination_pops=n_combo,
                        feasible=bool(n_combo > 0)))
    return out


def rank_report(k: int = K_PERT, r: int = R_DEFAULT) -> dict:
    """How many of the k(k-1)/2 pairs this rank can represent WITHOUT aliasing.

    Reported, never enforced. The pair basis saturates at min(r, k(k-1)/2), so r < 15 at
    k=6 means some pairs share an interaction direction. That is a real limit on what the
    interaction branch can express and a reader should see it -- but it does NOT predict
    held-out score (r=2 beat r=4 and r=6 on the benchmark), so the run proceeds either way
    and r is chosen by score instead.
    """
    n_pairs = k * (k - 1) // 2
    return dict(k=int(k), r=int(r), n_distinct_pairs=int(n_pairs),
                n_pairs_representable=int(min(r, n_pairs)),
                fully_unaliased=bool(r >= n_pairs),
                note=("pair basis rank saturates at min(r, k(k-1)/2); r below that "
                      "aliases some pairs onto a shared interaction direction. Reported "
                      "for interpretation -- r is selected by held-out score, since the "
                      "measured recovery trend does not follow this bound."))


def select_rank(pops: list[Pop], train_idx: np.ndarray, candidates=R_CANDIDATES,
                seed: int = 0, steps=STEPS, n_val_units: int = 8,
                verbose: bool = True, default_r: int = R_DEFAULT,
                min_margin: float = 0.05) -> dict:
    """Choose r by held-out DISTRIBUTIONAL score, never by CFM loss and never by formula.

    r IS TUNED PER FOLD, and the pool passed in must be that fold's TRAINING side only.
    This is what makes selection both leak-free and meaningful. A global tuning pool that
    excluded every screened combination would contain only singletons (all four pairs and
    the triple are screened), and with no combination in either the fit or the validation
    half the interaction branch is never exercised -- every candidate r would score the
    same and the "selection" would be noise. Selecting inside a fold fixes this: the
    fold's training side still contains the OTHER combinations, so r is chosen on genuine
    composition data while the fold's own held-out combination is never seen.

    The validation half is carved by holding out whole REPLICATE RECORDS that carry
    combinations, so the score is measured on control clouds the model never saw -- the
    same standard the screen itself uses.

    This function REFUSES to select on a pool with no combination populations, rather than
    returning a number that cannot discriminate its candidates.

    Selection is on normalised energy distance (reported with its denominator). CFM loss
    is deliberately not used: it is not comparable across configurations that change how
    trivial the regression target is.
    """
    rng = np.random.default_rng(seed + 77)
    combo_reps = np.array(sorted({pops[i].replicate for i in train_idx
                                  if pops[i].order >= 2}))
    n_combo_pops = int(sum(pops[i].order >= 2 for i in train_idx))
    if n_combo_pops == 0 or combo_reps.size < 2:
        return dict(selected_r=None, skipped=True,
                    reason=("tuning pool has %d combination populations across %d "
                            "records -- r cannot be discriminated without combinations, "
                            "so no selection was made and the default r is used"
                            % (n_combo_pops, combo_reps.size)),
                    n_combination_populations=n_combo_pops,
                    n_combination_records=int(combo_reps.size))
    basis = ("held-out replicate records containing combinations, drawn from this fold's "
             "training side; the fold's own held-out combination is absent")
    pick = rng.choice(combo_reps,
                      size=min(int(n_val_units), max(1, combo_reps.size // 4)),
                      replace=False)
    val = np.array([i for i in train_idx if pops[i].replicate in set(pick)])
    sub = np.array([i for i in train_idx if pops[i].replicate not in set(pick)])
    assert val.size and sub.size, "tuning split degenerate"
    rows = []
    for r in candidates:
        t0 = time.time()
        m = make_ihcfm(r=int(r), seed=seed)
        rep = train_hierarchical(m, pops, list(sub), objective=objective(seed),
                                 steps=steps, batch=BATCH, lr=3e-3, seed=seed,
                                 verbose=False)
        recs = predict_records(pops, val, ihcfm_predictor(m), f"tune_r{r}")
        sc = M.score_records(recs, seed=seed, max_n=EVAL_MAX_N, n_slices=64)
        summ = M.summarise_per_unit(sc, n_boot=300, seed=seed)
        rk = rank_report(K_PERT, r)
        rk.pop("r", None)          # `r` is set explicitly below; rank_report also emits
                                   # it, and dict(r=..., **rk) would raise TypeError
        rows.append(dict(r=int(r),
                         ed_normalised=summ["ed_normalised"]["point"],
                         ed_normalised_ci=[summ["ed_normalised"]["lo"],
                                           summ["ed_normalised"]["hi"]],
                         ed_absolute=summ["ed_absolute"]["point"],
                         ed_denominator=summ["ed_denominator"]["point"],
                         cosine_delta=summ.get("cosine_delta", {}).get("point"),
                         n_parameters=int(sum(p.numel() for p in m.parameters())),
                         skipped_nonfinite=int(rep["skipped_nonfinite_total"]),
                         wall_clock_s=float(time.time() - t0), **rk))
        if verbose:
            print(f"    r={r:3d}  val normED {rows[-1]['ed_normalised']:.4f} "
                  f"[{rows[-1]['ed_normalised_ci'][0]:.4f}, "
                  f"{rows[-1]['ed_normalised_ci'][1]:.4f}]  "
                  f"denom {rows[-1]['ed_denominator']:.4f}  "
                  f"pairs {rows[-1]['n_pairs_representable']}/"
                  f"{rows[-1]['n_distinct_pairs']}  ({rows[-1]['wall_clock_s']:.0f}s)",
                  flush=True)
    # A winner by a hair is not a winner. On one smoke fold the three candidates scored
    # 1.1095 / 1.2363 / 1.1094 with bootstrap CIs overlapping almost completely -- the
    # "best" r was decided by 1e-4, which is exactly the noise-selection failure this
    # function exists to avoid. So a candidate only displaces the default if it beats it
    # by at least `min_margin` AND its CI upper bound sits below the default's point
    # estimate (i.e. the improvement survives the fold's own uncertainty). Otherwise the
    # default stands and `decisive` is False, which is reported rather than hidden.
    ranked = sorted(rows, key=lambda x: x["ed_normalised"])
    best = ranked[0]
    dflt = next((x for x in rows if int(x["r"]) == int(default_r)), None)
    decisive = True
    if dflt is not None and int(best["r"]) != int(default_r):
        gain = float(dflt["ed_normalised"]) - float(best["ed_normalised"])
        decisive = bool(gain >= float(min_margin)
                        and best["ed_normalised_ci"][1] < dflt["ed_normalised"])
        if not decisive:
            best = dflt
    return dict(selected_r=int(best["r"]), criterion="held-out normalised energy distance",
                decisive=bool(decisive), default_r=int(default_r),
                min_margin=float(min_margin),
                selection_note=("a candidate displaces the default only if it wins by "
                                ">= min_margin AND its CI upper bound is below the "
                                "default's point estimate; otherwise the default stands "
                                "and decisive=False, because a sub-noise margin is not "
                                "evidence"),
                validation_basis=basis,
                n_validation_records=int(pick.size), n_validation_populations=int(val.size),
                n_fit_populations=int(sub.size), candidates=rows,
                note=("selected on a distributional metric over held-out RECORDS from the "
                      "training side only; CFM loss was not used because it is not "
                      "comparable across configurations"))


def shrink_fold(fold: dict, pops: list[Pop], n_train: int, n_held: int,
                seed: int = 0) -> dict:
    """Subsample a fold for the SMOKE run, keeping it a valid fold.

    Two invariants must survive the shrink or the smoke test stops exercising the real
    code path:
      * stage 1 of `train_hierarchical` fits singleton conditions and RAISES without any,
        so the training subsample is stratified to keep singletons (at least half the
        budget) alongside combinations.
      * the recorded population/unit counts are recomputed AFTER subsampling, so the
        smoke json does not report the full fold's sizes next to a shrunken fold's scores.
    """
    rng = np.random.default_rng(seed)
    tr = np.asarray(fold["train_idx"])
    orders = np.array([pops[i].order for i in tr])
    sing, comb = tr[orders == 1], tr[orders >= 2]
    n_s = min(sing.size, max(1, n_train // 2))
    n_c = min(comb.size, n_train - n_s)
    keep = [rng.choice(sing, size=n_s, replace=False)] if n_s else []
    if n_c:
        keep.append(rng.choice(comb, size=n_c, replace=False))
    f = dict(fold)
    f["train_idx"] = np.sort(np.concatenate(keep)) if keep else tr
    he = np.asarray(fold["held_idx"])
    f["held_idx"] = np.sort(rng.choice(he, size=min(n_held, he.size), replace=False))
    f["n_train"] = int(f["train_idx"].size)
    f["n_held"] = int(f["held_idx"].size)
    f["n_held_units"] = len({pops[i].replicate for i in f["held_idx"]})
    f["n_held_combination_pops"] = int(sum(pops[i].order >= 2 for i in f["held_idx"]))
    f["n_train_singletons"] = int(n_s)
    f["smoke_shrunk"] = True
    if "n_held_combination_pops" in fold:
        f["feasible"] = bool(f["n_held_combination_pops"] > 0)
    return f


def run_fold(pops: list[Pop], fold: dict, seed: int = 0, steps=STEPS, code_dim: int = 0,
             n_boot: int = 600, verbose: bool = True, r: int = R_DEFAULT,
             tune_r: bool = False,
             arms=("ihcfm", "perturbed_mean", "no_change")) -> dict:
    """Train IHC-FM on the fold's training index, then score every arm on the held-out set.

    Baselines are fitted on the IDENTICAL training index that IHC-FM trained on -- the
    reduced `tr_arr` after the validation records are carved out, NOT the full `tr`. This
    matters: giving the baselines the validation records too would hand them strictly more
    training data than the model they are compared against, and the paired per-record
    differences would then confound a data-quantity advantage with a modelling one. All
    arms are scored on the identical held-out populations from the same PRE clouds.
    """
    tr, he = fold["train_idx"], fold["held_idx"]
    res = dict(fold=fold["name"], n_train=int(len(tr)), n_held=int(len(he)),
               n_held_units=int(fold.get("n_held_units", 0)), code_dim=int(code_dim),
               doses=fold.get("doses"), r=int(r),
               dose_confounded=fold.get("dose_confounded"),
               feasible=bool(fold.get("feasible", True)))
    obj = objective(seed=seed)
    t0 = time.time()
    model = make_ihcfm(r=int(r), code_dim=code_dim, seed=seed)
    res["n_parameters"] = int(sum(p.numel() for p in model.parameters()))
    # val_idx must NOT be the screened held-out set. `train_hierarchical` documents
    # val_idx as used "for reporting and SELECTION", so feeding it the fold's held-out
    # combination would let the reported LOCO score be selected on its own test set.
    # Instead a validation slice is carved out of the TRAINING side by holding out whole
    # replicate records that contain combination populations (so it can actually speak to
    # composition), and the fold's held-out set is never seen during training.
    rng_v = np.random.default_rng(seed + 31)
    tr_arr = np.asarray(tr)
    combo_reps = np.array(sorted({pops[i].replicate for i in tr_arr
                                  if pops[i].order >= 2}))
    val_idx = None
    if combo_reps.size >= 4:
        pick = set(rng_v.choice(combo_reps, size=max(2, combo_reps.size // 8),
                                replace=False).tolist())
        v = [int(i) for i in tr_arr if pops[i].replicate in pick]
        f = [int(i) for i in tr_arr if pops[i].replicate not in pick]
        if v and any(pops[i].order == 1 for i in f):
            tr_arr, val_idx = np.array(f), v
    res["n_validation_populations"] = int(len(val_idx or []))
    res["validation_source"] = ("held-out replicate records from the TRAINING side; the "
                                "screened held-out combination is never used for "
                                "selection or monitoring")

    # r selected on THIS fold's training side, which still contains the other
    # combinations -- so the interaction branch is actually exercised during selection
    # while the fold's held-out combination stays unseen.
    if tune_r:
        tun = select_rank(pops, tr_arr, seed=seed, steps=steps,
                          candidates=R_CANDIDATES, verbose=verbose)
        res["rank_tuning"] = tun
        if tun.get("selected_r"):
            if verbose:
                print(f"      r={tun['selected_r']} selected"
                      f"{'' if tun.get('decisive') else ' (NOT decisive -- margin below '
                                                        'noise, default retained)'}",
                      flush=True)
            r = int(tun["selected_r"])
            res["r"] = r
            model = make_ihcfm(r=r, code_dim=code_dim, seed=seed)
            res["n_parameters"] = int(sum(p.numel() for p in model.parameters()))
        elif verbose:
            print(f"      rank tuning skipped: {tun.get('reason')}", flush=True)
    res["rank_report"] = rank_report(K_PERT, r)
    rep = train_hierarchical(model, pops, list(tr_arr), objective=obj, val_idx=val_idx,
                             steps=steps, batch=BATCH, lr=3e-3, seed=seed,
                             int_penalty=0.0, verbose=False)
    res["n_train"] = int(tr_arr.size)
    res["train_report"] = {s: {kk: vv for kk, vv in rep[s].items()
                               if kk != "loss_history"}
                           for s in ("stage1", "stage2", "stage3")}
    res["skipped_nonfinite_total"] = int(rep["skipped_nonfinite_total"])
    res["train_wall_clock_s"] = float(time.time() - t0)
    if verbose:
        print(f"    trained in {res['train_wall_clock_s']:.0f}s "
              f"(skipped {res['skipped_nonfinite_total']} non-finite steps)", flush=True)

    preds = {}
    if "ihcfm" in arms:
        preds["ihcfm"] = ihcfm_predictor(model)
    if "perturbed_mean" in arms:
        bm = PerturbedMean(LATENT_D, K_PERT, seed=seed)
        bm.fit(pops, list(tr_arr))          # SAME index the model trained on
        preds["perturbed_mean"] = baseline_predictor(bm)
    if "no_change" in arms:
        nc = NoChange(LATENT_D, K_PERT, seed=seed)
        nc.fit(pops, list(tr_arr))          # SAME index the model trained on
        preds["no_change"] = baseline_predictor(nc)

    # Reference scale for the denominator-collapse check: the median no-change energy
    # distance ED(pre, post) over a sample of the TRAINING populations. This is the
    # "healthy" denominator the held-out fold's is compared against, and it is computed
    # from training data only, so it leaks nothing about the held-out fold.
    rng_ref = np.random.default_rng(seed)
    ref_sel = rng_ref.choice(tr_arr, size=min(40, tr_arr.size), replace=False)
    den_ref = float(np.median([
        M.energy_distance(pops[i].pre, pops[i].post, max_n=EVAL_MAX_N, seed=seed)
        for i in ref_sel]))
    res["train_denominator_reference"] = den_ref
    res["train_denominator_reference_n"] = int(ref_sel.size)

    res["arms"], stored = {}, {}
    for name, fn in preds.items():
        t1 = time.time()
        recs = predict_records(pops, he, fn, fold["name"])
        stored[name] = recs
        res["arms"][name] = score_arm(recs, n_boot=n_boot, seed=seed,
                                      denominator_reference=den_ref)
        res["arms"][name]["predict_wall_clock_s"] = float(time.time() - t1)
        s = res["arms"][name]["summary"]
        if verbose and "ed_normalised" in s:
            c = s["ed_normalised"]
            print(f"      {name:15s} normED {c['point']:.4f} "
                  f"[{c['lo']:.4f}, {c['hi']:.4f}]  "
                  f"absED {s['ed_absolute']['point']:.4f}  "
                  f"denom {s['ed_denominator']['point']:.4f}  "
                  f"(n_units {c['n_groups']})", flush=True)

    # ---- PAIRED per-record differences, on the same units --------------------------
    res["paired"] = {}
    for a, b in (("ihcfm", "perturbed_mean"), ("ihcfm", "no_change"),
                 ("perturbed_mean", "no_change")):
        if a not in stored or b not in stored:
            continue
        # Pair on pop_index IDENTITY, never on list position: a paired test on
        # mis-aligned rows is silently wrong rather than noisy, and the alignment is
        # cheap to enforce.
        ma = {r["pop_index"]: r for r in res["arms"][a]["per_population"]}
        mb = {r["pop_index"]: r for r in res["arms"][b]["per_population"]}
        common = sorted(set(ma) & set(mb))
        for metric in ("ed_normalised", "ed_absolute", "cosine_delta"):
            va = [ma[i][metric] for i in common]
            vb = [mb[i][metric] for i in common]
            g = [ma[i]["unit"] for i in common]
            ok = [i for i in range(len(va))
                  if np.isfinite(va[i]) and np.isfinite(vb[i])]
            if len(ok) < 2:
                continue
            res["paired"][f"{a}_vs_{b}::{metric}"] = M.paired_group_difference(
                [va[i] for i in ok], [vb[i] for i in ok], [g[i] for i in ok],
                n_boot=n_boot, seed=seed, name_a=a, name_b=b)
    if verbose:
        key = "ihcfm_vs_perturbed_mean::ed_normalised"
        if key in res["paired"]:
            p = res["paired"][key]
            print(f"      PAIRED ihcfm - perturbed_mean (normED, lower better): "
                  f"{p['mean_diff']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}] "
                  f"sign p={p['sign_test_p']:.3g} over {p['n_units']} records",
                  flush=True)
    return res, stored


# ==================================================================================
# driver
# ==================================================================================
def main(smoke: bool = False, seed: int = 0, n_boot: int = 600,
         rep_folds: int = 3, save_predictions: bool = True, r: int = R_DEFAULT,
         tune_r: bool = False) -> dict:
    t_start = time.time()
    max_cells = 120 if smoke else MAX_CELLS
    steps = (40, 40, 20) if smoke else STEPS
    nb = 120 if smoke else n_boot
    print(f"preparing populations (max_cells={max_cells})", flush=True)
    pops, meta = prepare(seed=seed, max_cells=max_cells)
    print(f"  {meta['n_populations']} populations | {meta['n_records']} records | "
          f"PCA evr sum {meta['evr_cumulative']:.4f}", flush=True)

    combos = ("CSF", "CF") if smoke else tuple(COMBINATIONS)
    folds = loco_folds(pops, combos=combos)
    if smoke:
        folds = [shrink_fold(f, pops, n_train=90, n_held=8, seed=seed) for f in folds]

    # ---- rank: selected by held-out score if asked, otherwise the measured default ----
    # r is selected PER FOLD inside `run_fold` (never globally): a global pool that
    # excluded every screened combination would be singletons only, and r cannot be
    # discriminated without combinations. `r` here is the default / fallback.
    tuning = dict(policy=("r selected per fold on that fold's training side, which "
                          "contains the other combinations but never the fold's own "
                          "held-out one; see each fold's rank_tuning"),
                  enabled=bool(tune_r), default_r=int(r),
                  candidates=list(R_CANDIDATES))
    rk = rank_report(K_PERT, r)
    print(f"rank: k={K_PERT} r={r} -> {rk['n_pairs_representable']}/"
          f"{rk['n_distinct_pairs']} pairs representable "
          f"(fully unaliased: {rk['fully_unaliased']}) -- reported, not enforced",
          flush=True)

    out = dict(config=dict(smoke=bool(smoke), seed=int(seed), latent_d=LATENT_D,
                           ambient_d=AMBIENT_D, k=K_PERT, r=int(r), hidden=HIDDEN,
                           depth=DEPTH, max_cells=int(max_cells), steps=list(steps),
                           batch=BATCH, n_int_steps=N_INT_STEPS, n_boot=int(nb),
                           coupling=objective(seed).describe(),
                           rank_report=rk, rank_tuning=tuning,
                           rank_policy=("r is a tunable capacity hyperparameter; the "
                                        "r >= k(k-1)/2 requirement is retracted and the "
                                        "bound is reported for interpretation only"),
                           selection_metric=("normalised energy distance with its "
                                             "no-change denominator; CFM loss is NEVER "
                                             "used to compare couplings or ranks"),
                           run_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
               data=meta, loco=[], leave_replicate_out=[])

    all_preds = {}
    print(f"\n=== LOCO screen: {len(folds)} held-out combinations ===", flush=True)
    for f in folds:
        print(f"  [{f['name']}] order {f['order']} doses {f['doses']}"
              f"{' DOSE-CONFOUNDED' if f['dose_confounded'] else ' multi-dose'} | "
              f"train {f['n_train']} | held {f['n_held']} pops / "
              f"{f['n_held_units']} records | feasible {f['feasible']}", flush=True)
        if not f["feasible"]:
            out["loco"].append(dict(fold=f["name"], skipped="infeasible",
                                    missing_singles=f["missing_singles"]))
            continue
        fr, stored = run_fold(pops, f, seed=seed, steps=steps, code_dim=0,
                              n_boot=nb, r=r, tune_r=tune_r)
        fr["missing_singles"] = f["missing_singles"]
        out["loco"].append(fr)
        if save_predictions:
            all_preds[f"loco_{f['name']}"] = stored

    # ---- leave-replicate-out, code_dim 0 vs 16 as a matched pair -------------------
    # n_folds must be >= 2 even in the smoke run: with 1 fold every record is held out
    # and the training index is empty. The smoke run then keeps only the first fold.
    rfolds = replicate_folds(pops, n_folds=(3 if smoke else rep_folds), seed=seed)
    if smoke:
        rfolds = [shrink_fold(rfolds[0], pops, n_train=90, n_held=10, seed=seed + 1)]
    print(f"\n=== leave-replicate-out: {len(rfolds)} folds, code_dim 0 vs 16 ===",
          flush=True)
    for f in rfolds:
        print(f"  [{f['name']}] held {f['n_held']} pops / {f['n_held_units']} records "
              f"({f['n_held_combination_pops']} combination pops) | train {f['n_train']}",
              flush=True)
        if not f["feasible"]:
            out["leave_replicate_out"].append(dict(fold=f["name"],
                                                   skipped="no combination populations"))
            continue
        for cd in (0, 16):
            print(f"    code_dim={cd}", flush=True)
            # tune_r is deliberately OFF here: this arm is a matched code_dim 0-vs-16
            # comparison, and letting each member select its own r would confound the
            # conditioning effect with a capacity difference. Both run at the same r.
            fr, stored = run_fold(pops, f, seed=seed, steps=steps, code_dim=cd,
                                  n_boot=nb, r=r, tune_r=False,
                                  arms=("ihcfm", "no_change"))
            fr["fold"] = f["name"]
            out["leave_replicate_out"].append(fr)
            if save_predictions:
                all_preds[f"lro_{f['name']}_code{cd}"] = stored

    # ---- cross-fold roll-ups --------------------------------------------------------
    out["rollup"] = rollup(out)
    out["wall_clock_total_s"] = float(time.time() - t_start)

    OUT.mkdir(exist_ok=True)
    tag = "_smoke" if smoke else ""
    jf = OUT / f"loco_screen{tag}.json"
    json.dump(_jsonable(out), open(jf, "w"), indent=2)
    print(f"\nwrote {jf.relative_to(ROOT)}  ({out['wall_clock_total_s']:.0f}s total)",
          flush=True)
    if save_predictions and all_preds:
        pf = OUT / f"loco_screen{tag}_predictions.npz"
        flat = {}
        for arm_key, stored in all_preds.items():
            for arm, recs in stored.items():
                for j, rc in enumerate(recs):
                    for fld in ("pred", "true", "ctrl"):
                        flat[f"{arm_key}|{arm}|{j}|{fld}"] = rc[fld].astype(np.float32)
                    flat[f"{arm_key}|{arm}|{j}|meta"] = np.array(
                        [rc["unit"], rc["condition"], rc["dose"], rc["order"],
                         rc["pop_index"]], dtype=object)
        np.savez_compressed(pf, **flat)
        print(f"wrote {pf.relative_to(ROOT)} ({pf.stat().st_size/1e6:.1f} MB, "
              f"{len(flat)//4} prediction records) -- every score in the json is "
              "recomputable from these alone", flush=True)
    return out


def rollup(out: dict) -> dict:
    """Pool the per-combination folds, and pair code_dim 0 against 16 across records.

    The pooled LOCO number treats each HELD-OUT RECORD as one unit and each fold's records
    as distinct units (they are: a record held out in the CSF fold is a different
    observation from the same record held out in the CF fold, because the model differs).
    """
    r = {}
    for arm in ("ihcfm", "perturbed_mean", "no_change"):
        vals, groups, per_fold = [], [], {}
        for f in out["loco"]:
            if "arms" not in f or arm not in f["arms"]:
                continue
            rows = f["arms"][arm]["per_population"]
            per_fold[f["fold"]] = dict(
                normalised=f["arms"][arm]["summary"].get("ed_normalised", {}),
                absolute=f["arms"][arm]["summary"].get("ed_absolute", {}),
                denominator=f["arms"][arm]["summary"].get("ed_denominator", {}),
                cosine=f["arms"][arm]["summary"].get("cosine_delta", {}))
            for row in rows:
                if np.isfinite(row["ed_normalised"]):
                    vals.append(row["ed_normalised"])
                    groups.append(f"{f['fold']}:{row['unit']}")
        if vals:
            r[arm] = dict(per_fold=per_fold,
                          pooled_ed_normalised=M.grouped_bootstrap_ci(
                              vals, groups, n_boot=600, seed=0))
            r[arm]["pooled_ed_normalised"].pop("group_means", None)
            r[arm]["pooled_ed_normalised"].pop("groups", None)
    # pooled paired comparison across all LOCO folds
    for a, b in (("ihcfm", "perturbed_mean"), ("ihcfm", "no_change")):
        va, vb, gg = [], [], []
        for f in out["loco"]:
            if "arms" not in f or a not in f["arms"] or b not in f["arms"]:
                continue
            ma = {r["pop_index"]: r for r in f["arms"][a]["per_population"]}
            mb = {r["pop_index"]: r for r in f["arms"][b]["per_population"]}
            for pi in sorted(set(ma) & set(mb)):
                x, y = ma[pi], mb[pi]
                if np.isfinite(x["ed_normalised"]) and np.isfinite(y["ed_normalised"]):
                    va.append(x["ed_normalised"]); vb.append(y["ed_normalised"])
                    gg.append(f"{f['fold']}:{x['unit']}")
        if len(va) >= 2:
            r[f"paired_pooled_{a}_vs_{b}"] = M.paired_group_difference(
                va, vb, gg, n_boot=600, seed=0, name_a=a, name_b=b)
    # code_dim 0 vs 16 on the leave-replicate-out folds, PAIRED per record
    lro = out.get("leave_replicate_out", [])
    by = {}
    for e in lro:
        if "arms" not in e or "ihcfm" not in e["arms"]:
            continue
        by.setdefault(e["fold"], {})[int(e["code_dim"])] = e["arms"]["ihcfm"]
    va, vb, gg = [], [], []
    for fname, d in by.items():
        if 0 not in d or 16 not in d:
            continue
        m0 = {row["pop_index"]: row for row in d[0]["per_population"]}
        m16 = {row["pop_index"]: row for row in d[16]["per_population"]}
        for pi in sorted(set(m0) & set(m16)):
            x, y = m0[pi]["ed_normalised"], m16[pi]["ed_normalised"]
            if np.isfinite(x) and np.isfinite(y):
                va.append(x); vb.append(y); gg.append(f"{fname}:{m0[pi]['unit']}")
    if len(va) >= 2:
        r["paired_code0_vs_code16_leave_replicate_out"] = M.paired_group_difference(
            va, vb, gg, n_boot=600, seed=0, name_a="code_dim0", name_b="code_dim16")
        # and restricted to COMBINATION populations, where the claim actually lives
        va2, vb2, gg2 = [], [], []
        for fname, d in by.items():
            if 0 not in d or 16 not in d:
                continue
            m0 = {row["pop_index"]: row for row in d[0]["per_population"]}
            m16 = {row["pop_index"]: row for row in d[16]["per_population"]}
            for pi in sorted(set(m0) & set(m16)):
                if int(m0[pi].get("order", 1)) < 2:
                    continue
                x, y = m0[pi]["ed_normalised"], m16[pi]["ed_normalised"]
                if np.isfinite(x) and np.isfinite(y):
                    va2.append(x); vb2.append(y); gg2.append(f"{fname}:{m0[pi]['unit']}")
        if len(va2) >= 2:
            r["paired_code0_vs_code16_combinations_only"] = M.paired_group_difference(
                va2, vb2, gg2, n_boot=600, seed=0,
                name_a="code_dim0", name_b="code_dim16")
    return r


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=600)
    ap.add_argument("--rep-folds", type=int, default=3)
    ap.add_argument("--no-predictions", action="store_true")
    ap.add_argument("--r", type=int, default=R_DEFAULT,
                    help="interaction rank (tunable capacity, not a correctness bound)")
    ap.add_argument("--tune-r", action="store_true",
                    help="select r in R_CANDIDATES by held-out distributional score")
    a = ap.parse_args()
    main(smoke=a.smoke, seed=a.seed, n_boot=a.n_boot, rep_folds=a.rep_folds,
         save_predictions=not a.no_predictions, r=a.r, tune_r=a.tune_r)
