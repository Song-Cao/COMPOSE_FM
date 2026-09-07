"""Table 1: all NINE arms on IDENTICAL folds, identical protocol, identical coupling.

WHY THIS SCRIPT EXISTS
----------------------
IHC-FM and the seven baselines had never been scored on the same folds. `baselines.py`'s
`loco_fold_sweep` runs the seven baselines over every held-out pair; `loco_screen.py` runs
IHC-FM on the organoid data against two controls. Neither produces a head-to-head cell.
This script closes that gap: one benchmark instance, one fold list, one coupling, one step
budget, nine arms, three seeds.

THE POWER PROBLEM DRIVES THE DESIGN. The baselines track measured, pooled over the six
folds available at k = 4, that the three neural rows are inseparable: FactoredAdditive
leads DeepSets by +0.018 normalised ED (sem 0.255) and DeepSets leads Monolithic by +0.009
(sem 0.138), and the n required at that effect size is 270-450 folds. The ordering even
FLIPS between one fold (FactoredAdditive 0.647, apparently first by a wide margin) and the
pooled sweep (inseparable). Meanwhile the four controls are deterministic given the
benchmark -- measured across-seed std: NoChange and PerturbedMean bitwise 0.0, MatchingMean
1.11e-16, LinearResponse 1.20e-10 -- while every neural row carries std 0.27-0.37. A
single-fold table therefore reports which neural arm drew a lucky seed, not which model is
better, and it must not be the basis of any claim.

The cheapest available power is FOLDS, because a LOCO fold holds out one intervention pair
and the number of pairs is k(k-1)/2:

    k = 4   ->  6 folds  ->  12 held-out populations   (measured: cannot rank)
    k = 6   -> 15 folds  ->  30 held-out populations
    k = 7   -> 21 folds  ->  42 held-out populations   <- this script

`_run_stage` samples ONE population per optimisation step, so the training cost per fold is
set by the step budget and not by k: measured at k = 6, IHC-FM fits in 29.3 s, and raising k
adds populations to sample from without adding steps. Folds are therefore nearly free power,
and k = 7 x 3 seeds = 63 fold-seed fits x 9 arms is affordable on 8 CPU cores.

WHAT IS STILL NOT CLAIMED. 21 folds is a 3.5x improvement over 6, not the 270+ the measured
effect size demands. If IHC-FM full's margin over FactoredAdditiveCFM does not clear
`MARGIN_FOR_CLAIM` (0.25 normalised ED), the JSON and the LaTeX caption say in words that
the comparison is UNDERPOWERED and the two models are statistically indistinguishable. A
well-characterised negative is an acceptable outcome; an unseparated gap presented as a win
is not.

The within-model ablation (IHC-FM main-effects-only vs full) is better powered than any
cross-model row, because both arms share the fold, the seed, the initialisation seed, the
benchmark draw and the entire schedule, so fold difficulty and seed luck cancel in the
paired difference. It is reported separately and flagged as the stronger evidence.

METRIC POLICY (enforced here, learned from an audit that caught the original error)
----------------------------------------------------------------------------------
Ranking is on NORMALISED ENERGY DISTANCE, always reported with its no-change DENOMINATOR
and the absolute distance alongside, because the ratio hides the scale and the absolute is
not comparable across folds. CFM loss is recorded ONLY as a within-coupling training
diagnostic and is NEVER a ranking key: it tracks the identity-pair fraction a coupling
produces (paired 1.0, ot 0.795, sinkhorn 0.229, independent 0.0039), so it rewards a
coupling for making the regression target trivial rather than correct.

r IS A TUNABLE CAPACITY HYPERPARAMETER. The r >= k(k-1)/2 rule is retracted. r is re-tuned
at the new k on HELD-OUT SINGLETON conditions only, which is the project's documented
leak-free standard: the evaluation folds hold out PAIRS, and no pair score enters the
selection.

USAGE
    PYTHONPATH=src python experiments/table1.py            # full run
    PYTHONPATH=src python experiments/table1.py --smoke     # 3 folds, 1 seed, 60 steps
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time


import numpy as np

# Thread policy must be set BEFORE torch initialises its pools, and the environment
# variables must be set before the import so that spawned workers inherit them.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import torch

torch.set_num_threads(2)

if "src" not in sys.path:
    sys.path.insert(0, "src")

from composefm import baselines as B                                   # noqa: E402
from composefm import metrics as M                                     # noqa: E402
from composefm.ihcfm_geometry import (AmbientMetric, Decoder, HillDose,  # noqa: E402
                                      MetricRadialSat, PullbackField)
from composefm.ihcfm_hierarchy import (CFMObjective, ConditionalVelocity,  # noqa: E402
                                       integrate_velocity, train_hierarchical)
from composefm.synthetic import SyntheticSystem                        # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "results")
TABLES = os.path.join(RESULTS, "tables")

# ---- benchmark geometry ----------------------------------------------------------
D_LATENT = 6            # d; the benchmark observes the space it acts on (D = d)
K_PERT = 7              # 21 pairs -> 21 LOCO folds. See the power argument above.
R_TRUE = 2              # ground-truth interaction FACTOR rank of the benchmark
EPS_INT = 0.30          # calibrated interaction strength (~30% of the additive field)
NOISE = 0.02
N_CELLS = 200
TAUS = (0.6, 1.0)
SEEDS = (0, 1, 2)

# ---- shared protocol (identical for every arm) -----------------------------------
STEPS_IHCFM = (300, 300, 200)     # 800 total
STEPS_FLAT = 800                  # baselines spend the identical budget in one stage
BATCH = 128
LR = 3e-3
GRAD_CLIP = 5.0
N_INT_STEPS = 16
HIDDEN_IHCFM = 64
DEPTH = 2
MAX_COUPLING_N = 128

# ---- selected coupling: entropic UOT ("sinkhorn_uot" is the label, not a kwarg) ---
COUPLING = dict(coupling="sinkhorn", sinkhorn_eps=0.05, sinkhorn_iters=25,
                unbalanced_tau=1.0)

# ---- r tuning -------------------------------------------------------------------
R_CANDIDATES = (2, 8, 15)
R_TUNE_SEEDS = (0, 1)
R_TUNE_HELDOUT_SINGLETONS = 2     # held out for SELECTION only; folds hold out PAIRS

# ---- inference / reporting -------------------------------------------------------
N_BOOT = 2000
ALPHA = 0.05
# The margin below which the orchestrator requires an explicit "underpowered /
# indistinguishable" statement rather than a claim of a win.
MARGIN_FOR_CLAIM = 0.25

ARMS = ("no_change", "matching_mean", "linear", "displacement", "deepsets",
        "monolithic", "factored_additive", "ihcfm_main_only", "ihcfm_full")

ARM_LABELS = {
    "no_change": "NoChange",
    "matching_mean": "MatchingMean",
    "linear": "LinearResponse",
    "displacement": "PerturbedMean",
    "deepsets": "DeepSetsEndpoint",
    "monolithic": "MonolithicCFM",
    "factored_additive": "FactoredAdditiveCFM",
    "ihcfm_main_only": "IHC-FM (main effects only)",
    "ihcfm_full": "IHC-FM (full)",
}
BASELINE_OF = {
    "no_change": "NoChange", "matching_mean": "MatchingMean",
    "linear": "LinearResponse", "displacement": "PerturbedMean",
    "deepsets": "DeepSetsEndpoint", "monolithic": "MonolithicCFM",
    "factored_additive": "FactoredAdditiveCFM",
}
NEURAL_ARMS = ("deepsets", "monolithic", "factored_additive",
               "ihcfm_main_only", "ihcfm_full")
DETERMINISTIC_ARMS = ("no_change", "matching_mean", "linear", "displacement")


# ==================================================================================
# model construction
# ==================================================================================
class PullbackAdapter(torch.nn.Module):
    """Adapts `PullbackField(t, z, one_form=...)` to the `pullback(t, z, b)` contract.

    An nn.Module rather than a closure for a load-bearing reason: closures are invisible
    to nn.Module registration, so a PullbackField captured in one is NOT reached by
    `.to(dtype)`, and `to_inference_dtype(float64)` would leave the decoder single while
    the velocity net went double. Same rationale as `loco_screen.PullbackAdapter`.
    """

    def __init__(self, field: PullbackField):
        super().__init__()
        self.field = field

    def forward(self, t, z, b):
        return self.field(t, z, one_form=lambda tt, xx, cc: b)


def make_ihcfm(d: int = D_LATENT, k: int = K_PERT, r: int = 8, code_dim: int = 0,
               hidden: int = HIDDEN_IHCFM, depth: int = DEPTH, seed: int = 0,
               interaction_enabled: bool = True) -> ConditionalVelocity:
    """The FULL architecture: Hill dose, one-form hierarchy, pullback field, radial sat.

    Identical component wiring to `loco_screen.make_ihcfm` (D = d, decoder hidden 32,
    metric hidden 48 with w_min 0.25 / l_cap 1.5, metric-radial saturator wrapping the
    PULLBACK, HillDose). `interaction_enabled=False` is the documented ablation FLAG; it
    does not remove or resize anything, so the two IHC-FM arms are the same architecture
    with the same parameter allocation and the same initialisation.
    """
    torch.manual_seed(int(seed))
    dec = Decoder(int(d), int(d), hidden=32, s=0.5, seed=int(seed))
    met = AmbientMetric(int(d), hidden=48, w_min=0.25, l_cap=1.5)
    sat = MetricRadialSat(x_dim=int(d), latent_dim=int(d), hidden=32,
                          mode="metric_radial", kappa_init=2.0)
    pbf = PullbackField(dec, met, saturator=sat)
    dose = HillDose(int(k), amp_init=1.0, ec50_init=1.0, hill_init=1.0)
    return ConditionalVelocity(int(d), int(k), r=int(r), code_dim=int(code_dim),
                               hidden=int(hidden), depth=int(depth), dose=dose,
                               pullback=PullbackAdapter(pbf), decoder=dec,
                               interaction_enabled=bool(interaction_enabled))


def _lock_interaction_off(model: ConditionalVelocity) -> ConditionalVelocity:
    """Make the main-effects-only ablation hold across ALL THREE training stages.

    `train_hierarchical` calls `set_interaction_enabled(True)` at stage 2 by design, so
    constructing with `interaction_enabled=False` alone would be silently overridden. This
    shadows the method on the INSTANCE with a no-op, which leaves the architecture, the
    parameter allocation, the schedule and the step budget exactly as in the full arm and
    changes one thing only: the interaction branch is never activated, in training or at
    inference. Stages 2 and 3 still run, still see the combination conditions and still
    spend their steps -- the ablation removes the interaction PATHWAY, not the budget.
    """
    model.set_interaction_enabled(False)
    model.set_interaction_enabled = lambda flag: None     # type: ignore[assignment]
    return model


def _param_split(model: ConditionalVelocity) -> dict:
    """Total allocated parameters and the interaction-branch share of them.

    Both IHC-FM arms ALLOCATE the same tensors. They differ in what is ACTIVE: the
    main-effects-only arm never evaluates the interaction branch, so those parameters
    receive no gradient; the full arm trains all of them. The active count is therefore
    read off the model's ACTUAL interaction state rather than assumed -- crediting the
    full arm only total-minus-interaction would understate it by the whole branch.
    """
    total = sum(p.numel() for p in model.parameters())
    groups = model.parameter_groups()
    inter = sum(p.numel() for p in groups.get("interaction", []))
    enabled = bool(model.one_form.interaction_enabled)
    return dict(n_parameters_allocated=int(total),
                n_parameters_interaction_branch=int(inter),
                n_parameters_active=int(total if enabled else total - inter),
                interaction_branch_active=enabled)


def objective(seed: int = 0) -> CFMObjective:
    """The project's selected coupling. Every CFM arm gets exactly this object's settings."""
    return CFMObjective(seed=int(seed), max_coupling_n=MAX_COUPLING_N, **COUPLING)


# ==================================================================================
# benchmark instance (cached per worker process)
# ==================================================================================
_POPS_CACHE: dict = {}


def benchmark(k: int = K_PERT, n_cells: int = N_CELLS, draw_seed: int = 11):
    """The SINGLE benchmark instance every arm and every fold is scored on.

    The system structure is drawn from seed 0 and the populations from `draw_seed`, both
    fixed, so the fold list and the data are identical across arms, seeds and processes.
    Model seeds vary; the benchmark never does.
    """
    key = (int(k), int(n_cells), int(draw_seed))
    if key not in _POPS_CACHE:
        system = SyntheticSystem(d=D_LATENT, k=int(k), r=R_TRUE, eps=EPS_INT,
                                 noise=NOISE, seed=0)
        pops = system.sample_populations(n_cells=int(n_cells), n_replicates=1, taus=TAUS,
                                         n_pairs=None, n_triples=1, seed=int(draw_seed))
        _POPS_CACHE[key] = (system, pops)
    return _POPS_CACHE[key]


def fold_list(k: int = K_PERT) -> list[tuple]:
    """Every intervention PAIR is held out in turn: k(k-1)/2 folds."""
    return list(itertools.combinations(range(int(k)), 2))


# ==================================================================================
# one arm on one fold
# ==================================================================================
def _score_populations(pops, idx, predict, denominator_reference=None) -> list[dict]:
    """Normalised ED per held-out population, ALWAYS with absolute and denominator.

    `metrics.energy_report` is the single source of these three numbers; `max_n=None`
    disables subsampling (populations are 200 cells), so the values are deterministic
    given the prediction.
    """
    rows = []
    for i in idx:
        p = pops[i]
        pre = np.asarray(p.pre, dtype=np.float64)
        post = np.asarray(p.post, dtype=np.float64)
        er = M.energy_report(predict(p), post, pre, max_n=None, seed=0,
                             denominator_reference=denominator_reference)
        rows.append(dict(pop_index=int(i), label=str(p.label),
                         P=[int(q) for q in p.P], order=int(len(p.P)),
                         tau=float(p.tau),
                         ed_absolute=er["absolute"], ed_denominator=er["denominator"],
                         ed_normalised=er["normalised"],
                         ed_denominator_degenerate=bool(er["denominator_degenerate"]),
                         ed_denominator_vs_reference=er["denominator_vs_reference"]))
    return rows


def run_arm(arm: str, pops, system, train_idx, test_idx, seed: int, r: int,
            steps_ihcfm=STEPS_IHCFM, steps_flat=STEPS_FLAT,
            param_target: int | None = None,
            denominator_reference: float | None = None) -> dict:
    """Fit and score ONE arm on ONE fold. Same protocol object for every arm."""
    k = int(system.k)
    t_fit = time.time()
    extra: dict = {}

    if arm in ("ihcfm_full", "ihcfm_main_only"):
        model = make_ihcfm(d=D_LATENT, k=k, r=int(r), hidden=HIDDEN_IHCFM, depth=DEPTH,
                           seed=seed, interaction_enabled=(arm == "ihcfm_full"))
        if arm == "ihcfm_main_only":
            _lock_interaction_off(model)
        extra.update(_param_split(model))
        rep = train_hierarchical(model, pops, list(train_idx), objective=objective(seed),
                                 val_idx=None, steps=tuple(steps_ihcfm), batch=BATCH,
                                 lr=LR, lr_stage3=1e-3, anchor_weight=1.0,
                                 int_penalty=0.0, singleton_oversample=3.0, seed=seed)
        fit_s = time.time() - t_fit
        extra["skipped_nonfinite_total"] = int(rep.get("skipped_nonfinite_total", 0))
        extra["stage_losses"] = {s: rep[s].get("mean_loss_last_decile")
                                 for s in ("stage1", "stage2", "stage3")}
        extra["singleton_velocity_drift_rel"] = rep["stage3"].get(
            "singleton_velocity_drift_rel")
        extra["interaction_enabled_final"] = bool(model.one_form.interaction_enabled)
        t_pred = time.time()
        m64 = model.to_inference_dtype(torch.float64)
        rows = _score_populations(
            pops, test_idx,
            lambda p: integrate_velocity(m64, p.pre, p.P, p.tau, k,
                                         n_steps=N_INT_STEPS, mu0=p.pre),
            denominator_reference=denominator_reference)
        pred_s = time.time() - t_pred
        # CFM loss: within-coupling training diagnostic ONLY, never a ranking key.
        extra["cfm_loss_note"] = "within-coupling training diagnostic; NOT a ranking key"
        extra["n_parameters"] = extra["n_parameters_allocated"]
    else:
        name = BASELINE_OF[arm]
        cfg = B.TrainConfig(steps=int(steps_flat), batch=BATCH, lr=LR,
                            grad_clip=GRAD_CLIP, n_int_steps=N_INT_STEPS, seed=seed)
        model, match = B.build_baseline(
            name, D_LATENT, k, r=int(r), seed=seed, cfg=cfg,
            objective=objective(seed),
            param_target=(param_target if arm in ("deepsets", "monolithic",
                                                  "factored_additive") else None))
        rep = model.fit(pops, list(train_idx))
        fit_s = time.time() - t_fit
        extra["n_parameters"] = int(model.n_parameters())
        # `n_parameters()` counts TRAINABLE TORCH parameters, which is 0 for the four
        # closed-form controls -- but three of them do fit something, and reporting a
        # bare 0 would imply otherwise. LinearResponse in particular solves a ridge
        # coefficient matrix of (1 + 2k + d) x d = 126 entries. Count the fitted numpy
        # state so the table can distinguish "non-parametric" from "fitted in closed
        # form" rather than collapsing both to zero.
        fitted = {kk: vv for kk, vv in vars(model).items()
                  if isinstance(vv, np.ndarray)}
        extra["n_fitted_array_entries"] = int(sum(vv.size for vv in fitted.values()))
        extra["fitted_array_shapes"] = {kk: list(vv.shape)
                                        for kk, vv in fitted.items()}
        extra["parameter_matching"] = match
        extra["is_flow"] = bool(model.is_flow)
        extra["has_velocity"] = bool(model.has_velocity)
        if arm == "deepsets":
            # DOCUMENTED COUPLING EXCEPTION, carried forward from baselines.py. An
            # endpoint regressor has no integration step, so it applies the coupling's
            # barycentric projection directly as a map; under the shared entropic-UOT
            # coupling that inflates displacement 2.14x and it scores 1.923 on the
            # HELD-OUT combination and 4.99 on the TRAINING subset -- worse than no
            # change on both, and worse where it was fitted than where it generalised,
            # which is the signature of a wiring error rather than a hard problem.
            # Trained on the paired displacement -- which genuinely exists within a
            # population -- it scores 0.713 held-out and 1.06 on the training subset.
            # The exception is reported in the table footnote, not hidden.
            extra["coupling_exception"] = dict(
                supervision=rep.get("endpoint_supervision"),
                supervision_coupling=rep.get("supervision_coupling"),
                shared_objective_coupling=rep.get("shared_objective_coupling"),
                reason=rep.get("supervision_note"))
        t_pred = time.time()
        rows = _score_populations(pops, test_idx,
                                  lambda p: model.predict(p.pre, p.P, p.tau),
                                  denominator_reference=denominator_reference)
        pred_s = time.time() - t_pred

    return dict(arm=arm, seed=int(seed), r=int(r), fit_wall_clock_s=float(fit_s),
                predict_wall_clock_s=float(pred_s), per_population=rows, **extra)


# ==================================================================================
# r tuning -- on HELD-OUT SINGLETONS ONLY (no pair score enters the selection)
# ==================================================================================
def tune_r(k: int = K_PERT, candidates=R_CANDIDATES, seeds=R_TUNE_SEEDS,
           steps=STEPS_IHCFM, n_holdout: int = R_TUNE_HELDOUT_SINGLETONS,
           verbose: bool = True) -> dict:
    """Select r on held-out SINGLETON conditions -- the project's leak-free standard.

    The evaluation folds hold out PAIRS. Selecting r on singleton scores therefore uses no
    information from any evaluated fold's held-out condition. This mirrors the organoid
    screen, where r = 8 was selected on held-out singletons only.

    The retracted r >= k(k-1)/2 rule is NOT used to size r. It is reported for the record:
    the pair basis does saturate at min(r, k(k-1)/2), but measured recovery did not follow
    it (r = 2 beat r = 4 and r = 6; r = 8 measured best at k = 4), so r is chosen here on
    held-out distributional score and nothing else.
    """
    system, pops = benchmark(k=k)
    singles = sorted({tuple(p.P) for p in pops if len(p.P) == 1})
    held = singles[-int(n_holdout):]
    held_set = {tuple(sorted(h)) for h in held}
    train_idx = [i for i, p in enumerate(pops)
                 if tuple(sorted(p.P)) not in held_set]
    test_idx = [i for i, p in enumerate(pops) if tuple(sorted(p.P)) in held_set]
    n_train_singles = len({tuple(p.P) for i, p in enumerate(pops)
                           if i in set(train_idx) and len(p.P) == 1})
    out = {}
    for r in candidates:
        vals, secs = [], []
        for sd in seeds:
            res = run_arm("ihcfm_full", pops, system, train_idx, test_idx, seed=sd, r=r,
                          steps_ihcfm=steps)
            vals += [row["ed_normalised"] for row in res["per_population"]]
            secs.append(res["fit_wall_clock_s"])
        out[str(r)] = dict(r=int(r), mean_normalised_ed=float(np.mean(vals)),
                           # SD (population, ddof=0) over the scored populations -- a
                           # spread, NOT a standard error of the mean
                           sd_normalised_ed_population=float(np.std(vals)),
                           per_population=[float(v) for v in vals],
                           n_scored=len(vals), mean_fit_s=float(np.mean(secs)))
        if verbose:
            print(f"  r={r:>2}  held-out singleton normalised ED "
                  f"{out[str(r)]['mean_normalised_ed']:.4f}", flush=True)
    best = min(out, key=lambda s: out[s]["mean_normalised_ed"])
    return dict(selected_r=int(out[best]["r"]), candidates=list(candidates),
                per_candidate=out, seeds=list(seeds),
                selection_metric="mean normalised ED on HELD-OUT SINGLETON conditions",
                heldout_singletons=[list(h) for h in held],
                n_train_singleton_conditions=int(n_train_singles),
                n_heldout_populations=len(test_idx),
                retracted_rule=dict(
                    rule="r >= k(k-1)/2",
                    status="RETRACTED as a requirement; reported for the record only",
                    value_at_this_k=int(k * (k - 1) // 2),
                    rank_report=B.check_rank_sufficiency(int(k), int(out[best]["r"])),
                    note=("the pair basis does saturate at min(r, k(k-1)/2), but measured "
                          "recovery did not follow it (r=2 beat r=4 and r=6; r=8 best at "
                          "k=4), so r is selected on held-out score alone")),
                leakage_note=("folds hold out PAIRS; r is selected on SINGLETON scores, so "
                              "no evaluated fold's held-out condition informs r"))


# ==================================================================================
# the parallel sweep
# ==================================================================================
def _job(payload: tuple) -> dict:
    """One (fold, seed) cell: all nine arms on that exact split. Runs in a worker."""
    fold_i, held, seed, r, steps_ihcfm, steps_flat, k = payload
    torch.set_num_threads(2)
    system, pops = benchmark(k=k)
    train_idx, test_idx = system.loco_split(pops, [held])
    # held-out COMBINATIONS specifically -- the fold holds out one pair at both taus
    test_idx = [i for i in test_idx if len(pops[i].P) >= 2]
    # The reference denominator is the typical no-change ED on the TRAINING split, so the
    # collapse flag is a relative statement rather than an arbitrary absolute threshold.
    ref = float(np.median([
        M.energy_distance(np.asarray(pops[i].pre, dtype=np.float64),
                          np.asarray(pops[i].post, dtype=np.float64), max_n=None)
        for i in train_idx if len(pops[i].P) >= 2]))
    target = _param_split(make_ihcfm(d=D_LATENT, k=k, r=int(r), seed=seed))[
        "n_parameters_allocated"]
    arms = {}
    for arm in ARMS:
        arms[arm] = run_arm(arm, pops, system, train_idx, test_idx, seed=seed, r=int(r),
                            steps_ihcfm=steps_ihcfm, steps_flat=steps_flat,
                            param_target=target, denominator_reference=ref)
    return dict(fold_index=int(fold_i), held_out=[int(q) for q in held], seed=int(seed),
                r=int(r), n_train=len(train_idx), n_test=len(test_idx),
                train_denominator_reference=ref,
                parameter_matching_target=int(target), arms=arms)


def _jobs_for(k: int, seeds, r: int, folds, steps_ihcfm, steps_flat) -> list[tuple]:
    fl = fold_list(k) if folds is None else list(folds)
    return [(i, ho, sd, int(r), tuple(steps_ihcfm), int(steps_flat), int(k))
            for i, ho in enumerate(fl) for sd in seeds]


def sweep(k: int = K_PERT, seeds=SEEDS, r: int = 8, folds=None,
          steps_ihcfm=STEPS_IHCFM, steps_flat=STEPS_FLAT, workers: int = 4,
          verbose: bool = True, shard: int | None = None,
          n_shards: int = 1) -> list[dict]:
    """Every arm x every fold x every seed. Folds and seeds are the parallel axis.

    PARALLELISM IS BY SUBPROCESS SHARD, not by process pool. Two constraints force this:
    `ProcessPoolExecutor` is unavailable in this environment (its `_check_system_limits`
    reads SC_SEM_NSEMS_MAX, which raises PermissionError here), and fork-based pools are
    unsafe to inherit a torch process that has already run a multi-threaded fit. A shard
    is a FRESH interpreter given a disjoint slice of the (fold, seed) job list; each
    writes its own JSON and the parent merges. Deterministic either way: every cell's
    result depends only on (fold, seed, r) and the fixed benchmark, never on the
    scheduling order.
    """
    jobs = _jobs_for(k, seeds, r, folds, steps_ihcfm, steps_flat)
    if shard is not None:
        jobs = [j for i, j in enumerate(jobs) if i % int(n_shards) == int(shard)]
    cells: list[dict] = []
    t0 = time.time()
    for j in jobs:
        cells.append(_job(j))
        if verbose:
            print(f"  cell {len(cells)}/{len(jobs)} fold {cells[-1]['held_out']} "
                  f"seed {cells[-1]['seed']}  {time.time() - t0:.0f}s", flush=True)
    return cells


def sweep_sharded(k: int, seeds, r: int, folds, steps_ihcfm, steps_flat,
                  workers: int = 4, verbose: bool = True) -> list[dict]:
    """Fan the job list across `workers` fresh interpreters, then merge the shards."""
    import subprocess
    n_jobs = len(_jobs_for(k, seeds, r, folds, steps_ihcfm, steps_flat))
    n_sh = max(1, min(int(workers), n_jobs))
    if n_sh == 1:
        return sweep(k=k, seeds=seeds, r=r, folds=folds, steps_ihcfm=steps_ihcfm,
                     steps_flat=steps_flat, verbose=verbose)
    tmp = os.path.join(RESULTS, "logs")
    os.makedirs(tmp, exist_ok=True)
    env = dict(os.environ, PYTHONPATH="src", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    procs, paths = [], []
    smoke_flag = ["--smoke"] if folds is not None else []
    for s in range(n_sh):
        p = os.path.join(tmp, f"_table1_shard{s}_of{n_sh}.json")
        paths.append(p)
        cmd = [sys.executable, os.path.join("experiments", "table1.py"),
               "--shard", str(s), "--nshards", str(n_sh), "--r", str(int(r)),
               "--shard-out", p] + smoke_flag
        procs.append(subprocess.Popen(cmd, cwd=HERE, env=env,
                                      stdout=subprocess.DEVNULL,
                                      stderr=subprocess.PIPE))
    t0 = time.time()
    cells: list[dict] = []
    for s, (pr, p) in enumerate(zip(procs, paths)):
        _, err = pr.communicate()
        if pr.returncode != 0:
            raise RuntimeError(f"shard {s} failed ({pr.returncode}):\n"
                               f"{(err or b'').decode()[-3000:]}")
        with open(p) as fh:
            cells += json.load(fh)
        if verbose:
            print(f"  shard {s + 1}/{n_sh} merged ({len(cells)} cells, "
                  f"{time.time() - t0:.0f}s)", flush=True)
    # Deterministic ordering regardless of which shard finished first.
    cells.sort(key=lambda c: (c["fold_index"], c["seed"]))
    return cells


# ==================================================================================
# aggregation
# ==================================================================================
def _aligned(cells: list[dict]) -> dict:
    """Row-aligned per-arm vectors keyed by (fold, seed, population).

    Every arm is scored on the identical (fold, seed, held-out population) triple, so the
    vectors can be PAIRED row for row. The pairing is by explicit key, never by list order.
    """
    keys: list[tuple] = []
    for c in cells:
        for row in c["arms"][ARMS[0]]["per_population"]:
            keys.append((c["fold_index"], c["seed"], row["pop_index"]))
    keys = sorted(set(keys))
    index = {kk: j for j, kk in enumerate(keys)}
    out = {a: dict(norm=np.full(len(keys), np.nan), absolute=np.full(len(keys), np.nan),
                   denom=np.full(len(keys), np.nan)) for a in ARMS}
    for c in cells:
        for a in ARMS:
            for row in c["arms"][a]["per_population"]:
                j = index[(c["fold_index"], c["seed"], row["pop_index"])]
                out[a]["norm"][j] = row["ed_normalised"]
                out[a]["absolute"][j] = row["ed_absolute"]
                out[a]["denom"][j] = row["ed_denominator"]
    groups = np.array([f"fold{kk[0]}" for kk in keys])          # fold = independent unit
    return dict(keys=keys, groups=groups, per_arm=out,
                group_note=("the independent unit is the FOLD (one held-out intervention "
                            "pair); seeds and the two taus are nested within it, so the "
                            "bootstrap resamples folds and never rows"))


def _holm(pvals: dict) -> dict:
    """Holm-Bonferroni over a family of p-values. Reported alongside the raw values."""
    items = sorted(pvals.items(), key=lambda kv: (np.inf if not np.isfinite(kv[1])
                                                  else kv[1]))
    m = len(items)
    out, running = {}, 0.0
    for i, (name, p) in enumerate(items):
        adj = min(1.0, (m - i) * p) if np.isfinite(p) else float("nan")
        running = max(running, adj) if np.isfinite(adj) else running
        out[name] = dict(p_raw=float(p), p_holm=float(running),
                         reject_at_0_05=bool(np.isfinite(running) and running < 0.05))
    return out


def aggregate(cells: list[dict], n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Per-arm CIs, adjacent-pair separation, the letter display, and the key contrasts."""
    al = _aligned(cells)
    groups, per_arm = al["groups"], al["per_arm"]
    n_folds = int(np.unique(groups).size)

    rows = {}
    for a in ARMS:
        v = per_arm[a]["norm"]
        ci = M.grouped_bootstrap_ci(v, groups, n_boot=n_boot, seed=seed)
        ci.pop("group_means", None)
        ci.pop("groups", None)
        # across-seed spread at FIXED fold: the structural asymmetry the table must respect
        by_fold_seed: dict = {}
        for (fi, sd, _), x in zip(al["keys"], v):
            by_fold_seed.setdefault((fi, sd), []).append(x)
        per_fs = {}
        for (fi, sd), xs in by_fold_seed.items():
            per_fs.setdefault(fi, []).append(float(np.mean(xs)))
        seed_sd = [float(np.std(x, ddof=1)) for x in per_fs.values() if len(x) > 1]
        rows[a] = dict(
            label=ARM_LABELS[a],
            mean_normalised_ed=float(np.mean(v)),
            ci_lo=ci["lo"], ci_hi=ci["hi"], bootstrap_se=ci["se"],
            # SD and SEM are BOTH emitted, each named for what it is. They differ by
            # sqrt(n_folds) and conflating them inflates apparent precision by ~4.6x at
            # 21 folds -- the reason every dispersion field here carries its kind.
            sd_over_folds=float(np.std([np.mean(x) for x in per_fs.values()], ddof=1)),
            sem_over_folds=float(np.std([np.mean(x) for x in per_fs.values()], ddof=1)
                                 / math.sqrt(len(per_fs))),
            dispersion_note=("sd_over_folds is the SAMPLE SD (ddof=1) of the per-fold "
                             "means; sem_over_folds is that sd / sqrt(n_folds); "
                             "bootstrap_se is the grouped-bootstrap standard error of "
                             "the mean. All three are standard errors or deviations of "
                             "DIFFERENT things and are never interchangeable"),
            mean_absolute_ed=float(np.mean(per_arm[a]["absolute"])),
            mean_no_change_denominator=float(np.mean(per_arm[a]["denom"])),
            median_across_seed_sd_at_fixed_fold=(float(np.median(seed_sd)) if seed_sd
                                                 else 0.0),
            across_seed_sd_note=("SD (ddof=1) of the per-seed means within one fold, "
                                 "median over folds. A SD, not a sem."),
            n_folds=n_folds, n_scored_populations=int(v.size),
            ci_crosses_no_change=bool(ci["lo"] <= 1.0 <= ci["hi"]),
            worse_than_no_change=bool(float(np.mean(v)) > 1.0),
            n_parameters=int(np.median([c["arms"][a]["n_parameters"] for c in cells])),
            n_fitted_array_entries=int(np.median(
                [c["arms"][a].get("n_fitted_array_entries", 0) for c in cells])),
            parameter_count_note=("n_parameters counts trainable torch parameters; the "
                                  "closed-form controls have none but may still fit "
                                  "array state (n_fitted_array_entries), e.g. "
                                  "LinearResponse's ridge coefficient matrix"),
            mean_fit_wall_clock_s=float(np.mean([c["arms"][a]["fit_wall_clock_s"]
                                                 for c in cells])),
            mean_predict_wall_clock_s=float(np.mean([c["arms"][a]["predict_wall_clock_s"]
                                                     for c in cells])),
            deterministic_across_seeds=bool(a in DETERMINISTIC_ARMS),
        )
        if a in ("ihcfm_full", "ihcfm_main_only"):
            rows[a]["n_parameters_active"] = int(np.median(
                [c["arms"][a]["n_parameters_active"] for c in cells]))
            rows[a]["n_parameters_interaction_branch"] = int(np.median(
                [c["arms"][a]["n_parameters_interaction_branch"] for c in cells]))

    order = sorted(ARMS, key=lambda a: rows[a]["mean_normalised_ed"])

    def paired(a: str, b: str) -> dict:
        return M.paired_group_difference(per_arm[a]["norm"], per_arm[b]["norm"], groups,
                                         n_boot=n_boot, seed=seed,
                                         name_a=ARM_LABELS[a], name_b=ARM_LABELS[b])

    # ---- adjacent separation in the ranking ------------------------------------
    adjacent = []
    for a, b in zip(order[:-1], order[1:]):
        pd = paired(a, b)
        pd.pop("per_unit_diff", None)
        pd.pop("units", None)
        adjacent.append(dict(better=a, worse=b, better_label=ARM_LABELS[a],
                             worse_label=ARM_LABELS[b],
                             gap=float(rows[b]["mean_normalised_ed"]
                                       - rows[a]["mean_normalised_ed"]),
                             separated=bool(pd["excludes_zero"]), paired=pd))
    holm = _holm({f"{d['better']}<{d['worse']}": d["paired"]["sign_test_p"]
                  for d in adjacent})
    for d in adjacent:
        d["sign_test_holm"] = holm[f"{d['better']}<{d['worse']}"]

    # ---- compact letter display over ADJACENT pairs -----------------------------
    letters: dict[str, str] = {}
    cur = 0
    letters[order[0]] = "abcdefghi"[cur]
    for d in adjacent:
        if not d["separated"]:
            letters[d["worse"]] = letters[d["better"]]
        else:
            cur += 1
            letters[d["worse"]] = "abcdefghi"[cur]
    for a in ARMS:
        rows[a]["separation_group"] = letters[a]

    # ---- the contrasts the paper turns on ---------------------------------------
    contrasts = {}
    for tag, (a, b) in dict(
            ihcfm_full_vs_factored_additive=("ihcfm_full", "factored_additive"),
            ihcfm_full_vs_main_only=("ihcfm_full", "ihcfm_main_only"),
            ihcfm_full_vs_no_change=("ihcfm_full", "no_change"),
            ihcfm_full_vs_monolithic=("ihcfm_full", "monolithic"),
            ihcfm_full_vs_deepsets=("ihcfm_full", "deepsets"),
            ihcfm_main_only_vs_no_change=("ihcfm_main_only", "no_change"),
            factored_additive_vs_no_change=("factored_additive", "no_change"),
    ).items():
        pd = paired(a, b)
        pd.pop("units", None)
        contrasts[tag] = pd

    # ---- power / claim adjudication ---------------------------------------------
    hf = contrasts["ihcfm_full_vs_factored_additive"]
    margin = -float(hf["mean_diff"])        # positive = IHC-FM better (lower ED)
    separated = bool(hf["excludes_zero"])
    clears = bool(margin >= MARGIN_FOR_CLAIM)
    obs_sd = float(np.std(hf["per_unit_diff"], ddof=1))
    n_req = (int(math.ceil((2.8 * obs_sd / abs(margin)) ** 2)) if abs(margin) > 1e-12
             else None)
    if separated and clears:
        verdict = "separated"
        claim = (f"IHC-FM (full) beats FactoredAdditiveCFM by {margin:+.3f} normalised ED "
                 f"(paired CI excludes zero, margin clears the {MARGIN_FOR_CLAIM} "
                 f"threshold) over {n_folds} folds.")
    elif separated and not clears:
        verdict = "separated_but_below_margin"
        claim = (f"UNDERPOWERED FOR A HEADLINE CLAIM. The paired CI excludes zero, but the "
                 f"margin over FactoredAdditiveCFM is {margin:+.3f} normalised ED, below "
                 f"the {MARGIN_FOR_CLAIM} threshold the power analysis requires. Report as "
                 f"a small, precisely bounded difference, NOT as a win.")
    else:
        verdict = "indistinguishable"
        claim = (f"UNDERPOWERED. IHC-FM (full) and FactoredAdditiveCFM are STATISTICALLY "
                 f"INDISTINGUISHABLE on this benchmark at k={K_PERT}: paired difference "
                 f"{hf['mean_diff']:+.3f} normalised ED, 95% CI "
                 f"[{hf['lo']:+.3f}, {hf['hi']:+.3f}], which includes zero, over "
                 f"{n_folds} folds x {len(SEEDS)} seeds. The observed per-fold difference "
                 f"SD (ddof=1, not a sem) is {obs_sd:.3f}, so separating this effect "
                 f"would need about "
                 f"{n_req} folds. This gap is NOT a win and is not reported as one.")

    ab = contrasts["ihcfm_full_vs_main_only"]
    ablation = dict(
        paired=ab,
        interaction_pays_for_itself=bool(ab["excludes_zero"] and ab["mean_diff"] < 0),
        direction=("full better" if ab["mean_diff"] < 0 else "main-effects-only better"),
        power_note=("WITHIN-MODEL and therefore the better-powered comparison: both arms "
                    "share the fold, the seed, the initialisation, the benchmark draw and "
                    "the full three-stage schedule, so fold difficulty and seed luck "
                    "cancel in the paired difference. The cross-model rows do not enjoy "
                    "this cancellation."),
        architecture_note=("the ablation disables the interaction PATHWAY only: same "
                           "architecture, same parameter allocation, same step budget, "
                           "same stages. It is an ablation arm, never a simplification "
                           "of the main model."))

    return dict(rows=rows, ranking=order, adjacent_separation=adjacent,
                separation_letters=letters, contrasts=contrasts,
                headline=dict(verdict=verdict, statement=claim,
                              margin_normalised_ed=margin,
                              margin_threshold=MARGIN_FOR_CLAIM,
                              paired_ci=[hf["lo"], hf["hi"]],
                              n_folds=n_folds, n_seeds=len(SEEDS),
                              observed_per_fold_diff_sd=obs_sd,
                              folds_required_to_separate=n_req),
                ablation_main_only_vs_full=ablation,
                unit_definition=al["group_note"],
                letter_display_note=("letters are assigned by chaining ADJACENT paired "
                                     "tests: arms sharing a letter were not separated by "
                                     "the adjacent test. This is the standard compact "
                                     "display and is NOT a fully corrected all-pairs "
                                     "procedure; the pairwise tests are all in "
                                     "`contrasts` and `adjacent_separation`."),
                ranking_key="mean normalised energy distance over folds (lower is better)",
                cfm_loss_policy=("CFM loss is NOT reported as a ranking key anywhere in "
                                 "this table: it tracks the identity-pair fraction of the "
                                 "coupling, not predictive quality"))


# ==================================================================================
# oracle decomposition -- what is ACHIEVABLE, and what the interaction is WORTH
# ==================================================================================
def oracle_diagnosis(k: int = K_PERT, n_steps: int = 40) -> dict:
    """Bound the benchmark: the achievable floor, and the value of the interaction term.

    A null ablation has two very different explanations and this function separates them.

      ORACLE-FULL      integrate the TRUE field on the SAME pre cells a model sees.
                       This shares the target's sampling noise exactly as a model's
                       prediction does, so its nED is the achievable FLOOR on this
                       metric. (An independent redraw of the true distribution is NOT
                       the right comparison: it does not share the pre cells, so it
                       carries extra sampling noise no model has to pay.)
      ORACLE-ADDITIVE  integrate the true field with the INTERACTION TERM REMOVED.

    The gap ORACLE-ADDITIVE - ORACLE-FULL is a hard ceiling on what any interaction
    mechanism can win on these held-out populations. If that gap is ~0, a null ablation
    means the benchmark has no interaction signal to find and the ablation is
    uninformative. If the gap is large and the ablation is still null, the interaction
    branch is failing to capture available signal -- a real negative about the method,
    which is what must then be reported.

    Scored on ORDER-2 populations exactly, because that is what the folds score: a LOCO
    fold holds out one intervention PAIR at both taus, so every evaluated population has
    |P| = 2 (measured: 126 scored populations across 63 fold-seed cells, all order 2).
    Including the order-3 triple here would compare the oracle against a population no
    fold ever evaluates and would make the ceiling non-comparable to the ablation effect.
    """
    system, pops = benchmark(k=k)
    idx = [i for i, p in enumerate(pops) if len(p.P) == 2]
    full, add, ratio = [], [], []
    for i in idx:
        p = pops[i]
        pre = np.asarray(p.pre, dtype=np.float64)
        post = np.asarray(p.post, dtype=np.float64)
        tau = float(p.tau)
        er_f = M.energy_report(system.flow(p.P, pre, tau, n_steps=n_steps), post, pre,
                               max_n=None, seed=0)
        z = pre.copy()
        h = 1.0 / float(n_steps)
        for _ in range(n_steps):
            z = z + h * system.additive_field(p.P, z, tau)
        er_a = M.energy_report(z, post, pre, max_n=None, seed=0)
        v_a = system.additive_field(p.P, pre, tau)
        v_i = system.interaction_field(p.P, pre, tau)
        full.append(er_f["normalised"])
        add.append(er_a["normalised"])
        ratio.append(float(np.linalg.norm(v_i, axis=1).mean()
                           / max(1e-30, np.linalg.norm(v_a, axis=1).mean())))
    full, add = np.asarray(full), np.asarray(add)
    worth = float(add.mean() - full.mean())
    return dict(
        n_populations=len(idx),
        population_scope=("order-2 populations only, matching the folds exactly (a LOCO "
                          "fold holds out one intervention PAIR at both taus)"),
        oracle_full_normalised_ed=dict(mean=float(full.mean()),
                                       median=float(np.median(full)),
                                       min=float(full.min()), max=float(full.max())),
        oracle_additive_normalised_ed=dict(mean=float(add.mean()),
                                           median=float(np.median(add)),
                                           min=float(add.min()), max=float(add.max())),
        interaction_worth_normalised_ed=worth,
        interaction_over_additive_velocity_magnitude=float(np.mean(ratio)),
        definitions=dict(
            oracle_full=("true field integrated on the same pre cells; shares the "
                         "target's sampling noise exactly as a model prediction does, "
                         "so this is the achievable floor"),
            oracle_additive=("true field with the interaction term removed, same "
                             "integration"),
            interaction_worth=("oracle_additive - oracle_full: a hard CEILING on what "
                               "any interaction mechanism can win on these populations")),
        interpretation=("a null interaction ablation is only informative when "
                        "interaction_worth is materially above zero; compare the measured "
                        "ablation effect against this ceiling before concluding anything "
                        "about the mechanism"))


# ==================================================================================
# real-data columns (read back from the existing runs; nothing is re-derived)
# ==================================================================================
def real_data_columns() -> dict:
    """The organoid and finance cells, read from the runs that produced them.

    Only three arms were ever run on the real data (IHC-FM, PerturbedMean, NoChange), so
    those are the only cells that exist. The other six arms are reported as NOT EVALUATED
    rather than back-filled, imputed or left visually ambiguous.
    """
    out: dict = {"note": ("read back from results/loco_screen.json and "
                          "results/finance_blocked.json; NOT re-derived here"),
                 "arms_evaluated_on_real_data": ["ihcfm_full", "displacement",
                                                 "no_change"],
                 "arms_not_evaluated": [a for a in ARMS
                                        if a not in ("ihcfm_full", "displacement",
                                                     "no_change")]}
    loco_p = os.path.join(RESULTS, "loco_screen.json")
    if os.path.exists(loco_p):
        with open(loco_p) as fh:
            lo = json.load(fh)
        roll = lo.get("rollup", {})
        per_fold = []
        for f in lo.get("loco", []):
            arms = f.get("arms", {})
            cell = dict(fold=f.get("fold"), n_held=f.get("n_held"),
                        n_held_units=f.get("n_held_units"), r=f.get("r"))
            for tag, key in (("ihcfm_full", "ihcfm"), ("displacement", "perturbed_mean"),
                             ("no_change", "no_change")):
                s = arms.get(key, {}).get("summary", {}).get("ed_normalised")
                if s:
                    cell[tag] = dict(point=s["point"], lo=s["lo"], hi=s["hi"],
                                     n_units=s.get("n_groups"))
            per_fold.append(cell)
        out["organoid"] = dict(
            per_fold=per_fold,
            pooled=dict(
                ihcfm_full=roll.get("ihcfm", {}).get("pooled_ed_normalised"),
                displacement=roll.get("perturbed_mean", {}).get("pooled_ed_normalised"),
                no_change=roll.get("no_change", {}).get("pooled_ed_normalised")),
            paired_vs_displacement=roll.get(
                "paired_pooled_ihcfm_vs_perturbed_mean"),
            paired_vs_no_change=roll.get("paired_pooled_ihcfm_vs_no_change"),
            honest_negative=("IHC-FM beats the additive baseline over 136 paired "
                             "populations but LOSES to no-change on the mean, which is "
                             "dragged by a minority of large failures (sign test 72/136, "
                             "p=0.549). Both directions are reported."),
            config=lo.get("config"), data=lo.get("data"))
    fin_p = os.path.join(RESULTS, "finance_blocked.json")
    if os.path.exists(fin_p):
        with open(fin_p) as fh:
            fi = json.load(fh)
        splits = {}
        for sname, s in fi.get("results", {}).items():
            arms = s.get("arms", {})
            cell = dict(n_parameters=s.get("n_parameters"))
            for tag, key in (("ihcfm_full", "ihcfm"), ("displacement", "perturbed_mean"),
                             ("no_change", "no_change")):
                summ = arms.get(key, {}).get("summary", {}).get("ed_normalised")
                if summ:
                    cell[tag] = dict(point=summ["point"], lo=summ["lo"], hi=summ["hi"],
                                     n_units=summ.get("n_groups"))
            pr = s.get("paired", {}).get("ihcfm_vs_perturbed_mean::ed_normalised", {})
            cell["paired_vs_displacement"] = {kk: pr.get(kk) for kk in
                                              ("mean_diff", "lo", "hi", "excludes_zero",
                                               "n_units", "sign_test_p")}
            cell["shift_diagnosis"] = s.get("diagnosis", {})
            splits[sname] = cell
        out["finance"] = dict(
            splits=splits, config=fi.get("config"),
            framing=("OBSERVATIONAL order-flow conditioning, NOT causal intervention "
                     "estimation"),
            honest_negative=("the exposure-window split is a LOSS (+0.254 [+0.184,+0.332] "
                             "paired vs the additive baseline) and is diagnosed as "
                             "distribution shift: it carries the largest measured "
                             "pre-state shift of the four splits"))
    return out


# ==================================================================================
# LaTeX
# ==================================================================================
def _fmt(x, nd=3, dash="--"):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return dash
    return f"{x:.{nd}f}"


def latex_table(agg: dict, real: dict, meta: dict, orc: dict | None = None) -> str:
    """A real booktabs table, \\input-able, with the power verdict IN the caption."""
    rows, order = agg["rows"], agg["ranking"]
    hd = agg["headline"]
    n_folds, n_seeds = hd["n_folds"], hd["n_seeds"]
    org = real.get("organoid", {}).get("pooled", {})
    fin = real.get("finance", {}).get("splits", {})

    def org_cell(a):
        s = org.get(a)
        return _fmt(s["point"]) if isinstance(s, dict) and "point" in s else "--"

    def fin_cell(a):
        s = fin.get("chronological", {}).get(a)
        return _fmt(s["point"]) if isinstance(s, dict) and "point" in s else "--"

    L = []
    L.append("% Table 1 -- generated by experiments/table1.py. Do not edit by hand.")
    L.append("% Regenerate: PYTHONPATH=src python experiments/table1.py")
    L.append(r"\begin{table*}[t]")
    L.append(r"  \centering")
    L.append(r"  \small")
    L.append(r"  \setlength{\tabcolsep}{4pt}")
    L.append(r"  \begin{tabular}{llrrrrrr}")
    L.append(r"    \toprule")
    L.append(r"    & & \multicolumn{4}{c}{Synthetic composition benchmark} "
             r"& \multicolumn{2}{c}{Case studies} \\")
    L.append(r"    \cmidrule(lr){3-6}\cmidrule(lr){7-8}")
    L.append(r"    Model & Grp & Params & nED $[$95\% CI$]$ & abs.\ ED & denom. "
             r"& Organoid & Finance \\")
    L.append(r"    \midrule")
    for a in order:
        rr = rows[a]
        name = rr["label"].replace("&", r"\&")
        # Bold the row that actually MEASURED BEST, not the row the paper proposes. On
        # this benchmark that is the main-effects-only ablation, and bolding the full
        # model instead would assert a ranking the data does not support.
        if a == order[0]:
            name = r"\textbf{" + name + "}"
        star = r"$^{\dagger}$" if a == "deepsets" else ""
        if a in ("ihcfm_full", "ihcfm_main_only"):
            star += r"$^{\S}$"
        ned = (f"{rr['mean_normalised_ed']:.3f} "
               f"[{rr['ci_lo']:.3f}, {rr['ci_hi']:.3f}]")
        if rr["ci_crosses_no_change"]:
            ned += r"$^{\ddagger}$"
        # Parameters: ALLOCATED for every arm, with the ACTIVE count given too where the
        # two differ (the ablation allocates the interaction branch and never uses it).
        act = rr.get("n_parameters_active")
        if rr["n_parameters"] == 0:
            # closed-form arm: show its fitted array state, or an explicit dash when it
            # genuinely fits nothing (NoChange), never a bare "0" for both cases
            nfa = rr.get("n_fitted_array_entries", 0)
            pcell = (f"{nfa:,}$^{{c}}$" if nfa else "--")
        elif act in (None, rr["n_parameters"]):
            pcell = f"{rr['n_parameters']:,}"
        else:
            pcell = f"{rr['n_parameters']:,} ({act:,})"
        L.append(f"    {name}{star} & {rr['separation_group']} & "
                 f"{pcell} & {ned} & "
                 f"{_fmt(rr['mean_absolute_ed'])} & "
                 f"{_fmt(rr['mean_no_change_denominator'])} & "
                 f"{org_cell(a)} & {fin_cell(a)} \\\\")
    L.append(r"    \midrule")
    L.append(r"    \multicolumn{8}{l}{\emph{No-change reference: nED $=1.000$ by "
             r"construction; nED $>1$ is worse than predicting no change.}} \\")
    L.append(r"    \bottomrule")
    L.append(r"  \end{tabular}")

    ab = agg["ablation_main_only_vs_full"]["paired"]
    ab_sep = "excludes" if ab["excludes_zero"] else "includes"
    cap = []
    cap.append(
        rf"\textbf{{Nine arms on identical folds.}} All arms are fitted and scored on the "
        rf"SAME {n_folds} leave-one-combination-out folds of one synthetic benchmark "
        rf"instance ($d={D_LATENT}$, $k={K_PERT}$, ground-truth interaction rank "
        rf"{R_TRUE}), each fold repeated over {n_seeds} seeds, under one shared protocol "
        rf"({sum(STEPS_IHCFM)} optimisation steps, batch {BATCH}, lr {LR}, "
        rf"{N_INT_STEPS} integration steps, Adam) and one shared coupling "
        rf"(entropic unbalanced Sinkhorn, $\varepsilon=0.05$, 25 iterations). ")
    cap.append(
        r"nED is normalised energy distance, ED(pred, post)/ED(pre, post), so 1.0 is the "
        r"no-change predictor; the absolute distance and the denominator are given "
        r"alongside because the ratio hides the scale. Lower is better. ")
    cap.append(
        r"\textbf{Grp} is a compact letter display over adjacent paired tests: arms "
        r"sharing a letter were NOT statistically separated and must not be read as "
        r"ranked. ")
    cap.append(
        rf"$^{{\ddagger}}$ marks arms whose 95\% CI crosses the no-change line. ")
    cap.append(
        r"$^{\dagger}$ DeepSetsEndpoint is trained on the paired displacement rather than "
        r"the shared coupling: it has no integration step, so it would apply the "
        r"coupling's barycentric projection directly as a map (measured 2.14$\times$ "
        r"displacement inflation, scoring worse than no change on its own training "
        r"conditions). The exception is documented rather than hidden. ")
    cap.append(
        rf"$^{{\S}}$ The two IHC-FM rows are the SAME architecture with the same "
        rf"parameter allocation, the same initialisation and the same three-stage "
        rf"schedule; the ablation disables the interaction pathway only. Both therefore "
        rf"allocate {rows['ihcfm_full']['n_parameters']:,} parameters, but the ablation "
        rf"trains only the "
        rf"{rows['ihcfm_main_only'].get('n_parameters_active', 0):,} outside the "
        rf"interaction branch (shown in parentheses), while the full model trains all "
        rf"{rows['ihcfm_full'].get('n_parameters_active', 0):,}. ")
    cap.append(
        rf"\textbf{{Power.}} {hd['statement']} ")
    cap.append(
        rf"\textbf{{The interaction hierarchy does not pay for itself on this "
        rf"benchmark.}} The within-model ablation is the better-powered comparison, "
        rf"because both arms share fold, seed, initialisation and schedule so fold "
        rf"difficulty and seed luck cancel: the paired difference is "
        rf"{ab['mean_diff']:+.3f} nED, 95\% CI [{ab['lo']:+.3f}, {ab['hi']:+.3f}], "
        rf"which {ab_sep} zero ({ab['n_units_a_lower']}/{ab['n_units']} folds favour "
        rf"the full model, sign test $p={ab['sign_test_p']:.3g}$). The full model is "
        rf"nominally {abs(ab['mean_diff']):.3f} nED WORSE, and the effect is null rather "
        rf"than adverse. ")
    if orc:
        cap.append(
            rf"This null is not a floor effect: the achievable floor on these "
            rf"populations (true field integrated on the same pre cells) is "
            rf"{orc['oracle_full_normalised_ed']['mean']:.3f} nED and removing the true "
            rf"interaction term from that oracle costs "
            rf"{orc['interaction_worth_normalised_ed']:.3f} nED, so there IS interaction "
            rf"signal to capture and roughly "
            rf"{orc['interaction_over_additive_velocity_magnitude']:.2f} of the additive "
            rf"velocity magnitude is interaction. Both IHC-FM arms sit at "
            rf"{rows['ihcfm_full']['mean_normalised_ed']:.2f}, still above the "
            rf"interaction-free oracle at "
            rf"{orc['oracle_additive_normalised_ed']['mean']:.2f}, so the models have not "
            rf"yet reached the additive ceiling and the interaction branch has no "
            rf"headroom to demonstrate value here. We report this as an open negative "
            rf"rather than adjusting the architecture. ")
    cap.append(
        r"Case-study columns are pooled normalised ED read back from the organoid LOCO "
        r"screen and the chronological purged finance split; only three arms were run on "
        r"real data, and the remaining cells are marked \texttt{--} (not evaluated) "
        r"rather than back-filled. ")
    cap.append(
        rf"Parameter counts are matched to IHC-FM's allocation "
        rf"({meta.get('parameter_matching_target', 0):,}) by solving each neural "
        rf"baseline's width. $^{{c}}$ marks a closed-form arm with no trainable "
        rf"parameters: the figure given is its fitted array state (LinearResponse solves "
        rf"a $(1+2k+d)\times d$ ridge coefficient matrix; MatchingMean and PerturbedMean "
        rf"fit a mean displacement vector), and \texttt{{--}} means the arm fits nothing "
        rf"at all. ")
    L.append(r"  \caption{" + "".join(cap).rstrip() + "}")
    L.append(r"  \label{tab:main}")
    L.append(r"\end{table*}")
    return "\n".join(L) + "\n"


# ==================================================================================
# main
# ==================================================================================
def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o


def load_shards(k: int, n_shards: int) -> list[dict]:
    """Re-read the per-shard cell dumps from a completed sweep.

    The sweep is the expensive part (~1 hour at k=7). Aggregation, the oracle diagnosis
    and the LaTeX are cheap, so `--from-shards` rebuilds the table and JSON from the
    shard files a previous run left in results/logs/ without refitting a single model.
    The merged list is sorted by (fold, seed) so the result does not depend on which
    shard finished first.
    """
    cells: list[dict] = []
    for s in range(int(n_shards)):
        p = os.path.join(RESULTS, "logs", f"_table1_shard{s}_of{int(n_shards)}.json")
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        with open(p) as fh:
            cells += json.load(fh)
    cells.sort(key=lambda c: (c["fold_index"], c["seed"]))
    n_exp = len(fold_list(k)) * len(SEEDS)
    if len(cells) != n_exp:
        raise ValueError(f"expected {n_exp} cells at k={k}, found {len(cells)}")

    # Shard dumps written before `_param_split` consulted the model's actual interaction
    # state credited EVERY IHC-FM arm with total-minus-interaction, which understated the
    # FULL arm by its whole interaction branch (that branch is enabled and trained). The
    # correction needs no refit: the allocated and interaction-branch counts are already
    # in the dump, and which arm has the branch active is fixed by the arm name.
    for c in cells:
        for a, on in (("ihcfm_full", True), ("ihcfm_main_only", False)):
            arm = c["arms"].get(a)
            if arm is None or "n_parameters_allocated" not in arm:
                continue
            tot = int(arm["n_parameters_allocated"])
            inter = int(arm["n_parameters_interaction_branch"])
            want = tot if on else tot - inter
            if arm.get("n_parameters_active") != want:
                arm["n_parameters_active"] = want
                arm["active_count_corrected_on_load"] = True
            arm["interaction_branch_active"] = on

    # Shard dumps written before `n_fitted_array_entries` existed would otherwise
    # aggregate to 0 for every closed-form arm, which the table would render as "fits
    # nothing" -- wrong for LinearResponse. Refit the controls once (milliseconds each,
    # closed form, no optimisation) and backfill rather than reporting a false zero.
    missing = [a for a in DETERMINISTIC_ARMS
               if any("n_fitted_array_entries" not in c["arms"][a] for c in cells)]
    if missing:
        system, pops = benchmark(k=k)
        r = int(cells[0]["r"])
        train_idx, _ = system.loco_split(pops, [tuple(cells[0]["held_out"])])
        cfg = B.TrainConfig(steps=STEPS_FLAT, batch=BATCH, lr=LR, grad_clip=GRAD_CLIP,
                            n_int_steps=N_INT_STEPS, seed=0)
        for a in missing:
            mdl, _ = B.build_baseline(BASELINE_OF[a], D_LATENT, k, r=r, seed=0, cfg=cfg,
                                      objective=objective(0))
            mdl.fit(pops, list(train_idx))
            fitted = {kk: vv for kk, vv in vars(mdl).items()
                      if isinstance(vv, np.ndarray)}
            n = int(sum(vv.size for vv in fitted.values()))
            shapes = {kk: list(vv.shape) for kk, vv in fitted.items()}
            for c in cells:
                c["arms"][a].setdefault("n_fitted_array_entries", n)
                c["arms"][a].setdefault("fitted_array_shapes", shapes)
                c["arms"][a]["fitted_count_backfilled"] = True
        print(f"    backfilled fitted-array counts for {missing} "
              f"(shards predate the field)", flush=True)
    return cells


def main(smoke: bool = False, workers: int = 4, seed: int = 0,
         r_override: int | None = None, from_shards: int | None = None) -> dict:
    t0 = time.time()
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(TABLES, exist_ok=True)
    k = 4 if smoke else K_PERT
    seeds = (0,) if smoke else SEEDS
    steps_i = (20, 20, 20) if smoke else STEPS_IHCFM
    steps_f = 60 if smoke else STEPS_FLAT
    folds = fold_list(k)[:3] if smoke else None
    n_boot = 200 if smoke else N_BOOT

    print(f"=== table1: k={k}, {len(fold_list(k)) if folds is None else len(folds)} folds"
          f" x {len(seeds)} seeds, 9 arms ===", flush=True)

    if from_shards is not None:
        cells = load_shards(k, from_shards)
        r = int(cells[0]["r"])
        rtune = dict(selected_r=r,
                     note=(f"rebuilt from {from_shards} shard dumps of a completed "
                           f"sweep; r read back from the cells, not re-tuned"))
        print(f"--- rebuilt {len(cells)} cells from shards (r={r}) ---", flush=True)
    elif r_override is not None:
        rtune = dict(selected_r=int(r_override), note="r forced via --r; no tuning run")
    else:
        print("--- tuning r on HELD-OUT SINGLETONS (leak-free) ---", flush=True)
        rtune = tune_r(k=k, steps=steps_i,
                       seeds=(0,) if smoke else R_TUNE_SEEDS,
                       candidates=(2, 8) if smoke else R_CANDIDATES)
        print(f"    selected r = {rtune['selected_r']}", flush=True)
    r = int(rtune["selected_r"])

    if from_shards is None:
        print(f"--- sweep (r={r}, {workers} shards) ---", flush=True)
        cells = sweep_sharded(k=k, seeds=seeds, r=r, folds=folds, steps_ihcfm=steps_i,
                              steps_flat=steps_f, workers=workers)
    agg = aggregate(cells, n_boot=n_boot, seed=seed)
    real = real_data_columns()
    print("--- oracle decomposition (achievable floor, interaction ceiling) ---",
          flush=True)
    orc = oracle_diagnosis(k=k)
    print(f"    floor {orc['oracle_full_normalised_ed']['mean']:.4f} | "
          f"interaction-free oracle "
          f"{orc['oracle_additive_normalised_ed']['mean']:.4f} | interaction worth "
          f"{orc['interaction_worth_normalised_ed']:.4f} nED", flush=True)

    meta = dict(
        k=k, d=D_LATENT, r_selected=r, r_tuning=rtune, seeds=list(seeds),
        n_folds=len(cells) // len(seeds), n_cells=len(cells),
        interaction_rank_true=R_TRUE, interaction_strength_eps=EPS_INT, noise=NOISE,
        n_cells_per_population=N_CELLS, taus=list(TAUS),
        protocol=dict(steps_ihcfm=list(steps_i), steps_flat=steps_f, batch=BATCH, lr=LR,
                      grad_clip=GRAD_CLIP, n_int_steps=N_INT_STEPS,
                      hidden_ihcfm=HIDDEN_IHCFM, depth=DEPTH, optimiser="Adam",
                      note=("baselines spend the identical TOTAL budget in one stage; "
                            "IHC-FM stages it 300/300/200 because it has a hierarchy to "
                            "stage")),
        coupling=dict(**COUPLING, max_coupling_n=MAX_COUPLING_N,
                      label="sinkhorn_uot (a project LABEL, not a coupling= string)"),
        parameter_matching_target=int(np.median([c["parameter_matching_target"]
                                                 for c in cells])),
        parameter_matching_note=("target is IHC-FM's ACTUAL allocated parameter count at "
                                 "this (d, k, r, hidden), counting decoder, ambient "
                                 "metric, saturator, dose and hierarchy; each neural "
                                 "baseline's width is solved to it"),
        smoke=bool(smoke), workers=int(workers),
        power_design=("k raised from 4 to 7 to buy folds: a LOCO fold holds out one "
                      "intervention pair, so folds = k(k-1)/2 = 21 here against 6 at "
                      "k=4. _run_stage samples one population per step, so training cost "
                      "per fold does not grow with k. This is a 3.5x power improvement, "
                      "NOT the 270-450 folds the measured neural-row effect size needs."),
        seed_asymmetry=("the four controls are deterministic given the benchmark (measured "
                        "across-seed std: NoChange/PerturbedMean bitwise 0.0, MatchingMean "
                        "1.11e-16, LinearResponse 1.20e-10) while every neural arm carries "
                        "0.27-0.37; a single-fold table flatters whichever neural arm drew "
                        "well, which is why no cell here is a single fold"),
        metric_policy=("rank and select on normalised ED with its denominator reported; "
                       "CFM loss is NEVER a ranking key (it tracks the coupling's "
                       "identity-pair fraction: paired 1.0, ot 0.795, sinkhorn 0.229, "
                       "independent 0.0039)"),
        wall_clock_total_s=None)

    # The ablation verdict, adjudicated against the measured interaction CEILING rather
    # than against zero alone. A null is only evidence about the mechanism when there was
    # signal available to capture.
    abl = agg["ablation_main_only_vs_full"]
    apd = abl["paired"]
    worth = orc["interaction_worth_normalised_ed"]
    abl["oracle_context"] = dict(
        interaction_worth_normalised_ed=worth,
        measured_effect_normalised_ed=apd["mean_diff"],
        fraction_of_available_signal_captured=(float(-apd["mean_diff"] / worth)
                                               if abs(worth) > 1e-12 else None),
        both_arms_above_interaction_free_oracle=bool(
            agg["rows"]["ihcfm_full"]["mean_normalised_ed"]
            > orc["oracle_additive_normalised_ed"]["mean"]),
        verdict=(
            "OPEN NEGATIVE: the benchmark contains interaction signal worth "
            f"{worth:.3f} nED, but the interaction branch captures none of it "
            f"(paired effect {apd['mean_diff']:+.3f} nED, CI "
            f"[{apd['lo']:+.3f}, {apd['hi']:+.3f}], includes zero). Both IHC-FM arms "
            f"({agg['rows']['ihcfm_full']['mean_normalised_ed']:.3f} and "
            f"{agg['rows']['ihcfm_main_only']['mean_normalised_ed']:.3f}) remain ABOVE "
            f"the interaction-free oracle "
            f"({orc['oracle_additive_normalised_ed']['mean']:.3f}), i.e. neither has "
            "reached the ceiling that main effects alone could deliver, so the "
            "interaction hierarchy has no headroom in which to show value on this "
            "benchmark. This is diagnosed, not fixed: the architecture is unchanged and "
            "the negative is reported."
            if not apd["excludes_zero"] else
            "the ablation separated; see `paired` for the direction and magnitude"))

    out = dict(meta=meta, table=agg, oracle=orc, real_data=real, cells=cells)
    meta["wall_clock_total_s"] = time.time() - t0

    suffix = "_smoke" if smoke else ""
    jp = os.path.join(RESULTS, f"table1{suffix}.json")
    with open(jp, "w") as fh:
        json.dump(_jsonable(out), fh, indent=1)
    tp = os.path.join(TABLES, f"table1{suffix}.tex")
    with open(tp, "w") as fh:
        fh.write(latex_table(agg, real, meta, orc=orc))

    print(f"\n--- ranking ({meta['n_folds']} folds x {len(seeds)} seeds) ---", flush=True)
    for a in agg["ranking"]:
        rr = agg["rows"][a]
        flag = "  <- CI crosses no-change" if rr["ci_crosses_no_change"] else ""
        print(f"  {rr['separation_group']}  {rr['label']:<28} "
              f"{rr['mean_normalised_ed']:.4f} "
              f"[{rr['ci_lo']:.4f}, {rr['ci_hi']:.4f}]  "
              f"p={rr['n_parameters']:>7,}  {rr['mean_fit_wall_clock_s']:.1f}s{flag}",
              flush=True)
    print(f"\nVERDICT: {agg['headline']['verdict']}\n{agg['headline']['statement']}",
          flush=True)
    print(f"\nwrote {jp}\nwrote {tp}\ntotal {meta['wall_clock_total_s']:.0f}s", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--r", type=int, default=None)
    # shard mode: run a disjoint slice of the job list and write ONLY the raw cells
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--shard-out", type=str, default=None)
    # rebuild the table/JSON from a completed sweep's shard dumps -- no refitting
    ap.add_argument("--from-shards", type=int, default=None, dest="from_shards")
    a = ap.parse_args()
    if a.shard is not None:
        _k = 4 if a.smoke else K_PERT
        _seeds = (0,) if a.smoke else SEEDS
        _si = (20, 20, 20) if a.smoke else STEPS_IHCFM
        _sf = 60 if a.smoke else STEPS_FLAT
        _folds = fold_list(_k)[:3] if a.smoke else None
        _cells = sweep(k=_k, seeds=_seeds, r=int(a.r), folds=_folds, steps_ihcfm=_si,
                       steps_flat=_sf, verbose=False, shard=a.shard,
                       n_shards=a.nshards)
        with open(a.shard_out, "w") as _fh:
            json.dump(_jsonable(_cells), _fh)
    else:
        main(smoke=a.smoke, workers=a.workers, r_override=a.r,
             from_shards=a.from_shards)
