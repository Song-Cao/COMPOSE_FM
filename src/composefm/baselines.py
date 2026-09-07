"""The comparison ladder for IHC-FM: seven baselines behind ONE interface.

Everything in this module answers the same question that Table 1 of the paper asks -- given
the pre-population of a HELD-OUT intervention set, predict its post-population -- and every
model answers it through the same three calls, so the table is apples-to-apples:

    fit(populations, train_idx) -> dict      report; the model is fitted in place
    predict(Z0, P, tau)         -> (n, d)    the predicted post-population
    velocity(Z, P, tau, t)      -> (n, d)    the instantaneous field, WHERE MEANINGFUL
    n_parameters()              -> int       trainable scalars

`velocity` is not universal and the interface says so rather than faking one. A
displacement model has no velocity field; asking for it raises `NoVelocityField`. Where a
constant-displacement reading IS defensible (the shift baselines transport along a straight
line in unit time, so u = x1 - x0 is exactly the I-CFM conditional velocity of their own
prediction) the method is provided and `has_velocity` is True, but `is_flow` stays False:
these models never integrate an ODE, they only admit a straight-line reading of the map
they already committed to. Table 1 must not present them as flow models.

WHY THESE SEVEN
---------------
Two groups, answering two different objections.

SIMPLE / STRONG CONTROLS (1-4). Recent perturbation-prediction benchmarks have repeatedly
found that trivial predictors -- "nothing changed", "everything moves by the training mean"
-- match or beat published deep models once the metric is one the mean cannot game. They
are therefore not decoration; they are the floor a method must clear before any of its
structure is worth discussing. `NoChange` in particular calibrates the metric: normalised
energy distance is defined so that NoChange scores ~1.0 by construction, which turns every
other number in the table into "fraction of the null gap remaining".

CAPACITY-MATCHED NEURAL BASELINES (5-7). These decide the paper's actual claim. IHC-FM adds
a metric pullback and a moment hierarchy on top of a conditional velocity field; the
alternative explanation for any win is simply "it has more nonlinear set capacity". So the
neural baselines are PARAMETER-MATCHED to IHC-FM (widths solved for, not guessed), trained
with the SAME CFMObjective settings, the SAME optimiser, step count, batch size and
finite-step guards. What differs is only the structure:

ONE MEASURED EXCEPTION to "the same coupling", stated here so the header does not overclaim.
The two CFM rows share IHC-FM's validated entropic-UOT coupling exactly. `DeepSetsEndpoint`
does NOT: it draws index-aligned (paired) supervision, because it is not a flow model and
has no integration step to absorb a deliberately-wrong coupling. Under the shared UOT
coupling it scored 2.41 normalised on the held-out combination and 4.33 on the training
subset -- worse than predicting no change on both -- for a diagnosed reason written up in
that class's docstring. Both arms are fitted and both numbers are reported in
`deepsets_supervision_diagnosis`; the shared-coupling requirement carries a "where
applicable" clause and that row is where it does not apply.

    MonolithicCFM        one time-conditioned velocity net on the raw (indicator, exposure)
                         vector. No hierarchy, no factorisation, no geometry.
    FactoredAdditiveCFM  SUM of per-intervention velocity fields, each gated by its own
                         exposure. No interaction branch, no metric pullback. This is the
                         closest published-style competitor, so it gets the same dose
                         module and embedding table IHC-FM gets -- it is meant to be
                         strong, and if it wins that is the result.
    DeepSetsEndpoint     permutation-invariant DeepSets over the set, predicting the
                         endpoint displacement directly. NOT a flow model at all: it
                         isolates "does the flow formulation buy anything over regressing
                         the endpoint with the same set inductive bias".

PARAMETER MATCHING IS THE POINT, AND IT IS A CHOICE THAT MUST BE STATED. IHC-FM at
(d=6, k=4, r=8, hidden=64) has two counts one could match: the whole model including the
decoder, ambient metric and saturator (`reference_ihcfm_parameters()['total']`), or just the
velocity core (`['core']`). The neural baselines have no geometry, so matching the CORE
would hand IHC-FM its geometry parameters for free. This module matches the TOTAL, which
gives every neural baseline slightly MORE capacity than IHC-FM's velocity net -- the
conservative direction, since a baseline that then loses cannot be dismissed as undersized.
Both counts are reported.

r IS A TUNABLE CAPACITY HYPERPARAMETER, NOT A CORRECTNESS REQUIREMENT. An earlier version
of this module ENFORCED r >= k(k-1)/2 and refused to run below it. That requirement has
been retracted upstream and the enforcement is gone. The rank arithmetic behind it is still
correct -- the pair basis of the second moment saturates at min(r, k(k-1)/2), so at k=4
(6 pairs) r=2 gives rank 2, r=4 gives 4, r>=6 gives all 6 -- but under the correct coupling
the PERFORMANCE trend does not follow the rank: measured interaction recovery went r=2
+0.5930, r=4 +0.4414, r=6 +0.5121, r=8 +0.6427, so r=2 beats both r=4 and r=6 and only r=8
is clearly best. Aliasing distinct pairs onto a shared direction evidently costs less than
the extra parameters cost, at least at this k.

So: the default is r=8 at k=4 BECAUSE IT MEASURED BEST, not because a formula implies it,
and at larger k r should be picked on a held-out distributional score rather than scaled up
by k(k-1)/2. `check_rank_sufficiency` survives as a REPORTING helper -- it still computes
the saturation rank, which is a real and useful fact about the parameterisation -- but it no
longer gates anything and `sufficient` is not a pass criterion.

WHAT THIS MODULE DELIBERATELY DOES NOT DO
-----------------------------------------
No geometry. No pullback. No moment hierarchy. Not one of these seven may import
`ihcfm_geometry`; the only thing taken from `ihcfm_hierarchy` is the shared OBJECTIVE
machinery (`CFMObjective`, the batch drawer, the record packer), because sharing it is
exactly what makes the comparison fair. If a baseline needed the method's own components to
be competitive it would not be a baseline.
"""

from __future__ import annotations

import copy
import json
import math
import pathlib
import time
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(2)

# Shared objective machinery. Taken from the method's module ON PURPOSE: the coupling, the
# interpolant, the batch drawer and the record packer must be bit-identical to what IHC-FM
# trains against, or Table 1 is comparing objectives rather than models.
from composefm.ihcfm_hierarchy import (
    CFMObjective,
    _draw_batch,
    _pop_arrays,
    _time_features,
    N_TIME_FEATURES,
)

__all__ = [
    "NoVelocityField",
    "Baseline",
    "NoChange",
    "MatchingMean",
    "PerturbedMean",
    "LinearResponse",
    "DeepSetsEndpoint",
    "MonolithicCFM",
    "FactoredAdditiveCFM",
    "BASELINES",
    "build_baseline",
    "reference_ihcfm_parameters",
    "check_rank_sufficiency",
    "energy_distance",
    "normalised_energy_distance",
    "cfm_loss_of",
    "evaluate_baseline",
    "default_objective",
    "TrainConfig",
    "selftest",
    "main",
]


class NoVelocityField(Exception):
    """Raised by `velocity` on models that do not define one. Not an error to work around."""


# ==================================================================================
# shared scoring
# ==================================================================================
def energy_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Two-sample energy distance. Identical to the method module's, kept local so the
    baselines can be scored without importing a private symbol."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    def pd(A, B):
        return np.sqrt(np.maximum(
            np.sum(A ** 2, 1)[:, None] + np.sum(B ** 2, 1)[None, :] - 2 * A @ B.T, 0.0))

    return float(2 * pd(X, Y).mean() - pd(X, X).mean() - pd(Y, Y).mean())


def normalised_energy_distance(pred: np.ndarray, post: np.ndarray,
                              pre: np.ndarray) -> float:
    """ED(pred, post) / ED(pre, post) -- the NoChange-calibrated score.

    1.0 means "no better than predicting the pre-population", 0.0 means the predicted and
    observed post-populations are indistinguishable in energy distance. This normalisation
    is why `NoChange` is in the table: it fixes the scale, so a reader does not need to
    know the units of the benchmark to read the column. It is > 1 for a model that is
    actively worse than doing nothing, which does happen and must be reported.
    """
    denom = energy_distance(np.asarray(pre, dtype=np.float64),
                            np.asarray(post, dtype=np.float64))
    if denom <= 0.0:
        return float("nan")
    return energy_distance(pred, post) / denom


def cfm_loss_of(model: "Baseline", populations: Sequence, idx: Sequence[int], k: int,
                obj: CFMObjective, batch: int = 256, reps: int = 4,
                seed: int = 0) -> float:
    """Mean CFM loss of a baseline's velocity field over the given populations.

    Uses the SAME `_draw_batch` (hence the same coupling and the same interpolant) that
    trains IHC-FM, so the numbers in the CFM-loss column of Table 1 are directly
    comparable. Models without a velocity field return nan rather than a substitute score
    -- a displacement model has no CFM loss and printing one would be a fabrication.
    """
    if not model.has_velocity:
        return float("nan")
    recs = _pop_arrays(populations, idx, k)
    rng = np.random.default_rng(int(seed))
    tot, cnt = 0.0, 0
    for _ in range(int(reps)):
        for rec in recs:
            bt = _draw_batch(rec, obj, batch, rng)
            v = model.velocity(bt["x_t"].numpy().astype(np.float64), rec["P"], rec["tau"],
                               bt["t"].numpy().astype(np.float64).ravel())
            se = ((np.asarray(v, dtype=np.float64)
                   - bt["target"].numpy().astype(np.float64)) ** 2).sum(axis=1)
            tot += float(np.mean(se))
            cnt += 1
    return tot / max(cnt, 1)


def default_objective(seed: int = 0, max_coupling_n: int = 128) -> CFMObjective:
    """The project's selected coupling: entropic UOT. Every CFM baseline gets exactly this.

    `coupling='sinkhorn'` with eps=0.05, iters=25, unbalanced_tau=1.0 is what the project
    label "sinkhorn_uot" denotes. There is no `coupling='sinkhorn_uot'`; that string raises.

    HOW IT WAS SELECTED, AND HOW IT WAS NOT. An earlier selection ranked couplings by
    held-out CFM loss and reported an 8.14x spread. That comparison is INVALID and the
    number is withdrawn: CFM loss is not comparable across couplings, because it tracks the
    fraction of trivially identical source/target pairs a coupling produces (identity-pair
    fraction: paired 1.0, ot 0.795, sinkhorn 0.229, independent 0.0039 -- which is the
    whole reason a 'paired' arm can reach 6.43e-05). A coupling that pairs points with
    themselves scores a low loss for a reason that has nothing to do with the quality of
    the field it induces.

    On the DISTRIBUTIONAL metric the ranking inverts: normalised ED independent 0.8094,
    sinkhorn_uot 0.8387, ot 1.1188 -- so 'ot' is WORSE THAN PREDICTING NO CHANGE and must
    never be the default. Re-selected on ground-truth interaction recovery (scale-free and
    genuinely comparable across couplings, 3 seeds), sinkhorn_uot wins at every rank:
    r=2 +0.5930 vs +0.0799, r=4 +0.4414 vs +0.2586, r=8 +0.6427 vs +0.4361 (independent).

    CONSEQUENCE FOR THIS MODULE, ENFORCED THROUGHOUT: no coupling and no model may be
    selected, tuned, or ranked on CFM loss. Table 1 ranks on normalised energy distance,
    reported with its no-change denominator so the scale is visible. `cfm_loss_of` exists
    and its numbers are reported, but ONLY within a fixed coupling, as a training
    diagnostic -- never as the ranking key.
    """
    return CFMObjective(sigma=0.0, coupling="sinkhorn", sinkhorn_eps=0.05,
                        sinkhorn_iters=25, unbalanced_tau=1.0, seed=int(seed),
                        max_coupling_n=int(max_coupling_n))


class TrainConfig:
    """The optimisation settings shared by every neural baseline.

    `steps` is the TOTAL of IHC-FM's three-stage schedule (300 + 300 + 200 = 800). The
    baselines have no hierarchy to stage, so they spend the identical budget in one stage:
    matching wall-clock-equivalent gradient steps is the fair reading, and giving a
    flat model a staged schedule it has no structure to exploit would be a strawman in the
    other direction.
    """

    def __init__(self, steps: int = 800, batch: int = 128, lr: float = 3e-3,
                 grad_clip: float = 5.0, weight_decay: float = 0.0,
                 n_int_steps: int = 16, seed: int = 0):
        self.steps = int(steps)
        self.batch = int(batch)
        self.lr = float(lr)
        self.grad_clip = float(grad_clip)
        self.weight_decay = float(weight_decay)
        self.n_int_steps = int(n_int_steps)
        self.seed = int(seed)

    def describe(self) -> dict:
        return dict(steps=self.steps, batch=self.batch, lr=self.lr,
                    grad_clip=self.grad_clip, weight_decay=self.weight_decay,
                    n_int_steps=self.n_int_steps, seed=self.seed,
                    optimiser="Adam", note="steps = 300+300+200 total of IHC-FM's schedule")


# ==================================================================================
# parameter matching
# ==================================================================================
def reference_ihcfm_parameters(d: int = 6, k: int = 4, r: int = 8, hidden: int = 64,
                               depth: int = 2, code_dim: int = 0,
                               D: int | None = None) -> dict:
    """IHC-FM's own parameter count, measured by BUILDING it, never by a formula.

    Returns both candidate matching targets:
      'core'  velocity net only (one-form heads + embedding table + dose)
      'total' core + decoder + ambient metric + metric-radial saturator
    plus the per-component breakdown, because "parameter-matched" is only a meaningful
    claim if the reader can see what was counted. `D` is the observed dimension; it
    defaults to `d` (the synthetic benchmark observes the same space it acts on).

    This imports `ihcfm_geometry` -- the ONLY place in this module that does, and only to
    weigh the reference model. No baseline touches it.
    """
    from composefm.ihcfm_geometry import (Decoder, AmbientMetric, HillDose,
                                          MetricRadialSat, PullbackField)
    from composefm.ihcfm_hierarchy import ConditionalVelocity

    D = int(d) if D is None else int(D)
    torch.manual_seed(0)
    dec = Decoder(int(d), D, hidden=32)
    met = AmbientMetric(D, hidden=48)
    sat = MetricRadialSat(x_dim=D, latent_dim=int(d), hidden=32)
    pbf = PullbackField(dec, met, saturator=sat)

    def pullback(t, z, b):
        return pbf(t, z, one_form=lambda tt, xx, cc: b)

    dose = HillDose(int(k))
    m = ConditionalVelocity(int(d), int(k), r=int(r), code_dim=int(code_dim),
                            hidden=int(hidden), depth=int(depth), dose=dose,
                            pullback=pullback, decoder=dec)

    def n(mod) -> int:
        return int(sum(p.numel() for p in mod.parameters() if p.requires_grad))

    parts = dict(one_form=n(m.one_form), embed=n(m.embed), dose=n(m.dose),
                 encoder=n(m.encoder), decoder=n(dec), ambient_metric=n(met),
                 saturator=n(sat))
    core = parts["one_form"] + parts["embed"] + parts["dose"] + parts["encoder"]
    total = core + parts["decoder"] + parts["ambient_metric"] + parts["saturator"]
    return dict(config=dict(d=int(d), D=D, k=int(k), r=int(r), hidden=int(hidden),
                            depth=int(depth), code_dim=int(code_dim)),
                components=parts, core=int(core), total=int(total),
                matching_target="total",
                matching_note=("neural baselines are matched to 'total', which gives them "
                               "MORE capacity than IHC-FM's velocity core -- the "
                               "conservative direction for a baseline"))


def check_rank_sufficiency(k: int, r: int) -> dict:
    """Pair-basis saturation rank. REPORTING ONLY -- this does not gate anything.

    The arithmetic is sound: the pair basis saturates at min(r, k(k-1)/2), so at k=4 (6
    pairs) r=2 resolves 2 directions, r=4 resolves 4, and r>=6 resolves all 6. Below the
    bound distinct pairs share an interaction direction.

    WHAT IS RETRACTED is the inference that r must therefore be >= k(k-1)/2. Under the
    correct coupling the measured trend does not follow the rank (recovery r=2 +0.5930,
    r=4 +0.4414, r=6 +0.5121, r=8 +0.6427), so `sufficient` records where r sits relative
    to saturation and is NOT a correctness criterion. `full_pair_rank` is the neutral name;
    `sufficient` is kept as an alias for callers that predate the retraction.
    """
    need = int(k) * (int(k) - 1) // 2
    full = bool(int(r) >= need)
    return dict(k=int(k), r=int(r), saturation_rank=need,
                pair_basis_rank=int(min(int(r), need)), full_pair_rank=full,
                sufficient=full,
                is_correctness_requirement=False,
                note=("rank saturates at min(r, k(k-1)/2). REPORTING ONLY: the r >= "
                      "k(k-1)/2 requirement is RETRACTED -- r=2 outperformed r=4 and r=6 "
                      "on measured recovery, so r is a tunable capacity hyperparameter "
                      "selected on a held-out distributional score"))


def solve_hidden(build: Callable[[int], "Baseline"], target: int,
                 lo: int = 8, hi: int = 512, tol: float = 0.20) -> dict:
    """Smallest-|error| hidden width for a parameter budget, by construction and counting.

    Monotonicity of parameter count in width is assumed only for the search bracket, not
    for correctness: the width grid is swept and the exact minimiser of |count - target| is
    returned, together with whether it lands inside `tol`. If it does not, the caller must
    see that rather than silently shipping an unmatched baseline.
    """
    best = None
    counts = {}
    for h in range(int(lo), int(hi) + 1):
        c = build(h).n_parameters()
        counts[h] = c
        err = abs(c - int(target))
        if best is None or err < best[1]:
            best = (h, err, c)
        if c > 3 * int(target):        # far past the target; monotone enough to stop
            break
    h, err, c = best
    rel = err / max(int(target), 1)
    return dict(hidden=int(h), n_parameters=int(c), target=int(target),
                abs_error=int(err), rel_error=float(rel),
                within_tol=bool(rel <= float(tol)), tol=float(tol))


# ==================================================================================
# the interface
# ==================================================================================
class Baseline:
    """One shared contract. Subclasses override `_fit`, `predict` and optionally `velocity`.

    `is_flow` says whether the model transports by integrating a velocity field.
    `has_velocity` says whether `velocity(...)` is defined at all. They are NOT the same:
    the shift baselines admit a straight-line velocity reading of the map they committed
    to (so `has_velocity` is True and their CFM loss is a real, comparable number) but they
    never integrate anything (`is_flow` False). `DeepSetsEndpoint` has neither.
    """

    name: str = "baseline"
    is_flow: bool = False
    has_velocity: bool = False
    is_neural: bool = False

    def __init__(self, d: int, k: int, seed: int = 0):
        self.d, self.k, self.seed = int(d), int(k), int(seed)
        self.fitted = False
        self.report: dict = {}

    # -- contract -----------------------------------------------------------
    def fit(self, populations: Sequence, train_idx: Sequence[int]) -> dict:
        t0 = time.time()
        rep = self._fit(populations, train_idx) or {}
        rep = dict(rep)
        rep.setdefault("n_train_populations", len(list(train_idx)))
        rep["n_parameters"] = self.n_parameters()
        rep["fit_wall_clock_s"] = time.time() - t0
        self.report = rep
        self.fitted = True
        return rep

    def _fit(self, populations: Sequence, train_idx: Sequence[int]) -> dict:
        raise NotImplementedError

    def predict(self, Z0: np.ndarray, P: Sequence[int], tau: float) -> np.ndarray:
        raise NotImplementedError

    def velocity(self, Z: np.ndarray, P: Sequence[int], tau: float,
                 t: float | np.ndarray = 0.0) -> np.ndarray:
        raise NoVelocityField(
            f"{self.name} does not define a velocity field; it is a displacement model")

    def n_parameters(self) -> int:
        return 0

    # -- helpers ------------------------------------------------------------
    def _mask(self, P: Sequence[int]) -> np.ndarray:
        m = np.zeros(self.k, dtype=np.float64)
        for q in P:
            m[int(q)] = 1.0
        return m

    def _require_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError(f"{self.name}: call fit() before predict()")

    def describe(self) -> dict:
        return dict(name=self.name, is_flow=self.is_flow, has_velocity=self.has_velocity,
                    is_neural=self.is_neural, n_parameters=self.n_parameters(),
                    d=self.d, k=self.k)


# ==================================================================================
# 1-4  simple / strong controls
# ==================================================================================
class NoChange(Baseline):
    """Predict the pre-population unchanged. The metric's calibration point.

    Zero parameters, and by the definition of `normalised_energy_distance` it scores
    exactly 1.0 up to the estimator's own sampling noise. If it does not, the scoring is
    wired wrong and no other row of the table can be trusted -- which is precisely why it
    is in the table and why the self-test asserts on it.

    It also admits a velocity: the zero field. That is not a trick, it is the honest
    I-CFM conditional velocity of the identity map, and its CFM loss equals the mean
    squared displacement of the data -- a genuine, interpretable floor for the CFM column.
    """

    name = "NoChange"
    has_velocity = True

    def _fit(self, populations, train_idx) -> dict:
        return dict(model="identity map", note="zero parameters; calibrates the metric")

    def predict(self, Z0, P, tau) -> np.ndarray:
        return np.array(np.atleast_2d(np.asarray(Z0, dtype=np.float64)), copy=True)

    def velocity(self, Z, P, tau, t=0.0) -> np.ndarray:
        return np.zeros_like(np.atleast_2d(np.asarray(Z, dtype=np.float64)))


class MatchingMean(Baseline):
    """Shift every cell by ONE displacement: the mean over all training conditions.

    Condition-blind by construction -- it cannot tell a singleton from a triple, and its
    exposure response is flat. It is the control for "is the benchmark's signal just a
    global drift". The mean is taken over the per-condition mean displacements rather than
    over pooled cells, so a condition sampled with more cells does not dominate.
    """

    name = "MatchingMean"
    has_velocity = True

    def __init__(self, d: int, k: int, seed: int = 0):
        super().__init__(d, k, seed=seed)
        self.delta = np.zeros(self.d, dtype=np.float64)

    def _fit(self, populations, train_idx) -> dict:
        rows = []
        for i in train_idx:
            p = populations[i]
            rows.append(np.asarray(p.post, dtype=np.float64).mean(axis=0)
                        - np.asarray(p.pre, dtype=np.float64).mean(axis=0))
        self.delta = np.mean(np.stack(rows, axis=0), axis=0)
        return dict(model="x + mean training displacement",
                    delta=self.delta.tolist(), delta_norm=float(np.linalg.norm(self.delta)),
                    n_conditions_averaged=len(rows),
                    weighting="uniform over conditions, not over cells")

    def predict(self, Z0, P, tau) -> np.ndarray:
        self._require_fitted()
        return np.atleast_2d(np.asarray(Z0, dtype=np.float64)) + self.delta[None, :]

    def velocity(self, Z, P, tau, t=0.0) -> np.ndarray:
        self._require_fitted()
        Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
        return np.tile(self.delta[None, :], (Z.shape[0], 1))


class PerturbedMean(Baseline):
    """Per-intervention mean shift, SUMMED over the members of the set.

    This is the additive-displacement control in the style of the CPA family: learn one
    displacement per intervention from the conditions that contain it, then predict a
    combination as the sum of its members' displacements. It is the strongest thing you can
    do without ever modelling an interaction, which makes it the reference point for the
    question the interaction branch exists to answer -- any gap between IHC-FM and this row
    on held-out combinations IS the interaction claim.

    Exposure. Displacements are learned per (intervention, exposure) where an exposure was
    observed, and interpolated linearly in tau -- with the origin pinned, delta(0) = 0,
    which is the zero-dose identity the benchmark satisfies exactly. Extrapolation beyond
    the observed exposure range is clamped to the nearest observed level and flagged in the
    report rather than silently linearly extended.

    Singleton observations are used when available. A pair (p, q) contributes to p's
    displacement only if p is not otherwise identified, in which case the pair's total is
    split evenly -- recorded in the report as `derived_from_combinations`, because such a
    split is an assumption, not a measurement.
    """

    name = "PerturbedMean"
    has_velocity = True

    def __init__(self, d: int, k: int, seed: int = 0):
        super().__init__(d, k, seed=seed)
        self.table: dict[int, dict[float, np.ndarray]] = {}
        self.fallback = np.zeros(self.d, dtype=np.float64)

    def _fit(self, populations, train_idx) -> dict:
        # (intervention, tau) -> list of displacement vectors, from SINGLETONS first
        acc: dict[tuple[int, float], list[np.ndarray]] = {}
        derived: list[str] = []
        combos: list[tuple[tuple, float, np.ndarray]] = []
        for i in train_idx:
            p = populations[i]
            dlt = (np.asarray(p.post, dtype=np.float64).mean(axis=0)
                   - np.asarray(p.pre, dtype=np.float64).mean(axis=0))
            if len(p.P) == 1:
                acc.setdefault((int(p.P[0]), float(p.tau)), []).append(dlt)
            else:
                combos.append((tuple(int(q) for q in p.P), float(p.tau), dlt))
        identified = {q for (q, _) in acc}
        # only for interventions with NO singleton anywhere: split a combination evenly
        for P, tau, dlt in combos:
            missing = [q for q in P if q not in identified]
            if len(missing) == len(P):
                for q in missing:
                    acc.setdefault((q, tau), []).append(dlt / float(len(P)))
                    derived.append(f"X{q}@tau{tau:g} from {P}")
        self.table = {}
        for (q, tau), rows in acc.items():
            self.table.setdefault(q, {})[tau] = np.mean(np.stack(rows, 0), axis=0)
        allrows = [v for tab in self.table.values() for v in tab.values()]
        self.fallback = (np.mean(np.stack(allrows, 0), axis=0) if allrows
                         else np.zeros(self.d))
        return dict(model="sum of per-intervention mean displacements (CPA-style additive)",
                    n_interventions_identified=len(self.table),
                    exposures_per_intervention={int(q): sorted(t)
                                                for q, t in self.table.items()},
                    derived_from_combinations=derived,
                    dose_model="linear in tau through the origin, clamped outside range",
                    delta_norms={int(q): {float(t): float(np.linalg.norm(v))
                                          for t, v in tab.items()}
                                 for q, tab in self.table.items()})

    def _delta_one(self, q: int, tau: float) -> np.ndarray:
        tab = self.table.get(int(q))
        if not tab:
            return self.fallback * float(tau)
        taus = sorted(tab)
        tau = float(tau)
        if tau == 0.0:
            return np.zeros(self.d, dtype=np.float64)
        if tau <= taus[0]:                        # interpolate against delta(0) = 0
            return tab[taus[0]] * (tau / taus[0])
        if tau >= taus[-1]:                       # CLAMP, never extrapolate
            return tab[taus[-1]] * (taus[-1] / taus[-1])
        for a, b in zip(taus[:-1], taus[1:]):
            if a <= tau <= b:
                w = (tau - a) / (b - a)
                return (1.0 - w) * tab[a] + w * tab[b]
        return tab[taus[-1]]

    def _delta(self, P: Sequence[int], tau: float) -> np.ndarray:
        return np.sum([self._delta_one(int(q), float(tau)) for q in P], axis=0)

    def predict(self, Z0, P, tau) -> np.ndarray:
        self._require_fitted()
        return (np.atleast_2d(np.asarray(Z0, dtype=np.float64))
                + self._delta(P, tau)[None, :])

    def velocity(self, Z, P, tau, t=0.0) -> np.ndarray:
        self._require_fitted()
        Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
        return np.tile(self._delta(P, tau)[None, :], (Z.shape[0], 1))


class LinearResponse(Baseline):
    """Ridge regression from set features to the displacement. Linear-in-set, with a state term.

    Features per cell: [1, mask (k), mask * dose(tau) (k), z0 (d)]. Two things to note.

    The mask AND the dosed mask are both present. With only the dosed mask the model cannot
    represent a presence effect that saturates in exposure; with only the mask it cannot
    represent exposure at all. Including both makes it the honest linear model of the same
    (set, exposure) input the neural baselines see.

    The z0 block makes the displacement STATE-DEPENDENT, which the shift baselines above
    cannot be. That matters on this benchmark: the truth is a nonlinear flow, so its
    displacement genuinely varies over the population, and a linear-in-state predictor is a
    materially stronger control than a constant shift. It is still linear in the SET, so it
    has no interaction term by construction -- it can never distinguish X0+X1 from the sum
    of its parts, which is the whole point of including it.

    Ridge strength is selected by generalised cross-validation over a log grid on the
    training conditions only. Held-out populations are never touched.
    """

    name = "LinearResponse"
    has_velocity = True

    def __init__(self, d: int, k: int, seed: int = 0, n_sub: int = 200,
                 alphas: Sequence[float] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)):
        super().__init__(d, k, seed=seed)
        self.W = np.zeros((1 + 2 * self.k + self.d, self.d), dtype=np.float64)
        self.n_sub = int(n_sub)
        self.alphas = tuple(float(a) for a in alphas)
        self.alpha = float("nan")

    def _features(self, Z0: np.ndarray, P: Sequence[int], tau: float) -> np.ndarray:
        Z0 = np.atleast_2d(np.asarray(Z0, dtype=np.float64))
        n = Z0.shape[0]
        m = self._mask(P)
        rows = np.concatenate([np.ones((n, 1)),
                               np.tile(m[None, :], (n, 1)),
                               np.tile((m * float(tau))[None, :], (n, 1)),
                               Z0], axis=1)
        return rows

    def _fit(self, populations, train_idx) -> dict:
        g = np.random.default_rng(self.seed)
        Xs, Ys = [], []
        for i in train_idx:
            p = populations[i]
            pre = np.asarray(p.pre, dtype=np.float64)
            post = np.asarray(p.post, dtype=np.float64)
            n = min(pre.shape[0], post.shape[0], self.n_sub)
            sel = g.permutation(min(pre.shape[0], post.shape[0]))[:n]
            # within a population post[i] is the flow image of pre[i]; the paired
            # displacement is available and is what a regression should target
            Xs.append(self._features(pre[sel], p.P, p.tau))
            Ys.append(post[sel] - pre[sel])
        X = np.concatenate(Xs, 0)
        Y = np.concatenate(Ys, 0)
        XtX = X.T @ X
        XtY = X.T @ Y
        n_f = X.shape[1]
        best = None
        for a in self.alphas:
            R = np.eye(n_f) * a
            R[0, 0] = 0.0                       # never penalise the intercept
            try:
                W = np.linalg.solve(XtX + R, XtY)
            except np.linalg.LinAlgError:
                continue
            # GCV with the effective degrees of freedom trace(X (XtX+R)^-1 X^T)
            edf = float(np.trace(np.linalg.solve(XtX + R, XtX)))
            resid = float(np.mean(np.sum((X @ W - Y) ** 2, axis=1)))
            denom = (1.0 - edf / X.shape[0]) ** 2
            gcv = resid / max(denom, 1e-12)
            if best is None or gcv < best[0]:
                best = (gcv, a, W, edf)
        _, self.alpha, self.W, edf = best
        train_mse = float(np.mean(np.sum((X @ self.W - Y) ** 2, axis=1)))
        return dict(model="ridge: displacement ~ [1, mask, mask*tau, z0]",
                    n_features=int(n_f), n_rows=int(X.shape[0]),
                    alpha_selected=float(self.alpha), alphas_searched=list(self.alphas),
                    selection="generalised cross-validation on TRAINING conditions only",
                    effective_dof=float(edf), train_displacement_mse=train_mse,
                    structure_note=("linear in the set: no interaction term exists, so "
                                    "X0+X1 is forced to equal the sum of its parts"))

    def predict(self, Z0, P, tau) -> np.ndarray:
        self._require_fitted()
        Z0 = np.atleast_2d(np.asarray(Z0, dtype=np.float64))
        return Z0 + self._features(Z0, P, tau) @ self.W

    def velocity(self, Z, P, tau, t=0.0) -> np.ndarray:
        self._require_fitted()
        Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
        # the straight-line reading of its own displacement map, evaluated at the CURRENT
        # state -- not at z0, which it cannot see mid-path
        return self._features(Z, P, tau) @ self.W


# ==================================================================================
# neural machinery shared by 5-7
# ==================================================================================
def _mlp(n_in: int, hidden: int, n_out: int, depth: int = 2, ln: bool = True,
         act=nn.SiLU) -> nn.Sequential:
    """Same block the method uses (LayerNorm + SiLU), so width is comparable to width."""
    layers: list[nn.Module] = []
    d = int(n_in)
    for _ in range(int(depth)):
        layers.append(nn.Linear(d, hidden))
        if ln:
            layers.append(nn.LayerNorm(hidden))
        layers.append(act())
        d = hidden
    layers.append(nn.Linear(d, n_out))
    return nn.Sequential(*layers)


class _SoftplusDose(nn.Module):
    """A learned, monotone, zero-pinned dose response: a_p = amp_p * tanh(tau / ec50_p).

    Structurally a(0) = 0, which the benchmark's zero-dose identity requires, and monotone
    increasing and saturating in tau, which the Hill form the method uses also is. It is
    deliberately NOT `HillDose` itself: importing the method's dose module into a baseline
    would blur what is being compared. It has the same parameter budget shape (two scalars
    per intervention) so the comparison stays width-for-width.
    """

    def __init__(self, k: int):
        super().__init__()
        self.k = int(k)
        self.log_amp = nn.Parameter(torch.zeros(self.k))
        self.log_ec50 = nn.Parameter(torch.zeros(self.k))

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        tau = tau.reshape(tau.shape[0], -1)
        if tau.shape[1] == 1:
            tau = tau.expand(-1, self.k)
        ec = torch.exp(self.log_ec50).clamp_min(1e-6)
        return torch.exp(self.log_amp) * torch.tanh(tau / ec)


class _NeuralBaseline(Baseline):
    """Shared fit loop for the neural rows. Same objective, optimiser, guards as IHC-FM.

    The FINITE-STEP GUARDS are copied deliberately: a non-finite loss or a non-finite
    gradient norm SKIPS the step and is counted, never applied. A baseline that silently
    took nan steps would produce a table row that is not a fit of anything, and the count
    of skipped steps is reported so a reader can see it was zero (or not).
    """

    is_neural = True

    def __init__(self, d: int, k: int, seed: int = 0, cfg: TrainConfig | None = None,
                 objective: CFMObjective | None = None):
        super().__init__(d, k, seed=seed)
        self.cfg = cfg if cfg is not None else TrainConfig(seed=seed)
        self.obj = objective if objective is not None else default_objective(seed=seed)
        self.net: nn.Module | None = None

    def n_parameters(self) -> int:
        if self.net is None:
            return 0
        return int(sum(p.numel() for p in self.net.parameters() if p.requires_grad))

    # -- subclass hooks -----------------------------------------------------
    def _loss_on_batch(self, bt: dict) -> torch.Tensor:
        raise NotImplementedError

    def _train(self, recs: Sequence[dict]) -> dict:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        torch.manual_seed(cfg.seed)
        opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr,
                               weight_decay=cfg.weight_decay)
        hist: list[float] = []
        skipped = 0
        t0 = time.time()
        for _ in range(cfg.steps):
            j = int(rng.integers(0, len(recs)))
            bt = _draw_batch(recs[j], self.obj, cfg.batch, rng)
            loss = self._loss_on_batch(bt)
            if not torch.isfinite(loss):
                skipped += 1
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(self.net.parameters(), cfg.grad_clip)
            if not torch.isfinite(gn):
                skipped += 1
                opt.zero_grad(set_to_none=True)
                continue
            opt.step()
            hist.append(float(loss.detach()))
        tail = max(1, len(hist) // 10)
        return dict(steps=cfg.steps, applied=len(hist), skipped_nonfinite=int(skipped),
                    mean_loss_first_decile=(float(np.mean(hist[:tail])) if hist
                                            else float("nan")),
                    mean_loss_last_decile=(float(np.mean(hist[-tail:])) if hist
                                           else float("nan")),
                    final_loss=(hist[-1] if hist else float("nan")),
                    train_wall_clock_s=time.time() - t0,
                    config=cfg.describe(), objective=self.obj.describe())

    def _fit(self, populations, train_idx) -> dict:
        recs = _pop_arrays(populations, train_idx, self.k)
        rep = self._train(recs)
        self.net.eval()
        return rep

    # -- integration --------------------------------------------------------
    def _integrate(self, Z0: np.ndarray, P: Sequence[int], tau: float) -> np.ndarray:
        """RK4 over t in [0, 1] in float64 -- the same transport IHC-FM is scored under.

        float64 is not cosmetic here: project fact D is that float32 inference floors the
        chart test at ~3.1e-07 while float64 reaches ~7e-16. Scoring a baseline in float32
        against a method scored in float64 would be an unfair comparison in the baseline's
        disfavour.
        """
        Z = np.array(np.atleast_2d(np.asarray(Z0, dtype=np.float64)), copy=True)
        h = 1.0 / self.cfg.n_int_steps
        t = 0.0
        for _ in range(self.cfg.n_int_steps):
            k1 = self.velocity(Z, P, tau, t)
            k2 = self.velocity(Z + 0.5 * h * k1, P, tau, t + 0.5 * h)
            k3 = self.velocity(Z + 0.5 * h * k2, P, tau, t + 0.5 * h)
            k4 = self.velocity(Z + h * k3, P, tau, t + h)
            Z = Z + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            t += h
        return Z

    def _eval_net(self, fn: Callable[[], torch.Tensor]) -> np.ndarray:
        m64 = getattr(self, "_net64", None)
        if m64 is None:
            self._net64 = copy.deepcopy(self.net).to(torch.float64).eval()
        with torch.no_grad():
            return fn().numpy().astype(np.float64)


# ==================================================================================
# 5  DeepSets endpoint regression -- NOT a flow model
# ==================================================================================
class DeepSetsEndpoint(_NeuralBaseline):
    """Permutation-invariant DeepSets over the intervention set -> endpoint displacement.

        pooled = sum_{p in P} phi([e_p, a_p]),      a_p = dose_p(tau)
        x1_hat = z0 + rho([z0, pooled, tau])

    The pooling runs over MEMBERS ONLY (the sum is gated by the set mask), so it is exactly
    permutation-invariant in the set and a non-member contributes bit-exact zero -- the same
    structural property the method's first moment has. It has the set inductive bias but no
    flow: it regresses the endpoint directly, in one shot, with no time variable anywhere.

    Its role in the table is to separate two claims that are easy to conflate. If it matches
    the CFM baselines, the flow formulation is not what is buying the performance and the
    paper must say so. If it loses to them at equal parameter count, the flow formulation
    is doing work independent of the hierarchy.

    It has NO velocity field and does not pretend to: `velocity` raises, and its CFM-loss
    cell in Table 1 is nan, not a fabricated number.

    SUPERVISION: PAIRED, AND THIS IS A MEASURED CORRECTION, NOT A CONCESSION.
    `endpoint_supervision` defaults to 'paired'. It was first built to share IHC-FM's
    validated entropic-UOT coupling, on the principle that every neural baseline should see
    the same objective -- and at the default configuration it scored 1.92 normalised on the
    HELD-OUT combination and 4.99 on the training subset, i.e. worse than predicting no
    change on both, and 2.6x worse where it was FITTED than where it generalised. Losing
    on the training conditions is the signature of a wiring error rather than of a hard
    problem, so it was diagnosed rather than reported:

      * the UOT-coupled displacement has mean norm 0.868 against the true paired
        displacement's 0.406, a 2.1x inflation, because entropic OT agrees with the true
        pairing on only ~23% of samples on this benchmark (project-measured);
      * a CFM model is unharmed by that: it regresses E[u | x_t] and recovers the correct
        push-forward through INTEGRATION -- that is the entire point of the coupling being
        allowed to be wrong;
      * an endpoint regressor has no integration step. It applies E[u | x0] directly as a
        map, which is the barycentric projection of the coupling. That map overshoots (the
        inflated norm) and collapses variance (predicted per-coordinate sd 0.667-0.790
        against the observed post-population's 0.755-0.982);
      * switching to the paired displacement, which genuinely exists within a population
        (post[i] is the flow image of pre[i]), lifts the predicted sd to 0.714-0.888 and
        improves the DISTRIBUTIONAL score from 1.92 to 0.713 held-out and 4.99 to 1.06 on
        the training subset.

    All figures above are read back from `results/baselines_selftest.json` at the default
    configuration and move with it; the JSON is the authority, not this docstring.

    The evidence above is deliberately all distributional or structural. The two arms train
    under DIFFERENT couplings, so their training losses (1.575 vs 0.140) are NOT comparable
    and are not offered as evidence -- a paired coupling pairs every point with itself
    (identity-pair fraction 1.0 against sinkhorn's 0.229), which lowers the loss for a
    reason unrelated to the quality of the map. Those two numbers are recorded in the JSON
    as within-arm training diagnostics and are explicitly flagged non-comparable.

    So the "same coupling as IHC-FM" requirement carries a "where applicable" clause and
    this is where it does not apply: a minibatch coupling is a device for learning a
    velocity field, and imposing it on a model that does not integrate one is a strawman.
    The self-test fits BOTH arms and records both numbers, so the choice is visible rather
    than asserted.
    """

    name = "DeepSetsEndpoint"
    is_flow = False
    has_velocity = False

    def __init__(self, d: int, k: int, r: int = 8, hidden: int = 64, depth: int = 2,
                 seed: int = 0, cfg: TrainConfig | None = None,
                 objective: CFMObjective | None = None,
                 endpoint_supervision: str = "paired"):
        super().__init__(d, k, seed=seed, cfg=cfg, objective=objective)
        if endpoint_supervision not in ("paired", "coupled"):
            raise ValueError("endpoint_supervision must be 'paired' or 'coupled'")
        self.endpoint_supervision = str(endpoint_supervision)
        if self.endpoint_supervision == "paired":
            # a separate objective for the DATA DRAW only: same interpolant, same sigma,
            # index-aligned pairs. The shared objective is kept on self.obj for reporting.
            self._sup_obj = CFMObjective(sigma=self.obj.sigma, coupling="paired",
                                         seed=int(seed))
        else:
            self._sup_obj = self.obj
        self.r, self.hidden, self.depth = int(r), int(hidden), int(depth)
        torch.manual_seed(seed)
        g = torch.Generator().manual_seed(int(seed))
        self.net = nn.Module()
        self.net.embed = nn.Parameter(torch.randn(self.k, self.r, generator=g)
                                      / math.sqrt(self.r))
        self.net.dose = _SoftplusDose(self.k)
        self.net.phi = _mlp(self.r + 1, self.hidden, self.hidden, depth=self.depth, ln=True)
        self.net.rho = _mlp(self.d + self.hidden + 1, self.hidden, self.d,
                            depth=self.depth, ln=True)

    # -- forward ------------------------------------------------------------
    def _pooled(self, tau: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B = mask.shape[0]
        a = mask * self.net.dose(tau)                          # (B,k), exact zeros off-set
        E = self.net.embed.to(dtype=mask.dtype)                # (k,r)
        e = E.unsqueeze(0).expand(B, -1, -1)                   # (B,k,r)
        inp = torch.cat([e, a.unsqueeze(-1)], dim=-1)          # (B,k,r+1)
        h = self.net.phi(inp.reshape(B * self.k, self.r + 1)).reshape(B, self.k, -1)
        return (mask.unsqueeze(-1) * h).sum(dim=1)             # gated sum over MEMBERS

    def _displacement(self, z0: torch.Tensor, tau: torch.Tensor,
                      mask: torch.Tensor) -> torch.Tensor:
        pooled = self._pooled(tau, mask)
        return self.net.rho(torch.cat([z0, pooled, tau[:, :1]], dim=1))

    def _loss_on_batch(self, bt: dict) -> torch.Tensor:
        # Endpoint regression on the displacement. x0 is recovered exactly from the
        # interpolant the batch already carries: x_t = (1-t) x0 + t x1 + noise and
        # u = x1 - x0, so x0 = x_t - t*u - noise. No second draw, no re-coupling.
        u = bt["target"]
        x0 = bt["x_t"] - bt["t"] * u - bt["noise"]
        pred = self._displacement(x0, bt["tau"], bt["mask"])
        return ((pred - u) ** 2).sum(dim=1).mean()

    def _fit(self, populations, train_idx) -> dict:
        # draw under _sup_obj (see the supervision note in the class docstring); the
        # optimiser, step budget, batch size and guards are the shared ones
        recs = _pop_arrays(populations, train_idx, self.k)
        obj_shared, self.obj = self.obj, self._sup_obj
        try:
            rep = self._train(recs)
        finally:
            self.obj = obj_shared
        self.net.eval()
        rep["endpoint_supervision"] = self.endpoint_supervision
        rep["supervision_coupling"] = self._sup_obj.coupling
        rep["shared_objective_coupling"] = self.obj.coupling
        rep["supervision_note"] = (
            "paired: a minibatch coupling is a device for learning a velocity field; an "
            "endpoint regressor applies its barycentric projection directly as a map, "
            "which overshoots and collapses variance (diagnosed, both arms measured)")
        return rep

    def predict(self, Z0, P, tau) -> np.ndarray:
        self._require_fitted()
        Z0 = np.atleast_2d(np.asarray(Z0, dtype=np.float64))
        n = Z0.shape[0]
        m64 = copy.deepcopy(self.net).to(torch.float64).eval()
        saved, self.net = self.net, m64
        try:
            with torch.no_grad():
                z = torch.as_tensor(Z0, dtype=torch.float64)
                tv = torch.full((n, 1), float(tau), dtype=torch.float64)
                mk = torch.as_tensor(np.tile(self._mask(P), (n, 1)), dtype=torch.float64)
                out = self._displacement(z, tv, mk).numpy().astype(np.float64)
        finally:
            self.net = saved
        return Z0 + out


# ==================================================================================
# 6  monolithic CFM -- no hierarchy, no geometry
# ==================================================================================
class MonolithicCFM(_NeuralBaseline):
    """One time-conditioned velocity net on the RAW set indicator and exposure vector.

        v(t, z | P, tau) = MLP([z, timefeat(t), mask, mask * tau, tau])

    No moment hierarchy: the set enters as a k-dimensional indicator, so the network must
    learn permutation structure from data instead of having it by construction, and nothing
    forces a non-member intervention to contribute zero. No factorisation: there is one
    monolithic function of the whole condition vector, so the parameters that describe
    X0+X1 are not shared with those describing X0. No pullback: the field is read straight
    off the net in the coordinates it was given, with no metric and no decoder.

    This is the "just train a conditional velocity net" row. It is the row that decides
    whether the paper's structure is load-bearing or ornamental, so it gets the same
    objective, the same coupling, the same 800 steps, the same batch, the same optimiser and
    a width solved to match IHC-FM's parameter count.
    """

    name = "MonolithicCFM"
    is_flow = True
    has_velocity = True

    def __init__(self, d: int, k: int, hidden: int = 64, depth: int = 2, seed: int = 0,
                 cfg: TrainConfig | None = None, objective: CFMObjective | None = None):
        super().__init__(d, k, seed=seed, cfg=cfg, objective=objective)
        self.hidden, self.depth = int(hidden), int(depth)
        torch.manual_seed(seed)
        n_in = self.d + N_TIME_FEATURES + 2 * self.k + 1
        self.net = _mlp(n_in, self.hidden, self.d, depth=self.depth, ln=True)

    def _forward(self, t: torch.Tensor, z: torch.Tensor, tau: torch.Tensor,
                 mask: torch.Tensor) -> torch.Tensor:
        tf = _time_features(t).to(dtype=z.dtype)
        if tf.shape[0] == 1 and z.shape[0] > 1:
            tf = tf.expand(z.shape[0], -1)
        tau1 = tau[:, :1]
        return self.net(torch.cat([z, tf, mask, mask * tau1, tau1], dim=1))

    def _loss_on_batch(self, bt: dict) -> torch.Tensor:
        v = self._forward(bt["t"], bt["x_t"], bt["tau"], bt["mask"])
        return self.obj.loss(v, bt["target"])

    def velocity(self, Z, P, tau, t=0.0) -> np.ndarray:
        self._require_fitted()
        Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
        n = Z.shape[0]
        m64 = copy.deepcopy(self.net).to(torch.float64).eval()
        saved, self.net = self.net, m64
        try:
            with torch.no_grad():
                z = torch.as_tensor(Z, dtype=torch.float64)
                tt = np.full((n, 1), float(t)) if np.isscalar(t) else \
                    np.asarray(t, dtype=np.float64).reshape(-1, 1)
                if tt.shape[0] == 1 and n > 1:
                    tt = np.tile(tt, (n, 1))
                tv = torch.as_tensor(tt, dtype=torch.float64)
                tauv = torch.full((n, 1), float(tau), dtype=torch.float64)
                mk = torch.as_tensor(np.tile(self._mask(P), (n, 1)), dtype=torch.float64)
                return self._forward(tv, z, tauv, mk).numpy().astype(np.float64)
        finally:
            self.net = saved

    def predict(self, Z0, P, tau) -> np.ndarray:
        self._require_fitted()
        return self._integrate(Z0, P, tau)


# ==================================================================================
# 7  factored-additive CFM -- the closest published-style competitor
# ==================================================================================
class FactoredAdditiveCFM(_NeuralBaseline):
    """CFM velocity that SUMS per-intervention fields. No interaction branch, no pullback.

        v(t, z | P, tau) = sum_{p in P} a_p * f([z, timefeat(t), e_p, a_p]),
        a_p = dose_p(tau)

    This is factored velocity composition: one shared field conditioned on a per-
    intervention embedding, gated by that intervention's own exposure, summed over the set.
    It is the strongest structural competitor in the literature's style, and it is built to
    WIN if it can:

      * it gets its own learned embedding table at the SAME rank r as IHC-FM, sized to
        satisfy r >= k(k-1)/2 for the same reason;
      * it gets a learned monotone zero-pinned dose module, so its exposure response is as
        flexible as the method's;
      * the a_p gate makes zero-exposure vanishing and singleton reduction bit-exact, the
        same structural guarantees the method has -- it is not handicapped on those;
      * its width is solved for the same parameter budget.

    What it CANNOT do is represent an interaction. The sum over members is exact and
    unconditional: v(X0+X1) = v(X0) + v(X1) identically, for every state and every time. So
    the gap between this row and IHC-FM on held-out COMBINATIONS is, by construction, the
    interaction claim -- and the gap on singletons should be ~zero, which is a check on the
    comparison rather than a result.

    Honest caveat on that reading. Because velocities compose additively but FLOWS do not,
    an additive velocity field does NOT produce an additive displacement after integration:
    the composition defect (measured on this benchmark as comparable in magnitude to the
    true interaction and nearly orthogonal to it) is available to this baseline for free.
    So this row is a stronger competitor on the endpoint metric than "no interaction term"
    makes it sound, and any interaction claim must be stated against it, not against
    `PerturbedMean`.
    """

    name = "FactoredAdditiveCFM"
    is_flow = True
    has_velocity = True

    def __init__(self, d: int, k: int, r: int = 8, hidden: int = 64, depth: int = 2,
                 seed: int = 0, cfg: TrainConfig | None = None,
                 objective: CFMObjective | None = None):
        super().__init__(d, k, seed=seed, cfg=cfg, objective=objective)
        self.r, self.hidden, self.depth = int(r), int(hidden), int(depth)
        torch.manual_seed(seed)
        g = torch.Generator().manual_seed(int(seed))
        self.net = nn.Module()
        self.net.embed = nn.Parameter(torch.randn(self.k, self.r, generator=g)
                                      / math.sqrt(self.r))
        self.net.dose = _SoftplusDose(self.k)
        self.net.f = _mlp(self.d + N_TIME_FEATURES + self.r + 1, self.hidden, self.d,
                          depth=self.depth, ln=True)

    def _forward(self, t: torch.Tensor, z: torch.Tensor, tau: torch.Tensor,
                 mask: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        tf = _time_features(t).to(dtype=z.dtype)
        if tf.shape[0] == 1 and B > 1:
            tf = tf.expand(B, -1)
        a = mask * self.net.dose(tau)                                  # (B,k) exact zeros
        E = self.net.embed.to(dtype=z.dtype)
        base = torch.cat([z, tf], dim=1)                               # (B, d+T)
        base_r = base.unsqueeze(1).expand(-1, self.k, -1)              # (B,k,d+T)
        e_r = E.unsqueeze(0).expand(B, -1, -1)                         # (B,k,r)
        inp = torch.cat([base_r, e_r, a.unsqueeze(-1)], dim=-1)
        f = self.net.f(inp.reshape(B * self.k, inp.shape[-1])).reshape(B, self.k, self.d)
        f = torch.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
        # gate by a_p: exposure 0 (or non-member) contributes bit-exact zero, so singleton
        # reduction and zero-exposure identity hold structurally, as in the method
        return torch.einsum('bkd,bk->bd', f, a)

    def _loss_on_batch(self, bt: dict) -> torch.Tensor:
        v = self._forward(bt["t"], bt["x_t"], bt["tau"], bt["mask"])
        return self.obj.loss(v, bt["target"])

    def velocity(self, Z, P, tau, t=0.0) -> np.ndarray:
        self._require_fitted()
        Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
        n = Z.shape[0]
        m64 = copy.deepcopy(self.net).to(torch.float64).eval()
        saved, self.net = self.net, m64
        try:
            with torch.no_grad():
                z = torch.as_tensor(Z, dtype=torch.float64)
                tt = np.full((n, 1), float(t)) if np.isscalar(t) else \
                    np.asarray(t, dtype=np.float64).reshape(-1, 1)
                if tt.shape[0] == 1 and n > 1:
                    tt = np.tile(tt, (n, 1))
                tv = torch.as_tensor(tt, dtype=torch.float64)
                tauv = torch.full((n, 1), float(tau), dtype=torch.float64)
                mk = torch.as_tensor(np.tile(self._mask(P), (n, 1)), dtype=torch.float64)
                return self._forward(tv, z, tauv, mk).numpy().astype(np.float64)
        finally:
            self.net = saved

    def predict(self, Z0, P, tau) -> np.ndarray:
        self._require_fitted()
        return self._integrate(Z0, P, tau)


# ==================================================================================
# registry and construction
# ==================================================================================
BASELINES: dict[str, type] = {
    "NoChange": NoChange,
    "MatchingMean": MatchingMean,
    "PerturbedMean": PerturbedMean,
    "LinearResponse": LinearResponse,
    "DeepSetsEndpoint": DeepSetsEndpoint,
    "MonolithicCFM": MonolithicCFM,
    "FactoredAdditiveCFM": FactoredAdditiveCFM,
}

_CONTROLS = ("NoChange", "MatchingMean", "PerturbedMean", "LinearResponse")
_NEURAL = ("DeepSetsEndpoint", "MonolithicCFM", "FactoredAdditiveCFM")


def build_baseline(name: str, d: int, k: int, r: int = 8, hidden: int | None = None,
                   depth: int = 2, seed: int = 0, cfg: TrainConfig | None = None,
                   objective: CFMObjective | None = None,
                   param_target: int | None = None, tol: float = 0.20) -> tuple:
    """Construct one baseline, solving its width for `param_target` when it is neural.

    Returns (model, matching_report). `matching_report` is None for the controls (they have
    no width to solve) and otherwise carries the solved width, the achieved count, and
    whether it landed inside `tol` of the target. When `hidden` is given it is used as-is
    and no search runs -- for reproducing a specific published configuration.
    """
    if name not in BASELINES:
        raise KeyError(f"unknown baseline {name!r}; have {sorted(BASELINES)}")
    cls = BASELINES[name]
    if name in _CONTROLS:
        return cls(d, k, seed=seed), None
    kw = dict(seed=seed, cfg=cfg, objective=objective, depth=depth)
    if name in ("DeepSetsEndpoint", "FactoredAdditiveCFM"):
        kw["r"] = int(r)

    def build(h: int):
        return cls(d, k, hidden=int(h), **kw)

    if hidden is not None:
        m = build(int(hidden))
        return m, dict(hidden=int(hidden), n_parameters=m.n_parameters(),
                       target=param_target, searched=False)
    if param_target is None:
        m = build(64)
        return m, dict(hidden=64, n_parameters=m.n_parameters(), target=None,
                       searched=False)
    sol = solve_hidden(build, int(param_target), tol=tol)
    sol["searched"] = True
    return build(sol["hidden"]), sol


# ==================================================================================
# evaluation
# ==================================================================================
def evaluate_baseline(model: Baseline, populations: Sequence, idx: Sequence[int], k: int,
                      obj: CFMObjective, cfm_batch: int = 256, cfm_reps: int = 2,
                      seed: int = 0) -> dict:
    """Per-population energy distances plus a pooled CFM loss over `idx`.

    Reported by ORDER as well as pooled, because the interesting comparison is on held-out
    COMBINATIONS specifically -- an average over a set of held-out populations that includes
    singletons can be dominated by rows where every baseline does fine.
    """
    rows = []
    for i in idx:
        p = populations[i]
        pre = np.asarray(p.pre, dtype=np.float64)
        post = np.asarray(p.post, dtype=np.float64)
        pred = model.predict(pre, p.P, p.tau)
        ed = energy_distance(pred, post)
        ed_null = energy_distance(pre, post)
        rows.append(dict(label=p.label, P=[int(q) for q in p.P], order=len(p.P),
                         tau=float(p.tau), energy_distance=ed,
                         energy_distance_null=ed_null,
                         normalised_energy_distance=(ed / ed_null if ed_null > 0
                                                     else float("nan"))))
    out = dict(per_population=rows,
               cfm_loss=cfm_loss_of(model, populations, idx, k, obj,
                                    batch=cfm_batch, reps=cfm_reps, seed=seed))
    for tag, sel in (("all", rows),
                     ("singletons", [r for r in rows if r["order"] == 1]),
                     ("combinations", [r for r in rows if r["order"] >= 2])):
        if sel:
            v = [r["normalised_energy_distance"] for r in sel]
            out[f"mean_normalised_ed_{tag}"] = float(np.mean(v))
            out[f"std_normalised_ed_{tag}"] = float(np.std(v))
            out[f"n_{tag}"] = len(sel)
    return out


# ==================================================================================
# self-test
# ==================================================================================
def selftest(d: int = 6, k: int = 4, r_true: int = 2, r_emb: int = 8, hidden: int = 64,
             n_cells: int = 200, steps: int = 800, seed: int = 0,
             held_out: Sequence[tuple] = ((0, 1),), verbose: bool = True) -> dict:
    """Fit all seven on the synthetic benchmark and report the comparison table.

    Split is leave-one-combination-out: the pair (0, 1) is held out entirely, so no model
    ever sees it at any exposure or replicate. Scored on the held-out COMBINATION, with the
    training conditions reported alongside so a reader can see which rows are fitting and
    which are generalising.

    Sanity floor, asserted not assumed: NoChange must score ~1.0 normalised (it defines the
    scale), and at least one neural baseline must beat it. A table where the neural rows
    lose to doing nothing is a wiring bug, not a result, and is flagged as a failed check.
    """
    from composefm.synthetic import SyntheticSystem     # benchmark, not part of the method

    t_start = time.time()
    # REPORTED, NOT ENFORCED. The r >= k(k-1)/2 requirement is retracted; r is a tunable
    # capacity hyperparameter and this records only where it sits relative to saturation.
    rank = check_rank_sufficiency(k, r_emb)

    system = SyntheticSystem(d=d, k=k, r=r_true, eps=0.30, noise=0.02, seed=0)
    pops = system.sample_populations(n_cells=n_cells, n_replicates=1, taus=(0.6, 1.0),
                                     n_pairs=None, n_triples=1, seed=11)
    train_idx, test_idx = system.loco_split(pops, held_out)
    ref = reference_ihcfm_parameters(d=d, k=k, r=r_emb, hidden=hidden)
    target = ref[ref["matching_target"]]

    cfg = TrainConfig(steps=int(steps), batch=128, lr=3e-3, n_int_steps=16, seed=seed)
    obj = default_objective(seed=seed, max_coupling_n=128)
    eval_obj = default_objective(seed=seed + 1, max_coupling_n=128)

    rows: dict[str, dict] = {}
    cfm_row_objectives: list[CFMObjective] = []
    for name in BASELINES:
        # each model gets its OWN objective instance so the coupling RNG stream is
        # identical for every row rather than depending on fit order
        model_obj = default_objective(seed=seed, max_coupling_n=128)
        model, match = build_baseline(name, d, k, r=r_emb, seed=seed, cfg=cfg,
                                      objective=model_obj,
                                      param_target=(target if name in _NEURAL else None))
        t0 = time.time()
        fit_rep = model.fit(pops, train_idx)
        wall = time.time() - t0
        # the CFM rows -- the ones whose CFM-loss cell is a real number -- must all be
        # training under the same coupling for that column to mean anything
        if name in ("MonolithicCFM", "FactoredAdditiveCFM"):
            cfm_row_objectives.append(model.obj)
        held = evaluate_baseline(model, pops, test_idx, k, eval_obj, seed=seed)
        train_ev = evaluate_baseline(model, pops, train_idx[:8], k, eval_obj, seed=seed)
        rows[name] = dict(
            describe=model.describe(), parameter_matching=match,
            fit_report={kk: vv for kk, vv in fit_rep.items() if kk != "objective"},
            wall_clock_s=wall,
            heldout=held, train_subset=train_ev,
            heldout_normalised_ed=held.get("mean_normalised_ed_combinations",
                                           held.get("mean_normalised_ed_all")),
            heldout_cfm_loss=held["cfm_loss"])
        if verbose:
            print(f"  {name:22s} params={model.n_parameters():7d} "
                  f"nED={rows[name]['heldout_normalised_ed']:.4f} "
                  f"cfm={rows[name]['heldout_cfm_loss']:.4f} "
                  f"{wall:.1f}s", flush=True)

    # ---- structural checks on the additive baseline -------------------------
    # FactoredAdditiveCFM's velocity MUST be exactly additive over set members and MUST
    # vanish at zero exposure. These are bit-exact properties of its construction, so a
    # non-zero residual means the gate is mis-wired and its whole row is meaningless.
    # ---- DeepSets supervision arms -----------------------------------------
    # BOTH arms are fitted and reported. The 'coupled' arm is the one that shares IHC-FM's
    # UOT coupling; it is the arm that scored worse than NoChange and drove the diagnosis
    # written up in DeepSetsEndpoint's docstring. Reporting only the arm that works would
    # hide a real finding about what minibatch couplings are for.
    ds_arms = {}
    for arm in ("paired", "coupled"):
        m = DeepSetsEndpoint(d, k, r=r_emb,
                             hidden=rows["DeepSetsEndpoint"]["parameter_matching"]["hidden"],
                             seed=seed, cfg=cfg,
                             objective=default_objective(seed=seed, max_coupling_n=128),
                             endpoint_supervision=arm)
        frep = m.fit(pops, train_idx)
        held_a = evaluate_baseline(m, pops, test_idx, k, eval_obj, seed=seed)
        train_a = evaluate_baseline(m, pops, train_idx[:8], k, eval_obj, seed=seed)
        p0 = pops[test_idx[0]]
        pr = m.predict(np.asarray(p0.pre, dtype=np.float64), p0.P, p0.tau)
        ds_arms[arm] = dict(
            heldout_normalised_ed=held_a.get("mean_normalised_ed_combinations",
                                             held_a.get("mean_normalised_ed_all")),
            train_subset_normalised_ed=train_a.get("mean_normalised_ed_all"),
            train_loss_last_decile=frep["mean_loss_last_decile"],
            n_parameters=m.n_parameters(),
            pred_sd=np.asarray(pr).std(axis=0).tolist(),
            observed_post_sd=np.asarray(p0.post, dtype=np.float64).std(axis=0).tolist())
    # displacement-norm inflation under each coupling, on one training population
    rec0 = _pop_arrays(pops, [train_idx[0]], k)[0]
    disp_norm = {}
    for cname, o in (("sinkhorn_uot", default_objective(seed=seed, max_coupling_n=128)),
                     ("paired", CFMObjective(sigma=0.0, coupling="paired", seed=seed))):
        bt = _draw_batch(rec0, o, 256, np.random.default_rng(seed))
        u = bt["target"].numpy().astype(np.float64)
        disp_norm[cname] = float(np.mean(np.linalg.norm(u, axis=1)))
    ds_diag = dict(
        arms=ds_arms,
        mean_displacement_norm_by_coupling=disp_norm,
        inflation_factor=float(disp_norm["sinkhorn_uot"]
                               / max(disp_norm["paired"], 1e-300)),
        reading=("an endpoint regressor has no integration step, so it applies the "
                 "coupling's barycentric projection directly as a map: it inherits the "
                 "displacement inflation and collapses variance. A CFM model does not, "
                 "which is why the shared-coupling requirement carries a 'where "
                 "applicable' clause and this is where it does not apply."),
        arm_selected_on="held-out normalised energy distance (distributional)",
        train_loss_across_arms_comparable=False,
        train_loss_caveat=("the arms train under different couplings, so "
                           "train_loss_last_decile is a WITHIN-ARM diagnostic only. CFM "
                           "loss tracks the identity-pair fraction (paired 1.0 vs "
                           "sinkhorn 0.229) and must never rank arms or models."),
        default_arm="paired")

    fac_obj = default_objective(seed=seed, max_coupling_n=128)
    fac, _ = build_baseline("FactoredAdditiveCFM", d, k, r=r_emb, seed=seed, cfg=cfg,
                            objective=fac_obj, param_target=target)
    fac.fit(pops, train_idx[:4])
    Zp = np.asarray(pops[train_idx[0]].pre, dtype=np.float64)[:32]
    v01 = fac.velocity(Zp, (0, 1), 1.0, 0.3)
    v0 = fac.velocity(Zp, (0,), 1.0, 0.3)
    v1 = fac.velocity(Zp, (1,), 1.0, 0.3)
    additivity_defect = float(np.max(np.abs(v01 - (v0 + v1))))
    zero_dose_defect = float(np.max(np.abs(fac.velocity(Zp, (0, 1), 0.0, 0.3))))
    perm = float(np.max(np.abs(fac.velocity(Zp, (1, 0), 1.0, 0.3) - v01)))

    # ---- assemble ------------------------------------------------------------
    ranked = sorted(rows, key=lambda n: (float("inf")
                                         if rows[n]["heldout_normalised_ed"] is None
                                         else rows[n]["heldout_normalised_ed"]))
    # independently recomputed from the distributional column, to verify the reported
    # ranking is the nED ranking and not a CFM-loss ranking that happens to agree
    def _ned(n: str):
        h = rows[n]["heldout"]
        v = h.get("mean_normalised_ed_combinations", h.get("mean_normalised_ed_all"))
        return float("inf") if v is None else v

    ranked_by_ned = sorted(rows, key=_ned)
    ranked_by_cfm = sorted(rows, key=lambda n: (float("inf")
                                                if not np.isfinite(
                                                    rows[n]["heldout_cfm_loss"])
                                                else rows[n]["heldout_cfm_loss"]))

    # the distinct coupling settings the CFM rows actually trained under, read off the
    # live objectives so every coupling claim in the report is derived rather than typed
    _cfm_coupling_kwargs = [dict(t) for t in {
        tuple(sorted(dict(coupling=o.coupling, sinkhorn_eps=o.sinkhorn_eps,
                          sinkhorn_iters=o.sinkhorn_iters,
                          unbalanced_tau=o.unbalanced_tau, sigma=o.sigma).items()))
        for o in cfm_row_objectives}]

    nochange = rows["NoChange"]["heldout_normalised_ed"]
    neural_ed = {n: rows[n]["heldout_normalised_ed"] for n in _NEURAL}
    control_ed = {n: rows[n]["heldout_normalised_ed"] for n in _CONTROLS}
    matched = {n: rows[n]["parameter_matching"] for n in _NEURAL}

    checks = {
        "nochange_normalised_ed_is_one": bool(abs(nochange - 1.0) < 0.05),
        "nochange_has_zero_parameters": bool(rows["NoChange"]["describe"]["n_parameters"] == 0),
        "some_neural_beats_nochange": bool(min(neural_ed.values()) < nochange),
        "all_neural_beat_nochange": bool(max(neural_ed.values()) < nochange),
        "all_neural_within_20pct_of_ihcfm_params":
            bool(all(m["within_tol"] for m in matched.values())),
        # NOT a pass criterion any more: the r >= k(k-1)/2 requirement is retracted, so
        # full pair rank is recorded in `rank_sufficiency` and nothing gates on it. What IS
        # checked is that no model was ranked on CFM loss.
        "table_ranked_on_distributional_metric_not_cfm_loss":
            bool(ranked_by_ned == ranked),
        # the two CFM rows must train under IDENTICAL coupling kwargs, or their CFM-loss
        # column is not even a within-coupling diagnostic. Checked on the live objects.
        "cfm_rows_share_identical_coupling_kwargs":
            bool(len(_cfm_coupling_kwargs) == 1),
        "no_nonfinite_training_steps":
            bool(all(rows[n]["fit_report"].get("skipped_nonfinite", 0) == 0
                     for n in _NEURAL)),
        "factored_velocity_exactly_additive": bool(additivity_defect == 0.0),
        "factored_zero_exposure_vanishes": bool(zero_dose_defect == 0.0),
        "factored_permutation_invariant": bool(perm == 0.0),
        "deepsets_declines_velocity": True,
        "deepsets_paired_arm_beats_coupled_arm":
            bool(ds_arms["paired"]["heldout_normalised_ed"]
                 < ds_arms["coupled"]["heldout_normalised_ed"]),
        "flow_baselines_beat_best_control":
            bool(min(rows[n]["heldout_normalised_ed"] for n in
                     ("MonolithicCFM", "FactoredAdditiveCFM"))
                 < min(control_ed.values())),
    }
    try:
        rows_ds = build_baseline("DeepSetsEndpoint", d, k, r=r_emb, seed=seed, cfg=cfg)[0]
        rows_ds.velocity(np.zeros((2, d)), (0,), 1.0, 0.0)
        checks["deepsets_declines_velocity"] = False
    except NoVelocityField:
        pass


    out = dict(
        torch=torch.__version__,
        config=dict(d=d, k=k, r_true=r_true, r_emb=r_emb, hidden_reference=hidden,
                    n_cells=n_cells, steps=steps, seed=seed,
                    held_out=[list(h) for h in held_out],
                    n_train_populations=len(train_idx),
                    n_heldout_populations=len(test_idx),
                    taus=[0.6, 1.0]),
        rank_sufficiency=rank,
        ranking_key="heldout_combination_normalised_ed",
        metric_policy=dict(
            ranking_metric="normalised energy distance (ED(pred,post)/ED(pre,post))",
            cfm_loss_role=("reported as a WITHIN-COUPLING training diagnostic only; it is "
                           "never a ranking or selection key"),
            cfm_loss_comparable_across_couplings=False,
            reason=("CFM loss tracks the identity-pair fraction a coupling produces "
                    "(paired 1.0, ot 0.795, sinkhorn 0.229, independent 0.0039), so it "
                    "rewards couplings that pair points with themselves. On the "
                    "distributional metric the coupling ranking inverts and 'ot' (1.1188) "
                    "is worse than no change."),
            # DERIVED from the objectives the CFM rows actually trained under, not
            # asserted: if the default coupling ever changes, this reports the change
            # instead of continuing to claim the old settings.
            all_cfm_rows_share_one_coupling=bool(len(_cfm_coupling_kwargs) == 1),
            shared_coupling=(dict(sorted(_cfm_coupling_kwargs[0].items()))
                             if len(_cfm_coupling_kwargs) == 1
                             else [dict(sorted(c.items()))
                                   for c in _cfm_coupling_kwargs]),
            ranking_if_cfm_loss_were_used=ranked_by_cfm,
            rankings_agree=bool(ranked_by_cfm == ranked_by_ned),
            divergence_note=("recorded to make the point concrete rather than asserted: "
                             "if these two orderings differ, at least one model's place "
                             "in the table depends on which metric is chosen, and the "
                             "distributional one is the defensible choice")),
        reference_ihcfm_parameters=ref,
        parameter_matching_target=int(target),
        shared_protocol=dict(train=cfg.describe(), objective=obj.describe(),
                             note=("every neural baseline: same CFMObjective instance "
                                   "settings, same coupling, same Adam/lr/steps/batch, "
                                   "same finite-step guards, float64 RK4 at inference")),
        table=[dict(model=n,
                    n_parameters=rows[n]["describe"]["n_parameters"],
                    param_ratio_to_ihcfm=(rows[n]["describe"]["n_parameters"]
                                          / float(target)),
                    is_flow=rows[n]["describe"]["is_flow"],
                    has_velocity=rows[n]["describe"]["has_velocity"],
                    heldout_combination_normalised_ed=rows[n]["heldout_normalised_ed"],
                    heldout_combination_ed_std=rows[n]["heldout"].get(
                        "std_normalised_ed_combinations"),
                    heldout_cfm_loss=rows[n]["heldout_cfm_loss"],
                    train_subset_normalised_ed=rows[n]["train_subset"].get(
                        "mean_normalised_ed_all"),
                    wall_clock_s=rows[n]["wall_clock_s"])
               for n in BASELINES],
        ranking_best_to_worst=ranked,
        detail=rows,
        structural_checks_factored=dict(
            additivity_defect_max_abs=additivity_defect,
            zero_exposure_defect_max_abs=zero_dose_defect,
            permutation_defect_max_abs=perm,
            note=("these are bit-exact consequences of the a_p gate and the sum over "
                  "members; a non-zero value means the gate is mis-wired")),
        deepsets_supervision_diagnosis=ds_diag,
        control_normalised_ed=control_ed,
        neural_normalised_ed=neural_ed,
        checks=checks,
        wall_clock_s=time.time() - t_start,
    )
    return out


def seed_robustness(seeds: Sequence[int] = (0, 1, 2), d: int = 6, k: int = 4,
                    r_true: int = 2, r_emb: int = 8, hidden: int = 64,
                    n_cells: int = 200, steps: int = 800,
                    held_out: Sequence[tuple] = ((0, 1),), verbose: bool = True) -> dict:
    """Re-fit the whole ladder under several training seeds.

    WHY THIS IS NOT OPTIONAL. The held-out set is ONE combination at two exposures, i.e.
    two populations, so a single-seed ordering of two neural rows is not a result -- it is
    one draw. The benchmark itself is held FIXED (same system seed, same populations, same
    split); only the model initialisation and the training/coupling RNG move, which is
    exactly the variability a Table 1 ordering claim has to survive. Rows whose intervals
    overlap are reported as ordered-but-not-separated rather than ranked.
    """
    per_seed = []
    for s in seeds:
        if verbose:
            print(f"  seed {s}", flush=True)
        r = selftest(d=d, k=k, r_true=r_true, r_emb=r_emb, hidden=hidden,
                     n_cells=n_cells, steps=steps, seed=int(s), held_out=held_out,
                     verbose=False)
        per_seed.append({row["model"]: row["heldout_combination_normalised_ed"]
                         for row in r["table"]})
    agg = {}
    for name in BASELINES:
        v = np.array([p[name] for p in per_seed], dtype=np.float64)
        agg[name] = dict(mean=float(v.mean()), std=float(v.std()),
                         min=float(v.min()), max=float(v.max()),
                         values=v.tolist())
    order = sorted(agg, key=lambda n: agg[n]["mean"])
    # separation: is the better row's worst seed still below the worse row's best seed?
    sep = []
    for a, b in zip(order[:-1], order[1:]):
        sep.append(dict(better=a, worse=b,
                        separated=bool(agg[a]["max"] < agg[b]["min"]),
                        gap_of_means=float(agg[b]["mean"] - agg[a]["mean"])))
    return dict(seeds=[int(s) for s in seeds], per_seed=per_seed, aggregate=agg,
                ranking_by_mean=order, adjacent_separation=sep,
                held_fixed="benchmark system, populations and LOCO split",
                varied="model init + training/coupling RNG",
                note=("adjacent rows with separated=False are ordered but NOT "
                      "statistically separated at this seed count and must not be "
                      "reported as a ranking"))


def loco_fold_sweep(d: int = 6, k: int = 4, r_true: int = 2, r_emb: int = 8,
                    hidden: int = 64, n_cells: int = 200, steps: int = 800,
                    seed: int = 0, verbose: bool = True) -> dict:
    """Rotate the held-out combination over EVERY pair. The comparison Table 1 must use.

    WHY. `seed_robustness` showed that a single LOCO fold cannot order these models: the
    held-out set is one pair at two exposures (n = 2 populations), and the neural rows'
    across-seed spread (+-0.22 to +-0.30 normalised) swamps the between-model gaps. Every
    adjacent pair involving a neural row failed to separate; the one pair that did
    (NoChange < PerturbedMean) is two CONTROLS, which are deterministic given the
    benchmark and have exactly zero seed variance, so it separates trivially and says
    nothing about the models under test. That is a property of the ESTIMATOR, not of the
    models, and
    the fix is more held-out populations rather than more seeds: rotating the fold over all
    k(k-1)/2 pairs yields k(k-1) held-out populations, each scored by a model that never
    saw that pair.

    Note the asymmetry this exposes and the paper must respect: the four controls are
    deterministic given the benchmark, so their numbers have no seed variance at all, while
    every neural row does. A single-fold table therefore flatters whichever neural row got
    the lucky draw. Both the pooled mean and the per-fold values are returned.
    """
    from composefm.synthetic import SyntheticSystem
    import itertools as _it

    t_start = time.time()
    rank = check_rank_sufficiency(k, r_emb)      # reported, not enforced (see the helper)
    system = SyntheticSystem(d=d, k=k, r=r_true, eps=0.30, noise=0.02, seed=0)
    pops = system.sample_populations(n_cells=n_cells, n_replicates=1, taus=(0.6, 1.0),
                                     n_pairs=None, n_triples=1, seed=11)
    ref = reference_ihcfm_parameters(d=d, k=k, r=r_emb, hidden=hidden)
    target = ref[ref["matching_target"]]
    cfg = TrainConfig(steps=int(steps), batch=128, lr=3e-3, n_int_steps=16, seed=seed)
    eval_obj = default_objective(seed=seed + 1, max_coupling_n=128)

    folds = list(_it.combinations(range(k), 2))
    per_fold: dict[str, list[float]] = {n: [] for n in BASELINES}
    per_fold_cfm: dict[str, list[float]] = {n: [] for n in BASELINES}
    for fi, ho in enumerate(folds):
        train_idx, test_idx = system.loco_split(pops, [ho])
        for name in BASELINES:
            model, _ = build_baseline(
                name, d, k, r=r_emb, seed=seed, cfg=cfg,
                objective=default_objective(seed=seed, max_coupling_n=128),
                param_target=(target if name in _NEURAL else None))
            model.fit(pops, train_idx)
            ev = evaluate_baseline(model, pops, test_idx, k, eval_obj, cfm_reps=2,
                                   seed=seed)
            per_fold[name].append(float(ev.get("mean_normalised_ed_combinations",
                                               ev.get("mean_normalised_ed_all"))))
            per_fold_cfm[name].append(float(ev["cfm_loss"]))
        if verbose:
            print(f"  fold {fi + 1}/{len(folds)} held out {ho}", flush=True)

    agg = {}
    for name in BASELINES:
        v = np.array(per_fold[name], dtype=np.float64)
        c = np.array(per_fold_cfm[name], dtype=np.float64)
        agg[name] = dict(
            mean=float(v.mean()), std=float(v.std()),
            sem=float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else float("nan"),
            min=float(v.min()), max=float(v.max()), per_fold=v.tolist(),
            cfm_mean=(float(np.nanmean(c)) if not np.all(np.isnan(c)) else float("nan")),
            cfm_per_fold=c.tolist())
    order = sorted(agg, key=lambda n: agg[n]["mean"])
    # PAIRED comparison across folds: the fold is a shared nuisance, so the paired
    # difference is the right test and it is far more powerful than comparing means with
    # independent spreads.
    sep = []
    for a, b in zip(order[:-1], order[1:]):
        da = np.array(per_fold[b]) - np.array(per_fold[a])
        sd = float(da.std(ddof=1)) if len(da) > 1 else float("nan")
        sem = sd / math.sqrt(len(da)) if len(da) > 1 else float("nan")
        sep.append(dict(better=a, worse=b, mean_paired_gap=float(da.mean()),
                        paired_gap_sem=sem,
                        folds_where_better_wins=int(np.sum(da > 0)), n_folds=len(da),
                        separated_2sem=bool(len(da) > 1 and da.mean() > 2.0 * sem)))
    return dict(config=dict(d=d, k=k, r_true=r_true, r_emb=r_emb, steps=steps, seed=seed,
                            n_cells=n_cells, taus=[0.6, 1.0]),
                folds=[list(f) for f in folds], n_folds=len(folds),
                n_heldout_populations_total=2 * len(folds),
                reference_ihcfm_parameters=ref, parameter_matching_target=int(target),
                aggregate=agg, ranking_by_mean=order, adjacent_separation_paired=sep,
                ranking_key="mean normalised energy distance over folds",
                cfm_mean_role=("within-coupling training diagnostic; NOT the ranking key "
                               "and not comparable across couplings"),
                comparison="paired across folds; fold is a shared nuisance factor",
                note=("controls are deterministic given the benchmark and have zero seed "
                      "variance; neural rows do not, so a single-fold table flatters "
                      "whichever neural row drew well"),
                wall_clock_s=time.time() - t_start)


def main(seeds: Sequence[int] = (0, 1, 2), folds: bool = True) -> dict:
    print("baselines self-test (d=6, k=4, r_true=2, r_emb=8) -- CPU, 2 threads")
    rep = selftest()
    print("\nseed robustness")
    rep["seed_robustness"] = seed_robustness(seeds=seeds)
    # THESE TWO ARE EXPECTED TO FAIL AND ARE KEPT BECAUSE THEY FAIL. Both ask for a
    # WORST-CASE guarantee on a single LOCO fold: that every neural row beats NoChange at
    # every seed, and that the best flow baseline's worst seed still beats the best
    # control's best seed. Neither holds -- the neural rows span roughly 0.65 to 1.51
    # normalised across three seeds on one fold, so their worst seed loses to doing
    # nothing. That is a real limitation of single-fold evaluation on this benchmark, not a
    # bug, and `loco_fold_sweep` is the answer to it: pooled over all six folds the same
    # models sit at 0.79/0.81/0.82 against NoChange's 1.0. Reporting these as PASS by
    # loosening them to means would hide the variance that motivates the sweep.
    rep["checks"]["neural_beat_nochange_every_seed"] = bool(all(
        rep["seed_robustness"]["aggregate"][n]["max"] < 1.0 for n in _NEURAL))
    rep["checks"]["best_flow_baseline_separated_from_best_control"] = bool(
        max(rep["seed_robustness"]["aggregate"][n]["max"]
            for n in ("FactoredAdditiveCFM",)) <
        min(rep["seed_robustness"]["aggregate"][c]["min"] for c in _CONTROLS))
    rep["expected_failures"] = {
        "neural_beat_nochange_every_seed": (
            "single-fold worst-case guarantee; fails because the neural rows' across-seed "
            "range on one fold (n=2 held-out populations) exceeds the gap to NoChange. "
            "Pooled over 6 folds all three beat NoChange on the mean."),
        "best_flow_baseline_separated_from_best_control": (
            "same cause. The controls are deterministic (zero seed variance) while the "
            "flow rows are not, so a worst-vs-best comparison on one fold cannot "
            "separate them. See loco_fold_sweep.adjacent_separation_paired for the "
            "correctly-powered paired test, which also does not separate them -- with "
            "6 folds the required n is ~270+."),
    }
    if folds:
        print("\nLOCO fold sweep (all pairs held out in turn)")
        rep["loco_fold_sweep"] = loco_fold_sweep()
        fs = rep["loco_fold_sweep"]
        # every adjacent pair that INVOLVES a neural row must fail to separate. Pairs of
        # controls separate trivially (zero seed variance) and are excluded, or this check
        # would depend on where the deterministic rows happen to land in the ordering.
        rep["checks"]["single_fold_cannot_order_neural_models"] = bool(
            not any(s["separated"] for s in rep["seed_robustness"]["adjacent_separation"]
                    if s["better"] in _NEURAL or s["worse"] in _NEURAL))
        rep["checks"]["fold_sweep_neural_mean_beats_nochange"] = bool(
            min(fs["aggregate"][n]["mean"] for n in _NEURAL)
            < fs["aggregate"]["NoChange"]["mean"])
        rep["checks"]["fold_sweep_best_flow_beats_best_control_paired"] = bool(
            fs["aggregate"]["FactoredAdditiveCFM"]["mean"]
            < min(fs["aggregate"][c]["mean"] for c in _CONTROLS))
    outp = pathlib.Path("results/baselines_selftest.json")
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(rep, indent=2, sort_keys=False))
    print(f"\nparameter target (IHC-FM total) = {rep['parameter_matching_target']}")
    for row in rep["table"]:
        print(f"  {row['model']:22s} {row['n_parameters']:7d} "
              f"({row['param_ratio_to_ihcfm']:.2f}x)  "
              f"nED={row['heldout_combination_normalised_ed']:.4f}  "
              f"cfm={row['heldout_cfm_loss']:.4f}")
    sr = rep["seed_robustness"]
    print(f"\nheld-out combination nED over seeds {sr['seeds']}:")
    for n in sr["ranking_by_mean"]:
        a = sr["aggregate"][n]
        print(f"  {n:22s} {a['mean']:.4f} +- {a['std']:.4f}  "
              f"[{a['min']:.4f}, {a['max']:.4f}]")
    print("  adjacent separation: " + ", ".join(
        f"{s['better']}<{s['worse']}={'yes' if s['separated'] else 'NO'}"
        for s in sr["adjacent_separation"]))
    if "loco_fold_sweep" in rep:
        fs = rep["loco_fold_sweep"]
        print(f"\npooled over {fs['n_folds']} LOCO folds "
              f"({fs['n_heldout_populations_total']} held-out populations):")
        for n in fs["ranking_by_mean"]:
            a = fs["aggregate"][n]
            print(f"  {n:22s} {a['mean']:.4f} +- {a['sem']:.4f} (sem)  "
                  f"[{a['min']:.4f}, {a['max']:.4f}]  cfm={a['cfm_mean']:.4f}")
        print("  paired adjacent gaps:")
        for s in fs["adjacent_separation_paired"]:
            print(f"    {s['better']:22s} < {s['worse']:22s} "
                  f"gap={s['mean_paired_gap']:+.4f}+-{s['paired_gap_sem']:.4f} "
                  f"wins {s['folds_where_better_wins']}/{s['n_folds']} "
                  f"{'SEPARATED' if s['separated_2sem'] else 'not separated'}")
    print("\nchecks:")
    for kk, vv in rep["checks"].items():
        print(f"  {'PASS' if vv else 'FAIL'}  {kk}")
    print(f"\nwrote {outp}  ({rep['wall_clock_s']:.1f}s)")
    return rep


if __name__ == "__main__":
    main()
