"""Evaluation metrics for IHC-FM, computed from SAVED PREDICTIONS on CPU.

DESIGN CONTRACT (this module never trains anything)
---------------------------------------------------
Every function here takes point clouds or per-unit scalars that were written to disk by an
experiment script, and returns numbers. Nothing here builds a model, touches torch, or
re-runs a flow. That separation is deliberate: a metric that can only be recomputed by
retraining cannot be audited, and every number in the paper has to be reproducible from
the saved predictions alone.

WHAT A "UNIT" IS, AND WHY IT IS NOT A CELL
------------------------------------------
The uncertainty functions (`grouped_bootstrap_ci`, `paired_group_difference`) demand an
explicit group label and resample GROUPS, never rows. This is not conservatism, it is the
only correct choice for either of this project's domains:

  * BIOLOGY. Cells inside one organoid population share a patient, a plate, a replicate
    and -- critically -- the SAME control draw that the prediction is pushed forward from.
    Two cells from one population are not two independent tests of the model; they are two
    reads of one test. The independent unit is the POPULATION.
  * FINANCE. Windows are built by binning a continuous trade tape, and consecutive windows
    share a state (the post-state of window i is the pre-state of window i+1), an
    order-splitting parent order, and the same slow-moving volatility regime. Overlapping
    or adjacent windows are not independent. The independent unit is a PURGED CONTIGUOUS
    BLOCK of time.

Treating rows as replicates inflates the effective sample size from n_units to n_rows and
shrinks every confidence interval by roughly sqrt(n_rows / n_units) -- on the organoid
study that is sqrt(800 * 666 / 666) ~ 28x too narrow. Any interval computed that way would
declare significance that the data does not contain. The bootstrap here therefore refuses
to run without a group label.

PAIRED COMPARISONS. `paired_group_difference` compares two models on the SAME units and
bootstraps the per-unit DIFFERENCE. Unit-level difficulty (a patient who responds weakly,
a quiet hour of tape) is the dominant variance component in both domains, and it cancels
in the pairing; comparing two independently-bootstrapped marginal intervals throws that
cancellation away and is far less powerful.

NORMALISED SCORES ALWAYS SHIP WITH THEIR DENOMINATOR
----------------------------------------------------
`energy_report` returns the absolute energy distance, the no-change denominator
ED(control, observed), and the ratio -- always all three. A normalised number alone is
uninterpretable and actively misleading when the denominator collapses: on the finance
k=4 split the denominator measured 0.154 against 0.351 on train, which inflated every
model's ratio several-fold for reasons that had nothing to do with the models. That trap
is only visible if the denominator is printed next to the ratio, so this module makes it
impossible to report one without the other.

Detecting that collapse AUTOMATICALLY needs a comparison, not a threshold: 0.154 is an
unremarkable number in isolation and alarming only beside 0.351. So `energy_report` only
raises `denominator_is_small` when it is GIVEN a reference scale, and
`flag_denominator_collapse` does the job across a set of scored records (reference =
median denominator, or the train split's). A fixed absolute cutoff would have missed the
finance case entirely.

Run `python -m composefm.metrics` for the unit tests (each metric against a case with a
known closed-form answer).
"""
from __future__ import annotations

import json
import pathlib
import time
from typing import Callable, Sequence

import numpy as np

__all__ = [
    "energy_distance", "energy_report", "flag_denominator_collapse",
    "sliced_wasserstein", "mmd_rbf",
    "delta_agreement", "residualise_shifts", "residualised_report",
    "crps_ensemble", "pit_values", "pit_coverage", "quantile_loss",
    "grouped_bootstrap_ci", "paired_group_difference",
    "score_record", "score_records", "summarise_per_unit",
]


# ======================================================================== helpers
def _as2d(X, name="X") -> np.ndarray:
    A = np.asarray(X, dtype=np.float64)
    if A.ndim == 1:
        A = A[:, None]
    if A.ndim != 2:
        raise ValueError(f"{name} must be (n, d); got shape {A.shape}")
    if A.shape[0] < 1:
        raise ValueError(f"{name} is empty")
    return A


def _subsample(A: np.ndarray, max_n: int | None, rng) -> np.ndarray:
    if max_n is None or A.shape[0] <= max_n:
        return A
    take = rng.choice(A.shape[0], size=int(max_n), replace=False)
    return A[np.sort(take)]


def _pdist_mean(A: np.ndarray, B: np.ndarray, exclude_diag: bool = False) -> float:
    """Mean Euclidean distance between rows of A and rows of B.

    Computed in blocks so a large cloud never materialises an (n x m) matrix in full;
    the organoid study calls this with n = m = 800 (5 MB) but the finance blocks can be
    several thousand windows wide.
    """
    A = np.ascontiguousarray(A)
    B = np.ascontiguousarray(B)
    n, m = A.shape[0], B.shape[0]
    if exclude_diag:
        if n != m:
            raise ValueError("exclude_diag needs square")
        if n < 2:
            return 0.0
    tot = 0.0
    block = max(1, int(4_000_000 // max(m, 1)))
    for s in range(0, n, block):
        e = min(s + block, n)
        D = np.sqrt(np.maximum(
            ((A[s:e, None, :] - B[None, :, :]) ** 2).sum(-1), 0.0))
        if exclude_diag:
            for i in range(s, e):
                D[i - s, i] = 0.0
        tot += float(D.sum())
    denom = (n * m - n) if exclude_diag else (n * m)
    return tot / max(denom, 1)


# ======================================================================== distributional
def energy_distance(X, Y, unbiased: bool = False, max_n: int | None = 2000,
                    seed: int = 0) -> float:
    """Energy distance between two point clouds.

        ED(X, Y) = 2 E||x - y|| - E||x - x'|| - E||y - y'||

    `unbiased=False` (default) is the V-statistic: all pairs including the diagonal in the
    within-cloud terms. It is what every previously-reported number in this project used
    (organoid held-out triple 0.903 / 1.283, finance 0.2866 / 2.7877), so it stays the
    default for comparability. It is >= 0 by construction but has a positive O(1/n) bias
    when the two distributions are identical.

    `unbiased=True` is the U-statistic (diagonals excluded), whose expectation is exactly
    0 for identical distributions; it can therefore go slightly negative on finite samples.
    Use it when the question is "are these two clouds the same distribution"; use the
    default when the question is "how far is my prediction from the truth", where the O(1/n)
    offset is common to every model being compared and cancels.

    The clouds are subsampled to `max_n` rows (seeded, without replacement) because the
    cost is O(n*m); at n = 2000 the three terms cost ~50 ms.
    """
    rng = np.random.default_rng(seed)
    X = _subsample(_as2d(X, "X"), max_n, rng)
    Y = _subsample(_as2d(Y, "Y"), max_n, rng)
    if X.shape[1] != Y.shape[1]:
        raise ValueError(f"dim mismatch: X {X.shape[1]} vs Y {Y.shape[1]}")
    xy = _pdist_mean(X, Y)
    xx = _pdist_mean(X, X, exclude_diag=unbiased)
    yy = _pdist_mean(Y, Y, exclude_diag=unbiased)
    return float(2.0 * xy - xx - yy)


def energy_report(pred, true, ctrl, unbiased: bool = False, max_n: int | None = 2000,
                  seed: int = 0, denominator_reference: float | None = None,
                  collapse_ratio: float = 0.5) -> dict:
    """Absolute AND normalised energy distance, ALWAYS with the denominator alongside.

    Returns
    -------
    absolute        ED(pred, true)                -- the scale-carrying number
    denominator     ED(ctrl, true)                -- the no-change predictor's score
    normalised      absolute / denominator        -- 1.0 = no better than no change
    denominator_degenerate  True only when the denominator is so close to zero that the
                    RATIO IS NUMERICALLY meaningless (< 1e-9). This is a floating-point
                    guard and nothing more.
    denominator_is_small  True when the denominator has COLLAPSED RELATIVE to a reference
                    scale, i.e. den < collapse_ratio * denominator_reference. It is None
                    when no reference is supplied, because collapse is a RELATIVE
                    statement and a single record cannot see it: the finance k=4 case in
                    the module docstring had a denominator of 0.154 against 0.351 on
                    train, which no fixed absolute threshold can distinguish from a
                    legitimately small-scale split. Pass the reference (the typical
                    denominator on the comparison split), or use
                    `flag_denominator_collapse` to compute the flag across records.
    denominator_vs_reference  den / reference, or None.

    Both the absolute and the normalised number are reported because they answer different
    questions and each one alone can mislead: the absolute distance is not comparable
    across splits with different spread, and the ratio hides the scale.
    """
    a = energy_distance(pred, true, unbiased=unbiased, max_n=max_n, seed=seed)
    den = energy_distance(ctrl, true, unbiased=unbiased, max_n=max_n, seed=seed)
    a_cl = max(a, 0.0)
    den_cl = max(den, 1e-12)
    out = dict(absolute=float(a), denominator=float(den),
               normalised=float(a_cl / den_cl),
               denominator_degenerate=bool(den < 1e-9),
               denominator_is_small=None, denominator_vs_reference=None)
    if denominator_reference is not None:
        ref = float(denominator_reference)
        if ref > 1e-12:
            out["denominator_vs_reference"] = float(den / ref)
            out["denominator_is_small"] = bool(den < float(collapse_ratio) * ref)
    return out


def flag_denominator_collapse(rows: Sequence[dict], key: str = "ed_denominator",
                              reference: float | None = None,
                              collapse_ratio: float = 0.5) -> dict:
    """Detect a COLLAPSED no-change denominator across records -- the finance k=4 trap.

    Collapse is only visible in comparison, which is why it cannot live inside
    `energy_report`: a denominator of 0.154 is unremarkable on its own and alarming only
    next to 0.351 from the split the model was trained on. This helper takes the scored
    rows, uses `reference` (typically the median denominator on the TRAIN split) or the
    median of the rows themselves, and reports which records fall below
    `collapse_ratio` * reference along with how far the ratio is consequently inflated.

    A True `collapsed` means: do not read the normalised column on those records without
    also reading `ed_absolute`, because the division is by a number that is small for
    reasons unrelated to any model. `flagged` NAMES those records -- their row positions,
    their identity columns (unit / condition / block, whichever are present) and their
    individual inflation factors -- so a caller can act on the flag rather than merely
    knowing that something somewhere collapsed.
    """
    den = np.array([float(r[key]) for r in rows if r.get(key) is not None
                    and np.isfinite(r[key])], dtype=np.float64)
    if den.size == 0:
        return dict(n_records=0, collapsed=False, flagged=[])
    ref = float(np.median(den)) if reference is None else float(reference)
    thr = float(collapse_ratio) * ref
    id_keys = ("unit", "condition", "block", "pop_index", "dose", "day")
    flagged = []
    for i, r in enumerate(rows):
        v = r.get(key)
        if v is None or not np.isfinite(v) or float(v) >= thr:
            continue
        ent = dict(row=int(i), denominator=float(v),
                   inflation=float(ref / max(float(v), 1e-12)))
        for kk in id_keys:
            if kk in r:
                ent[kk] = r[kk]
        flagged.append(ent)
    return dict(n_records=int(len(rows)), reference=ref, threshold=thr,
                collapse_ratio=float(collapse_ratio),
                median_denominator=float(np.median(den)),
                min_denominator=float(den.min()), max_denominator=float(den.max()),
                n_below_threshold=int(len(flagged)),
                fraction_below=float(len(flagged) / len(rows)),
                inflation_at_min=float(ref / max(den.min(), 1e-12)),
                collapsed=bool(len(flagged) > 0),
                flagged=flagged,
                note=("normalised scores on the records listed in `flagged` divide by a "
                      "denominator smaller than the reference split's; read ed_absolute "
                      "for those rows"))


def sliced_wasserstein(X, Y, n_slices: int = 128, seed: int = 0, p: int = 2,
                       max_n: int | None = 4000, dirs: np.ndarray | None = None) -> float:
    """Sliced Wasserstein distance, returned as the p-th ROOT (a distance, not a cost).

    Projects both clouds onto `n_slices` fixed unit directions, sorts each 1-d projection,
    and averages the 1-d W_p^p over directions; the return value is that average to the
    power 1/p. Unequal cloud sizes are handled by quantile interpolation onto a common
    grid of min(n, m) points, which is the standard empirical-quantile coupling.

    Directions are drawn from a SEEDED generator and are identical for every call with the
    same seed, so two models are scored on the same slices. Pass `dirs` (d, n_slices) to
    supply your own.

    Known answer used in the unit tests: for Y = X + t (a pure translation),
    W_2^2 along direction u is exactly (u . t)^2, so SW_2^2 = mean_i (u_i . t)^2.
    """
    rng = np.random.default_rng(seed)
    X = _subsample(_as2d(X, "X"), max_n, rng)
    Y = _subsample(_as2d(Y, "Y"), max_n, rng)
    d = X.shape[1]
    if Y.shape[1] != d:
        raise ValueError("dim mismatch")
    if dirs is None:
        g = np.random.default_rng(int(seed) + 991)
        U = g.normal(size=(d, int(n_slices)))
        U /= np.linalg.norm(U, axis=0, keepdims=True) + 1e-300
    else:
        U = np.asarray(dirs, dtype=np.float64)
        if U.shape[0] != d:
            raise ValueError(f"dirs must be (d={d}, n_slices)")
    A = np.sort(X @ U, axis=0)
    B = np.sort(Y @ U, axis=0)
    n, m = A.shape[0], B.shape[0]
    if n != m:
        q = (np.arange(min(n, m)) + 0.5) / min(n, m)
        A = np.stack([np.quantile(A[:, j], q) for j in range(A.shape[1])], axis=1)
        B = np.stack([np.quantile(B[:, j], q) for j in range(B.shape[1])], axis=1)
    cost = float(np.mean(np.abs(A - B) ** p))
    return float(cost ** (1.0 / p))


def mmd_rbf(X, Y, bandwidth: float | str = "median", unbiased: bool = True,
            max_n: int | None = 2000, seed: int = 0) -> dict:
    """Maximum mean discrepancy with a Gaussian kernel, k(a,b) = exp(-||a-b||^2 / (2 s^2)).

    bandwidth : "median" uses the median heuristic (median pairwise distance of the POOLED
                sample). Pooling matters: a bandwidth taken from one cloud only makes the
                statistic asymmetric in its arguments.
    unbiased  : U-statistic (diagonals excluded), expectation exactly 0 under H0; may be
                slightly negative on finite samples. `unbiased=False` is the V-statistic,
                which is exactly 0 when X and Y are the same array (used as a unit test)
                and strictly positive otherwise.

    Returns dict(mmd2, mmd, bandwidth) -- mmd is sqrt(max(mmd2, 0)).
    """
    rng = np.random.default_rng(seed)
    X = _subsample(_as2d(X, "X"), max_n, rng)
    Y = _subsample(_as2d(Y, "Y"), max_n, rng)
    if X.shape[1] != Y.shape[1]:
        raise ValueError("dim mismatch")
    if bandwidth == "median":
        Z = np.concatenate([X, Y], 0)
        sub = Z if Z.shape[0] <= 1000 else Z[rng.choice(Z.shape[0], 1000, replace=False)]
        D = np.sqrt(np.maximum(((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1), 0.0))
        iu = np.triu_indices(sub.shape[0], 1)
        s = float(np.median(D[iu])) if iu[0].size else 1.0
        s = s if s > 1e-12 else 1.0
    else:
        s = float(bandwidth)

    def K(A, B):
        D2 = np.maximum(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1), 0.0)
        return np.exp(-D2 / (2.0 * s * s))

    Kxx, Kyy, Kxy = K(X, X), K(Y, Y), K(X, Y)
    n, m = X.shape[0], Y.shape[0]
    if unbiased:
        if n < 2 or m < 2:
            raise ValueError("unbiased MMD needs n, m >= 2")
        txx = (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1))
        tyy = (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1))
        mmd2 = float(txx + tyy - 2.0 * Kxy.mean())
    else:
        mmd2 = float(Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean())
    return dict(mmd2=mmd2, mmd=float(np.sqrt(max(mmd2, 0.0))), bandwidth=float(s))


# ======================================================================== coordinate level
def _rank(a: np.ndarray) -> np.ndarray:
    """Average ranks (ties shared), so Spearman on tied data matches the textbook value."""
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="stable")
    r = np.empty(a.shape[0], dtype=np.float64)
    r[order] = np.arange(1, a.shape[0] + 1, dtype=np.float64)
    # average ties
    vals, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        sums = np.zeros(vals.shape[0])
        np.add.at(sums, inv, r)
        r = (sums / cnt)[inv]
    return r


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-300 or nb < 1e-300:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-300 or nb < 1e-300:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def delta_agreement(pred, true, ctrl, top_m: int | None = None) -> dict:
    """Coordinate-level (marker-level) agreement of the PREDICTED vs OBSERVED mean shift.

    Both shifts are measured against the same control cloud:
        observed  = mean(true) - mean(ctrl)
        predicted = mean(pred) - mean(ctrl)

    Reported
    --------
    pearson_delta   Pearson r over coordinates. Centres both shifts, so it is blind to a
                    shared additive offset across markers.
    spearman_delta  the same on average ranks -- robust to one dominant marker.
    r2_delta        1 - SS(observed - predicted) / SS(observed - mean(observed)). This is
                    the only one of the three that punishes a WRONG MAGNITUDE: a prediction
                    that is 2x the truth still has pearson 1.0 but R^2 < 0.
    cosine_delta    uncentred cosine -- the direction agreement of the response vector,
                    which is the quantity the composition claim is about.
    top_overlap     fraction of the top-`top_m` coordinates by |shift| that the prediction
                    also puts in its own top-`top_m`. Default top_m = max(1, d // 4).
    l2_ratio        ||predicted|| / ||observed|| -- reported so a good cosine with a
                    collapsed magnitude cannot pass unnoticed.
    """
    P = _as2d(pred, "pred"); T = _as2d(true, "true"); C = _as2d(ctrl, "ctrl")
    d = C.shape[1]
    if P.shape[1] != d or T.shape[1] != d:
        raise ValueError("dim mismatch")
    obs = T.mean(0) - C.mean(0)
    prd = P.mean(0) - C.mean(0)
    m = int(top_m) if top_m else max(1, d // 4)
    m = min(m, d)
    t_obs = set(np.argsort(-np.abs(obs))[:m].tolist())
    t_prd = set(np.argsort(-np.abs(prd))[:m].tolist())
    ss_tot = float(((obs - obs.mean()) ** 2).sum())
    ss_res = float(((obs - prd) ** 2).sum())
    return dict(
        pearson_delta=_pearson(obs, prd),
        spearman_delta=_pearson(_rank(obs), _rank(prd)),
        r2_delta=float(1.0 - ss_res / ss_tot) if ss_tot > 1e-300 else float("nan"),
        cosine_delta=_cosine(obs, prd),
        top_overlap=float(len(t_obs & t_prd) / m),
        l2_ratio=float(np.linalg.norm(prd) / max(np.linalg.norm(obs), 1e-300)),
        observed_shift_l2=float(np.linalg.norm(obs)),
        top_m=m,
    )


# ======================================================================== residualised
def residualise_shifts(obs_shifts: np.ndarray, pred_shifts: np.ndarray,
                       global_dir: np.ndarray | None = None) -> dict:
    """Regress out the mean response SHARED across conditions, then score what remains.

    Motivation. Across the organoid ladder every cytotoxic treatment pushes cells in a
    broadly similar direction (a stress/death axis), so a model that predicts the AVERAGE
    response for every condition already scores well on cosine and on normalised energy
    distance. That is not perturbation-specific skill, and a composition claim rests
    entirely on the perturbation-SPECIFIC part. This function removes the shared component
    and rescores.

    Method. Let g be the shared direction: the mean OBSERVED shift over conditions
    (unit-normalised). g is computed from the OBSERVED shifts only, never from the
    predictions, so a model cannot influence the basis it is scored in. Each shift row is
    then replaced by its component orthogonal to g:

        x_res = x - (x . g) g

    and the agreement metrics are recomputed on those residuals. A model that emits the
    same shift for every condition has pred_res ~ 0 and scores cosine ~ 0.

    Parameters
    ----------
    obs_shifts, pred_shifts : (n_cond, d)
    global_dir : optionally supply g (e.g. fitted on TRAIN conditions only, which is the
        stricter choice when the held-out set is small). Not normalised for you.

    Returns dict with the residual cosine / pearson (pooled over all conditions and also
    per condition), the fraction of observed shift norm that the shared direction carried,
    and g itself.
    """
    O = np.atleast_2d(np.asarray(obs_shifts, dtype=np.float64))
    P = np.atleast_2d(np.asarray(pred_shifts, dtype=np.float64))
    if O.shape != P.shape:
        raise ValueError(f"shape mismatch {O.shape} vs {P.shape}")
    g = O.mean(0) if global_dir is None else np.asarray(global_dir, dtype=np.float64)
    ng = float(np.linalg.norm(g))
    if ng < 1e-300:
        g_hat = np.zeros_like(g)
    else:
        g_hat = g / ng
    Ores = O - np.outer(O @ g_hat, g_hat)
    Pres = P - np.outer(P @ g_hat, g_hat)
    shared_frac = float(np.linalg.norm(O @ g_hat) / max(np.linalg.norm(O), 1e-300))
    per = [dict(cosine_res=_cosine(Ores[i], Pres[i]),
                obs_res_l2=float(np.linalg.norm(Ores[i])),
                pred_res_l2=float(np.linalg.norm(Pres[i])))
           for i in range(O.shape[0])]
    # A mean-response predictor has Pres == 0 exactly, so EVERY per-condition cosine is
    # nan (0/0). That is the correct answer, not an error: guard the mean so it reports
    # nan without a warning rather than raising on an empty slice.
    _cos_ok = [r["cosine_res"] for r in per if np.isfinite(r["cosine_res"])]
    return dict(
        cosine_res_pooled=_cosine(Ores.ravel(), Pres.ravel()),
        pearson_res_pooled=_pearson(Ores.ravel(), Pres.ravel()),
        cosine_res_mean=float(np.mean(_cos_ok)) if _cos_ok else float("nan"),
        n_conditions_with_residual=int(len(_cos_ok)),
        shared_direction_fraction=shared_frac,
        residual_norm_ratio=float(np.linalg.norm(Pres) / max(np.linalg.norm(Ores), 1e-300)),
        global_dir=[float(x) for x in g_hat],
        per_condition=per,
    )


def residualised_report(records: Sequence[dict], unbiased: bool = False,
                        max_n: int | None = 2000, seed: int = 0,
                        global_dir: np.ndarray | None = None) -> dict:
    """Residualised scoring at BOTH the shift level and the distribution level.

    Takes the same record list as `score_records` (each with pred / true / ctrl) and adds
    the one thing a per-record metric cannot see: what the other conditions look like.

    Two residualised numbers come out:
      * shift level  -- `residualise_shifts` on the per-condition mean shifts.
      * distribution level -- normalised energy distance against a MEAN-RESPONSE
        DENOMINATOR: ED(pred, true) / ED(ctrl + g_bar, true), where g_bar is the average
        observed shift over records. A value below 1 means the model beats "predict the
        average response"; the ordinary no-change ratio can be far below 1 while this one
        is above 1, and that gap is exactly the claim a composition paper must defend.
    """
    O, Pr = [], []
    for r in records:
        C = _as2d(r["ctrl"]); T = _as2d(r["true"]); P = _as2d(r["pred"])
        O.append(T.mean(0) - C.mean(0))
        Pr.append(P.mean(0) - C.mean(0))
    O = np.array(O); Pr = np.array(Pr)
    shift = residualise_shifts(O, Pr, global_dir=global_dir)
    g_bar = O.mean(0) if global_dir is None else np.asarray(global_dir, dtype=np.float64)
    rows = []
    for r in records:
        C = _as2d(r["ctrl"]); T = _as2d(r["true"]); P = _as2d(r["pred"])
        num = max(energy_distance(P, T, unbiased=unbiased, max_n=max_n, seed=seed), 0.0)
        den = max(energy_distance(C + g_bar, T, unbiased=unbiased, max_n=max_n, seed=seed),
                  1e-12)
        rows.append(dict(unit=r.get("unit"), condition=r.get("condition"),
                         ed_absolute=float(num), ed_mean_response_denominator=float(den),
                         ed_vs_mean_response=float(num / den)))
    return dict(shift_level=shift, distribution_level=rows,
                mean_response_shift=[float(x) for x in g_bar])


# ======================================================================== finance / probabilistic
def crps_ensemble(samples, y) -> float:
    """CRPS of an ensemble forecast against one scalar observation.

        CRPS(F, y) = E|X - y| - 0.5 E|X - X'|,   X, X' ~ F independent

    This is the standard energy form of the continuous ranked probability score, and it is
    the 1-d energy distance between the forecast and a point mass at y (up to the factor
    2), which is why it sits naturally next to `energy_distance` in this module.

    Known answer used in the unit tests: a deterministic forecast (all samples equal to a)
    has E|X - X'| = 0 and so CRPS = |a - y|; CRPS therefore reduces to absolute error for a
    point forecast, which is the property that makes it a proper generalisation of MAE.
    Lower is better; the units are those of y.
    """
    s = np.asarray(samples, dtype=np.float64).ravel()
    if s.size == 0:
        return float("nan")
    y = float(y)
    term1 = float(np.mean(np.abs(s - y)))
    ss = np.sort(s)
    n = ss.size
    # E|X - X'| for the empirical measure, in O(n log n). On the SORTED sample
    #   sum_{i<j} (ss[j] - ss[i]) = sum_j ss[j] * (2j - n + 1)
    # and E|X - X'| averages over all n^2 ORDERED pairs, so the i<j sum is doubled.
    # The factor of 2 here is load-bearing and was caught by the two-point unit test:
    # omitting it returned 0.75 for the closed-form value 0.5.
    j = np.arange(n, dtype=np.float64)
    pair_sum = float(np.sum(ss * (2.0 * j - n + 1.0)))
    term2 = 2.0 * pair_sum / (n * n)    # V-statistic form; -> E|X-X'| as n grows
    return float(term1 - 0.5 * term2)


def pit_values(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Probability-integral-transform values, one per observation.

    PIT_i = F_i(y_i) estimated from the ensemble as
        (#{s < y} + 0.5 #{s == y}) / n_samples
    The half-weight on ties is the standard randomised-PIT correction in expectation; it
    keeps the transform uniform for discrete or duplicated ensembles instead of piling
    mass at 0 and 1.

    `samples` is (n_obs, n_samples), `y` is (n_obs,). A calibrated forecast gives PIT
    values uniform on [0, 1].
    """
    S = np.asarray(samples, dtype=np.float64)
    if S.ndim == 1:
        S = S[None, :]
    yy = np.asarray(y, dtype=np.float64).ravel()
    if S.shape[0] != yy.shape[0]:
        raise ValueError(f"samples {S.shape} vs y {yy.shape}")
    lt = (S < yy[:, None]).sum(1)
    eq = (S == yy[:, None]).sum(1)
    return (lt + 0.5 * eq) / float(S.shape[1])


def pit_coverage(samples: np.ndarray, y: np.ndarray,
                 levels: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9)) -> dict:
    """Calibration: empirical coverage of one-sided predictive quantiles, and central bands.

    For each nominal level q, `coverage[q]` is the fraction of observations with PIT <= q.
    A calibrated forecast has coverage[q] = q for every q; coverage below the nominal level
    means the forecast is too wide on that side, above means too narrow.

    Also returned:
      central_<w>  empirical coverage of the central w band (PIT in [(1-w)/2, (1+w)/2]).
      ks_uniform   Kolmogorov-Smirnov distance of the PIT sample from uniform -- one
                   scalar summary of miscalibration.
    """
    pit = pit_values(samples, y)
    out = dict(levels=[float(q) for q in levels],
               coverage=[float(np.mean(pit <= q)) for q in levels],
               mean_pit=float(np.mean(pit)))
    for w in (0.5, 0.9):
        lo, hi = (1.0 - w) / 2.0, (1.0 + w) / 2.0
        out[f"central_{w:g}"] = float(np.mean((pit >= lo) & (pit <= hi)))
    p = np.sort(pit)
    n = p.size
    ecdf = (np.arange(1, n + 1)) / n
    out["ks_uniform"] = float(np.max(np.abs(ecdf - p))) if n else float("nan")
    out["n_obs"] = int(n)
    return out


def quantile_loss(samples: np.ndarray, y: np.ndarray,
                  quantiles: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9)) -> dict:
    """Pinball (quantile) loss of ensemble quantiles against the observations.

        L_q(y, yhat) = q * (y - yhat)      if y >= yhat
                       (1 - q) * (yhat - y) otherwise

    Minimised in expectation by the true q-quantile, so it scores a specific tail rather
    than the whole distribution -- which is what a risk application asks for. Returns the
    per-quantile mean loss and their average.
    """
    S = np.asarray(samples, dtype=np.float64)
    if S.ndim == 1:
        S = S[None, :]
    yy = np.asarray(y, dtype=np.float64).ravel()
    per = []
    for q in quantiles:
        yhat = np.quantile(S, q, axis=1)
        diff = yy - yhat
        per.append(float(np.mean(np.where(diff >= 0, q * diff, (q - 1.0) * diff))))
    return dict(quantiles=[float(q) for q in quantiles], loss=per,
                mean_loss=float(np.mean(per)))


# ======================================================================== uncertainty
def grouped_bootstrap_ci(values, groups, stat: Callable[[np.ndarray], float] = np.mean,
                         n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Bootstrap CI that resamples INDEPENDENT UNITS (groups), never rows.

    WHY THE GROUP LABEL IS MANDATORY. The rows handed to this function are usually not
    independent: cells share a population (same patient, same plate, same control draw),
    and time windows share a volatility regime and even a state vector with their
    neighbours. The bootstrap's validity rests on resampling i.i.d. units, so the unit has
    to be the population or the purged time block -- the thing that could have been drawn
    again independently. Resampling rows instead treats n_rows correlated reads as n_rows
    independent tests and shrinks the interval by roughly sqrt(n_rows / n_groups), which on
    the organoid study is a factor of ~28. There is no default group label for that reason:
    the caller must say what an independent unit is.

    Procedure: draw n_groups groups WITH replacement, concatenate their rows, apply `stat`.
    Percentile interval at level 1 - alpha.

    Returns dict(point, lo, hi, se, n_groups, n_rows, n_boot, alpha, group_means).
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    g = np.asarray(groups).ravel()
    if v.shape[0] != g.shape[0]:
        raise ValueError(f"values {v.shape} and groups {g.shape} must align")
    if v.size == 0:
        raise ValueError("no values")
    uniq, inv = np.unique(g, return_inverse=True)
    idx_by_group = [np.nonzero(inv == j)[0] for j in range(uniq.size)]
    rng = np.random.default_rng(seed)
    point = float(stat(v))
    boots = np.empty(int(n_boot), dtype=np.float64)
    ng = uniq.size
    for b in range(int(n_boot)):
        pick = rng.integers(0, ng, size=ng)
        rows = np.concatenate([idx_by_group[j] for j in pick])
        boots[b] = stat(v[rows])
    lo, hi = np.quantile(boots, [alpha / 2.0, 1.0 - alpha / 2.0])
    gm = [float(np.mean(v[ix])) for ix in idx_by_group]
    return dict(point=point, lo=float(lo), hi=float(hi), se=float(np.std(boots, ddof=1)),
                n_groups=int(ng), n_rows=int(v.size), n_boot=int(n_boot),
                alpha=float(alpha), group_means=gm,
                groups=[str(u) for u in uniq])


def paired_group_difference(values_a, values_b, groups, n_boot: int = 2000,
                            alpha: float = 0.05, seed: int = 0,
                            name_a: str = "a", name_b: str = "b") -> dict:
    """PAIRED per-unit difference between two models, bootstrapped over units.

    Both models must be scored on the SAME units (same `groups` vector, aligned row for
    row). The statistic is the mean over units of (unit mean of a) - (unit mean of b), and
    the bootstrap resamples UNITS. Pairing is the point: in both of this project's domains
    the dominant variance component is unit difficulty -- a weakly-responding patient, a
    quiet hour of tape -- and it is common to both models, so it cancels in the per-unit
    difference. Comparing two separately-bootstrapped marginal intervals discards that
    cancellation and will call a real difference insignificant.

    Sign convention: `mean_diff` is a - b, so with a loss-type metric (energy distance,
    where lower is better) a NEGATIVE mean_diff means model `a` is better.

    Also returns a sign test: `n_units_a_lower` / `n_units` and the exact two-sided
    binomial p-value, which needs no distributional assumption at all and is the honest
    fallback when the per-unit differences are heavy-tailed.
    """
    a = np.asarray(values_a, dtype=np.float64).ravel()
    b = np.asarray(values_b, dtype=np.float64).ravel()
    g = np.asarray(groups).ravel()
    if not (a.shape == b.shape == g.shape):
        raise ValueError(f"shapes must match: {a.shape}, {b.shape}, {g.shape}")
    uniq, inv = np.unique(g, return_inverse=True)
    ng = uniq.size
    da = np.array([float(np.mean(a[inv == j])) for j in range(ng)])
    db = np.array([float(np.mean(b[inv == j])) for j in range(ng)])
    diff = da - db
    rng = np.random.default_rng(seed)
    boots = np.empty(int(n_boot))
    for i in range(int(n_boot)):
        pick = rng.integers(0, ng, size=ng)
        boots[i] = float(np.mean(diff[pick]))
    lo, hi = np.quantile(boots, [alpha / 2.0, 1.0 - alpha / 2.0])
    n_lower = int(np.sum(diff < 0))
    n_eff = int(np.sum(diff != 0))
    p_sign = _binom_two_sided(n_lower, n_eff)
    return dict(name_a=name_a, name_b=name_b,
                mean_a=float(np.mean(da)), mean_b=float(np.mean(db)),
                mean_diff=float(np.mean(diff)), lo=float(lo), hi=float(hi),
                se=float(np.std(boots, ddof=1)),
                excludes_zero=bool((lo > 0) or (hi < 0)),
                n_units=int(ng), n_units_a_lower=n_lower, n_units_tied=int(ng - n_eff),
                sign_test_p=float(p_sign),
                per_unit_diff=[float(x) for x in diff],
                units=[str(u) for u in uniq], n_boot=int(n_boot), alpha=float(alpha))


def _binom_two_sided(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial p-value (sum of outcomes no more likely than observed)."""
    if n <= 0:
        return float("nan")
    from math import comb
    pmf = np.array([comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(n + 1)])
    obs = pmf[int(k)]
    return float(pmf[pmf <= obs + 1e-15].sum())


# ======================================================================== drivers
def score_record(rec: dict, seed: int = 0, max_n: int | None = 2000,
                 n_slices: int = 128, top_m: int | None = None) -> dict:
    """Full metric suite for ONE saved prediction record.

    A record is dict(unit=str, condition=str, pred=(n,d), true=(m,d), ctrl=(n0,d)).
    `unit` is the independent unit (population id / purged block id) and is carried
    through to the output so the bootstrap can group on it.
    """
    P, T, C = _as2d(rec["pred"]), _as2d(rec["true"]), _as2d(rec["ctrl"])
    er = energy_report(P, T, C, max_n=max_n, seed=seed)
    row = dict(unit=rec.get("unit"), condition=rec.get("condition"),
               n_pred=int(P.shape[0]), n_true=int(T.shape[0]), n_ctrl=int(C.shape[0]),
               ed_absolute=er["absolute"], ed_denominator=er["denominator"],
               ed_normalised=er["normalised"],
               ed_denominator_degenerate=er["denominator_degenerate"],
               sw_pred=sliced_wasserstein(P, T, n_slices=n_slices, seed=seed, max_n=max_n),
               sw_denominator=sliced_wasserstein(C, T, n_slices=n_slices, seed=seed,
                                                 max_n=max_n),
               mmd_pred=mmd_rbf(P, T, max_n=max_n, seed=seed)["mmd2"],
               mmd_denominator=mmd_rbf(C, T, max_n=max_n, seed=seed)["mmd2"])
    row["sw_normalised"] = float(row["sw_pred"] / max(row["sw_denominator"], 1e-12))
    row.update(delta_agreement(P, T, C, top_m=top_m))
    # Provenance columns pass through untouched so downstream code can PAIR rows across
    # models by identity (`pop_index` / `block`) instead of trusting list order.
    for extra in ("dose", "replicate", "order", "split", "pop_index", "block",
                  "day", "window_index"):
        if extra in rec:
            row[extra] = rec[extra]
    return row


def score_records(records: Sequence[dict], seed: int = 0, max_n: int | None = 2000,
                  n_slices: int = 128, top_m: int | None = None) -> list[dict]:
    """`score_record` over a list, preserving order."""
    return [score_record(r, seed=seed, max_n=max_n, n_slices=n_slices, top_m=top_m)
            for r in records]


def summarise_per_unit(rows: Sequence[dict], metrics: Sequence[str] | None = None,
                       group_key: str = "unit", n_boot: int = 1000,
                       seed: int = 0) -> dict:
    """Grouped-bootstrap summary of per-record metric rows.

    Each row is one record; `group_key` names the column holding the independent unit.
    When one record IS one unit the bootstrap is an ordinary unit bootstrap; when several
    records share a unit (several doses of one population, several windows of one block)
    the unit mean is taken first, which is the correct hierarchical reduction.
    """
    if not rows:
        return {}
    if metrics is None:
        metrics = ["ed_normalised", "ed_absolute", "ed_denominator", "sw_normalised",
                   "mmd_pred", "cosine_delta", "pearson_delta", "spearman_delta",
                   "r2_delta", "top_overlap", "l2_ratio"]
    groups = np.array([str(r.get(group_key)) for r in rows])
    out = {}
    for m in metrics:
        vals = np.array([float(r[m]) if r.get(m) is not None and np.isfinite(r[m])
                         else np.nan for r in rows])
        ok = np.isfinite(vals)
        if ok.sum() == 0:
            continue
        out[m] = grouped_bootstrap_ci(vals[ok], groups[ok], n_boot=n_boot, seed=seed)
        out[m].pop("group_means", None)
        out[m].pop("groups", None)
    out["n_records"] = len(rows)
    out["n_units"] = int(np.unique(groups).size)
    return out


# ======================================================================== unit tests
def _t(name, ok, **kw):
    return dict(test=name, passed=bool(ok), **{k: (float(v) if isinstance(v, (int, float,
                np.floating)) else v) for k, v in kw.items()})


def selftest(verbose: bool = True) -> dict:
    """Unit-test every metric against a case with a KNOWN closed-form answer."""
    res = []
    rng = np.random.default_rng(0)

    # ---- energy distance -------------------------------------------------------
    # (1) two point masses in R^d separated by c: E|x-x'| = E|y-y'| = 0, so ED = 2c.
    a = np.zeros((50, 3)); b = np.zeros((60, 3)); b[:, 0] = 2.5
    ed = energy_distance(a, b)
    res.append(_t("energy_point_masses_equals_2c", abs(ed - 5.0) < 1e-12,
                  got=ed, expected=5.0))
    # (2) identical arrays -> V-statistic exactly 0 (all three terms equal).
    X = rng.normal(size=(200, 4))
    res.append(_t("energy_identical_arrays_zero", abs(energy_distance(X, X)) < 1e-12,
                  got=energy_distance(X, X)))
    # (3) unbiased estimator on two independent draws from ONE distribution: E = 0.
    #     Averaged over 40 replicates the mean must sit within a few SE of 0, and the
    #     V-statistic must be systematically LARGER (its O(1/n) bias is positive).
    u, vv = [], []
    for s in range(40):
        g = np.random.default_rng(100 + s)
        A = g.normal(size=(80, 3)); B = g.normal(size=(80, 3))
        u.append(energy_distance(A, B, unbiased=True))
        vv.append(energy_distance(A, B, unbiased=False))
    mu, se = float(np.mean(u)), float(np.std(u, ddof=1) / np.sqrt(len(u)))
    res.append(_t("energy_unbiased_mean_zero_under_H0", abs(mu) < 4 * se,
                  mean=mu, se=se))
    res.append(_t("energy_V_stat_biased_above_U_stat", np.mean(vv) > np.mean(u),
                  v_mean=float(np.mean(vv)), u_mean=mu))
    # (4) permutation of coordinates (an orthogonal map) leaves it invariant, bit-close.
    perm = rng.permutation(4)
    Y = rng.normal(size=(150, 4)) + 0.7
    e0 = energy_distance(X, Y); e1 = energy_distance(X[:, perm], Y[:, perm])
    res.append(_t("energy_permutation_invariant", abs(e0 - e1) < 1e-10,
                  delta=abs(e0 - e1)))
    # (5) translation invariance: shifting BOTH clouds equally changes nothing.
    sh = np.array([1.0, -2.0, 0.5, 3.0])
    e2 = energy_distance(X + sh, Y + sh)
    res.append(_t("energy_translation_invariant", abs(e0 - e2) < 1e-10,
                  delta=abs(e0 - e2)))

    # ---- energy_report ---------------------------------------------------------
    ctrl = rng.normal(size=(120, 3))
    true = ctrl + np.array([1.0, 0.0, 0.0])
    rep_perfect = energy_report(true, true, ctrl)
    res.append(_t("energy_report_perfect_prediction_zero",
                  abs(rep_perfect["normalised"]) < 1e-12 and rep_perfect["denominator"] > 0,
                  normalised=rep_perfect["normalised"], denom=rep_perfect["denominator"]))
    rep_nochange = energy_report(ctrl, true, ctrl)
    res.append(_t("energy_report_nochange_prediction_is_one",
                  abs(rep_nochange["normalised"] - 1.0) < 1e-12,
                  normalised=rep_nochange["normalised"]))
    res.append(_t("energy_report_always_ships_denominator",
                  set(("absolute", "denominator", "normalised")) <= set(rep_nochange),
                  keys=sorted(rep_nochange)))
    # The collapse flag must fire on the ACTUAL finance k=4 case the docstring cites
    # (denominator 0.154 against a train reference of 0.351) and stay quiet on the
    # exposure split (0.320 against the same reference). A fixed 1e-3 absolute cutoff --
    # what this originally shipped -- reports False for both, which is the bug this test
    # exists to prevent regressing.
    ref_train = 0.351
    fake = [dict(ed_denominator=0.154, unit="k4", condition="k=4"),
            dict(ed_denominator=0.320, unit="expo", condition="exposure"),
            dict(ed_denominator=0.351, unit="tr", condition="train")]
    fl = flag_denominator_collapse(fake, reference=ref_train, collapse_ratio=0.5)
    res.append(_t("denominator_collapse_flag_fires_on_finance_k4_case",
                  fl["collapsed"] and fl["n_below_threshold"] == 1
                  and abs(fl["inflation_at_min"] - ref_train / 0.154) < 1e-9,
                  n_below=fl["n_below_threshold"], threshold=fl["threshold"],
                  inflation=fl["inflation_at_min"]))
    # the flag must NAME the offending record, not just count it
    res.append(_t("denominator_collapse_flag_names_the_records",
                  len(fl["flagged"]) == 1 and fl["flagged"][0]["row"] == 0
                  and fl["flagged"][0]["unit"] == "k4"
                  and abs(fl["flagged"][0]["inflation"] - ref_train / 0.154) < 1e-9,
                  flagged=fl["flagged"]))
    fl_ok = flag_denominator_collapse([dict(ed_denominator=0.320),
                                       dict(ed_denominator=0.351)],
                                      reference=ref_train, collapse_ratio=0.5)
    res.append(_t("denominator_collapse_flag_quiet_on_healthy_split",
                  not fl_ok["collapsed"], n_below=fl_ok["n_below_threshold"]))
    # and energy_report's own flag is None without a reference (collapse is relative),
    # True with one -- never a silent False.
    rep_noref = energy_report(ctrl, true, ctrl)
    rep_ref = energy_report(ctrl, true, ctrl,
                            denominator_reference=10.0 * rep_nochange["denominator"])
    res.append(_t("energy_report_flag_none_without_reference_true_with",
                  rep_noref["denominator_is_small"] is None
                  and rep_ref["denominator_is_small"] is True
                  and not rep_noref["denominator_degenerate"],
                  noref=str(rep_noref["denominator_is_small"]),
                  withref=str(rep_ref["denominator_is_small"])))

    # ---- sliced Wasserstein ----------------------------------------------------
    # pure translation: W2^2 along u is exactly (u.t)^2 -> SW2^2 = mean_i (u_i.t)^2.
    d, ns = 4, 64
    g2 = np.random.default_rng(7 + 991)
    U = g2.normal(size=(d, ns)); U /= np.linalg.norm(U, axis=0, keepdims=True)
    t_vec = np.array([0.8, -0.3, 0.0, 1.2])
    Z = rng.normal(size=(300, d))
    sw = sliced_wasserstein(Z, Z + t_vec, n_slices=ns, seed=7)
    expect = float(np.mean((U.T @ t_vec) ** 2) ** 0.5)
    res.append(_t("sliced_w2_translation_closed_form", abs(sw - expect) < 1e-10,
                  got=sw, expected=expect))
    res.append(_t("sliced_w2_identical_zero", sliced_wasserstein(Z, Z, seed=7) < 1e-12,
                  got=sliced_wasserstein(Z, Z, seed=7)))
    # unequal cloud sizes must still work (quantile coupling), and stay near the truth
    sw_uneq = sliced_wasserstein(Z, (Z + t_vec)[:137], n_slices=ns, seed=7)
    res.append(_t("sliced_w2_unequal_sizes_close", abs(sw_uneq - expect) < 0.08,
                  got=sw_uneq, expected=expect))

    # ---- MMD -------------------------------------------------------------------
    m_same = mmd_rbf(X, X, unbiased=False)
    res.append(_t("mmd_V_stat_identical_arrays_zero", abs(m_same["mmd2"]) < 1e-12,
                  got=m_same["mmd2"]))
    mu2, vals = None, []
    for s in range(30):
        g = np.random.default_rng(500 + s)
        A = g.normal(size=(120, 3)); B = g.normal(size=(120, 3))
        vals.append(mmd_rbf(A, B, unbiased=True)["mmd2"])
    mu2 = float(np.mean(vals)); se2 = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
    res.append(_t("mmd_unbiased_mean_zero_under_H0", abs(mu2) < 4 * se2,
                  mean=mu2, se=se2))
    far = mmd_rbf(X, X + 5.0, unbiased=True)
    res.append(_t("mmd_far_clouds_near_upper_bound", far["mmd2"] > 0.5, got=far["mmd2"]))
    mp0 = mmd_rbf(X, Y, unbiased=True)["mmd2"]
    mp1 = mmd_rbf(X[:, perm], Y[:, perm], unbiased=True)["mmd2"]
    res.append(_t("mmd_permutation_invariant", abs(mp0 - mp1) < 1e-10, delta=abs(mp0 - mp1)))

    # ---- delta agreement -------------------------------------------------------
    C = rng.normal(size=(200, 8))
    obs_shift = np.array([2.0, -1.0, 0.5, 0.0, 1.5, -0.2, 0.1, -3.0])
    T = C + obs_shift
    da_perfect = delta_agreement(T, T, C)
    res.append(_t("delta_perfect_r2_one_cosine_one",
                  abs(da_perfect["r2_delta"] - 1.0) < 1e-12
                  and abs(da_perfect["cosine_delta"] - 1.0) < 1e-12
                  and abs(da_perfect["top_overlap"] - 1.0) < 1e-12,
                  r2=da_perfect["r2_delta"], cos=da_perfect["cosine_delta"],
                  top=da_perfect["top_overlap"]))
    # doubled shift: cosine and pearson stay 1, R^2 must FALL and l2_ratio = 2.
    da_double = delta_agreement(C + 2 * obs_shift, T, C)
    ss_tot = float(((obs_shift - obs_shift.mean()) ** 2).sum())
    r2_expect = 1.0 - float((obs_shift ** 2).sum()) / ss_tot
    res.append(_t("delta_doubled_shift_cosine_one_r2_known",
                  abs(da_double["cosine_delta"] - 1.0) < 1e-12
                  and abs(da_double["r2_delta"] - r2_expect) < 1e-10
                  and abs(da_double["l2_ratio"] - 2.0) < 1e-12,
                  cos=da_double["cosine_delta"], r2=da_double["r2_delta"],
                  r2_expected=r2_expect, l2=da_double["l2_ratio"]))
    # sign-flipped shift: cosine = -1 exactly.
    da_flip = delta_agreement(C - obs_shift, T, C)
    res.append(_t("delta_flipped_shift_cosine_minus_one",
                  abs(da_flip["cosine_delta"] + 1.0) < 1e-12, cos=da_flip["cosine_delta"]))
    # no-change prediction: predicted shift is 0 -> cosine undefined (nan), l2_ratio 0.
    da_zero = delta_agreement(C, T, C)
    res.append(_t("delta_nochange_gives_nan_cosine_zero_norm",
                  np.isnan(da_zero["cosine_delta"]) and da_zero["l2_ratio"] < 1e-12,
                  cos=str(da_zero["cosine_delta"]), l2=da_zero["l2_ratio"]))
    # coordinate permutation applied to ALL THREE clouds leaves every scalar unchanged
    pe = rng.permutation(8)
    da_perm = delta_agreement((C + 2 * obs_shift)[:, pe], T[:, pe], C[:, pe])
    res.append(_t("delta_permutation_invariant",
                  abs(da_perm["cosine_delta"] - da_double["cosine_delta"]) < 1e-12
                  and abs(da_perm["r2_delta"] - da_double["r2_delta"]) < 1e-12
                  and abs(da_perm["top_overlap"] - da_double["top_overlap"]) < 1e-12,
                  cos_delta=abs(da_perm["cosine_delta"] - da_double["cosine_delta"])))
    # spearman is rank based: a monotone but nonlinear distortion keeps it at 1 while
    # pearson drops. This is the property that makes reporting both worthwhile.
    mono = np.sign(obs_shift) * np.abs(obs_shift) ** 1.7
    da_mono = delta_agreement(C + mono, T, C)
    res.append(_t("spearman_survives_monotone_distortion_pearson_drops",
                  abs(da_mono["spearman_delta"] - 1.0) < 1e-9
                  and da_mono["pearson_delta"] < 0.999,
                  spearman=da_mono["spearman_delta"], pearson=da_mono["pearson_delta"]))

    # ---- residualised ----------------------------------------------------------
    # Build 5 conditions sharing a large common direction plus a small specific part.
    dd = 6
    shared = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    spec = rng.normal(size=(5, dd)) * 0.3
    O = shared[None, :] + spec
    r_perfect = residualise_shifts(O, O)
    res.append(_t("residual_perfect_prediction_cosine_one",
                  abs(r_perfect["cosine_res_pooled"] - 1.0) < 1e-10,
                  got=r_perfect["cosine_res_pooled"]))
    # A model that predicts the AVERAGE response for every condition must score ~0:
    # its residual is (mean - mean.g g) which is identically zero for the mean itself.
    P_avg = np.tile(O.mean(0), (5, 1))
    r_avg = residualise_shifts(O, P_avg)
    res.append(_t("residual_mean_response_predictor_scores_zero",
                  r_avg["residual_norm_ratio"] < 1e-12,
                  ratio=r_avg["residual_norm_ratio"],
                  cos=str(r_avg["cosine_res_pooled"])))
    # the shared direction should carry most of the observed norm here (by construction)
    res.append(_t("residual_shared_fraction_large_by_construction",
                  r_perfect["shared_direction_fraction"] > 0.8,
                  frac=r_perfect["shared_direction_fraction"]))
    # residual is orthogonal to g by construction
    gh = np.array(r_perfect["global_dir"])
    Ores = O - np.outer(O @ gh, gh)
    res.append(_t("residual_orthogonal_to_global_dir",
                  float(np.abs(Ores @ gh).max()) < 1e-12,
                  max_abs_proj=float(np.abs(Ores @ gh).max())))

    # ---- CRPS / PIT / quantile loss --------------------------------------------
    # point forecast -> CRPS = |a - y| exactly
    cp = crps_ensemble(np.full(64, 3.0), 5.0)
    res.append(_t("crps_point_forecast_equals_abs_error", abs(cp - 2.0) < 1e-12, got=cp))
    # a two-point forecast {0, 2} with y = 1: E|X-y| = 1, E|X-X'| (V-stat, n=2) = 1
    #  -> CRPS = 1 - 0.5 = 0.5
    cp2 = crps_ensemble(np.array([0.0, 2.0]), 1.0)
    res.append(_t("crps_two_point_closed_form", abs(cp2 - 0.5) < 1e-12, got=cp2))
    # (3) INDEPENDENT closed form: for a Gaussian forecast N(mu, sigma) and observation y,
    #     CRPS = sigma * [ z(2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi) ],  z = (y - mu)/sigma.
    #     This validates the estimator against an analytic value derived outside this
    #     module, not just against its own algebra.
    from math import erf, exp, pi, sqrt
    gz = np.random.default_rng(4242)
    ens_g = gz.normal(0.0, 2.0, 200_000)
    for y_ in (0.0, 1.5, -3.0):
        z_ = (y_ - 0.0) / 2.0
        Phi = 0.5 * (1.0 + erf(z_ / sqrt(2.0)))
        phi = exp(-0.5 * z_ * z_) / sqrt(2.0 * pi)
        want = 2.0 * (z_ * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / sqrt(pi))
        got = crps_ensemble(ens_g, y_)
        res.append(_t(f"crps_gaussian_closed_form_y{y_:g}", abs(got - want) < 0.01,
                      got=got, expected=want))
    # sharper (correct) forecast must beat a wider one on a well-centred observation
    gs = np.random.default_rng(3)
    tight = gs.normal(0, 1, 4000); wide = gs.normal(0, 4, 4000)
    c_t = float(np.mean([crps_ensemble(tight, y) for y in gs.normal(0, 1, 60)]))
    c_w = float(np.mean([crps_ensemble(wide, y) for y in gs.normal(0, 1, 60)]))
    res.append(_t("crps_prefers_correctly_sharp_forecast", c_t < c_w,
                  tight=c_t, wide=c_w))
    # PIT: put y exactly at the empirical q-quantile -> PIT = q
    S = np.arange(1, 101, dtype=float)[None, :]          # 1..100
    p60 = pit_values(S, np.array([60.5]))[0]             # 60 below, none equal
    res.append(_t("pit_equals_target_quantile", abs(p60 - 0.60) < 1e-12, got=p60))
    # calibrated ensemble -> coverage ~ nominal
    gs2 = np.random.default_rng(11)
    ens = gs2.normal(size=(3000, 200)); ys = gs2.normal(size=3000)
    cov = pit_coverage(ens, ys)
    dev = max(abs(c - q) for c, q in zip(cov["coverage"], cov["levels"]))
    res.append(_t("pit_calibrated_coverage_matches_nominal", dev < 0.03,
                  max_deviation=dev, ks=cov["ks_uniform"]))
    # over-narrow ensemble -> PIT piles in the tails -> central band under-covers
    narrow = gs2.normal(size=(3000, 200)) * 0.25
    cov_n = pit_coverage(narrow, ys)
    res.append(_t("pit_detects_overconfident_forecast",
                  cov_n["central_0.9"] < 0.75 and cov_n["ks_uniform"] > 0.1,
                  central90=cov_n["central_0.9"], ks=cov_n["ks_uniform"]))
    # quantile loss: median forecast, y above it -> loss = 0.5 * (y - yhat)
    ql = quantile_loss(np.array([[0.0, 1.0, 2.0]]), np.array([5.0]), quantiles=(0.5,))
    res.append(_t("quantile_loss_median_closed_form", abs(ql["loss"][0] - 2.0) < 1e-12,
                  got=ql["loss"][0], expected=2.0))
    ql9 = quantile_loss(np.array([[0.0, 1.0, 2.0]]), np.array([5.0]), quantiles=(0.9,))
    # np.quantile(.,0.9) of [0,1,2] = 1.8 ; y - yhat = 3.2 ; loss = 0.9*3.2 = 2.88
    res.append(_t("quantile_loss_upper_tail_closed_form",
                  abs(ql9["loss"][0] - 0.9 * (5.0 - 1.8)) < 1e-12, got=ql9["loss"][0]))

    # ---- grouped bootstrap -----------------------------------------------------
    # 20 groups x 40 rows, group effect dominant. Row bootstrap (wrong) must be far
    # narrower than the group bootstrap (right) -- this is the quantitative statement
    # the module docstring makes.
    gb = np.random.default_rng(21)
    ng_, nr_ = 20, 40
    geff = gb.normal(0, 1.0, ng_)
    vals_ = np.concatenate([geff[j] + gb.normal(0, 0.1, nr_) for j in range(ng_)])
    grp_ = np.concatenate([np.full(nr_, f"g{j}") for j in range(ng_)])
    ci_grp = grouped_bootstrap_ci(vals_, grp_, n_boot=800, seed=1)
    ci_row = grouped_bootstrap_ci(vals_, np.arange(vals_.size), n_boot=800, seed=1)
    wid_g = ci_grp["hi"] - ci_grp["lo"]; wid_r = ci_row["hi"] - ci_row["lo"]
    res.append(_t("row_bootstrap_understates_interval_vs_group",
                  wid_r < 0.35 * wid_g,
                  width_group=wid_g, width_row=wid_r, ratio=wid_r / wid_g))
    # coverage check: the group bootstrap must cover the true mean of the group effects
    res.append(_t("group_bootstrap_covers_truth",
                  ci_grp["lo"] <= float(np.mean(geff)) <= ci_grp["hi"],
                  lo=ci_grp["lo"], truth=float(np.mean(geff)), hi=ci_grp["hi"]))
    # constant values -> zero-width interval, point exactly the constant
    ci_c = grouped_bootstrap_ci(np.full(50, 2.5), np.repeat(np.arange(10), 5),
                                n_boot=200, seed=0)
    res.append(_t("bootstrap_constant_values_zero_width",
                  abs(ci_c["point"] - 2.5) < 1e-12 and (ci_c["hi"] - ci_c["lo"]) < 1e-12,
                  point=ci_c["point"], width=ci_c["hi"] - ci_c["lo"]))
    # misaligned inputs must RAISE rather than silently broadcast
    try:
        grouped_bootstrap_ci(np.zeros(10), np.zeros(9))
        raised = False
    except ValueError:
        raised = True
    res.append(_t("bootstrap_rejects_misaligned_groups", raised))

    # ---- paired difference -----------------------------------------------------
    # b = a + 0.3 for every row: per-unit difference is exactly -0.3 with ZERO variance,
    # so the bootstrap interval must collapse onto -0.3 and the sign test must be maximal.
    pd_ = paired_group_difference(vals_, vals_ + 0.3, grp_, n_boot=400, seed=2)
    res.append(_t("paired_constant_offset_exact",
                  abs(pd_["mean_diff"] + 0.3) < 1e-12
                  and (pd_["hi"] - pd_["lo"]) < 1e-12
                  and pd_["n_units_a_lower"] == ng_,
                  mean_diff=pd_["mean_diff"], width=pd_["hi"] - pd_["lo"],
                  n_lower=pd_["n_units_a_lower"], p=pd_["sign_test_p"]))
    # identical models -> difference exactly 0, interval contains 0, sign test p = 1
    pd_same = paired_group_difference(vals_, vals_, grp_, n_boot=200, seed=2)
    res.append(_t("paired_identical_models_zero",
                  abs(pd_same["mean_diff"]) < 1e-14 and not pd_same["excludes_zero"]
                  and pd_same["n_units_tied"] == ng_,
                  mean_diff=pd_same["mean_diff"], p=pd_same["sign_test_p"]))
    # antisymmetry: swapping the arguments flips the sign of everything
    pd_swap = paired_group_difference(vals_ + 0.3, vals_, grp_, n_boot=400, seed=2)
    res.append(_t("paired_antisymmetric",
                  abs(pd_swap["mean_diff"] + pd_["mean_diff"]) < 1e-12,
                  a=pd_["mean_diff"], b=pd_swap["mean_diff"]))
    # pairing is more powerful: a small consistent offset buried in large unit variance
    # is detected by the paired interval but NOT by two marginal intervals.
    a_ = vals_; b_ = vals_ + 0.05
    pd_small = paired_group_difference(a_, b_, grp_, n_boot=800, seed=5)
    ci_a = grouped_bootstrap_ci(a_, grp_, n_boot=800, seed=5)
    ci_b = grouped_bootstrap_ci(b_, grp_, n_boot=800, seed=5)
    marginals_overlap = not (ci_a["hi"] < ci_b["lo"] or ci_b["hi"] < ci_a["lo"])
    res.append(_t("pairing_detects_offset_marginals_cannot",
                  pd_small["excludes_zero"] and marginals_overlap,
                  paired_lo=pd_small["lo"], paired_hi=pd_small["hi"],
                  marginals_overlap=marginals_overlap))
    # exact binomial sign test against a hand value: k=0, n=5 -> 2 * 0.5^5 = 0.0625
    res.append(_t("sign_test_exact_small_n", abs(_binom_two_sided(0, 5) - 0.0625) < 1e-12,
                  got=_binom_two_sided(0, 5)))

    # ---- driver round trip -----------------------------------------------------
    recs = []
    for j in range(6):
        gj = np.random.default_rng(900 + j)
        Cj = gj.normal(size=(120, 6))
        sj = shared + gj.normal(size=6) * 0.3
        recs.append(dict(unit=f"pop{j}", condition=f"cond{j % 3}",
                         ctrl=Cj, true=Cj + sj, pred=Cj + sj))
    rows = score_records(recs, seed=0, max_n=200)
    summ = summarise_per_unit(rows, n_boot=200, seed=0)
    res.append(_t("driver_perfect_records_score_zero_ed",
                  max(abs(r["ed_normalised"]) for r in rows) < 1e-10
                  and summ["n_units"] == 6,
                  max_ed=max(abs(r["ed_normalised"]) for r in rows),
                  n_units=summ["n_units"]))
    rr = residualised_report(recs, max_n=200)
    res.append(_t("residualised_report_perfect_is_one",
                  abs(rr["shift_level"]["cosine_res_pooled"] - 1.0) < 1e-8,
                  got=rr["shift_level"]["cosine_res_pooled"]))
    # and the mean-response predictor must LOSE on the residualised distributional score
    recs_avg = [dict(r, pred=r["ctrl"] + np.array([np.mean([
        (rr2["true"].mean(0) - rr2["ctrl"].mean(0))[i] for rr2 in recs])
        for i in range(6)])) for r in recs]
    rr_avg = residualised_report(recs_avg, max_n=200)
    mean_resid = float(np.mean([x["ed_vs_mean_response"]
                                for x in rr_avg["distribution_level"]]))
    res.append(_t("mean_response_predictor_scores_one_on_residualised",
                  abs(mean_resid - 1.0) < 0.05, got=mean_resid))

    n_pass = sum(1 for r in res if r["passed"])
    out = dict(n_tests=len(res), n_passed=n_pass, n_failed=len(res) - n_pass, tests=res,
               run_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    if verbose:
        for r in res:
            print(("  PASS  " if r["passed"] else "  FAIL  ") + r["test"]
                  + "".join(f"  {k}={v}" for k, v in r.items()
                            if k not in ("test", "passed")))
        print(f"\n{n_pass}/{len(res)} metric unit tests passed")
    return out


if __name__ == "__main__":
    rep = selftest()
    p = pathlib.Path(__file__).resolve().parents[2] / "results"
    p.mkdir(exist_ok=True)
    json.dump(rep, open(p / "metrics_selftest.json", "w"), indent=2)
