"""Chart-covariance harness -- TWO tests that must never be conflated.

Why two tests
-------------
"Our method is chart-covariant" can mean two very different things, and the weaker one is
routinely reported as if it were the stronger.

  TEST A -- OPERATOR / IMPLEMENTATION covariance.
      Take ONE trained model. Its prediction is the flow of a velocity field. Compute the
      prediction two ways: (a) integrate natively in the original chart, then map the
      answer forward with psi; (b) push the FIELD forward with the Jacobian,
      Xtilde(w, t) = Dpsi(psi^{-1}(w)) . X(psi^{-1}(w), t), and integrate that in the new
      chart from psi(z0). In exact arithmetic these are identically equal -- that is the
      covariance of ODE flows under a diffeomorphism. So Test A measures whether the
      transformation law is IMPLEMENTED correctly, and its floor is integrator error.
      It says nothing whatsoever about learning.

      Test A ships with TWO control arms, which measure different things:
        * NAIVE: apply the trained field in the new chart with NO Jacobian factor (what a
          chart-agnostic model effectively does). Establishes that the test has power.
          It is large for affine charts too, so it does NOT distinguish valid from
          invalid charts.
        * FROZEN-JACOBIAN: linearise psi at the trajectory start and hold Dpsi fixed.
          This is exact iff D2psi == 0, so it isolates the SECOND-order part of the chart
          change and is the arm that separates genuinely nonlinear charts from affine
          ones.

      MEASURED PRECISION LIMIT. Test A's achievable error is set by the precision of the
      model's velocity evaluation, not by the transformation. In float32 the error does
      still decrease under refinement, but at a badly degraded RATE -- measured
      3.51e-08 (n=8), 1.70e-08 (n=32), 5.60e-09 (n=128), 1.02e-09 (n=512), i.e. only
      ~1.4-2.1x per doubling where RK4 should give ~16x -- so it behaves as a soft
      precision wall rather than a hard floor, and reaches only ~1e-9 at 512 steps.
      Casting the same trained weights to float64 for inference restores the expected
      convergence: 2.88e-09 (n=8), 1.12e-11 (n=32), 4.46e-14 (n=128), 6.92e-15 (n=512).
      The placeholder therefore evaluates in float64 by default (`f64_inference`), and a
      Phase-2 model that evaluates in float32 will floor Test A at ~1e-8 for that reason
      and not because its transformation is wrong -- diagnose before concluding.

  TEST B -- OPTIMISATION / IDENTIFIABILITY covariance.
      Train the model INDEPENDENTLY on the same observations expressed in two charts,
      map both prediction sets back into one common chart, and measure the discrepancy.
      This asks whether two separately optimised models learn the SAME function. It is
      strictly stronger than Test A: an exactly covariant operator can still be paired
      with a loss, a parameterisation and an optimiser whose joint minimiser is
      chart-dependent, in which case Test A passes at 1e-14 and Test B fails.

      Test B is meaningless without a NOISE FLOOR. Two models retrained in the SAME chart
      with different initialisation seeds already disagree by some amount. We measure that
      and report the RATIO discrepancy/floor. A ratio near 1 means the chart change costs
      no more than reshuffling the random seed; a large ratio means chart-dependence.

Chart validity
--------------
The chart MUST be genuinely nonlinear. For an affine chart (PCA, whitening, any linear
map) the second derivative D2psi vanishes identically, every composition variant is exact,
and the test measures nothing -- this was verified earlier in the project (affine chart:
S+2G 1.396e-16, S-2G 1.149e-16, S alone 4.842e-17, i.e. even the WRONG forms pass). So
this harness ASSERTS a nonzero Hessian scale for every chart it uses, and additionally
runs an affine chart as a labelled NEGATIVE CONTROL to demonstrate that affine charts
cannot detect the defect.

Model interface (model-agnostic, IHC-FM plugs in at Phase 2)
------------------------------------------------------------
`factory(d, k, seed, **kw) -> model` where model provides:

    model.fit(populations, idx)               -> None    (trains on populations[i] for i in idx)
    model.velocity(Z, P, tau, t) -> (n, d)              instantaneous velocity at flow time t
    model.predict(Z0, P, tau)    -> (n, d)              flow of `velocity` from t=0 to t=1

`velocity` is the only structural requirement: Test A pushes it forward, and `predict` is
just its integral (the harness supplies `integrate_field` if a model wants to reuse it).
A model that cannot expose an instantaneous velocity is not testable by Test A, which is
itself an informative fact about that model class.

The placeholder below is a small conditional-flow-matching MLP -- present ONLY so the
harness can be shown to run end to end. It is not a baseline and its numbers are not a
result about IHC-FM.
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys
import time
from typing import Callable, Sequence

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from composefm.synthetic import NonlinearChart, Population, SyntheticSystem  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results"


# ==================================================================================
# integration of a possibly time-dependent field
# ==================================================================================
def integrate_field(field: Callable[[np.ndarray, float], np.ndarray], Z0: np.ndarray,
                    n_steps: int = 32) -> np.ndarray:
    """RK4 over flow time t in [0, 1] for a field f(Z, t). float64."""
    Z = np.array(Z0, dtype=np.float64, copy=True)
    h = 1.0 / int(n_steps)
    t = 0.0
    for _ in range(int(n_steps)):
        k1 = field(Z, t)
        k2 = field(Z + 0.5 * h * k1, t + 0.5 * h)
        k3 = field(Z + 0.5 * h * k2, t + 0.5 * h)
        k4 = field(Z + h * k3, t + h)
        Z = Z + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += h
    return Z


# ==================================================================================
# placeholder model -- a small conditional flow-matching MLP
# ==================================================================================
class PlaceholderCFM:
    """Conditional flow matching on the paired coupling. Deliberately minimal.

    velocity(z, P, tau, t) = MLP([z_std, multihot(P) * tau, tau, t]) in standardised
    coordinates, rescaled back. Trained on z_t = (1-t) z0 + t z1 against target z1 - z0.

    `standardise` exists because it is a genuine confound for Test B: without it, a chart
    that rescales the data changes the effective learning rate and input conditioning, so
    a Test B discrepancy would partly measure optimiser conditioning rather than
    chart-dependence of the minimiser. Standardisation is itself an AFFINE, data-estimated
    re-chart, so it removes the affine part of the chart change but not the nonlinear part
    -- exactly the confound we want gone. Both settings are reported.
    """

    def __init__(self, d: int, k: int, seed: int = 0, hidden: int = 64, n_steps: int = 600,
                 lr: float = 3e-3, batch: int = 256, standardise: bool = True,
                 n_int_steps: int = 32, f64_inference: bool = True):
        import torch
        self.torch = torch
        self.d, self.k = int(d), int(k)
        self.n_steps, self.lr, self.batch = int(n_steps), float(lr), int(batch)
        self.standardise, self.n_int_steps = bool(standardise), int(n_int_steps)
        # f64_inference casts the trained weights to float64 for evaluation only.
        # MEASURED REASON: with float32 inference the Test A pushforward error converges
        # far more slowly than the integrator's O(h^4) -- 3.51e-08 (n=8), 1.70e-08 (n=32),
        # 5.60e-09 (n=128), 1.02e-09 (n=512), only ~1.4-2.1x per doubling -- because
        # network evaluation noise, not the integrator, dominates. In float64 the same
        # model gives 1.12e-11 (n=32) and 4.46e-14 (n=128). Test A is a statement about
        # the TRANSFORMATION, so it must not be limited by the placeholder's inference
        # precision. Training stays float32.
        self.f64_inference = bool(f64_inference)
        self.seed = int(seed)
        torch.manual_seed(self.seed)
        torch.set_num_threads(2)
        n_in = d + k + 2
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_in, hidden), torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden), torch.nn.Tanh(),
            torch.nn.Linear(hidden, d))
        self.mu = np.zeros(d)
        self.sd = np.ones(d)
        self.final_loss = float("nan")

    # -- condition encoding -------------------------------------------------
    def _cond(self, P: Sequence[int], tau: float, n: int) -> np.ndarray:
        c = np.zeros((n, self.k + 1), dtype=np.float64)
        for p in P:
            c[:, int(p)] = float(tau)
        c[:, self.k] = float(tau)
        return c

    # -- training -----------------------------------------------------------
    def fit(self, populations: Sequence[Population], idx: Sequence[int]) -> None:
        torch = self.torch
        Z0 = np.concatenate([populations[i].pre for i in idx], axis=0)
        Z1 = np.concatenate([populations[i].post for i in idx], axis=0)
        C = np.concatenate([self._cond(populations[i].P, populations[i].tau,
                                       populations[i].pre.shape[0]) for i in idx], axis=0)
        if self.standardise:
            self.mu = Z0.mean(axis=0)
            self.sd = Z0.std(axis=0) + 1e-8
        z0 = torch.tensor((Z0 - self.mu) / self.sd, dtype=torch.float32)
        z1 = torch.tensor((Z1 - self.mu) / self.sd, dtype=torch.float32)
        cc = torch.tensor(C, dtype=torch.float32)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        gen = torch.Generator().manual_seed(self.seed + 1)
        n = z0.shape[0]
        for _ in range(self.n_steps):
            b = torch.randint(0, n, (min(self.batch, n),), generator=gen)
            t = torch.rand((b.shape[0], 1), generator=gen)
            zt = (1.0 - t) * z0[b] + t * z1[b]
            target = z1[b] - z0[b]
            pred = self.net(torch.cat([zt, cc[b], t], dim=1))
            loss = ((pred - target) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.final_loss = float(loss.item())
        if self.f64_inference:
            self.net = self.net.double()

    # -- inference ----------------------------------------------------------
    def velocity(self, Z: np.ndarray, P: Sequence[int], tau: float, t: float) -> np.ndarray:
        torch = self.torch
        Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
        n = Z.shape[0]
        zs = (Z - self.mu) / self.sd
        x = np.concatenate([zs, self._cond(P, tau, n),
                            np.full((n, 1), float(t))], axis=1)
        dt = torch.float64 if self.f64_inference else torch.float32
        with torch.no_grad():
            v = self.net(torch.tensor(x, dtype=dt)).numpy().astype(np.float64)
        return v * self.sd            # chain rule back to raw coordinates

    def predict(self, Z0: np.ndarray, P: Sequence[int], tau: float) -> np.ndarray:
        return integrate_field(lambda Z, t: self.velocity(Z, P, tau, t), Z0,
                               n_steps=self.n_int_steps)


def placeholder_factory(d: int, k: int, seed: int = 0, **kw):
    return PlaceholderCFM(d=d, k=k, seed=seed, **kw)


# ==================================================================================
# TEST A -- operator / implementation covariance
# ==================================================================================
def test_A(model, chart: NonlinearChart, Z0: np.ndarray, P: Sequence[int], tau: float,
           n_int_steps: int = 128) -> dict:
    """One trained model, two routes to the same answer in the new chart.

    route_native  : integrate in chart 1, then map with psi         -> psi(Phi_z(z0))
    route_pushfwd : push the field forward with Dpsi, integrate in chart 2 from psi(z0)
    route_naive   : integrate the field in chart 2 with NO Jacobian factor (control)

    Errors are relative to the mean displacement norm in chart 2, so they are scale-free.
    """
    Z0 = np.atleast_2d(np.asarray(Z0, dtype=np.float64))
    W0 = chart.psi(Z0)

    native = chart.psi(integrate_field(lambda Z, t: model.velocity(Z, P, tau, t), Z0,
                                       n_steps=n_int_steps))

    def pushed(W: np.ndarray, t: float) -> np.ndarray:
        Z = chart.inv(W)
        return np.einsum('nij,nj->ni', chart.jac(Z), model.velocity(Z, P, tau, t))

    pushfwd = integrate_field(pushed, W0, n_steps=n_int_steps)
    naive = integrate_field(lambda W, t: model.velocity(W, P, tau, t), W0,
                            n_steps=n_int_steps)

    # FROZEN-JACOBIAN arm -- isolates the SECOND-order part of the chart change.
    # Use Dpsi and psi^{-1} linearised at the trajectory's starting point and held fixed:
    #   Z ~ Z0 + J0^{-1}(W - W0),   Xtilde(W) ~ J0 . X(Z).
    # For an AFFINE chart Dpsi is constant and psi^{-1} is exactly this linear map, so
    # this arm equals the pushforward arm to integrator precision. For a nonlinear chart
    # it is wrong at O(D2psi). This is the arm that distinguishes charts which can
    # exercise the covariance defect from charts which cannot -- the naive (no-Jacobian)
    # arm cannot make that distinction, because an affine chart still rotates and scales
    # and so breaks the no-Jacobian route just as badly.
    J0 = chart.jac(Z0)
    J0inv = np.linalg.inv(J0)

    def frozen(W: np.ndarray, t: float) -> np.ndarray:
        Z = Z0 + np.einsum('nij,nj->ni', J0inv, W - W0)
        return np.einsum('nij,nj->ni', J0, model.velocity(Z, P, tau, t))

    frozen_pred = integrate_field(frozen, W0, n_steps=n_int_steps)

    scale = float(np.mean(np.linalg.norm(pushfwd - W0, axis=1))) + 1e-300
    err = float(np.mean(np.linalg.norm(native - pushfwd, axis=1))) / scale
    err_naive = float(np.mean(np.linalg.norm(native - naive, axis=1))) / scale
    err_frozen = float(np.mean(np.linalg.norm(native - frozen_pred, axis=1))) / scale
    return dict(err_pushforward=err, err_naive=err_naive, err_frozen_jacobian=err_frozen,
                power_ratio=err_naive / max(err, 1e-300),
                second_order_ratio=err_frozen / max(err, 1e-300),
                displacement_scale=scale,
                hessian_scale=chart.hessian_scale(Z0))


# ==================================================================================
# TEST B -- optimisation / identifiability covariance
# ==================================================================================
def test_B(factory: Callable, system: SyntheticSystem, chart: NonlinearChart,
           pops: Sequence[Population], train_idx: Sequence[int],
           eval_conditions: Sequence[tuple], Z0: np.ndarray, tau: float,
           seeds: tuple = (0, 1), model_kw: dict | None = None) -> dict:
    """Retrain independently in both charts, compare in the common (original) chart.

    Discrepancy and noise floor are computed on IDENTICAL prediction targets, so the
    ratio is the quantity to read. The floor is two same-chart models differing only in
    initialisation/minibatch seed.
    """
    model_kw = dict(model_kw or {})
    d, k = system.d, system.k
    pops_w = [p.mapped(chart) for p in pops]

    # chart 1, two seeds -> noise floor; chart 2, first seed -> chart discrepancy
    m_z = []
    for s in seeds:
        m = factory(d, k, seed=s, **model_kw)
        m.fit(pops, train_idx)
        m_z.append(m)
    m_w = factory(d, k, seed=seeds[0], **model_kw)
    m_w.fit(pops_w, train_idx)

    W0 = chart.psi(Z0)
    rows = []
    for P in eval_conditions:
        pz = m_z[0].predict(Z0, P, tau)                       # chart-1 model, chart-1 answer
        pz_alt = m_z[1].predict(Z0, P, tau)                   # same chart, different seed
        pw = m_w.predict(W0, P, tau)                          # chart-2 model, chart-2 answer
        pw_back = chart.inv(pw)                               # mapped back to chart 1
        scale = float(np.mean(np.linalg.norm(pz - Z0, axis=1))) + 1e-300
        disc = float(np.mean(np.linalg.norm(pz - pw_back, axis=1))) / scale
        floor = float(np.mean(np.linalg.norm(pz - pz_alt, axis=1))) / scale
        truth = system.flow(P, Z0, tau)
        err_z = float(np.mean(np.linalg.norm(pz - truth, axis=1))) / scale
        err_w = float(np.mean(np.linalg.norm(pw_back - truth, axis=1))) / scale
        rows.append(dict(condition="+".join(f"X{p}" for p in P), order=len(P),
                         discrepancy=disc, seed_floor=floor,
                         ratio=disc / max(floor, 1e-300),
                         err_vs_truth_chart1=err_z, err_vs_truth_chart2=err_w))
    agg = dict(
        discrepancy_mean=float(np.mean([r["discrepancy"] for r in rows])),
        seed_floor_mean=float(np.mean([r["seed_floor"] for r in rows])),
        err_vs_truth_chart1_mean=float(np.mean([r["err_vs_truth_chart1"] for r in rows])),
        err_vs_truth_chart2_mean=float(np.mean([r["err_vs_truth_chart2"] for r in rows])),
        final_loss_chart1=m_z[0].final_loss, final_loss_chart2=m_w.final_loss)
    agg["ratio_mean"] = agg["discrepancy_mean"] / max(agg["seed_floor_mean"], 1e-300)

    # POWER PRECONDITION. A Test B "pass" is only meaningful if the model actually
    # learned something. If err_vs_truth ~ 1 the model predicts roughly no displacement,
    # both charts are equally wrong, and the discrepancy is small for a trivial reason.
    # We therefore report a skill score against the zero-displacement predictor and flag
    # the test as uninformative when skill is negligible. This is a precondition on the
    # MODEL, not on covariance -- with the placeholder it is expected to be weak.
    agg["skill_vs_zero_displacement"] = float(1.0 - agg["err_vs_truth_chart1_mean"])
    agg["test_B_informative"] = bool(agg["skill_vs_zero_displacement"] > 0.5)
    return dict(rows=rows, agg=agg)


# ==================================================================================
# sweep
# ==================================================================================
def run(k_values: Sequence[int] = (2, 3, 4), strengths: Sequence[float] = (0.1, 0.3, 0.6),
        kinds: Sequence[str] = ("tanh", "cubic"), d: int = 6, r: int = 2, eps: float = 0.15,
        anisotropy: float = 5.0, n_cells: int = 150, n_states: int = 120,
        # 2500 steps is the MEASURED budget at which the placeholder acquires enough skill
        # for Test B to be informative. Skill vs the zero-displacement predictor:
        # 150 steps -0.120, 600 steps +0.833, 2500 steps +0.917, 6000 steps +0.931.
        # Below ~600 the model predicts essentially no displacement and Test B is vacuous.
        tau: float = 1.0, n_steps: int = 2500, hidden: int = 64,
        include_affine_control: bool = True, seed: int = 0, verbose: bool = True) -> dict:
    model_kw = dict(hidden=hidden, n_steps=n_steps)
    report: dict = dict(config=dict(
        k_values=list(k_values), strengths=list(strengths), kinds=list(kinds), d=d, r=r,
        eps=eps, anisotropy=anisotropy, n_cells=n_cells, n_states=n_states, tau=tau,
        train_steps=n_steps, hidden=hidden, model="PlaceholderCFM"),
        test_A=[], test_B=[], chart_validity=[])

    for k in k_values:
        system = SyntheticSystem(d=d, k=max(k, 2), r=min(r, max(k, 2)), eps=eps,
                                 anisotropy=anisotropy, seed=seed)
        pops = system.sample_populations(n_cells=n_cells, taus=(0.5, 1.0), n_triples=1,
                                         seed=11)
        train_idx = list(range(len(pops)))
        Z0 = system.control(n_states, seed=77)
        # one trained model per k, reused across charts for Test A
        mA = placeholder_factory(d, system.k, seed=0, **model_kw)
        mA.fit(pops, train_idx)

        # evaluation conditions: the highest-order sets available (where interaction lives)
        eval_conditions = [c for c in system.conditions(n_triples=1) if len(c) >= 2][:3]
        P_A = eval_conditions[-1]

        kinds_here = list(kinds) + (["affine"] if include_affine_control else [])
        for kind in kinds_here:
            for s in strengths:
                chart = NonlinearChart(d, strength=s, kind=kind, seed=101)
                hs = chart.hessian_scale(Z0)
                invres = chart.inverse_residual(Z0)
                report["chart_validity"].append(dict(kind=kind, strength=s,
                                                     hessian_scale=hs,
                                                     inverse_residual=invres,
                                                     is_valid_nonlinear=bool(hs > 0.0)))
                if kind != "affine" and hs <= 0.0:
                    raise AssertionError(
                        f"chart {kind} s={s} has zero Hessian -- invalid as a covariance test")

                a = test_A(mA, chart, Z0, P_A, tau)
                a.update(k=k, kind=kind, strength=s, condition="+".join(f"X{p}" for p in P_A),
                         negative_control=(kind == "affine"), inverse_residual=invres)
                report["test_A"].append(a)
                if verbose:
                    print(f"[A] k={k} {kind:>6} s={s:.2f} | pushfwd {a['err_pushforward']:.3e}"
                          f" | naive {a['err_naive']:.3e} | power {a['power_ratio']:.2e}"
                          f" | |D2psi| {hs:.2e}"
                          + ("  <- NEGATIVE CONTROL" if kind == "affine" else ""))

        # Test B is the expensive arm (3 trainings per chart) -- one nonlinear kind, all s
        for kind in [kk for kk in kinds if kk != "affine"][:1]:
            for s in strengths:
                chart = NonlinearChart(d, strength=s, kind=kind, seed=101)
                for std in (True, False):
                    b = test_B(placeholder_factory, system, chart, pops, train_idx,
                               eval_conditions, Z0, tau, seeds=(0, 1),
                               model_kw=dict(model_kw, standardise=std))
                    b["agg"].update(k=k, kind=kind, strength=s, standardise=std,
                                    hessian_scale=chart.hessian_scale(Z0))
                    report["test_B"].append(dict(agg=b["agg"], rows=b["rows"]))
                    if verbose:
                        g = b["agg"]
                        print(f"[B] k={k} {kind:>6} s={s:.2f} std={int(std)} | "
                              f"discrepancy {g['discrepancy_mean']:.3e} | "
                              f"seed floor {g['seed_floor_mean']:.3e} | "
                              f"ratio {g['ratio_mean']:.2f} | "
                              f"err_truth c1 {g['err_vs_truth_chart1_mean']:.3f} "
                              f"c2 {g['err_vs_truth_chart2_mean']:.3f}")
    return report


def trend_analysis(report: dict) -> dict:
    """Is the Test B discrepancy SYSTEMATIC in chart nonlinearity, or just noise?

    This exists because the "ratio <= 2x seed noise" criterion can PASS while hiding a
    clean monotone dependence on chart strength. Two things are measured:

      * corr(|D2psi|, discrepancy) and the regression slope. The intercept is the
        chart-independent part; a positive slope means the chart change itself is doing
        work.
      * a SIGN TEST on accuracy: in how many configurations is the chart-2-trained model
        WORSE against ground truth than the chart-1-trained model? Under true
        chart-independence this should be ~50%. A near-sweep is evidence that the
        optimisation prefers the original chart, independently of the discrepancy scale.

    MEASURED with the placeholder on the default sweep, over ALL 18 Test B rows (both
    standardise settings, which is what this function aggregates): corr 0.9246,
    slope 2.1785, intercept 1.854e-02; chart-2 worse in 17/18 configurations; mean
    accuracy degradation +0.0020 (s=0.1), +0.0089 (s=0.3), +0.0221 (s=0.6). Restricting
    to the standardise=True subset gives corr 0.9424 and slope 2.1929 -- the same
    conclusion, but do not confuse the two row sets when quoting. So the placeholder's
    optimisation is NOT chart-independent even though its operator is covariant to
    8.2e-11 -- exactly the Test A / Test B distinction this harness exists to draw.
    """
    B = [x["agg"] for x in report["test_B"]]
    if len(B) < 3:
        return dict(insufficient_data=True, n=len(B))
    hs = np.array([b["hessian_scale"] for b in B], dtype=np.float64)
    disc = np.array([b["discrepancy_mean"] for b in B], dtype=np.float64)
    slope, intercept = np.polyfit(hs, disc, 1)
    worse = int(sum(1 for b in B
                    if b["err_vs_truth_chart2_mean"] > b["err_vs_truth_chart1_mean"]))
    by_s = {}
    for s in sorted({b["strength"] for b in B}):
        rows = [b for b in B if b["strength"] == s]
        by_s[str(s)] = dict(
            discrepancy=float(np.mean([b["discrepancy_mean"] for b in rows])),
            seed_floor=float(np.mean([b["seed_floor_mean"] for b in rows])),
            err_chart1=float(np.mean([b["err_vs_truth_chart1_mean"] for b in rows])),
            err_chart2=float(np.mean([b["err_vs_truth_chart2_mean"] for b in rows])),
        )
        by_s[str(s)]["ratio"] = (by_s[str(s)]["discrepancy"]
                                 / max(by_s[str(s)]["seed_floor"], 1e-300))
        by_s[str(s)]["accuracy_degradation"] = (by_s[str(s)]["err_chart2"]
                                               - by_s[str(s)]["err_chart1"])
    return dict(
        corr_hessian_discrepancy=float(np.corrcoef(hs, disc)[0, 1]),
        regression_slope=float(slope), regression_intercept=float(intercept),
        chart2_worse_count=worse, n_configs=len(B),
        chart2_worse_fraction=float(worse / len(B)),
        by_strength=by_s,
        systematic_chart_dependence=bool(
            np.corrcoef(hs, disc)[0, 1] > 0.8 and slope > 0.0 and worse >= 0.8 * len(B)),
    )


def summarise(report: dict) -> dict:
    A = [r for r in report["test_A"] if not r["negative_control"]]
    Aaff = [r for r in report["test_A"] if r["negative_control"]]
    B = report["test_B"]
    Bstd = [r["agg"] for r in B if r["agg"]["standardise"]]
    Braw = [r["agg"] for r in B if not r["agg"]["standardise"]]
    s = dict(
        test_A_max_pushforward_err=float(max(r["err_pushforward"] for r in A)),
        test_A_min_naive_err=float(min(r["err_naive"] for r in A)),
        test_A_min_power_ratio=float(min(r["power_ratio"] for r in A)),
        test_A_affine_control_max_naive_err=(
            float(max(r["err_naive"] for r in Aaff)) if Aaff else None),
        # the discriminating quantity: second-order (frozen-Jacobian) defect
        test_A_min_frozen_jac_err_nonlinear=float(min(r["err_frozen_jacobian"] for r in A)),
        test_A_affine_control_max_frozen_jac_err=(
            float(max(r["err_frozen_jacobian"] for r in Aaff)) if Aaff else None),
        test_A_min_second_order_ratio_nonlinear=float(
            min(r["second_order_ratio"] for r in A)),
        test_B_max_ratio_standardised=(float(max(r["ratio_mean"] for r in Bstd))
                                       if Bstd else None),
        test_B_max_ratio_raw=(float(max(r["ratio_mean"] for r in Braw)) if Braw else None),
        test_B_max_discrepancy_standardised=(
            float(max(r["discrepancy_mean"] for r in Bstd)) if Bstd else None),
        test_B_mean_seed_floor=(float(np.mean([r["seed_floor_mean"] for r in Bstd]))
                                if Bstd else None),
        test_B_max_skill_vs_zero_displacement=(
            float(max(r["skill_vs_zero_displacement"] for r in Bstd)) if Bstd else None),
        test_B_any_informative=(bool(any(r["test_B_informative"] for r in Bstd))
                                if Bstd else None),
        trend=trend_analysis(report),
    )
    # Pre-registered pass conditions.
    s["verdict"] = dict(
        # A: the transformation is implemented right (near integrator floor) AND the test
        #    has power (the naive arm is orders of magnitude worse).
        test_A_implementation_correct=bool(s["test_A_max_pushforward_err"] < 1e-6),
        test_A_has_power=bool(s["test_A_min_power_ratio"] > 1e3),
        # B: chart change costs no more than ~2x a seed change. This is a claim about the
        #    PLACEHOLDER model here; Phase 2 re-runs it for IHC-FM.
        test_B_chart_within_2x_seed_noise=(
            bool(s["test_B_max_ratio_standardised"] <= 2.0)
            if s["test_B_max_ratio_standardised"] is not None else None),
        # ... but the above is only meaningful if the model has predictive skill. With
        # the placeholder it typically does NOT, so Test B is reported as UNINFORMATIVE
        # rather than as a pass. Do not quote the ratio as evidence when this is False.
        test_B_precondition_model_has_skill=s["test_B_any_informative"],
        # HONEST QUALIFIER. Even when the ratio criterion passes, a positive result here
        # means the discrepancy tracks chart nonlinearity systematically, so the model's
        # OPTIMISATION is chart-dependent and the ratio pass must not be quoted as
        # "chart-independent training". Reported as its own line, not folded into a pass.
        test_B_systematic_chart_dependence_detected=s["trend"][
            "systematic_chart_dependence"],
        # The affine chart must be BLIND to the second-order defect: with D2psi == 0 the
        # frozen-Jacobian arm is exact, so it cannot detect a chart-covariance error.
        # This is what makes affine charts (PCA, whitening) invalid as covariance tests.
        # NOTE: it is deliberately NOT stated in terms of the naive no-Jacobian arm --
        # that arm is large for affine charts too (measured 1.25e+00), because rotation
        # and scaling alone already break the no-Jacobian route.
        affine_control_blind_to_second_order=(
            bool(s["test_A_affine_control_max_frozen_jac_err"] < 1e-6)
            if s["test_A_affine_control_max_frozen_jac_err"] is not None else None),
        nonlinear_charts_expose_second_order=bool(
            s["test_A_min_second_order_ratio_nonlinear"] > 1e3),
    )
    return s


def main(argv: Sequence[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny configuration to prove end-to-end execution")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    t0 = time.time()
    if args.smoke:
        # smoke uses 600 steps, not 150: at 150 the placeholder has NEGATIVE skill
        # (-0.120) and Test B reports itself uninformative, which is correct behaviour
        # but proves less about the harness executing meaningfully.
        report = run(k_values=(2,), strengths=(0.3,), kinds=("tanh",), d=4, r=2,
                     n_cells=60, n_states=40, n_steps=600, hidden=32)
        out = pathlib.Path(args.out) if args.out else OUT / "chart_test_smoke.json"
    else:
        report = run()
        out = pathlib.Path(args.out) if args.out else OUT / "chart_test.json"

    report["summary"] = summarise(report)
    report["wall_clock_s"] = time.time() - t0
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out, "w"), indent=2)

    s = report["summary"]
    print("\n=== SUMMARY ===")
    print(f"TEST A  max pushforward err   {s['test_A_max_pushforward_err']:.3e}   "
          f"(implementation floor)")
    print(f"TEST A  min naive err         {s['test_A_min_naive_err']:.3e}   "
          f"(control: no Jacobian)")
    print(f"TEST A  min power ratio       {s['test_A_min_power_ratio']:.3e}")
    print(f"TEST A  min frozen-Jac err    {s['test_A_min_frozen_jac_err_nonlinear']:.3e}   "
          f"(nonlinear charts: 2nd-order defect is visible)")
    if s["test_A_affine_control_max_frozen_jac_err"] is not None:
        print(f"AFFINE  max frozen-Jac err    "
              f"{s['test_A_affine_control_max_frozen_jac_err']:.3e}   "
              f"(negative control: D2psi=0, so affine charts CANNOT see the defect)")
        print(f"AFFINE  max naive err         "
              f"{s['test_A_affine_control_max_naive_err']:.3e}   "
              f"(large even for affine -- rotation/scale alone; not a defect probe)")
    if s["test_B_max_ratio_standardised"] is not None:
        print(f"TEST B  max ratio (std)       {s['test_B_max_ratio_standardised']:.3f}   "
              f"discrepancy {s['test_B_max_discrepancy_standardised']:.3e}, "
              f"floor {s['test_B_mean_seed_floor']:.3e}")
        print(f"TEST B  max ratio (raw)       {s['test_B_max_ratio_raw']:.3f}")
        print(f"TEST B  max skill vs zero-disp {s['test_B_max_skill_vs_zero_displacement']:+.3f}"
              f"   informative: {s['test_B_any_informative']}")
        if not s["test_B_any_informative"]:
            print("        -> Test B is UNINFORMATIVE with this model: it has no "
                  "predictive skill, so both charts are equally wrong and a low ratio "
                  "is trivial. Not evidence of chart-independent optimisation.")
    tr = s["trend"]
    if not tr.get("insufficient_data"):
        print(f"TREND   corr(|D2psi|,disc) {tr['corr_hessian_discrepancy']:+.4f}  "
              f"slope {tr['regression_slope']:.4f}  "
              f"intercept {tr['regression_intercept']:.3e}")
        print(f"TREND   chart-2 worse than chart-1 in "
              f"{tr['chart2_worse_count']}/{tr['n_configs']} configs")
        for sk, sv in tr["by_strength"].items():
            print(f"        s={sk}: disc {sv['discrepancy']:.4e}  ratio {sv['ratio']:.3f}"
                  f"  accuracy degradation {sv['accuracy_degradation']:+.4f}")
    for kk, vv in s["verdict"].items():
        print(f"  {kk}: {vv}")
    if s["verdict"].get("test_B_systematic_chart_dependence_detected"):
        print("  NOTE: the Test B ratio criterion passes, but the discrepancy scales "
              "systematically with chart nonlinearity and the chart-2 model is almost "
              "always less accurate. Do NOT report this as chart-independent training.")
    print(f"wrote {out} in {report['wall_clock_s']:.1f}s")
    return report


if __name__ == "__main__":
    main()
