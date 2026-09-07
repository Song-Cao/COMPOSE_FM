"""Controlled synthetic system with a KNOWN low-rank pairwise interaction structure.

Purpose
-------
This module is the only place in the project where the ground-truth interaction field is
available in closed form. It exists so that "does the interaction hierarchy recover the
true interaction?" can be answered by *recovery scoring against a known target*, rather
than by distributional distance between predicted and observed populations (which a model
can improve for reasons that have nothing to do with interaction structure).

The construction is DOMAIN-FREE. Nothing below refers to cells, genes, assets or doses in
any substantive sense: `tau` is an exposure scalar, an "intervention" is an index into a
finite set of primitive vector fields. Biology and finance are validation case studies
elsewhere and play no part in the definition.

Generative law
--------------
Latent state z in R^d. There are k primitive vector fields, all analytic and bounded:

    X_p(z) = M_p . tanh(z),          DX_p(z) = M_p . diag(1 - tanh(z)^2)

with each M_p normalised to unit spectral norm and then scaled by amp_p. The primitives
are deliberately non-commuting ([X_p, X_q] != 0 generically), so the kinematic cross term
of generator composition is nonzero and must be distinguished from a genuine interaction.

An intervention set P at exposure tau induces a dose vector a in R^k,

    a_p(tau) = A_p * (1 - exp(-tau / tau_p))   for p in P,     a_p = 0 for p not in P

which is bounded (-> A_p), strictly monotone increasing in tau, and satisfies a_p(0) = 0
EXACTLY (not approximately: the expression is identically zero at tau = 0).

The realised field is a dosed sum of primitives plus an interaction residual:

    X_P^tau(z) = sum_p a_p(tau) X_p(z)  +  (eps * c_cal) * I(z, a)

    I(z, a) = sum_{m=1..r} h_m(a) B_m(z)
    h_m(a)  = sum_{p<q} L_pm L_qm a_p a_q

c_cal is a FIXED scalar computed once per system (see `_calibration_constant`) that makes
the knob `eps` mean "median interaction-to-additive norm ratio over pair conditions at
tau = 1". It is a units choice, not extra structure: c_cal multiplies the whole
interaction uniformly and changes nothing about rank, singleton-vanishing or
non-kinematicity. It is needed because the raw h_m scale depends on k, r and the random
signs of L, so an uncalibrated eps = 0.6 in fact produced a ratio of 0.042 (measured on
the d=6, k=4, r=2 system, whose fitted c_cal is 16.51 -- so uncalibrated eps = 0.15 there
gives only ~0.009). With calibration on, eps = 0.15 measures 0.149 (measured).
Pass calibrate_eps=False for c_cal = 1.

Three properties of I hold BY CONSTRUCTION and are checked numerically in `self_test`:

  (i)  SINGLETON-VANISHING, exactly. If a has a single nonzero entry then every term in
       h_m carries a factor a_p * a_q with p != q, at least one of which is exactly 0.0,
       so h_m == 0.0 bit-exactly and I == 0 with no floating point residual at all.
       Interaction is therefore unidentifiable from singletons and
       leave-one-combination-out is a real test.
       The strict p<q sum is used for exactly this reason: the algebraically identical
       form 0.5*[(L_.m . a)^2 - sum_p L_pm^2 a_p^2] rounds fl(L_pm*a_p)^2 and
       fl(L_pm^2)*fl(a_p^2) separately and leaves a ~1 ulp residual (measured 5.3e-17
       before the change), which would trip the `== 0.0` check in `self_test`.
  (ii) RANK r. The coefficient tensor is T_pq = sum_m L_pm L_qm (p != q), i.e. the
       off-diagonal part of L L^T, and the whole interaction family factors through the
       r basis fields B_m. The recovery target is therefore an r-dimensional space of
       fields; `interaction_field_rank()` measures it as the numerical rank of the matrix
       of realised interaction fields over conditions and states, which equals r.
       HONEST CAVEAT: the *matrix* off-diag(L L^T) can have numerical rank above r,
       because zeroing a diagonal is not a rank-preserving operation. The r that is
       recoverable, and the r this module claims, is the FACTOR rank (number of basis
       fields), exposed as `.r` and verified by `interaction_field_rank()`. The numerical
       rank of the coefficient matrix is reported separately as
       `coefficient_matrix_numerical_rank()` and is NOT the recovery target.
  (iii) NON-KINEMATIC. The B_m are built from independently drawn (R_m, Q_m, o_m) and are
       not in the span of the kinematic cross fields DX_p[X_q] + DX_q[X_p] nor of the
       primitives themselves. `nonkinematic_fraction()` measures the least-squares
       residual of the true interaction against that span, so the claim is a measured
       number, not an assertion. This matters because generator composition already
       reproduces the kinematic cross term for free; only the residual above it is new
       structure a model has to learn.

Exposure convention
-------------------
Integration time is FIXED at T = 1. Exposure enters only through the bounded dose
response. Consequently tau = 0 gives the identity map exactly, and the field is
autonomous in integration time for fixed tau (so the semigroup in T holds, while the
dose response is deliberately non-semigroup in tau -- saturation is the point).

What this module does NOT model
-------------------------------
No measurement model beyond additive isotropic Gaussian noise, no dropout, no batch
effects, no unobserved confounding. It is a controlled test of interaction recovery and
chart behaviour, and its passes are necessary but not sufficient for real data.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field as _dcfield
from typing import Callable, Iterable, Sequence

import numpy as np

__all__ = [
    "SyntheticSystem",
    "NonlinearChart",
    "Population",
    "rk4",
]


# ----------------------------------------------------------------------------------
# integration
# ----------------------------------------------------------------------------------
def rk4(f: Callable[[np.ndarray], np.ndarray], Z0: np.ndarray, T: float = 1.0,
        n_steps: int = 40) -> np.ndarray:
    """Classical RK4 for an autonomous field. float64 throughout."""
    Z = np.array(Z0, dtype=np.float64, copy=True)
    if T == 0.0 or n_steps == 0:
        return Z
    h = float(T) / int(n_steps)
    for _ in range(int(n_steps)):
        k1 = f(Z)
        k2 = f(Z + 0.5 * h * k1)
        k3 = f(Z + 0.5 * h * k2)
        k4 = f(Z + h * k3)
        Z = Z + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return Z


def _spectral_normalise(M: np.ndarray) -> np.ndarray:
    s = np.linalg.svd(M, compute_uv=False)[0]
    return M / max(s, 1e-12)


# ----------------------------------------------------------------------------------
# nonlinear charts
# ----------------------------------------------------------------------------------
class NonlinearChart:
    """A genuinely nonlinear, globally invertible re-chart psi: R^d -> R^d.

    Two families, both with NONZERO second derivative -- this is the whole point. An
    affine chart (PCA, whitening, any linear map) has D2psi == 0 and every composition
    variant is then exact, so affine charts cannot exercise the covariance defect and are
    invalid as tests. `hessian_scale()` returns a positive number for these charts and is
    asserted nonzero by the harness.

    kind='tanh' : psi(z) = z + s * A tanh(z),  A spectral-normalised, invertible for s < 1.
    kind='cubic': psi(z) = Q m(Q^T z),  m_i(y) = y_i + s y_i^3, monotone per coordinate
                  hence globally invertible for s >= 0; strongly nonlinear in the tails.
    """

    def __init__(self, d: int, strength: float = 0.3, kind: str = "tanh", seed: int = 7):
        if kind not in ("tanh", "cubic", "affine"):
            raise ValueError(f"unknown chart kind {kind!r}")
        g = np.random.default_rng(seed)
        self.d, self.s, self.kind = int(d), float(strength), kind
        self.A = _spectral_normalise(g.normal(size=(d, d)))
        Q, _ = np.linalg.qr(g.normal(size=(d, d)))
        self.Q = Q
        if kind == "tanh" and not (0.0 <= self.s < 1.0):
            raise ValueError("tanh chart needs 0 <= strength < 1 for invertibility")

    # -- forward ---------------------------------------------------------------
    def psi(self, Z: np.ndarray) -> np.ndarray:
        Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
        if self.kind == "affine":
            return Z @ self.A.T
        if self.kind == "tanh":
            return Z + self.s * (np.tanh(Z) @ self.A.T)
        Y = Z @ self.Q
        return (Y + self.s * Y ** 3) @ self.Q.T

    def jac(self, Z: np.ndarray) -> np.ndarray:
        Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
        n = Z.shape[0]
        if self.kind == "affine":
            return np.broadcast_to(self.A, (n, self.d, self.d)).copy()
        if self.kind == "tanh":
            sech2 = 1.0 - np.tanh(Z) ** 2                        # (n,d)
            return np.eye(self.d)[None] + self.s * (self.A[None] * sech2[:, None, :])
        Y = Z @ self.Q
        dm = 1.0 + 3.0 * self.s * Y ** 2                         # (n,d)
        return np.einsum('ia,na,ja->nij', self.Q, dm, self.Q)

    def hess(self, Z: np.ndarray) -> np.ndarray:
        """D2psi[n, i, j, k] = d^2 psi_i / dz_j dz_k. Symmetric in (j, k) by construction."""
        Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
        n = Z.shape[0]
        if self.kind == "affine":
            return np.zeros((n, self.d, self.d, self.d))
        if self.kind == "tanh":
            t = np.tanh(Z)
            ddm = -2.0 * t * (1.0 - t ** 2)                      # (n,d)
            H = np.zeros((n, self.d, self.d, self.d))
            for a in range(self.d):
                H[:, :, a, a] += self.s * self.A[None, :, a] * ddm[:, a][:, None]
            return H
        Y = Z @ self.Q
        ddm = 6.0 * self.s * Y                                   # (n,d)
        return np.einsum('ia,na,ja,ka->nijk', self.Q, ddm, self.Q, self.Q)

    def hessian_scale(self, Z: np.ndarray) -> float:
        """RMS magnitude of D2psi over the given states. Zero iff the chart is affine."""
        return float(np.sqrt(np.mean(self.hess(Z) ** 2)))

    # -- inverse ---------------------------------------------------------------
    def inv(self, W: np.ndarray, iters: int = 80, tol: float = 1e-14) -> np.ndarray:
        W = np.atleast_2d(np.asarray(W, dtype=np.float64))
        if self.kind == "affine":
            return np.linalg.solve(self.A, W.T).T
        if self.kind == "cubic":
            # per-coordinate monotone cubic y + s y^3 = v, Newton from a safe start
            V = W @ self.Q
            Y = np.sign(V) * np.minimum(np.abs(V), (np.abs(V) / max(self.s, 1e-12)) ** (1 / 3)
                                        if self.s > 0 else np.abs(V))
            for _ in range(iters):
                r = Y + self.s * Y ** 3 - V
                if np.max(np.abs(r)) < tol:
                    break
                Y = Y - r / (1.0 + 3.0 * self.s * Y ** 2)
            return Y @ self.Q.T
        Z = W.copy()
        for _ in range(iters):
            r = self.psi(Z) - W
            if np.max(np.abs(r)) < tol:
                break
            Z = Z - np.linalg.solve(self.jac(Z), r[:, :, None])[:, :, 0]
        return Z

    def inverse_residual(self, Z: np.ndarray) -> float:
        """max |psi^{-1}(psi(z)) - z| -- a check that the Newton inverse actually converged."""
        Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
        return float(np.max(np.abs(self.inv(self.psi(Z)) - Z)))

    # -- pushforward -----------------------------------------------------------
    def pushforward_field(self, field_z: Callable[[np.ndarray], np.ndarray]
                          ) -> Callable[[np.ndarray], np.ndarray]:
        """Xtilde(w) = Dpsi(psi^{-1}(w)) . X(psi^{-1}(w)) -- the covariant transform of a field."""
        def field_w(W: np.ndarray) -> np.ndarray:
            Z = self.inv(W)
            return np.einsum('nij,nj->ni', self.jac(Z), field_z(Z))
        return field_w


# ----------------------------------------------------------------------------------
# populations
# ----------------------------------------------------------------------------------
@dataclass
class Population:
    """One observed (pre, post) pair for one condition. `P` is the intervention set."""
    label: str
    P: tuple
    tau: float
    pre: np.ndarray
    post: np.ndarray
    replicate: int = 0
    meta: dict = _dcfield(default_factory=dict)

    @property
    def order(self) -> int:
        return len(self.P)

    def mapped(self, chart: NonlinearChart) -> "Population":
        """The same observation expressed in another chart. Populations transform as points."""
        return Population(label=self.label, P=self.P, tau=self.tau,
                          pre=chart.psi(self.pre), post=chart.psi(self.post),
                          replicate=self.replicate, meta=dict(self.meta, chart="mapped"))


# ----------------------------------------------------------------------------------
# the system
# ----------------------------------------------------------------------------------
class SyntheticSystem:
    """Controlled ground truth: k known primitive fields + a known rank-r interaction.

    Parameters
    ----------
    d            latent dimension
    k            number of primitive interventions
    r            FACTOR rank of the pairwise interaction (number of basis fields)
    eps          interaction strength. With calibrate_eps=True (default) it is calibrated
                 to MEAN the median interaction-to-additive norm ratio over pair
                 conditions at tau=1, so eps=0.15 gives an interaction about 15% of the
                 additive field. eps=0 turns the interaction off exactly (null system).
    anisotropy   condition number of the control covariance (1 = isotropic)
    noise        std of additive isotropic observation noise on the post population
    pre_noise    std of additive noise on the pre population (0 = pre observed exactly)
    amp_spread   spread of primitive amplitudes
    seed         all structure is drawn from this seed and is then fixed
    """

    def __init__(self, d: int = 6, k: int = 4, r: int = 2, eps: float = 0.6,
                 anisotropy: float = 1.0, noise: float = 0.02, pre_noise: float = 0.0,
                 amp_spread: float = 0.4, seed: int = 0, calibrate_eps: bool = True):
        if k < 2:
            raise ValueError("need k >= 2 for pairwise interaction to exist")
        if not (1 <= r <= k):
            raise ValueError("need 1 <= r <= k")
        g = np.random.default_rng(seed)
        self.d, self.k, self.r = int(d), int(k), int(r)
        self.eps, self.noise, self.pre_noise = float(eps), float(noise), float(pre_noise)
        self.anisotropy = float(anisotropy)
        self.seed = int(seed)

        # ---- primitive fields: X_p(z) = amp_p * Mhat_p tanh(z) ----
        self.M = np.stack([_spectral_normalise(g.normal(size=(d, d))) for _ in range(k)])
        self.amp = 1.0 + amp_spread * (2.0 * g.random(k) - 1.0)
        self.M = self.M * self.amp[:, None, None]

        # ---- dose response: bounded, monotone, a(0) = 0 exactly ----
        self.dose_A = 0.9 + 0.5 * g.random(k)          # saturation ceiling
        self.dose_tau = 0.6 + 0.9 * g.random(k)        # half-saturation scale

        # ---- interaction: rank-r factor loadings + r basis fields ----
        self.L = g.normal(size=(k, r)) / np.sqrt(r)
        self.B_R = np.stack([_spectral_normalise(g.normal(size=(d, d))) for _ in range(r)])
        self.B_Q = np.stack([_spectral_normalise(g.normal(size=(d, d))) for _ in range(r)])
        self.B_o = g.normal(size=(r, d)) * 0.5

        # ---- control distribution with controlled anisotropy ----
        Qc, _ = np.linalg.qr(g.normal(size=(d, d)))
        lam = np.geomspace(1.0, max(self.anisotropy, 1.0), d)
        lam = lam / np.exp(np.mean(np.log(lam)))       # unit geometric mean -> scale-free
        self.cov_sqrt = Qc @ np.diag(np.sqrt(lam)) @ Qc.T
        self.base_scale = 0.8

        # ---- calibrate eps so it MEANS the interaction-to-additive norm ratio ----
        # Without this, eps is an opaque knob: the raw h_m coefficients depend on k, r and
        # the random signs of L, so eps=0.6 produced an actual ratio of 0.042 (measured).
        # We divide out the median realised ratio over all pair conditions at tau=1 on a
        # fixed reference sample, so that afterwards eps=0.1 means "interaction is about
        # 10% of the additive field in norm". Set calibrate_eps=False for the raw scale.
        self._eps_cal = 1.0
        if calibrate_eps and self.eps != 0.0:
            self._eps_cal = self._calibration_constant()

    def _calibration_constant(self, n_states: int = 256, tau: float = 1.0) -> float:
        Zc = self.control(n_states, seed=self.seed + 90210)
        saved, self._eps_cal = self._eps_cal, 1.0
        try:
            ratios = []
            for (p, q) in itertools.combinations(range(self.k), 2):
                add = self.additive_field((p, q), Zc, tau)
                inter = self.interaction_field((p, q), Zc, tau)
                ratios.append(np.linalg.norm(inter) / max(np.linalg.norm(add), 1e-300))
            med = float(np.median(ratios)) / max(abs(self.eps), 1e-300)
        finally:
            self._eps_cal = saved
        return 1.0 / max(med, 1e-300)

    # ---- dose response ------------------------------------------------------
    def dose(self, p: int, tau: float) -> float:
        """a_p(tau): bounded, strictly monotone, exactly 0 at tau = 0."""
        return float(self.dose_A[p] * (1.0 - np.exp(-float(tau) / self.dose_tau[p])))

    def dose_vector(self, P: Iterable[int], tau: float) -> np.ndarray:
        a = np.zeros(self.k, dtype=np.float64)
        for p in P:
            a[int(p)] = self.dose(int(p), tau)
        return a

    # ---- primitive fields ---------------------------------------------------
    def primitive_field(self, p: int, Z: np.ndarray) -> np.ndarray:
        Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
        return np.tanh(Z) @ self.M[int(p)].T

    def primitive_jacobian(self, p: int, Z: np.ndarray) -> np.ndarray:
        Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
        sech2 = 1.0 - np.tanh(Z) ** 2
        return self.M[int(p)][None] * sech2[:, None, :]

    def additive_field(self, P: Iterable[int], Z: np.ndarray, tau: float) -> np.ndarray:
        a = self.dose_vector(P, tau)
        Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
        out = np.zeros_like(Z)
        for p in np.nonzero(a)[0]:
            out += a[p] * self.primitive_field(p, Z)
        return out

    # ---- interaction --------------------------------------------------------
    def basis_field(self, m: int, Z: np.ndarray) -> np.ndarray:
        Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
        return np.tanh(Z @ self.B_Q[int(m)].T + self.B_o[int(m)]) @ self.B_R[int(m)].T

    def interaction_coeffs(self, a: np.ndarray) -> np.ndarray:
        """h_m(a) = sum_{p<q} L_pm L_qm a_p a_q. Exactly 0.0 for singleton a.

        NOTE ON THE FORM. The algebraically identical expression
        0.5 * [(L_.m . a)^2 - sum_p L_pm^2 a_p^2] is NOT exactly zero in floating point
        for a singleton, because (L_pm * a_p)^2 and L_pm^2 * a_p^2 are not bit-identical:
        measured residual 5.3e-17. The strict upper-triangular form below is exactly zero
        instead, because every term carries a factor a_p * a_q with p != q, at least one
        of which is exactly 0.0, and a sum of exact zeros is exactly 0.0. Since exact
        singleton-vanishing is a structural claim the benchmark makes, we use the form
        that delivers it exactly rather than to machine precision.
        """
        a = np.asarray(a, dtype=np.float64)
        outer = a[:, None] * a[None, :]                        # (k,k)
        upper = np.triu(np.ones((self.k, self.k)), k=1)
        return np.einsum('pm,qm,pq,pq->m', self.L, self.L, outer, upper)

    def interaction_field(self, P: Iterable[int], Z: np.ndarray, tau: float) -> np.ndarray:
        """The TRUE interaction residual field, in closed form. Recovery target."""
        Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
        h = self.interaction_coeffs(self.dose_vector(P, tau))
        out = np.zeros_like(Z)
        for m in range(self.r):
            if h[m] != 0.0:
                out += h[m] * self.basis_field(m, Z)
        return (self.eps * self._eps_cal) * out

    def interaction_tensor(self) -> np.ndarray:
        """Coefficient matrix T_pq = sum_m L_pm L_qm with the diagonal zeroed."""
        T = self.L @ self.L.T
        np.fill_diagonal(T, 0.0)
        return T

    def coefficient_matrix_numerical_rank(self, tol: float = 1e-10) -> int:
        """Numerical rank of off-diag(L L^T). NOT the recovery target -- see module docstring."""
        s = np.linalg.svd(self.interaction_tensor(), compute_uv=False)
        return int(np.sum(s > tol * max(s[0], 1e-300)))

    # ---- total field and flow ----------------------------------------------
    def field(self, P: Iterable[int], Z: np.ndarray, tau: float) -> np.ndarray:
        return self.additive_field(P, Z, tau) + self.interaction_field(P, Z, tau)

    def flow(self, P: Iterable[int], Z: np.ndarray, tau: float, n_steps: int = 40
             ) -> np.ndarray:
        """Integrate the tau-dosed field for unit time. tau = 0 -> identity exactly."""
        if float(tau) == 0.0:
            return np.atleast_2d(np.asarray(Z, dtype=np.float64)).copy()
        return rk4(lambda Q: self.field(P, Q, tau), Z, T=1.0, n_steps=n_steps)

    # ---- populations --------------------------------------------------------
    def control(self, n: int, seed: int = 1) -> np.ndarray:
        g = np.random.default_rng(seed)
        return self.base_scale * (g.normal(size=(n, self.d)) @ self.cov_sqrt.T)

    def conditions(self, n_pairs: int | None = None, n_triples: int = 1
                   ) -> list[tuple]:
        """Singletons, pairs and triples. Ordering is deterministic so splits are stable."""
        singles = [(p,) for p in range(self.k)]
        pairs = list(itertools.combinations(range(self.k), 2))
        if n_pairs is not None:
            pairs = pairs[:int(n_pairs)]
        triples = list(itertools.combinations(range(self.k), 3))[:int(n_triples)]
        return singles + pairs + triples

    @staticmethod
    def label(P: Iterable[int], tau: float) -> str:
        return "+".join(f"X{int(p)}" for p in P) + f"@tau{float(tau):g}"

    def sample_populations(self, n_cells: int = 200, n_replicates: int = 1,
                           taus: Sequence[float] = (1.0,), n_pairs: int | None = None,
                           n_triples: int = 1, seed: int = 11, n_steps: int = 40
                           ) -> list[Population]:
        """(pre, post) population pairs for every (condition, tau, replicate).

        `pre` is an independent draw from the control distribution for each population, so
        pre-populations are NOT shared across conditions (a model may not exploit paired
        identity across conditions). Within one population, post[i] is the flow image of
        pre[i], which makes paired supervision available to placeholder models; a method
        that only consumes the two marginals is free to ignore that.
        """
        pops: list[Population] = []
        for ci, P in enumerate(self.conditions(n_pairs=n_pairs, n_triples=n_triples)):
            for tau in taus:
                for rep in range(int(n_replicates)):
                    sd = int(seed) + 7919 * ci + 104729 * rep + int(round(1e4 * float(tau)))
                    g = np.random.default_rng(sd + 1)
                    pre_clean = self.control(n_cells, seed=sd)
                    post_clean = self.flow(P, pre_clean, tau, n_steps=n_steps)
                    pre = pre_clean + (self.pre_noise * g.normal(size=pre_clean.shape)
                                       if self.pre_noise > 0 else 0.0)
                    post = post_clean + (self.noise * g.normal(size=post_clean.shape)
                                         if self.noise > 0 else 0.0)
                    pops.append(Population(label=self.label(P, tau), P=tuple(P), tau=float(tau),
                                           pre=pre, post=post, replicate=rep,
                                           meta=dict(order=len(P), seed=sd)))
        return pops

    @staticmethod
    def loco_split(pops: Sequence[Population], held_out: Iterable[tuple]
                   ) -> tuple[list[int], list[int]]:
        """Leave-one-combination-out: hold out whole intervention SETS (all taus, all reps)."""
        held = {tuple(sorted(P)) for P in held_out}
        train = [i for i, p in enumerate(pops) if tuple(sorted(p.P)) not in held]
        test = [i for i, p in enumerate(pops) if tuple(sorted(p.P)) in held]
        return train, test

    # ---- recovery scoring ---------------------------------------------------
    def score_interaction_recovery(self, pred_interaction: Callable[[tuple, np.ndarray, float],
                                                                    np.ndarray],
                                   Z: np.ndarray, conditions: Sequence[tuple] | None = None,
                                   tau: float = 1.0) -> dict:
        """Score a predicted interaction field against the known truth.

        Returns relative L2 error, cosine similarity and the optimal global scale (the
        least-squares c in truth ~ c * pred), all pooled over the given conditions and
        states. A method that predicts zero interaction gets rel_l2 = 1 and cosine 0.
        """
        if conditions is None:
            conditions = [c for c in self.conditions() if len(c) >= 2]
        T, Pr = [], []
        for P in conditions:
            T.append(self.interaction_field(P, Z, tau).ravel())
            Pr.append(np.asarray(pred_interaction(tuple(P), Z, tau), dtype=np.float64).ravel())
        t = np.concatenate(T)
        p = np.concatenate(Pr)
        nt, npn = np.linalg.norm(t), np.linalg.norm(p)
        return dict(rel_l2=float(np.linalg.norm(t - p) / max(nt, 1e-300)),
                    cosine=float(t @ p / max(nt * npn, 1e-300)),
                    optimal_scale=float((t @ p) / max(npn ** 2, 1e-300)),
                    truth_norm=float(nt), pred_norm=float(npn),
                    n_conditions=len(conditions), n_states=int(np.atleast_2d(Z).shape[0]))

    # ---- structural diagnostics --------------------------------------------
    def interaction_field_rank(self, Z: np.ndarray, taus: Sequence[float] = (0.4, 1.0, 1.8),
                               tol: float = 1e-9) -> tuple[int, list[float]]:
        """Numerical rank of the realised interaction fields over (condition, tau).

        Rows are vec(I(., a_c)); the row space is spanned by the r basis fields, so the
        rank equals r whenever the coefficient vectors h(a_c) span R^r. This is the FACTOR
        rank -- the quantity a recovery method can actually identify.
        """
        rows = []
        for P in self.conditions():
            if len(P) < 2:
                continue
            for tau in taus:
                rows.append(self.interaction_field(P, Z, tau).ravel())
        Mrows = np.stack(rows)
        s = np.linalg.svd(Mrows, compute_uv=False)
        rank = int(np.sum(s > tol * max(s[0], 1e-300)))
        return rank, [float(x) for x in s[:min(len(s), self.k + 3)]]

    def nonkinematic_fraction(self, Z: np.ndarray, tau: float = 1.0) -> dict:
        """How much of the true interaction is OUTSIDE the kinematic/primitive span.

        For each pair condition (p, q) the basis is
            {DX_p[X_q] + DX_q[X_p]}  (the symmetric cross term that generator composition
                                      already produces for free)
          + {X_p, X_q}               (anything a dosed-additive model can absorb)
          + {DX_p[X_p], DX_q[X_q]}   (self-kinematic terms, for good measure)
        and we report the least-squares residual fraction of the true interaction on that
        span. A value near 1 means the interaction is genuinely new structure; a value
        near 0 would mean the "interaction" is a relabelled kinematic term and the
        benchmark would be worthless.
        """
        Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
        fracs, labels = [], []
        for (p, q) in itertools.combinations(range(self.k), 2):
            a = self.dose_vector((p, q), tau)
            Xp, Xq = self.primitive_field(p, Z), self.primitive_field(q, Z)
            Jp, Jq = self.primitive_jacobian(p, Z), self.primitive_jacobian(q, Z)
            cross = (np.einsum('nij,nj->ni', Jp, a[q] * Xq)
                     + np.einsum('nij,nj->ni', Jq, a[p] * Xp))
            selfp = np.einsum('nij,nj->ni', Jp, a[p] * Xp)
            selfq = np.einsum('nij,nj->ni', Jq, a[q] * Xq)
            basis = np.stack([cross.ravel(), Xp.ravel(), Xq.ravel(),
                              selfp.ravel(), selfq.ravel()], axis=1)
            y = self.interaction_field((p, q), Z, tau).ravel()
            coef, *_ = np.linalg.lstsq(basis, y, rcond=None)
            resid = y - basis @ coef
            fracs.append(float(np.linalg.norm(resid) / max(np.linalg.norm(y), 1e-300)))
            labels.append(f"X{p}+X{q}")
        return dict(per_pair=dict(zip(labels, fracs)),
                    min=float(min(fracs)), median=float(np.median(fracs)),
                    max=float(max(fracs)))

    def singleton_interaction_max(self, Z: np.ndarray, taus: Sequence[float] = (0.5, 1.0, 2.0)
                                  ) -> float:
        """max |I| over all singletons. Exactly 0.0 by construction."""
        worst = 0.0
        for p in range(self.k):
            for tau in taus:
                worst = max(worst, float(np.max(np.abs(self.interaction_field((p,), Z, tau)))))
        return worst

    def loco_identifiability(self, Z: np.ndarray, held_out: tuple = (0, 1),
                             tau: float = 1.0) -> dict:
        """Is the held-out combination's interaction identifiable from the training set?

        Leave-one-combination-out is only a well-posed generalisation test if the held-out
        interaction is (a) reconstructible from the OTHER observed combinations -- else no
        method could succeed and the benchmark is unfair -- and (b) NOT reconstructible
        from singletons alone -- else it is not an interaction test at all.

        Measured on the d=6, k=4, r=2 system: residual fraction from other pairs
        1.664e-16 (fully reconstructible via the rank-r basis), singleton interaction
        basis norm exactly 0.0 (impossible), other-pair span rank 2 == r.
        """
        Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
        ho = tuple(sorted(held_out))
        others = [P for P in itertools.combinations(range(self.k), 2)
                  if tuple(sorted(P)) != ho]
        y = self.interaction_field(ho, Z, tau).ravel()
        B = np.stack([self.interaction_field(P, Z, tau).ravel() for P in others], axis=1)
        c, *_ = np.linalg.lstsq(B, y, rcond=None)
        sv = np.linalg.svd(B, compute_uv=False)
        Bs = np.stack([self.interaction_field((p,), Z, tau).ravel()
                       for p in range(self.k)], axis=1)
        return dict(
            held_out=list(ho),
            resid_frac_from_other_pairs=float(
                np.linalg.norm(y - B @ c) / max(np.linalg.norm(y), 1e-300)),
            singleton_basis_norm=float(np.linalg.norm(Bs)),
            other_pair_span_rank=int(np.sum(sv > 1e-9 * max(sv[0], 1e-300))),
            r=self.r)

    def dose_diagnostics(self, taus: Sequence[float] = tuple(np.linspace(0.0, 6.0, 121))
                         ) -> dict:
        """a_p(0) == 0 exactly, monotone increasing, bounded by A_p."""
        taus = np.asarray(taus, dtype=np.float64)
        A = np.stack([[self.dose(p, t) for t in taus] for p in range(self.k)])
        return dict(max_abs_at_tau0=float(np.max(np.abs(A[:, 0]))),
                    min_increment=float(np.min(np.diff(A, axis=1))),
                    max_over_ceiling=float(np.max(A - self.dose_A[:, None])),
                    ceilings=[float(x) for x in self.dose_A])

    def self_test(self, n_states: int = 300, seed: int = 3) -> dict:
        """All structural claims of the module, measured. Cheap enough to run in CI."""
        Z = self.control(n_states, seed=seed)
        rank, svals = self.interaction_field_rank(Z)
        out = dict(
            d=self.d, k=self.k, r_declared=self.r,
            interaction_field_rank=rank,
            interaction_field_singular_values=svals,
            coefficient_matrix_numerical_rank=self.coefficient_matrix_numerical_rank(),
            singleton_interaction_max_abs=self.singleton_interaction_max(Z),
            dose=self.dose_diagnostics(),
            nonkinematic=self.nonkinematic_fraction(Z),
            tau0_flow_identity_max_abs=float(np.max(np.abs(self.flow((0, 1), Z, 0.0) - Z))),
            loco=self.loco_identifiability(Z),
        )
        # interaction magnitude relative to the additive part, so eps is interpretable
        add = self.additive_field((0, 1), Z, 1.0)
        inter = self.interaction_field((0, 1), Z, 1.0)
        out["interaction_to_additive_norm_ratio"] = float(
            np.linalg.norm(inter) / max(np.linalg.norm(add), 1e-300))
        _rat = [float(np.linalg.norm(self.interaction_field((p, q), Z, 1.0))
                      / max(np.linalg.norm(self.additive_field((p, q), Z, 1.0)), 1e-300))
                for (p, q) in itertools.combinations(range(self.k), 2)]
        out["interaction_to_additive_ratio_median"] = float(np.median(_rat))
        out["interaction_to_additive_ratio_range"] = [float(min(_rat)), float(max(_rat))]
        out["eps_requested"] = self.eps
        out["eps_calibration_constant"] = float(self._eps_cal)
        null_system = (self.eps == 0.0)
        out["null_system"] = null_system
        out["checks"] = dict(
            # eps=0 is the deliberate NULL system: rank must be 0, not r, and the
            # nonkinematic fraction is 0/0 and therefore not a meaningful check.
            rank_equals_r=(rank == (0 if null_system else self.r)),
            singletons_vanish_exactly=(out["singleton_interaction_max_abs"] == 0.0),
            dose_zero_at_zero_exactly=(out["dose"]["max_abs_at_tau0"] == 0.0),
            dose_monotone=(out["dose"]["min_increment"] > 0.0),
            dose_bounded=(out["dose"]["max_over_ceiling"] <= 0.0),
            interaction_nonkinematic=(True if null_system
                                      else out["nonkinematic"]["min"] > 0.5),
            tau0_is_identity=(out["tau0_flow_identity_max_abs"] == 0.0),
            # LOCO well-posedness: solvable from other pairs, impossible from singletons.
            loco_well_posed=(
                True if null_system else
                (out["loco"]["resid_frac_from_other_pairs"] < 1e-8
                 and out["loco"]["singleton_basis_norm"] == 0.0)),
        )
        # eps means the median interaction/additive ratio when calibrated; check it does.
        if not null_system:
            out["checks"]["eps_calibration_within_2x"] = bool(
                0.5 * abs(self.eps) <= out["interaction_to_additive_ratio_median"]
                <= 2.0 * abs(self.eps))
        return out


# ----------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------
def main() -> dict:
    import json
    import pathlib
    import time

    t0 = time.time()
    report: dict = {"systems": []}
    for (d, k, r, aniso, eps) in [(6, 4, 2, 1.0, 0.15),
                                  (6, 4, 3, 20.0, 0.15),
                                  (8, 5, 2, 5.0, 0.30),
                                  (6, 4, 2, 1.0, 0.0)]:
        sysm = SyntheticSystem(d=d, k=k, r=r, anisotropy=aniso, eps=eps, seed=0)
        st = sysm.self_test(n_states=300)
        st["config"] = dict(d=d, k=k, r=r, anisotropy=aniso, eps=eps)
        report["systems"].append(st)
        print(f"d={d} k={k} r={r} aniso={aniso:>4g} eps={eps} | "
              f"field rank {st['interaction_field_rank']} (declared {r}) | "
              f"coef-matrix rank {st['coefficient_matrix_numerical_rank']} | "
              f"singleton |I| {st['singleton_interaction_max_abs']:.1e} | "
              f"nonkinematic min {st['nonkinematic']['min']:.4f} | "
              f"I/A median {st['interaction_to_additive_ratio_median']:.4f} | "
              f"all checks {all(st['checks'].values())}")
        if not all(st["checks"].values()):
            print("   FAILED:", [c for c, v in st["checks"].items() if not v])

    # chart sanity: nonzero Hessian for the nonlinear families, exactly zero for affine
    sysm = SyntheticSystem(d=6, k=4, r=2, seed=0)
    Z = sysm.control(200, seed=5)
    chart_rows = []
    for kind in ("tanh", "cubic", "affine"):
        for s in (0.1, 0.3, 0.6):
            ch = NonlinearChart(6, strength=s, kind=kind, seed=7)
            chart_rows.append(dict(kind=kind, strength=s,
                                   hessian_scale=ch.hessian_scale(Z),
                                   inverse_residual=ch.inverse_residual(Z)))
            print(f"chart {kind:>6} s={s} | |D2psi| {chart_rows[-1]['hessian_scale']:.3e} "
                  f"| inv residual {chart_rows[-1]['inverse_residual']:.2e}")
    report["charts"] = chart_rows

    pops = sysm.sample_populations(n_cells=120, n_replicates=2, taus=(0.5, 1.0), n_triples=1)
    tr, te = SyntheticSystem.loco_split(pops, held_out=[(0, 1)])
    report["sampler"] = dict(n_populations=len(pops), n_train=len(tr), n_test=len(te),
                             orders=sorted({p.order for p in pops}),
                             labels_held_out=sorted({pops[i].label for i in te}))
    print(f"sampler: {len(pops)} populations, LOCO holding X0+X1 -> "
          f"{len(tr)} train / {len(te)} test, orders {report['sampler']['orders']}")

    report["wall_clock_s"] = time.time() - t0
    out = pathlib.Path(__file__).resolve().parents[2] / "results" / "synthetic_selftest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out, "w"), indent=2)
    print(f"wrote {out} in {report['wall_clock_s']:.2f}s")
    return report


if __name__ == "__main__":
    main()
