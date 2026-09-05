"""Toy ground-truth system for COMPOSE-FM prototype gates.

Design principle: the toy is calibrated to a measurement made earlier in this project on
the real Norman 2019 Perturb-seq doubles (131 pairs, pseudobulk mean-shift estimator):

    cos(V_p + V_q, V_pq)          mean   0.960     -> direction well predicted
    saturation scalar c            mean   0.924, median 0.895, IQR [0.699, 1.145]
    pairs with c < 1               0.63   (83/131) -> sub-additive, but only in ~2/3

i.e. additive composition gets the *direction* of a combined perturbation nearly right and
overshoots its *magnitude* by roughly 10% at the median. The mechanism below reproduces that
phenomenology from a saturating joint drive (standard receptor/pathway saturation).

PROVENANCE AND HONEST CAVEATS -- read before trusting the toy:
  * Those numbers are from the pseudobulk estimator. A per-cell OT-based estimator on the
    same data gave a much stronger c (mean 0.635, 131/131 sub-additive), but that was traced
    to regression dilution from per-cell noise, not to real biology: the mean-level estimator
    on the same OT fields gave c mean 0.916 / median 0.881 and agreed with pseudobulk at
    Spearman +0.993. The defensible effect size is therefore the ~10% shortfall quoted above,
    NOT 37%. The toy's `kappa` is set to produce a shortfall in that range.
  * The toy's *state-dependence* of the deficit is a DESIGN CHOICE, not a fitted Norman
    quantity. On real Norman a learned state-dependent gate beat a single global constant by
    only +0.0085 R^2 -- i.e. essentially all the composition signal there was captured by one
    scalar. The toy deliberately makes state-dependence recoverable so the prototype can test
    whether the architecture *can* exploit it; whether real data rewards that is exactly what
    the later gates must decide, and Norman alone says it barely does.

Ground truth
------------
Each intervention p has a direction u_p and a state-dependent gain g_p(z). The *raw* drive
of an intervention set P adds linearly, but the realised field saturates in its own norm:

    R_P(z) = sum_{p in P} g_p(z) u_p ,      g_p(z) = sigmoid(w_p . z + b_p)
    X_P(z) = R_P(z) / (1 + ||R_P(z)|| / kappa)

Consequences (all verified in tests/test_toy.py):
  * singles are recoverable: X_{p} is a saturated version of a single drive;
  * composition is SUB-ADDITIVE and the deficit depends on z (through g) and on the
    alignment of u_p, u_q -- so a *global constant* contraction cannot capture it, while a
    state-dependent gate gamma(z,P) can;
  * the field is autonomous, so exposure == integration time and the semigroup holds
    exactly: Psi^{tau+tau'} = Psi^{tau'} o Psi^{tau}.

The last point is what makes the toy able to test dose/exposure extrapolation, and the
second is what makes a learned gate *necessary* rather than decorative.
"""
from __future__ import annotations
import numpy as np


class ToySystem:
    def __init__(self, d: int = 4, n_pert: int = 6, kappa: float = 1.2, seed: int = 0):
        g = np.random.default_rng(seed)
        self.d, self.n_pert, self.kappa = d, n_pert, kappa
        # directions: unit norm, deliberately non-orthogonal so alignment matters
        U = g.normal(size=(n_pert, d))
        self.U = U / np.linalg.norm(U, axis=1, keepdims=True)
        self.W = g.normal(size=(n_pert, d)) * 0.9      # state-dependent gain weights
        self.b = g.normal(size=n_pert) * 0.5
        self.amp = 1.0 + 0.6 * g.random(n_pert)        # per-perturbation strength

    # ---- ground-truth fields -------------------------------------------------
    def gain(self, P, Z):
        """(|P|, N) state-dependent gains."""
        Z = np.atleast_2d(Z)
        return np.stack([1.0 / (1.0 + np.exp(-(Z @ self.W[p] + self.b[p]))) for p in P])

    def raw_drive(self, P, Z):
        Z = np.atleast_2d(Z)
        G = self.gain(P, Z)                                    # (|P|,N)
        return np.einsum('pn,pd->nd', G * self.amp[list(P)][:, None], self.U[list(P)])

    def field(self, P, Z):
        """The true velocity field X_P(z). Autonomous in t."""
        R = self.raw_drive(P, Z)
        return R / (1.0 + np.linalg.norm(R, axis=1, keepdims=True) / self.kappa)

    def flow(self, P, Z, tau: float = 1.0, n_steps: int = 60):
        """RK4 integration of X_P for exposure tau."""
        Z = np.atleast_2d(Z).astype(float).copy()
        h = tau / n_steps
        for _ in range(n_steps):
            k1 = self.field(P, Z); k2 = self.field(P, Z + h * k1 / 2)
            k3 = self.field(P, Z + h * k2 / 2); k4 = self.field(P, Z + h * k3)
            Z = Z + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return Z

    # ---- populations ---------------------------------------------------------
    def control(self, n: int, seed: int = 1):
        return np.random.default_rng(seed).normal(size=(n, self.d)) * 0.8

    def observe(self, P, n: int, tau: float = 1.0, seed: int = 1, noise: float = 0.05):
        """An observed perturbed population: control pushed through the flow, plus noise."""
        g = np.random.default_rng(seed + 977 * int(sum(P)) + int(1000 * tau))
        Z0 = self.control(n, seed=seed)
        Z1 = self.flow(P, Z0, tau=tau)
        return Z1 + g.normal(size=Z1.shape) * noise

    # ---- diagnostics used by the gates --------------------------------------
    def saturation_scalar(self, p, q, Z):
        """The c in X_pq ~ c * (X_p + X_q); ground-truth, per state."""
        Xp, Xq, Xpq = self.field([p], Z), self.field([q], Z), self.field([p, q], Z)
        A = Xp + Xq
        return np.sum(A * Xpq, axis=1) / np.maximum(np.sum(A * A, axis=1), 1e-12)


def diffeomorphism(d: int, strength: float = 0.35, seed: int = 7):
    """A smooth invertible latent re-chart phi(z) = z + c * A sin(z), with jacobian."""
    A = np.random.default_rng(seed).normal(size=(d, d)) * 0.6

    def phi(Z):
        return Z + strength * (np.sin(np.atleast_2d(Z)) @ A.T)

    def dphi(Z):
        Z = np.atleast_2d(Z)
        return np.eye(d)[None] + strength * (A[None] * np.cos(Z)[:, None, :])

    def phi_inv(W, iters: int = 60):
        Z = np.atleast_2d(W).astype(float).copy()
        for _ in range(iters):
            r = phi(Z) - W
            J = dphi(Z)
            Z = Z - np.linalg.solve(J, r[:, :, None])[:, :, 0]
        return Z

    return phi, dphi, phi_inv
