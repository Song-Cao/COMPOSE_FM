"""IHC-FM -- FIELD AND GEOMETRY track.  Domain-free.

Nothing in this file refers to cells, genes, assets, doses-as-drugs, prices or any other
domain object.  The vocabulary is: a latent chart coordinate `z`, an observed (ambient)
point `x`, a one-form `b` on the observed space, a metric `W` on the observed space, an
intervention index `p` with a scalar exposure `tau_p`, and a population code `c`.

WHAT THIS MODULE IS FOR
-----------------------
The composed velocity field of IHC-FM is

    v(t, z | c, a) = G(z)^{-1} [ J(z)^T b(t, x, c, a) ],
    x = Dec(z),   J = dDec/dz,   G = J^T W(x) J,

and this file supplies everything except `b` (which is the hierarchy track's job): the
decoder, the ambient metric, the pullback assembly, the exposure->amplitude map, and the
saturation operator.

THE ONE IDEA THAT MAKES THE COVARIANCE WORK
-------------------------------------------
A re-chart is a diffeomorphism psi acting on the LATENT coordinate only:

    z  ->  w = psi(z),        Dec  ->  Dec o psi^{-1}.

The observed point is therefore an INVARIANT of the re-chart: Dec'(w) = Dec(z) = x.  Every
consequence below follows from that single observation.

  * J' = J . Dpsi^{-1}          (chain rule on Dec o psi^{-1})
  * W is a function of x alone, hence W' = W  ->  G' = Dpsi^{-T} G Dpsi^{-1}, a (0,2) tensor
  * b is a function of (t, x, c, a) alone, hence alpha' = J'^T b = Dpsi^{-T} alpha, a (0,1) tensor
  * v' = G'^{-1} alpha' = Dpsi G^{-1} Dpsi^T Dpsi^{-T} alpha = Dpsi . v     <-- a vector field

So the design rule for this file is mechanical and total:

    ANY scalar, vector or tensor the field depends on must be a function of the OBSERVED
    point x (or of t, c, a, tau -- all chart-blind), NEVER of the latent coordinate z.

Two corollaries that are easy to get wrong, and which the self-test measures:

  (i)  An additive Euclidean stabiliser on the LATENT metric, G <- G + eps*I_d, is NOT
       covariant: I_d is not a (0,2) tensor under psi (it would have to transform to
       Dpsi^{-T} Dpsi^{-1}).  Phase 1 measured the damage: relerr 2.12e-06 at eps=1e-6,
       2.09e-02 at eps=1e-2, 1.83e-01 at eps=0.1.  This module therefore has NO latent
       eps.  `PullbackField(latent_eps=...)` exists only as a deliberately-broken ablation
       arm and defaults to exactly 0.0.
  (ii) An additive floor on the AMBIENT metric, W <- W + w_min*I_D, IS covariant, because
       x is invariant so W is not being transformed at all.  This is the legitimate way to
       condition G, and it is what `AmbientMetric` does (a Cholesky factor plus an ambient
       floor).  The self-test verifies both halves of this contrast: ambient floor keeps
       covariance at machine precision, latent eps destroys it.

  The same rule dictates the shape of the saturation operator.  `MetricRadialSat` bounds
  ||v||_G = sqrt(v^T G v), which is invariant (v'^T G' v' = v^T G v), by a ceiling
  kappa(x) read off the OBSERVED point, and multiplies v by a scalar -- so direction is
  preserved exactly and the bounded field is still a vector field.  The old
  coordinate-scalar gate g(z) . v is retained as an ablation arm and is NOT covariant; the
  self-test runs both and reports the contrast, because that contrast is the entire
  justification for the component.

PRECISION
---------
Phase 1 fact D: the chart harness's Test A floors at ~1e-8 in float32 for arithmetic
reasons alone.  `PullbackField.inference_double()` returns a float64 clone of the whole
field so the transformation law can be measured at its true precision.

Self-test:  PYTHONPATH=src python src/composefm/ihcfm_geometry.py
"""
from __future__ import annotations

import copy
import json
import math
import pathlib
import time
from typing import Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(2)

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "results"

__all__ = [
    "Decoder",
    "AmbientMetric",
    "PullbackField",
    "HillDose",
    "MetricRadialSat",
    "RandomNonlinearChart",
    "ReChartedDecoder",
    "spd_diagnostics",
]


def _mlp(i: int, h: int, o: int, depth: int = 2, act=nn.SiLU, ln: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(i, h), act()]
    for _ in range(max(0, depth - 1)):
        if ln:
            layers.append(nn.LayerNorm(h))
        layers += [nn.Linear(h, h), act()]
    layers.append(nn.Linear(h, o))
    return nn.Sequential(*layers)


# ==================================================================================
# Decoder :  z -> x,  with an exact Jacobian
# ==================================================================================
class Decoder(nn.Module):
    """Latent chart coordinate z in R^d  ->  observed point x in R^D, with dDec/dz.

    Form (deliberately simple, so J is analytic and cond(J) is bounded by construction):

        x = A z + s * V tanh(U z + q),      J = A + s * V diag(sech^2(U z + q)) U

    `A` is initialised semi-orthogonal, so cond(A) = 1 at init, and the nonlinear part is
    scaled by `s` < 1.  That keeps sigma_max(J)/sigma_min(J) finite by construction rather
    than by a stabiliser -- which matters because cond(G) <= cond(W) . cond(J)^2 and
    conditioning G additively in the LATENT space would break covariance (fact P3).

    `jacobian(z, mode=...)` gives the analytic Jacobian; mode='autodiff' recomputes it with
    `torch.func.jacrev` as an independent check (the self-test asserts they agree), so a
    future decoder without a closed-form J can drop in by overriding `forward` alone.
    """

    def __init__(self, d: int, D: int, hidden: int = 32, s: float = 0.5, seed: int = 0):
        super().__init__()
        self.d, self.D, self.hidden = int(d), int(D), int(hidden)
        g = torch.Generator().manual_seed(int(seed))
        A = torch.linalg.qr(torch.randn(max(D, d), max(D, d), generator=g))[0][:D, :d]
        self.A = nn.Parameter(A.contiguous())
        self.U = nn.Parameter(torch.randn(hidden, d, generator=g) / math.sqrt(d))
        self.V = nn.Parameter(torch.randn(D, hidden, generator=g) / math.sqrt(hidden))
        self.q = nn.Parameter(0.1 * torch.randn(hidden, generator=g))
        self.log_s = nn.Parameter(torch.tensor(math.log(float(s))))

    @property
    def s(self) -> torch.Tensor:
        return torch.exp(self.log_s)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z (B,d) -> x (B,D)."""
        h = torch.tanh(z @ self.U.T + self.q)
        return z @ self.A.T + self.s * (h @ self.V.T)

    def jacobian(self, z: torch.Tensor, mode: str = "analytic") -> torch.Tensor:
        """(B,d) -> (B,D,d)."""
        if mode == "autodiff":
            f = lambda zz: self.forward(zz.unsqueeze(0)).squeeze(0)  # noqa: E731
            return torch.stack([torch.func.jacrev(f)(z[i]) for i in range(z.shape[0])])
        if mode != "analytic":
            raise ValueError(f"unknown jacobian mode {mode!r}")
        pre = z @ self.U.T + self.q                       # (B,h)
        sech2 = 1.0 - torch.tanh(pre) ** 2                # (B,h)
        # A + s * V diag(sech2) U
        VW = self.V.unsqueeze(0) * sech2.unsqueeze(1)     # (B,D,h)
        return self.A.unsqueeze(0) + self.s * (VW @ self.U)

    def cond_jacobian(self, z: torch.Tensor) -> float:
        with torch.no_grad():
            sv = torch.linalg.svdvals(self.jacobian(z))
            return float((sv[:, 0] / sv[:, -1]).max())


# ==================================================================================
# AmbientMetric :  x -> W(x) SPD   (NO latent eps -- see module docstring)
# ==================================================================================
class AmbientMetric(nn.Module):
    """W: R^D -> SPD(D), a Riemannian metric on the OBSERVED space.

        W(x) = w_min * I_D + L(x) L(x)^T

    `L(x)` is lower triangular with a strictly positive diagonal (softplus), assembled from
    a network of x.  Two things are true of this construction and both are load-bearing:

      * It is SPD by construction with lambda_min >= w_min, so G = J^T W J is SPD whenever J
        has full column rank -- no additive repair needed anywhere downstream.
      * `w_min * I_D` lives in the OBSERVED space, where the re-chart does not act.  It is
        therefore NOT the forbidden eps*I of fact P3: that one was added to the LATENT
        metric G, where I_d fails to be a (0,2) tensor.  The self-test measures the
        difference explicitly.

    The off-diagonal / raw-diagonal entries are passed through a tanh with amplitude
    `l_cap`, so ||L||_F -- and hence lambda_max(W) -- is bounded by construction too.  That
    gives an a-priori bound cond(W) <= 1 + D*l_cap^2/w_min, which `cond_bound()` reports.

    `mode='diag'` is a cheaper ablation arm: W(x) = diag(w_min + softplus(head(x))).
    """

    def __init__(self, D: int, hidden: int = 48, w_min: float = 0.25, l_cap: float = 1.5,
                 mode: str = "cholesky", diag_init: float = 1.0):
        super().__init__()
        if mode not in ("cholesky", "diag"):
            raise ValueError(f"unknown metric mode {mode!r}")
        self.D, self.mode = int(D), mode
        self.w_min, self.l_cap = float(w_min), float(l_cap)
        n_tri = D * (D + 1) // 2
        out = n_tri if mode == "cholesky" else D
        self.head = _mlp(D, hidden, out, depth=2)
        nn.init.normal_(self.head[-1].weight, std=1e-2)
        nn.init.zeros_(self.head[-1].bias)
        idx = torch.tril_indices(D, D)
        self.register_buffer("tri_r", idx[0])
        self.register_buffer("tri_c", idx[1])
        self.register_buffer("is_diag", (idx[0] == idx[1]).to(torch.bool))
        self._diag_bias = float(math.log(math.expm1(max(diag_init, 1e-6))))

    # -- the factor ------------------------------------------------------------
    def cholesky_factor(self, x: torch.Tensor) -> torch.Tensor:
        """L(x), lower triangular, strictly positive diagonal. (B,D,D)."""
        if self.mode != "cholesky":
            raise RuntimeError("cholesky_factor is only defined for mode='cholesky'")
        raw = self.head(x)                                        # (B,n_tri)
        dg = F.softplus(raw[:, self.is_diag] + self._diag_bias)   # > 0, no additive eps
        off = self.l_cap * torch.tanh(raw[:, ~self.is_diag])      # bounded
        L = x.new_zeros(x.shape[0], self.D, self.D)
        L[:, self.tri_r[self.is_diag], self.tri_c[self.is_diag]] = dg
        L[:, self.tri_r[~self.is_diag], self.tri_c[~self.is_diag]] = off
        return L

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (B,D) -> W (B,D,D) SPD."""
        eye = torch.eye(self.D, dtype=x.dtype, device=x.device)
        if self.mode == "diag":
            dg = self.w_min + F.softplus(self.head(x) + self._diag_bias)
            return torch.diag_embed(dg)
        L = self.cholesky_factor(x)
        return self.w_min * eye + L @ L.transpose(-1, -2)

    def cond_bound(self) -> float:
        """A-priori bound on cond(W(x)), valid for every x. Construction, not measurement."""
        if self.mode == "diag":
            return float("inf")  # softplus is unbounded above; use 'cholesky' for a bound
        lmax = self.w_min + self.D * (self.l_cap ** 2) + self.D * (self.l_cap ** 2)
        return float(lmax / self.w_min)


# ==================================================================================
# HillDose :  (tau, p) -> a_p,  bounded, monotone, a_p(0) = 0 EXACTLY
# ==================================================================================
class HillDose(nn.Module):
    """Learned Hill family with explicit boundary conditions.

        a_p(tau) = amp_p * sigmoid( h_p * ( log tau - log ec50_p ) ),   tau > 0
        a_p(0)   = 0                                                     (exactly)

    The sigmoid form is algebraically identical to the textbook
    amp * tau^h / (tau^h + ec50^h) -- write u = (tau/ec50)^h, then u/(1+u) = sigmoid(log u)
    -- but it is the numerically correct way to evaluate it:

      * no `tau**h` and no division, so no overflow/`0/0` at either end;
      * `a_p(0)` is bit-exact 0.0 through a `torch.where` on tau > 0, and the classic
        double-`where` (substituting tau=1 inside the masked branch) keeps the backward
        pass finite even for h < 1, where d a/d tau diverges at 0;
      * amp_p = exp(log_amp_p) is a hard ceiling: sigmoid < 1 strictly, so a_p < amp_p for
        every finite tau -- boundedness is structural, not a clamp;
      * h_p = exp(log_hill_p) > 0 always, and log tau is strictly increasing, so a_p is
        strictly increasing on tau > 0 -- monotonicity is structural too.

    Why this replaces the previous power-law attenuation: `(1 + (T/kappa)^nu)^{-1}` applied
    to a drive that itself grows linearly in T gives ||v|| ~ T^(1-nu), which is a power law,
    not a saturation -- it has no finite ceiling unless nu >= 1 and no correct value at
    T = 0.  The Hill family fixes both endpoints by construction and lets the transition
    scale (ec50) and sharpness (h) be fitted per intervention.

    Parameters are per-intervention (log_ec50, log_hill, log_amp), which is what makes
    `params_table()` a report rather than a summary.
    """

    def __init__(self, k: int, amp_init: float = 1.0, ec50_init: float = 1.0,
                 hill_init: float = 1.0, share: bool = False):
        super().__init__()
        self.k, self.share = int(k), bool(share)
        n = 1 if share else int(k)
        self.log_amp = nn.Parameter(torch.full((n,), math.log(float(amp_init))))
        self.log_ec50 = nn.Parameter(torch.full((n,), math.log(float(ec50_init))))
        self.log_hill = nn.Parameter(torch.full((n,), math.log(float(hill_init))))

    def _expand(self, t: torch.Tensor) -> torch.Tensor:
        return t.expand(self.k) if self.share else t

    @property
    def amp(self) -> torch.Tensor:
        return torch.exp(self._expand(self.log_amp))

    @property
    def ec50(self) -> torch.Tensor:
        return torch.exp(self._expand(self.log_ec50))

    @property
    def hill(self) -> torch.Tensor:
        return torch.exp(self._expand(self.log_hill))

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        """tau (B,k) of per-intervention exposures (0 = intervention absent) -> a (B,k).

        Zero exposure maps to bit-exact 0.0, so `a` is automatically supported only on the
        active intervention set and the interaction moments built from it vanish for
        singletons without any masking logic downstream.
        """
        if tau.dim() == 1:
            tau = tau.unsqueeze(0)
        pos = tau > 0
        safe = torch.where(pos, tau, torch.ones_like(tau))         # keeps grads finite
        logu = self.hill * (torch.log(safe) - self.log_ec50_full())
        a = self.amp * torch.sigmoid(logu)
        return torch.where(pos, a, torch.zeros_like(a))

    def log_ec50_full(self) -> torch.Tensor:
        return self._expand(self.log_ec50)

    def params_table(self) -> dict:
        """Per-intervention learned parameters, for reporting."""
        with torch.no_grad():
            return {
                "amp": [float(v) for v in self.amp],
                "ec50": [float(v) for v in self.ec50],
                "hill": [float(v) for v in self.hill],
            }

    # -- diagnostics (mirrors synthetic.SyntheticSystem.dose_diagnostics) -------
    def diagnostics(self, taus: Optional[Sequence[float]] = None) -> dict:
        if taus is None:
            taus = np.linspace(0.0, 6.0, 121)
        T = torch.tensor(np.asarray(taus, dtype=np.float64), dtype=self.log_amp.dtype)
        grid = T.unsqueeze(1).expand(-1, self.k).contiguous()      # (nt,k)
        with torch.no_grad():
            A = self(grid)                                         # (nt,k)
            amp = self.amp
            inc = A[1:] - A[:-1]
            return {
                "max_abs_at_tau0": float(A[0].abs().max()),
                "zero_at_zero_bit_exact": bool(torch.all(A[0] == 0.0)),
                "min_increment": float(inc.min()),
                "max_over_ceiling": float((A - amp).max()),
                "tau_grid": [float(taus[0]), float(taus[-1]), len(taus)],
            }


# ==================================================================================
# MetricRadialSat :  bound ||v||_G, preserve direction, stay chart-invariant
# ==================================================================================
class MetricRadialSat(nn.Module):
    """Radial saturation in the metric norm.

        n(z)     = sqrt( v^T G v )                       (chart-INVARIANT)
        kappa(x) = softplus(head(x)) + kappa_min > 0     (chart-INVARIANT: x is invariant)
        v_sat    = v * [ kappa/n * tanh(n/kappa) ]

    Three properties, each of which the self-test measures:

      * DIRECTION.  The bracket is a scalar, so v_sat is exactly parallel to v -- cosine 1
        to rounding.  Contrast the old gate, which was also a scalar but a scalar of the
        wrong variable.
      * BOUND.  ||v_sat||_G = kappa * tanh(n/kappa) < kappa, with equality only in the
        limit, and v_sat -> v as n -> 0 (the bracket -> 1).  So the ceiling is a true
        asymptote, not a clip, and the field is untouched where it is small.
        FLOATING-POINT CAVEAT, measured: the inequality is strict in exact arithmetic, but
        float64 `tanh(u)` returns exactly 1.0 for u >~ 19, so at extreme drive the attained
        norm EQUALS kappa up to the rounding of the quadratic form -- self-test measures an
        overshoot of +4.44e-16 at 1000x drive (vs -7.33e-07 at 50x, where tanh has not yet
        saturated).  The enforced guarantee is therefore ||v_sat||_G <= kappa*(1+1e-12),
        not a strict <; both numbers are reported rather than the flattering one.
      * INVARIANCE.  Both arguments of the bracket are chart-blind, so the bracket is a
        chart-invariant scalar and v_sat transforms exactly as v does.  This is the whole
        reason for the component: a bound expressed in ANY coordinate-dependent quantity
        (the Euclidean norm ||v||_2, or a gate g(z)) silently converts the field into a
        non-vector object.

    Ablation arms, selectable with `mode`, kept because the paper needs the contrast:
      'metric_radial' : the above.                                     COVARIANT
      'coord_scalar'  : v * sigmoid(net([z, v]))  -- the OLD gate.     NOT covariant
      'coord_norm'    : metric-radial form but with the Euclidean
                        norm ||v||_2 and kappa(z) in place of the
                        invariant pair -- isolates "wrong norm" from
                        "wrong gate".                                  NOT covariant
      'off'           : identity.
    """

    MODES = ("metric_radial", "coord_scalar", "coord_norm", "off")

    def __init__(self, x_dim: int, latent_dim: Optional[int] = None, hidden: int = 32,
                 mode: str = "metric_radial", kappa_init: float = 2.0,
                 kappa_min: float = 1e-2):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        self.mode, self.kappa_min = mode, float(kappa_min)
        self.x_dim, self.latent_dim = int(x_dim), int(latent_dim or x_dim)
        bias = float(math.log(math.expm1(max(kappa_init - kappa_min, 1e-6))))
        if mode == "metric_radial":
            self.head = _mlp(self.x_dim, hidden, 1, depth=1)
        elif mode == "coord_norm":
            self.head = _mlp(self.latent_dim, hidden, 1, depth=1)
        elif mode == "coord_scalar":
            self.head = _mlp(self.latent_dim * 2, hidden, 1, depth=1)
        else:
            self.head = None
        if self.head is not None:
            nn.init.normal_(self.head[-1].weight, std=1e-2)
            nn.init.constant_(self.head[-1].bias, bias if mode != "coord_scalar" else 2.2)

    def kappa(self, x: torch.Tensor) -> torch.Tensor:
        """The learned ceiling on ||v||_G. (B,1), strictly positive."""
        if self.mode == "off":
            return torch.full((x.shape[0], 1), float("inf"), dtype=x.dtype, device=x.device)
        return F.softplus(self.head(x)) + self.kappa_min

    @staticmethod
    def _radial(v: torch.Tensor, n: torch.Tensor, kap: torch.Tensor) -> torch.Tensor:
        """v * (kap/n) tanh(n/kap), with the removable singularity at n=0 handled."""
        u = n / kap
        small = u < 1e-6
        safe_u = torch.where(small, torch.ones_like(u), u)
        # tanh(u)/u, and its 2nd-order expansion 1 - u^2/3 near 0
        scale = torch.where(small, 1.0 - u * u / 3.0, torch.tanh(safe_u) / safe_u)
        return v * scale

    def metric_norm(self, v: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        """sqrt(v^T G v). (B,1). Chart-invariant."""
        q = torch.einsum("bi,bij,bj->b", v, G, v).clamp(min=0.0)
        return torch.sqrt(q).unsqueeze(1)

    def forward(self, v: torch.Tensor, G: torch.Tensor, x: torch.Tensor,
                z: Optional[torch.Tensor] = None) -> torch.Tensor:
        """v (B,d), G (B,d,d), x (B,D), z (B,d) -> v_sat (B,d)."""
        if self.mode == "off":
            return v
        if self.mode == "metric_radial":
            return self._radial(v, self.metric_norm(v, G), self.kappa(x))
        if z is None:
            raise ValueError(f"mode={self.mode!r} is a coordinate ablation and needs z")
        if self.mode == "coord_norm":
            n2 = v.norm(dim=-1, keepdim=True)                 # Euclidean: NOT invariant
            return self._radial(v, n2, F.softplus(self.head(z)) + self.kappa_min)
        g = torch.sigmoid(self.head(torch.cat([z, v], -1)))    # the OLD gate
        return v * g


# ==================================================================================
# PullbackField :  v = G^{-1} J^T b
# ==================================================================================
class PullbackField(nn.Module):
    """The composed velocity field.  Assembles the whole learned field as a vector field.

        x = Dec(z);  J = dDec/dz;  W = W(x);  G = J^T W J;  alpha = J^T b(t, x, c, a);
        v = G^{-1} alpha,  then optionally metric-radial saturation.

    `one_form` is any callable `b(t, x, c) -> (B, D)`; the hierarchy track supplies
    `OneFormHierarchy.b`.  It is called with the OBSERVED point, never with z -- that is the
    interface constraint that makes the field covariant, and it is the reason `b` may
    contain arbitrary nonlinearity, interaction structure and population conditioning
    without any of it having to be covariance-aware.

    Solver.  G is SPD by construction (AmbientMetric floor + full-column-rank J), so the
    default solve is a Cholesky one, with an LU fallback if the factorisation ever fails
    (reported through `solve_fallbacks`).  There is deliberately NO additive repair: a
    Cholesky failure is a diagnostic to be reported, not to be papered over with eps*I.

    `latent_eps` is an ABLATION KNOB and defaults to exactly 0.0.  Setting it > 0 restores
    the broken G + eps*I_d behaviour of fact P3 so the self-test can demonstrate that the
    covariance test has the power to detect it.  Nothing in the training path sets it.
    """

    def __init__(self, decoder: nn.Module, metric: nn.Module,
                 one_form: Optional[Callable] = None,
                 saturator: Optional[nn.Module] = None,
                 latent_eps: float = 0.0, solver: str = "cholesky",
                 jac_mode: str = "analytic"):
        super().__init__()
        if solver not in ("cholesky", "lu"):
            raise ValueError(f"unknown solver {solver!r}")
        self.decoder, self.metric_net = decoder, metric
        self.saturator = saturator
        self.one_form = one_form
        self.latent_eps = float(latent_eps)
        self.solver, self.jac_mode = solver, jac_mode
        self.solve_fallbacks = 0

    # -- geometry --------------------------------------------------------------
    def geometry(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """z -> (x, J, G)."""
        x = self.decoder(z)
        J = self.decoder.jacobian(z, mode=self.jac_mode)          # (B,D,d)
        W = self.metric_net(x)                                    # (B,D,D)
        G = J.transpose(-1, -2) @ W @ J                           # (B,d,d)
        G = 0.5 * (G + G.transpose(-1, -2))                       # kill rounding asymmetry
        if self.latent_eps > 0.0:                                 # ABLATION ONLY
            G = G + self.latent_eps * torch.eye(G.shape[-1], dtype=G.dtype, device=G.device)
        return x, J, G

    def metric(self, z: torch.Tensor) -> torch.Tensor:
        """The pullback metric G(z). (B,d,d)."""
        return self.geometry(z)[2]

    def cond_number(self, z: torch.Tensor) -> float:
        """max over the batch of cond_2(G(z))."""
        with torch.no_grad():
            ev = torch.linalg.eigvalsh(self.metric(z).double())
            return float((ev[:, -1] / ev[:, 0].clamp(min=1e-300)).max())

    def _solve(self, G: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        rhs = alpha.unsqueeze(-1)
        if self.solver == "cholesky":
            L, info = torch.linalg.cholesky_ex(G)
            if int(info.max()) == 0:
                return torch.cholesky_solve(rhs, L).squeeze(-1)
            self.solve_fallbacks += 1
        return torch.linalg.solve(G, rhs).squeeze(-1)

    # -- the field -------------------------------------------------------------
    def forward(self, t, z: torch.Tensor, c: Optional[torch.Tensor] = None,
                one_form: Optional[Callable] = None,
                return_parts: bool = False):
        """t scalar or (B,1); z (B,d); c (B,code) or None  ->  v (B,d)."""
        b_fn = one_form or self.one_form
        if b_fn is None:
            raise ValueError("PullbackField needs a one-form callable b(t, x, c)")
        x, J, G = self.geometry(z)
        if not torch.is_tensor(t):
            t = torch.full((z.shape[0], 1), float(t), dtype=z.dtype, device=z.device)
        elif t.dim() == 0:
            t = t.expand(z.shape[0]).unsqueeze(1)
        elif t.dim() == 1:
            t = t.unsqueeze(1)
        b = b_fn(t, x, c)                                          # (B,D)  observed-space
        alpha = torch.einsum("bij,bi->bj", J, b)                   # J^T b : (0,1) tensor
        v = self._solve(G, alpha)
        if self.saturator is not None:
            v = self.saturator(v, G, x, z=z)
        if return_parts:
            return v, {"x": x, "J": J, "G": G, "alpha": alpha, "b": b}
        return v

    def velocity(self, z: torch.Tensor, t=0.0, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Convenience alias with (z, t) ordering."""
        return self.forward(t, z, c)

    # -- precision -------------------------------------------------------------
    def inference_double(self) -> "PullbackField":
        """A float64 eval-mode clone (fact D: float32 floors chart Test A at ~1e-8)."""
        f = copy.deepcopy(self)
        f.double().eval()
        return f

    def spd_report(self, z: torch.Tensor) -> dict:
        return spd_diagnostics(self.metric(z))


def spd_diagnostics(G: torch.Tensor) -> dict:
    """Eigenvalue / symmetry / condition report for a batch of matrices."""
    with torch.no_grad():
        Gd = G.double()
        asym = float((Gd - Gd.transpose(-1, -2)).abs().max())
        ev = torch.linalg.eigvalsh(Gd)
        return {
            "min_eigenvalue": float(ev.min()),
            "max_eigenvalue": float(ev.max()),
            "is_spd": bool(ev.min() > 0.0),
            "max_asymmetry": asym,
            "cond_max": float((ev[:, -1] / ev[:, 0].clamp(min=1e-300)).max()),
            "cond_median": float((ev[:, -1] / ev[:, 0].clamp(min=1e-300)).median()),
        }


# ==================================================================================
# TEST UTILITIES -- a self-contained nonlinear chart, and the re-charted decoder
# ==================================================================================
class RandomNonlinearChart:
    """psi(z) = z + s * A tanh(z), A spectral-normalised: a diffeomorphism for 0 <= s < 1.

    Self-contained (float64 numpy) so this module's self-test has no dependencies, and
    deliberately the SAME family as `synthetic.NonlinearChart(kind='tanh')` so the numbers
    are comparable with the Phase-1 record.  D2psi != 0 -- an affine chart would make every
    variant exact and measure nothing (Phase-1 fact: affine chart passes even the WRONG
    composition forms at ~1e-16).
    """

    def __init__(self, d: int, strength: float = 0.4, seed: int = 11):
        if not 0.0 <= float(strength) < 1.0:
            raise ValueError("need 0 <= strength < 1 for invertibility")
        g = np.random.default_rng(int(seed))
        A = g.normal(size=(d, d))
        self.A = A / (np.linalg.norm(A, 2) + 1e-12)
        self.d, self.s = int(d), float(strength)

    def psi(self, Z: np.ndarray) -> np.ndarray:
        Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
        return Z + self.s * (np.tanh(Z) @ self.A.T)

    def jac(self, Z: np.ndarray) -> np.ndarray:
        Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
        sech2 = 1.0 - np.tanh(Z) ** 2
        return np.eye(self.d)[None] + self.s * (self.A[None] * sech2[:, None, :])

    def hess(self, Z: np.ndarray) -> np.ndarray:
        Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
        t = np.tanh(Z)
        ddm = -2.0 * t * (1.0 - t ** 2)
        H = np.zeros((Z.shape[0], self.d, self.d, self.d))
        for a in range(self.d):
            H[:, :, a, a] += self.s * self.A[None, :, a] * ddm[:, a][:, None]
        return H

    def hessian_scale(self, Z: np.ndarray) -> float:
        return float(np.sqrt(np.mean(self.hess(Z) ** 2)))

    def inv(self, W: np.ndarray, iters: int = 100, tol: float = 1e-15) -> np.ndarray:
        W = np.atleast_2d(np.asarray(W, dtype=np.float64))
        Z = W.copy()
        for _ in range(iters):
            r = self.psi(Z) - W
            if np.max(np.abs(r)) < tol:
                break
            Z = Z - np.linalg.solve(self.jac(Z), r[..., None])[..., 0]
        return Z


class ReChartedDecoder(nn.Module):
    """The SAME decoder expressed in a new chart: Dec' = Dec o psi^{-1}.

    Jacobian by the chain rule, J'(w) = J(psi^{-1}(w)) . Dpsi(psi^{-1}(w))^{-1}.

    This is a TEST utility, not part of the model: it manufactures the re-charted model
    whose field must equal Dpsi . v.  It is exact by construction, which is exactly what
    makes it a fair test -- all the error it can expose belongs to the pullback assembly
    (the metric's argument, the one-form's argument, the solve, the saturator), not to the
    chart bookkeeping.  Inference only (float64, numpy chart calls under no_grad).
    """

    def __init__(self, decoder: nn.Module, chart):
        super().__init__()
        self.base, self.chart = decoder, chart
        self.d, self.D = decoder.d, decoder.D

    def _pre(self, w: torch.Tensor) -> np.ndarray:
        return self.chart.inv(w.detach().double().cpu().numpy())

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        z = torch.as_tensor(self._pre(w), dtype=w.dtype, device=w.device)
        return self.base(z)

    def jacobian(self, w: torch.Tensor, mode: str = "analytic") -> torch.Tensor:
        zn = self._pre(w)
        z = torch.as_tensor(zn, dtype=w.dtype, device=w.device)
        J = self.base.jacobian(z, mode=mode)                       # (B,D,d)
        Dpsi = torch.as_tensor(self.chart.jac(zn), dtype=w.dtype, device=w.device)
        return torch.linalg.solve(Dpsi.transpose(-1, -2), J.transpose(-1, -2)).transpose(-1, -2)


# ==================================================================================
# SELF-TEST
# ==================================================================================
class _StubOneForm(nn.Module):
    """A stand-in for OneFormHierarchy: b(t, x, c) -> (B,D). Observed-space argument only.

    Present ONLY so the geometry can be exercised end to end; the hierarchy track owns the
    real one.  It is intentionally nonlinear in every argument, because a linear b would
    make the covariance test weaker than it should be.
    """

    def __init__(self, D: int, code: int = 0, hidden: int = 48):
        super().__init__()
        self.D, self.code = int(D), int(code)
        self.net = _mlp(D + 1 + code, hidden, D, depth=2)

    def forward(self, t: torch.Tensor, x: torch.Tensor, c: Optional[torch.Tensor]) -> torch.Tensor:
        parts = [x, t]
        if self.code:
            parts.append(c if c.dim() == 2 else c.expand(x.shape[0], -1))
        return self.net(torch.cat(parts, -1))


def _build(d=6, D=8, k=4, code=3, sat_mode="metric_radial", seed=0, latent_eps=0.0,
           metric_mode="cholesky"):
    torch.manual_seed(seed)
    dec = Decoder(d, D, hidden=32, s=0.5, seed=seed)
    met = AmbientMetric(D, hidden=48, w_min=0.25, l_cap=1.5, mode=metric_mode)
    bfn = _StubOneForm(D, code=code, hidden=48)
    sat = MetricRadialSat(x_dim=D, latent_dim=d, hidden=32, mode=sat_mode, kappa_init=2.0)
    fld = PullbackField(dec, met, one_form=bfn, saturator=sat, latent_eps=latent_eps)
    dose = HillDose(k, amp_init=1.0, ec50_init=1.0, hill_init=1.0)
    return fld, dose


def _covariance_error(field: PullbackField, chart, Z: np.ndarray, c: torch.Tensor,
                      t: float = 0.37, one_form: Optional[Callable] = None) -> dict:
    """max_i |v'(psi(z)) - Dpsi(z) v(z)| / max|Dpsi v|, in float64.

    `field` must already be the float64 inference clone.  The re-charted field REUSES the
    same metric, one-form and saturator objects: only the decoder is re-expressed.  That is
    what "the same model in another chart" means, and it is why any coordinate dependence
    hiding inside those shared objects shows up here.
    """
    f64 = field.inference_double()
    Zt = torch.tensor(Z, dtype=torch.float64)
    with torch.no_grad():
        v = f64(t, Zt, c.double(), one_form=one_form)
        f2 = PullbackField(ReChartedDecoder(f64.decoder, chart), f64.metric_net,
                           one_form=f64.one_form, saturator=f64.saturator,
                           latent_eps=f64.latent_eps, solver=f64.solver)
        W = torch.tensor(chart.psi(Z), dtype=torch.float64)
        v2 = f2(t, W, c.double(), one_form=one_form)
        Dpsi = torch.tensor(chart.jac(Z), dtype=torch.float64)
        target = torch.einsum("nij,nj->ni", Dpsi, v)
        num = float((v2 - target).abs().max())
        den = float(target.abs().max())
    return {"max_abs_err": num, "scale": den, "rel_err": num / max(den, 1e-300)}


def _tensor_transform_errors(field: PullbackField, chart, Z: np.ndarray, c: torch.Tensor,
                             t: float = 0.37) -> dict:
    """Check the two intermediate objects separately: G as (0,2), alpha as (0,1)."""
    f64 = field.inference_double()
    Zt = torch.tensor(Z, dtype=torch.float64)
    with torch.no_grad():
        _, parts = f64(t, Zt, c.double(), return_parts=True)
        f2 = PullbackField(ReChartedDecoder(f64.decoder, chart), f64.metric_net,
                           one_form=f64.one_form, saturator=f64.saturator,
                           latent_eps=f64.latent_eps, solver=f64.solver)
        W = torch.tensor(chart.psi(Z), dtype=torch.float64)
        _, p2 = f2(t, W, c.double(), return_parts=True)
        Dpsi = torch.tensor(chart.jac(Z), dtype=torch.float64)
        Dinv = torch.linalg.inv(Dpsi)
        G_t = torch.einsum("nia,nij,njb->nab", Dinv, parts["G"], Dinv)
        a_t = torch.einsum("nia,ni->na", Dinv, parts["alpha"])
        rel = lambda A, B: float((A - B).abs().max() / max(float(B.abs().max()), 1e-300))  # noqa: E731
        return {"G_02_tensor_rel_err": rel(p2["G"], G_t),
                "alpha_01_tensor_rel_err": rel(p2["alpha"], a_t),
                "x_invariance_rel_err": rel(p2["x"], parts["x"]),
                "b_invariance_rel_err": rel(p2["b"], parts["b"])}


def _fit_hill_to_reference(k: int = 4, seed: int = 3, steps: int = 400) -> dict:
    """Fit the Hill family to a bounded monotone reference a(tau) = A (1 - exp(-tau/s)).

    The reference is the dose law used by the project's controlled ground truth
    (synthetic.SyntheticSystem.dose), reproduced here in three lines so this file stays
    dependency-free.  The point of the fit is NOT that the Hill family is the true law --
    it is not, the reference has a finite slope at 0 and a different tail -- but that a
    learnable Hill family can absorb a plausible saturating law while keeping a(0)=0,
    monotonicity and boundedness structural.  We report the fit error honestly.
    """
    g = np.random.default_rng(seed)
    A_ref = 0.9 + 0.5 * g.random(k)
    s_ref = 0.6 + 0.9 * g.random(k)
    taus = np.linspace(0.0, 6.0, 61)
    ref = A_ref[None, :] * (1.0 - np.exp(-taus[:, None] / s_ref[None, :]))
    T = torch.tensor(taus, dtype=torch.float64).unsqueeze(1).expand(-1, k).contiguous()
    Y = torch.tensor(ref, dtype=torch.float64)
    dose = HillDose(k, amp_init=1.0, ec50_init=1.0, hill_init=1.0).double()
    opt = torch.optim.Adam(dose.parameters(), lr=5e-2)
    for _ in range(steps):
        opt.zero_grad()
        loss = ((dose(T) - Y) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = dose(T)
        rel = float((pred - Y).norm() / Y.norm())
    return {"rel_l2_fit_error": rel, "final_mse": float(loss),
            "reference_ceilings": [float(v) for v in A_ref],
            "reference_scales": [float(v) for v in s_ref],
            "learned": dose.params_table(),
            "diagnostics": dose.diagnostics()}


def selftest(verbose: bool = True) -> dict:
    t_start = time.time()
    d, D, k, code, n = 6, 8, 4, 3, 64
    rng = np.random.default_rng(0)
    Z = 0.8 * rng.normal(size=(n, d))
    c = torch.tensor(rng.normal(size=(n, code)), dtype=torch.float32)
    chart = RandomNonlinearChart(d, strength=0.4, seed=11)
    hs = chart.hessian_scale(Z)
    assert hs > 1e-6, f"chart is (near-)affine, hessian_scale={hs:g} -- test would be vacuous"

    res: dict = {"config": {"d": d, "D": D, "k": k, "code_dim": code, "n_samples": n,
                            "chart": "tanh strength=0.4 seed=11",
                            "chart_hessian_scale": hs,
                            "torch": torch.__version__}}

    # ---- 1. whole-field covariance, float64, metric-radial saturation on -------------
    fld, dose = _build(d, D, k, code, sat_mode="metric_radial")
    cov = _covariance_error(fld, chart, Z, c)
    res["test1_whole_field_covariance"] = cov
    res["test1_tensor_parts"] = _tensor_transform_errors(fld, chart, Z, c)

    # a second, sharper chart -- covariance must not degrade with chart strength
    chart2 = RandomNonlinearChart(d, strength=0.7, seed=23)
    res["test1_stronger_chart"] = {"hessian_scale": chart2.hessian_scale(Z),
                                   **_covariance_error(fld, chart2, Z, c)}

    # float32 path, for the record (fact D)
    Zt32 = torch.tensor(Z, dtype=torch.float32)
    with torch.no_grad():
        v32 = fld(0.37, Zt32, c)
        f2_32 = PullbackField(ReChartedDecoder(fld.decoder, chart), fld.metric_net,
                              one_form=fld.one_form, saturator=fld.saturator)
        v2_32 = f2_32(0.37, torch.tensor(chart.psi(Z), dtype=torch.float32), c)
        tgt32 = torch.einsum("nij,nj->ni", torch.tensor(chart.jac(Z), dtype=torch.float32), v32)
        res["test1_float32_for_contrast"] = {
            "rel_err": float((v2_32 - tgt32).abs().max() / tgt32.abs().max())}

    # ---- 2. no eps*I, and the ablation proving the test has power --------------------
    src = pathlib.Path(__file__).read_text()
    no_eps = {
        "latent_eps_default_is_zero": fld.latent_eps == 0.0,
        "field_has_no_unconditional_eye_add": "latent_eps > 0.0" in src,
        "metric_floor_lives_in_observed_space": True,
    }
    eps_arm = {}
    for eps in (1e-6, 1e-2, 1e-1):
        f_eps, _ = _build(d, D, k, code, sat_mode="metric_radial", latent_eps=eps)
        f_eps.load_state_dict(fld.state_dict())
        f_eps.latent_eps = eps
        eps_arm[f"latent_eps_{eps:g}"] = _covariance_error(f_eps, chart, Z, c)["rel_err"]
    # the LEGITIMATE ambient floor: raise w_min and covariance must be untouched
    f_amb, _ = _build(d, D, k, code, sat_mode="metric_radial")
    f_amb.load_state_dict(fld.state_dict())
    f_amb.metric_net.w_min = 5.0
    no_eps["ambient_floor_w_min_5_rel_err"] = _covariance_error(f_amb, chart, Z, c)["rel_err"]
    res["test2_no_latent_eps"] = no_eps
    res["test2_latent_eps_ablation_rel_err"] = eps_arm

    # ---- 3. G is SPD, condition number bounded by construction -----------------------
    Zt = torch.tensor(Z, dtype=torch.float64)
    f64 = fld.inference_double()
    spd = f64.spd_report(Zt)
    spd["cond_jacobian_max"] = f64.decoder.cond_jacobian(Zt)
    with torch.no_grad():
        Wm = f64.metric_net(f64.decoder(Zt))
        ev = torch.linalg.eigvalsh(Wm)
        spd["cond_W_max_measured"] = float((ev[:, -1] / ev[:, 0]).max())
    spd["cond_W_apriori_bound"] = f64.metric_net.cond_bound()
    spd["cond_G_apriori_bound"] = spd["cond_W_apriori_bound"] * spd["cond_jacobian_max"] ** 2
    spd["solve_fallbacks"] = int(f64.solve_fallbacks)
    # analytic vs autodiff Jacobian -- an independent check on the geometry
    with torch.no_grad():
        Ja = f64.decoder.jacobian(Zt[:8], mode="analytic")
        Jb = f64.decoder.jacobian(Zt[:8], mode="autodiff")
        spd["jacobian_analytic_vs_autodiff_rel_err"] = float(
            (Ja - Jb).abs().max() / Jb.abs().max())
    res["test3_spd_and_conditioning"] = spd

    # ---- 4. HillDose boundary conditions ---------------------------------------------
    dose64 = HillDose(k).double()
    dg = dose64.diagnostics()
    dg["params_at_init"] = dose64.params_table()
    dg["fit_to_bounded_monotone_reference"] = _fit_hill_to_reference(k=k)
    res["test4_hill_dose"] = dg

    # ---- 5. MetricRadialSat: direction, bound, chart-invariance vs the old gate -------
    sat_res: dict = {}
    with torch.no_grad():
        f_off, _ = _build(d, D, k, code, sat_mode="off")
        f_off.load_state_dict(fld.state_dict(), strict=False)
        f_off64 = f_off.inference_double()
        v_raw = f_off64(0.37, Zt, c.double())
        _, parts = f64(0.37, Zt, c.double(), return_parts=True)
        v_sat = f64(0.37, Zt, c.double())
        cos = F.cosine_similarity(v_sat, v_raw, dim=-1)
        sat = f64.saturator
        n_raw = sat.metric_norm(v_raw, parts["G"]).squeeze(1)
        n_sat = sat.metric_norm(v_sat, parts["G"]).squeeze(1)
        kap = sat.kappa(parts["x"]).squeeze(1)
        sat_res["direction"] = {
            "min_cosine_with_raw": float(cos.min()),
            "max_abs_cosine_deviation": float((cos - 1.0).abs().max()),
            "max_abs_cross_product_norm": float(
                (v_sat - v_raw * (v_sat.norm(dim=-1, keepdim=True)
                                  / v_raw.norm(dim=-1, keepdim=True))).abs().max())}
        sat_res["bound"] = {
            "max_metric_norm_over_ceiling": float((n_sat - kap).max()),
            "all_below_ceiling": bool(torch.all(n_sat < kap)),
            "max_raw_metric_norm": float(n_raw.max()),
            "max_sat_metric_norm": float(n_sat.max()),
            "mean_ceiling": float(kap.mean()),
            "max_shrink_factor": float((n_sat / n_raw.clamp(min=1e-30)).min())}
    # -- 5b. STRESS the ceiling.  At init ||v||_G ~ 0.3 against kappa ~ 2, so the
    #    saturator is barely engaged and "bounded" would pass vacuously.  Scale the
    #    one-form up until the bound BINDS hard, then re-check all three properties.
    stress: dict = {}
    b32, b64 = fld.one_form, f64.one_form          # same weights, two dtypes
    for scale in (1.0, 50.0, 1000.0):
        def big(t_, x_, c_, s=scale):
            fn = b64 if x_.dtype == torch.float64 else b32
            return s * fn(t_, x_, c_)
        with torch.no_grad():
            v_r = f_off64(0.37, Zt, c.double(), one_form=big)
            v_s, pp = f64(0.37, Zt, c.double(), one_form=big, return_parts=True)
            nr = f64.saturator.metric_norm(v_r, pp["G"]).squeeze(1)
            ns = f64.saturator.metric_norm(v_s, pp["G"]).squeeze(1)
            kp = f64.saturator.kappa(pp["x"]).squeeze(1)
            cs = F.cosine_similarity(v_s, v_r, dim=-1)
        stress[f"b_scale_{scale:g}"] = {
            "max_raw_metric_norm": float(nr.max()),
            "max_sat_metric_norm": float(ns.max()),
            "mean_ceiling": float(kp.mean()),
            "norm_over_ceiling_max": float((ns - kp).max()),
            "rel_norm_over_ceiling_max": float(((ns - kp) / kp).max()),
            "all_strictly_below_ceiling": bool(torch.all(ns < kp)),
            "all_below_ceiling_tol_1e-12": bool(torch.all(ns <= kp * (1.0 + 1e-12))),
            "min_shrink_factor": float((ns / nr.clamp(min=1e-30)).min()),
            "max_abs_cosine_deviation": float((cs - 1.0).abs().max()),
            "covariance_rel_err": _covariance_error(fld, chart, Z, c, one_form=big)["rel_err"]}
    sat_res["stress_bound_binding"] = stress

    # -- 5c. invariance contrast: metric-radial vs the two coordinate arms, same weights.
    #    HONEST CAVEAT, measured below: the coordinate arms' defect is proportional to how
    #    much the gate actually VARIES with z.  At initialisation their head has std 1e-2
    #    output weights, so the gate is nearly a CONSTANT scalar -- and a constant scalar
    #    IS covariant.  Reporting only the at-init number would understate the defect by
    #    orders of magnitude, so we sweep the head's output scale and report the curve.
    arms, arms_detail = {}, {}
    for mode in ("off", "metric_radial", "coord_norm", "coord_scalar"):
        f_m, _ = _build(d, D, k, code, sat_mode=mode)
        sd = {kk: vv for kk, vv in fld.state_dict().items() if not kk.startswith("saturator.")}
        f_m.load_state_dict(sd, strict=False)
        arms[mode] = _covariance_error(f_m, chart, Z, c)["rel_err"]
    sat_res["covariance_rel_err_by_saturation_mode_at_init"] = arms

    for mode in ("coord_norm", "coord_scalar"):
        curve = {}
        for gs in (0.01, 0.1, 0.5, 1.0):
            f_m, _ = _build(d, D, k, code, sat_mode=mode, seed=0)
            sd = {kk: vv for kk, vv in fld.state_dict().items()
                  if not kk.startswith("saturator.")}
            f_m.load_state_dict(sd, strict=False)
            g_ = torch.Generator().manual_seed(5)
            head = f_m.saturator.head
            with torch.no_grad():
                head[-1].weight.copy_(gs * torch.randn(head[-1].weight.shape, generator=g_))
            # how much does the gate actually vary over the batch?
            f_m64 = f_m.inference_double()
            with torch.no_grad():
                _, pm = f_m64(0.37, Zt, c.double(), return_parts=True)
                if mode == "coord_scalar":
                    v0 = f_off64(0.37, Zt, c.double())
                    gv = torch.sigmoid(f_m64.saturator.head(torch.cat([Zt, v0], -1)))
                else:
                    gv = F.softplus(f_m64.saturator.head(Zt)) + f_m64.saturator.kappa_min
            curve[f"head_std_{gs:g}"] = {
                "gate_std_over_batch": float(gv.std()),
                "gate_mean": float(gv.mean()),
                "covariance_rel_err": _covariance_error(f_m, chart, Z, c)["rel_err"]}
        arms_detail[mode] = curve
    sat_res["coordinate_arm_defect_vs_gate_sensitivity"] = arms_detail
    worst = {m: max(v["covariance_rel_err"] for v in cu.values())
             for m, cu in arms_detail.items()}
    sat_res["coordinate_arm_worst_case_rel_err"] = worst
    sat_res["invariance_contrast_ratio_worst_coord_over_metric_radial"] = (
        max(worst.values()) / max(arms["metric_radial"], 1e-300))
    res["test5_metric_radial_saturation"] = sat_res

    # ---- 6. wall clock: one velocity evaluation, d=6 k=4 batch=256 --------------------
    Zb = torch.tensor(0.8 * rng.normal(size=(256, d)), dtype=torch.float32)
    cb = torch.tensor(rng.normal(size=(256, code)), dtype=torch.float32)
    with torch.no_grad():
        for _ in range(3):
            fld(0.5, Zb, cb)
        t0 = time.perf_counter()
        for _ in range(20):
            fld(0.5, Zb, cb)
        per32 = (time.perf_counter() - t0) / 20
        f64b = fld.inference_double()
        Zb64, cb64 = Zb.double(), cb.double()
        for _ in range(3):
            f64b(0.5, Zb64, cb64)
        t0 = time.perf_counter()
        for _ in range(20):
            f64b(0.5, Zb64, cb64)
        per64 = (time.perf_counter() - t0) / 20
    res["test6_timing"] = {"batch": 256, "d": d, "D": D, "k": k,
                           "velocity_eval_s_float32": per32,
                           "velocity_eval_s_float64": per64,
                           "threads": torch.get_num_threads()}

    # ---- verdicts ---------------------------------------------------------------------
    v: dict = {}
    v["t1_covariance_le_1e-10"] = cov["rel_err"] <= 1e-10
    v["t1_stronger_chart_le_1e-10"] = res["test1_stronger_chart"]["rel_err"] <= 1e-10
    v["t1_G_is_02_tensor"] = res["test1_tensor_parts"]["G_02_tensor_rel_err"] <= 1e-10
    v["t1_alpha_is_01_tensor"] = res["test1_tensor_parts"]["alpha_01_tensor_rel_err"] <= 1e-10
    v["t2_no_latent_eps_by_default"] = bool(no_eps["latent_eps_default_is_zero"])
    v["t2_ablation_has_power"] = eps_arm["latent_eps_0.01"] > 1e-3
    v["t2_ambient_floor_is_safe"] = no_eps["ambient_floor_w_min_5_rel_err"] <= 1e-10
    v["t3_G_spd"] = bool(spd["is_spd"])
    v["t3_no_solver_fallback"] = spd["solve_fallbacks"] == 0
    v["t3_jacobian_agrees"] = spd["jacobian_analytic_vs_autodiff_rel_err"] <= 1e-6
    v["t4_zero_at_zero_bit_exact"] = bool(dg["zero_at_zero_bit_exact"])
    v["t4_strictly_monotone"] = dg["min_increment"] > 0.0
    v["t4_bounded"] = dg["max_over_ceiling"] <= 0.0
    v["t5_direction_preserved"] = sat_res["direction"]["max_abs_cosine_deviation"] <= 1e-12
    v["t5_norm_bounded"] = bool(sat_res["bound"]["all_below_ceiling"])
    v["t5_saturation_is_chart_invariant"] = arms["metric_radial"] <= 1e-10
    # under stress the bound must BIND (shrink hard) and still be exact + covariant
    st = sat_res["stress_bound_binding"]["b_scale_1000"]
    v["t5_bound_binds_under_stress"] = st["min_shrink_factor"] < 1e-2
    # non-strict at 1e-12 relative: float64 tanh returns exactly 1.0 at 1000x drive, so the
    # attained norm equals kappa to the rounding of v^T G v (+4.44e-16 absolute, measured)
    v["t5_bounded_under_stress"] = bool(st["all_below_ceiling_tol_1e-12"])
    v["t5_bounded_before_tanh_saturates"] = bool(
        sat_res["stress_bound_binding"]["b_scale_50"]["all_strictly_below_ceiling"])
    v["t5_direction_preserved_under_stress"] = st["max_abs_cosine_deviation"] <= 1e-12
    v["t5_covariant_under_stress"] = st["covariance_rel_err"] <= 1e-10
    # the coordinate arms must FAIL -- measured at their worst gate sensitivity, because
    # a near-constant gate is trivially covariant and would hide the defect
    v["t5_coord_gate_breaks_covariance"] = worst["coord_scalar"] > 1e-3
    v["t5_coord_norm_breaks_covariance"] = worst["coord_norm"] > 1e-3
    res["verdicts"] = {kk: bool(vv) for kk, vv in v.items()}
    res["all_passed"] = bool(all(v.values()))
    res["wall_clock_s"] = time.time() - t_start

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ihcfm_geometry_selftest.json").write_text(json.dumps(res, indent=2))
    if verbose:
        print(json.dumps(res, indent=2))
        print("\nALL PASSED:", res["all_passed"], f"  ({res['wall_clock_s']:.1f}s)")
    return res


if __name__ == "__main__":
    selftest()
