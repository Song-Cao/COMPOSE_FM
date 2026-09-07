"""Numerical verification of the three correctness propositions of IHC-FM,
plus the double-counting diagnosis.

Everything here is DOMAIN-FREE: a latent chart Z = R^d, a chart change psi,
a decoder into an ambient R^Dim, and vector fields on Z.  No biology, no finance.

Differentiation strategy
------------------------
Every derivative is available by TWO independent routes:

  (a) analytic  -- hand-coded Jacobians / Hessians of the closed-form maps;
  (b) complex-step -- f'(x) = Im f(x + i h) / h with h = 1e-20.

Route (b) has NO truncation error (the O(h^2) term is real, so it does not
contaminate the imaginary part) and no subtractive cancellation, so it returns
derivatives at full float64 precision -- unlike a finite difference, which
would floor every residual below at ~1e-8 and make the 1e-16 claims
unverifiable.  All maps below are implemented with complex-analytic primitives
(matmul, tanh, log1p/exp, linear solve) so route (b) is valid.

Proposition 3 is the one place where an algebraic identity could be smuggled in
as an assumption: J_{D o psi^-1}(z') = J_D(z) Dpsi(z)^{-1} is the chain rule, and
if we simply *used* it the covariance test would be a tautology.  So the chart-2
decoder Jacobian is ALSO obtained by complex-step differentiating the actually
composed map z' -> D(newton_psi_inv(z')), with the Newton solve carried through
in complex arithmetic.  Check P3.2 reports the agreement.

Run:  python experiments/verify_propositions.py
Writes: results/propositions.json
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

# float64 / complex128 throughout; no float32 anywhere in this file.
DT = np.float64
CT = np.complex128
CSTEP_H = 1e-20


# --------------------------------------------------------------------------- #
#  complex-analytic primitives                                                #
# --------------------------------------------------------------------------- #
def tanh(x):
    return np.tanh(x)


def sech2(x):
    """d/dx tanh."""
    t = np.tanh(x)
    return 1.0 - t * t


def dsech2(x):
    """d^2/dx^2 tanh = -2 tanh sech^2."""
    t = np.tanh(x)
    return -2.0 * t * (1.0 - t * t)


def softplus(x):
    """log(1+exp(x)), overflow-safe and complex-analytic on the used domain."""
    xr = np.real(x)
    big = xr > 30.0
    xs = np.where(big, np.zeros_like(x), x)
    return np.where(big, x, np.log1p(np.exp(xs)))


# --------------------------------------------------------------------------- #
#  complex-step differentiation                                               #
# --------------------------------------------------------------------------- #
def cs_jac(f, z, h=CSTEP_H):
    """Jacobian of f: R^d -> R^m at a single point z, by complex step.

    Returns (m, d).  Exact to float64 -- no truncation, no cancellation.
    """
    z = np.asarray(z, dtype=CT)
    d = z.shape[0]
    cols = []
    for j in range(d):
        dz = np.zeros(d, dtype=CT)
        dz[j] = 1j * h
        cols.append(np.imag(np.asarray(f(z + dz), dtype=CT)) / h)
    return np.stack(cols, axis=-1)


def cs_dir(f, z, u, h=CSTEP_H):
    """Directional derivative Df(z)[u] by complex step (one evaluation)."""
    z = np.asarray(z, dtype=CT)
    u = np.asarray(u, dtype=CT)
    return np.imag(np.asarray(f(z + 1j * h * u), dtype=CT)) / h


def cs_dmat(f, z, h=CSTEP_H):
    """d_l F_{ij}(z) for a matrix-valued F, returned as (d, i, j)."""
    z = np.asarray(z, dtype=CT)
    d = z.shape[0]
    out = []
    for l in range(d):
        dz = np.zeros(d, dtype=CT)
        dz[l] = 1j * h
        out.append(np.imag(np.asarray(f(z + dz), dtype=CT)) / h)
    return np.stack(out, axis=0)


# --------------------------------------------------------------------------- #
#  random nonlinear chart change  psi: R^d -> R^d                             #
# --------------------------------------------------------------------------- #
class Chart:
    """psi(z) = A z + c + s * C tanh(B z + e).

    GLOBAL DIFFEOMORPHISM BY CONSTRUCTION.  Because |tanh'| <= 1, the nonlinear
    part has Lipschitz constant at most s ||C||_2 ||B||_2, so psi is a global
    diffeomorphism of R^d as soon as

        s * ||C||_2 ||B||_2  <  sigma_min(A).

    A random draw does NOT satisfy this automatically: at some seeds the drawn
    s ||C|| ||B|| exceeds sigma_min(A), Dpsi becomes near-singular somewhere on
    the sample (measured sigma_min down to 2.0e-3), and the Newton inverse then
    converges to a DIFFERENT branch -- psi(psi^-1(z')) = z' still holds to 3e-16
    while psi^-1(psi(z)) - z = 8.2e-1, so a psi-space residual check does not
    catch it.  Every covariance identity below is a statement about a single
    chart, so a wrong branch breaks them (measured: P2 law 2.46, P3 field 0.63).
    We therefore RESCALE the nonlinear block at construction to enforce the
    contraction bound with a safety factor, and assert it.  Second derivative

        D2psi^a_ij = s * sum_m C_am * dsech2(Bz+e)_m * B_mi * B_mj

    is symmetric in (i, j) by construction, as Clairaut requires.
    `affine=True` zeroes s, giving the exactness control of Proposition 1.
    """

    def __init__(self, d, rng, s=0.35, m=6, affine=False, cond=3.0):
        self.d = d
        self.affine = affine
        self.s = 0.0 if affine else s
        # well-conditioned A with a prescribed condition number
        Q1, _ = np.linalg.qr(rng.standard_normal((d, d)))
        Q2, _ = np.linalg.qr(rng.standard_normal((d, d)))
        sv = np.logspace(0.0, -np.log10(cond), d)
        self.A = (Q1 * sv) @ Q2.T
        self.c = 0.2 * rng.standard_normal(d)
        self.B = rng.standard_normal((m, d)) / np.sqrt(d)
        self.C = rng.standard_normal((d, m)) / np.sqrt(m)
        self.e = 0.3 * rng.standard_normal(m)
        # Enforce the global-diffeomorphism contraction bound, seed-independently:
        #   s ||C|| ||B|| <= safety * sigma_min(A),  safety < 1.
        # This keeps psi genuinely nonlinear (s stays O(0.1), D2psi != 0) while
        # making the inverse branch unique for EVERY draw.
        self.safety = 0.6
        if not affine:
            lip = np.linalg.norm(self.C, 2) * np.linalg.norm(self.B, 2)
            smin = np.linalg.svd(self.A, compute_uv=False)[-1]
            cap = self.safety * smin / max(lip, 1e-300)
            self.s = float(min(self.s, cap))
        self.lipschitz_ratio = (0.0 if affine else float(
            self.s * np.linalg.norm(self.C, 2) * np.linalg.norm(self.B, 2)
            / np.linalg.svd(self.A, compute_uv=False)[-1]))
        assert self.lipschitz_ratio < 1.0, "chart is not a global diffeomorphism"

    def __call__(self, z):
        z = np.asarray(z)
        lin = z @ self.A.T + self.c
        if self.affine:
            return lin
        return lin + self.s * (tanh(z @ self.B.T + self.e) @ self.C.T)

    def jac(self, z):
        """Dpsi(z), analytic. (d, d)"""
        if self.affine:
            return self.A.astype(np.asarray(z).dtype)
        a = np.asarray(z) @ self.B.T + self.e
        return self.A + self.s * (self.C * sech2(a)) @ self.B

    def hess(self, z):
        """D2psi(z) as (a, i, j), analytic and exactly symmetric in (i, j)."""
        dt = np.asarray(z).dtype
        if self.affine:
            return np.zeros((self.d, self.d, self.d), dtype=dt)
        a = np.asarray(z) @ self.B.T + self.e
        w = self.s * dsech2(a)                       # (m,)
        # sum_m (s C_am w_m) B_mi B_mj
        return np.einsum('am,mi,mj->aij', self.C * w, self.B, self.B)

    def h2(self, z, u, v):
        """D2psi(z)[u, v] -- contraction on VECTORS (not fields). (d,)"""
        return np.einsum('aij,i,j->a', self.hess(z), u, v)

    def inv(self, zp, iters=60):
        """psi^{-1}(zp) by Newton; carried through in the input dtype so that
        complex-step differentiation of the composed map is valid."""
        zp = np.asarray(zp)
        z = np.linalg.solve(self.A, (zp - self.c))
        for _ in range(iters):
            r = self(z) - zp
            z = z - np.linalg.solve(self.jac(z), r)
        return z

    def margin(self, Z):
        """Diffeomorphism margin: min over samples of sigma_min(Dpsi)."""
        return float(min(np.linalg.svd(self.jac(z), compute_uv=False)[-1] for z in Z))


# --------------------------------------------------------------------------- #
#  random nonlinear decoder  D: R^d -> R^Dim  (an immersion)                  #
# --------------------------------------------------------------------------- #
class Decoder:
    """D(z) = W2 tanh(W1 z + b1) + W3 z + b2,  Dim >= d.

    The linear skip W3 keeps D an immersion (rank d Jacobian) robustly: the
    tanh block alone could lose rank in saturation.  `min_sigma()` measures
    sigma_min(J_D) over the sample so the immersion assumption of Proposition 3
    is a MEASURED number, not an assertion.
    """

    def __init__(self, d, Dim, rng, hidden=24, scale=1.0):
        self.d, self.Dim = d, Dim
        self.W1 = rng.standard_normal((hidden, d)) / np.sqrt(d)
        self.b1 = 0.3 * rng.standard_normal(hidden)
        self.W2 = scale * rng.standard_normal((Dim, hidden)) / np.sqrt(hidden)
        self.W3 = rng.standard_normal((Dim, d)) / np.sqrt(d)
        self.b2 = 0.1 * rng.standard_normal(Dim)

    def __call__(self, z):
        z = np.asarray(z)
        return tanh(z @ self.W1.T + self.b1) @ self.W2.T + z @ self.W3.T + self.b2

    def jac(self, z):
        """J_D(z), analytic. (Dim, d)"""
        a = np.asarray(z) @ self.W1.T + self.b1
        return (self.W2 * sech2(a)) @ self.W1 + self.W3

    def min_sigma(self, Z):
        return float(min(np.linalg.svd(self.jac(z), compute_uv=False)[-1] for z in Z))


class PulledBackDecoder:
    """D' = D o psi^{-1}, the SAME map read in the new chart z' = psi(z).

    Its Jacobian is obtained by complex-step differentiation of the composed
    map, with the Newton inverse carried through in complex arithmetic.  We do
    NOT substitute J_D(z) Dpsi(z)^{-1}: that is the chain rule, and asserting it
    would make the Proposition 3 test circular.  Check P3.2 compares the two.
    """

    def __init__(self, dec, chart):
        self.dec, self.chart = dec, chart

    def __call__(self, zp):
        return self.dec(self.chart.inv(zp))

    def jac_cs(self, zp):
        return cs_jac(self.__call__, zp)


# --------------------------------------------------------------------------- #
#  ambient PD weight W(x)  and the pullback construction                      #
# --------------------------------------------------------------------------- #
class AmbientWeight:
    """State-dependent ambient positive-definite weight W(x), x in R^Dim.

    W(x) = U diag(softplus(V x + w) + w_floor) U^T  with U a fixed orthogonal
    frame.  Strictly PD by construction, and a genuine function of the AMBIENT
    point -- which is the whole reason the pullback metric G = J^T W J is a
    chart-covariant object: W is evaluated at x = D(z) = D'(z'), the same
    ambient point in either chart.
    """

    def __init__(self, Dim, rng, floor=0.5, gain=0.4):
        self.Dim, self.floor = Dim, floor
        self.U, _ = np.linalg.qr(rng.standard_normal((Dim, Dim)))
        self.V = gain * rng.standard_normal((Dim, Dim)) / np.sqrt(Dim)
        self.w = 0.2 * rng.standard_normal(Dim)

    def __call__(self, x):
        s = softplus(np.asarray(x) @ self.V.T + self.w) + self.floor
        return (self.U * s) @ self.U.T


class AmbientOneForm:
    """Ambient covector field b(x) in (R^Dim)^*.

    It is a ONE-FORM: its components are indexed downstairs, and it is pulled
    back to the latent chart by alpha = J^T b, with NO Jacobian inverse.  This
    is what makes the construction of Proposition 3 work -- the pullback of a
    one-form needs no invertibility of J (which is Dim x d and not square).
    """

    def __init__(self, Dim, rng, gain=0.6):
        self.Dim = Dim
        self.P = gain * rng.standard_normal((Dim, Dim)) / np.sqrt(Dim)
        self.q = 0.3 * rng.standard_normal(Dim)
        self.c = rng.standard_normal(Dim) / np.sqrt(Dim)

    def __call__(self, x):
        x = np.asarray(x)
        return tanh(x @ self.P.T + self.q) + self.c


def pullback_field(dec, W, b, z, eps=0.0, euclid_stab=False):
    """The learned intrinsic field of Proposition 3, in whatever chart `dec` reads.

        G(z)     = J_D(z)^T W(D(z)) J_D(z)          (pullback metric, d x d, PD)
        alpha(z) = J_D(z)^T b(D(z))                 (intrinsic lift, a one-form)
        v(z)     = G(z)^{-1} alpha(z)               (a vector field)

    `eps > 0` with euclid_stab=True adds the RAW EUCLIDEAN stabiliser
    G -> G + eps I.  That term is NOT covariant (I is not a (0,2) tensor), and
    the ablation in check P3.4 measures exactly how much covariance it destroys.

    Returns (v, G, alpha, cond(G)).
    """
    J = dec.jac(z)
    x = dec(z)
    Wx = W(x)
    G = J.T @ Wx @ J
    G = 0.5 * (G + G.T)                    # symmetrise: kills asymmetric round-off
    if euclid_stab and eps > 0.0:
        G = G + eps * np.eye(G.shape[0], dtype=G.dtype)
    alpha = J.T @ b(x)
    v = np.linalg.solve(G, alpha)
    condG = float(np.linalg.cond(np.real(G))) if np.iscomplexobj(G) else float(
        np.linalg.cond(G))
    return v, G, alpha, condG


# --------------------------------------------------------------------------- #
#  learned metric + its Levi-Civita connection (Proposition 2)                #
# --------------------------------------------------------------------------- #
class Metric:
    """Learned Riemannian metric G(z) = L(z) L(z)^T on the latent chart.

    Mirrors src/composefm/compose.py::Connection.metric (Cholesky parameterisation,
    softplus diagonal + floor), but in float64/complex128 so the residuals below
    are not limited by float32.
    """

    def __init__(self, d, rng, floor=0.3, gain=0.5):
        self.d, self.floor = d, floor
        self.n_tri = d * (d + 1) // 2
        self.i, self.j = np.tril_indices(d)
        self.N = gain * rng.standard_normal((self.n_tri, d)) / np.sqrt(d)
        self.m = 0.4 * rng.standard_normal(self.n_tri)

    def L(self, z):
        vals = tanh(np.asarray(z) @ self.N.T + self.m)
        L = np.zeros((self.d, self.d), dtype=np.asarray(vals).dtype)
        L[self.i, self.j] = vals
        dg = np.arange(self.d)
        L[dg, dg] = softplus(L[dg, dg]) + self.floor
        return L

    def __call__(self, z):
        L = self.L(z)
        G = L @ L.T
        return 0.5 * (G + G.T)


def christoffel(metric, z):
    """Levi-Civita connection coefficients of `metric` at z.

        Gamma^k_ij = 1/2 G^{kl} ( d_i G_lj + d_j G_li - d_l G_ij )

    Derivatives by complex step, so this is exact to float64.  Returned as
    (k, i, j) and SYMMETRISED in (i, j) -- the Levi-Civita connection is
    torsion-free, and the transformation law of Proposition 2 only holds for the
    symmetric part (fact B: an unsymmetrised test gives wrong answers).
    """
    G = metric(np.asarray(z, dtype=DT))
    dG = cs_dmat(metric, z)                                  # (l, i, j) = d_l G_ij
    term = (np.einsum('ilj->lij', dG)
            + np.einsum('jli->lij', dG)
            - np.einsum('lij->lij', dG))
    Gam = 0.5 * np.linalg.solve(G, term.reshape(G.shape[0], -1)).reshape(
        G.shape[0], G.shape[0], G.shape[0])
    return 0.5 * (Gam + np.swapaxes(Gam, 1, 2))


def gamma_vec(metric, z, u, v):
    """Gamma(z)[u, v]^k -- contraction on VECTORS, matching compose.py::gamma."""
    return np.einsum('kij,i,j->k', christoffel(metric, z), u, v)


# --------------------------------------------------------------------------- #
#  primitive vector fields (same family as src/composefm/synthetic.py)        #
# --------------------------------------------------------------------------- #
class Primitive:
    """X_p(z) = M_p tanh(z), spectrally normalised then scaled -- the primitive
    family of src/composefm/synthetic.py.  Analytic Jacobian
    DX_p(z) = M_p diag(sech^2 z), and complex-analytic, so complex-step applies.
    """

    def __init__(self, d, rng, amp=1.0):
        M = rng.standard_normal((d, d))
        M = M / np.linalg.svd(M, compute_uv=False)[0]
        self.M = amp * M

    def __call__(self, z):
        return tanh(np.asarray(z)) @ self.M.T

    def jac(self, z):
        return self.M * sech2(np.asarray(z))

    def pushforward(self, chart):
        """The SAME field read in the chart z' = psi(z):  X'(z') = Dpsi . X(psi^-1 z')."""
        def Xp(zp):
            z = chart.inv(zp)
            return chart.jac(z) @ self(z)
        return Xp


def rk4_flow(f, z0, T, n):
    """Integrate dz/dt = f(z) for time T in n RK4 steps.  Local error O(h^5), so
    with n large the integrator is far below every O(T) effect measured here."""
    z = np.array(z0, dtype=DT)
    h = T / n
    for _ in range(n):
        k1 = f(z)
        k2 = f(z + 0.5 * h * k1)
        k3 = f(z + 0.5 * h * k2)
        k4 = f(z + h * k3)
        z = z + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return z


# --------------------------------------------------------------------------- #
#  reporting helpers                                                          #
# --------------------------------------------------------------------------- #
def relerr(a, b):
    """||a - b|| / max(||b||, tiny) -- relative discrepancy of a against b."""
    a = np.asarray(a, dtype=DT).ravel()
    b = np.asarray(b, dtype=DT).ravel()
    den = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / (den if den > 1e-300 else 1.0))


ROWS = []


def record(pid, name, value, thresh, kind="rel", note=""):
    """Log one check.  `thresh` is the pass ceiling; `value` must be <= it."""
    ok = bool(np.isfinite(value) and value <= thresh)
    ROWS.append(dict(prop=pid, check=name, value=float(value),
                     threshold=float(thresh), kind=kind, passed=ok, note=note))
    return ok


# --------------------------------------------------------------------------- #
#  PROPOSITION 1 -- the displacement defect                                   #
# --------------------------------------------------------------------------- #
def prop1(rng, d=4, n_states=8, n_rep=60):
    """psi(z + delta) - psi(z) = Dpsi(z) delta + O(||delta||^2).

    Checks
      P1.1  the O(||delta||^2) rate: log-log slope of the defect norm -> 2
      P1.2  the leading coefficient is exactly 1/2 D2psi[delta, delta], tested as
            CONSTANCY of relerr/||delta|| across the usable window (see below)
      P1.3  AFFINE CONTROL: defect identically 0 (not small -- machine zero)
      P1.4  growth with intervention count k          (ensemble-averaged)
      P1.5  growth with exposure tau                  (ensemble-averaged)
      P1.6  growth with latent anisotropy at FIXED ||delta|| (see note)

    Two measurement subtleties, both diagnosed numerically rather than tuned away:

    * WINDOW FOR P1.2.  The defect is formed as psi(z+delta) - psi(z) - Dpsi delta,
      a difference of O(1) quantities whose true size is O(||delta||^2).  Below
      ||delta|| ~ 1e-4 the float64 round-off of the O(1) terms (~1e-16 absolute)
      is no longer negligible against that, so the measured relative error stops
      following the theoretical O(||delta||) law and rises again.  This is
      SUBTRACTIVE CANCELLATION IN THE MEASUREMENT, not a failure of the
      proposition.  We therefore test the coefficient on the window
      ||delta|| in [1e-4, 1e-2], where relerr/||delta|| is constant to about
      three significant digits (measured spread 1.42e-3 relative), and
      separately RECORD the cancellation floor below it as a diagnostic.

    * ANISOTROPY (P1.6) -- a partially NEGATIVE result, reported as measured.
      Sweeps are averaged over `n_rep` primitive draws and `n_chart` chart draws,
      because a single draw is too noisy to read a trend from.  Three channels
      were separated:

        (a) ||delta||-INFLATION channel -- CONFIRMED.  Anisotropic latent geometry
            makes the same field amplitudes produce a LONGER displacement, and
            since the relative defect is proportional to ||delta|| (P1.2), the
            defect grows.  Monotone over 12 chart draws in the unsaturated
            regime.  This is the channel the paper may claim.

        (b) DIRECTION-CONCENTRATION channel at FIXED ||delta|| -- NOT SUPPORTED.
            Concentrating delta along the stretched axes at fixed length has no
            chart-independent effect.  Averaged over a random anisotropy frame
            and 12 chart draws (`P1_aniso_fixed_norm_random_frame`, state scale
            ZS, exposure 0.05) the relative defect is FLAT across a 20x stretch:
            5.582e-4 -> 5.605e-4, a relative range of 8.1e-3 with no trend.
            With the stretch axes AXIS-ALIGNED an apparent trend does appear
            (`P1_aniso_fixed_norm_axis_aligned`: 6.415e-4 -> 5.082e-4 here) but
            its SIGN is not stable across chart draws -- an independent probe at
            a larger state scale gave the opposite, INCREASING, trend.  What the
            aligned sweep measures is alignment between the stretch axes and the
            eigendirections of D2psi, i.e. a property of the particular chart,
            not an effect of anisotropy.  Both keys are reported.

        (c) SATURATION turnover.  At a large state scale the naive sweep is
            NON-monotone: this chart's nonlinearity is a bounded tanh, so
            stretching z drives its argument into saturation where D2psi -> 0.
            Recorded with the saturation diagnostic that explains it.
    """
    out = {}
    chart = Chart(d, rng, s=0.35)
    Z = 0.6 * rng.standard_normal((n_states, d))
    out["chart_diffeo_margin_sigma_min"] = chart.margin(Z)

    dirs = rng.standard_normal((n_states, d))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    # --- P1.1 : quadratic rate ---------------------------------------------
    # Fitted on ||delta|| <= 1e-2.  At 1e-1 the O(||delta||^3) remainder is
    # already ~7% (recorded below), which biases the fitted slope off 2.
    fit_scales = [1e-2, 1e-3, 1e-4]
    all_scales = [1e-1] + fit_scales + [1e-5, 1e-6]
    rate, coef, lead = {}, {}, {}
    for s in all_scales:
        dn, qn, le = [], [], []
        for z, u in zip(Z, dirs):
            delta = s * u
            defect = chart(z + delta) - chart(z) - chart.jac(z) @ delta
            dn.append(np.linalg.norm(defect))
            qn.append(np.linalg.norm(defect) / s ** 2)
            le.append(relerr(defect, 0.5 * chart.h2(z, delta, delta)))
        rate[f"{s:g}"] = float(np.mean(dn))
        coef[f"{s:g}"] = float(np.mean(qn))
        lead[f"{s:g}"] = float(np.mean(le))
    out["P1_defect_norm_by_scale"] = rate
    out["P1_defect_over_delta2_by_scale"] = coef
    out["P1_leading_coeff_relerr_by_scale"] = lead

    slope = np.polyfit(np.log(fit_scales),
                       np.log([rate[f"{s:g}"] for s in fit_scales]), 1)[0]
    out["P1_quadratic_slope"] = float(slope)
    out["P1_quadratic_slope_fit_window"] = [min(fit_scales), max(fit_scales)]
    out["P1_cubic_remainder_at_delta_1e-1"] = lead["0.1"]
    record("P1", "O(||d||^2) rate: log-log slope vs 2", abs(slope - 2.0), 2e-3,
           kind="abs", note=f"slope={slope:.6f} on ||d||<=1e-2")

    # --- P1.2 : leading coefficient, via constancy of relerr/||delta|| ------
    norm_coef = {f"{s:g}": lead[f"{s:g}"] / s for s in fit_scales}
    out["P1_leading_coeff_relerr_over_delta"] = norm_coef
    vals = np.array(list(norm_coef.values()))
    spread = float(np.max(np.abs(vals - vals.mean())) / vals.mean())
    out["P1_leading_coeff_constancy_spread"] = spread
    out["P1_leading_coeff_limit"] = float(vals.mean())
    out["P1_cancellation_floor_relerr"] = {f"{s:g}": lead[f"{s:g}"]
                                           for s in (1e-5, 1e-6)}
    record("P1", "leading term == 1/2 D2psi[d,d] (O(d) rate)", spread, 5e-3,
           note=f"relerr/||d|| = {vals.mean():.5f} const over [1e-4,1e-2]")
    # Falsifiable form of the cancellation diagnosis.  Two competing explanations
    # for the rise in relative error below ||d|| ~ 1e-4:
    #
    #   (H_roundoff) the expansion is fine and we are seeing float64 round-off of
    #       the O(1) terms in the measurement.  Then the ABSOLUTE residual is
    #       pinned near eps_mach * ||psi(z)|| (~1e-16) REGARDLESS of ||d||, and
    #       the relative error rises only because we divide by a shrinking signal.
    #   (H_extra) the expansion is missing a term larger than the cubic one.
    #       Then the absolute residual would be LARGER than that floor, growing
    #       with the size of the missing term.
    #
    # The two are distinguished by the absolute residual, so we test it directly:
    # observed / (eps_mach ||psi||) must sit within 2 decades of 1.  For scale,
    # the genuine cubic term here is far BELOW the floor and hence invisible: it
    # is calibrated from the measured cubic remainder at ||d||=1e-1 and scaled by
    # ||d||^3, giving ~1e-20 at ||d||=1e-6 -- four decades under eps_mach ||psi||,
    # which is exactly why the floor, not the cubic term, sets what we observe.
    abs_resid, floor_pred = [], []
    for z, u in zip(Z, dirs):
        delta = 1e-6 * u
        defect = chart(z + delta) - chart(z) - chart.jac(z) @ delta
        abs_resid.append(float(np.linalg.norm(
            defect - 0.5 * chart.h2(z, delta, delta))))
        floor_pred.append(float(np.finfo(DT).eps * np.linalg.norm(chart(z))))
    ratio = float(np.mean(abs_resid) / np.mean(floor_pred))
    out["P1_cancellation_abs_residual_at_1e-6"] = float(np.mean(abs_resid))
    out["P1_cancellation_predicted_floor"] = float(np.mean(floor_pred))
    out["P1_cancellation_ratio_to_eps_floor"] = ratio
    # cubic size at 1e-6, CALIBRATED from the measured remainder at 1e-1
    # (relative remainder there x defect norm there) and scaled by (1e-6/1e-1)^3
    cubic_1e1 = lead["0.1"] * rate["0.1"]
    out["P1_cubic_abs_term_at_delta_1e-1"] = float(cubic_1e1)
    out["P1_cubic_abs_term_at_1e-6_extrapolated"] = float(cubic_1e1 * 1e-15)
    record("P1", "sub-window rise is round-off (abs err ~ eps*||psi||)",
           abs(np.log10(max(ratio, 1e-300))), 2.0, kind="abs",
           note=f"abs {np.mean(abs_resid):.2e} vs eps floor "
                f"{np.mean(floor_pred):.2e} (ratio {ratio:.2f}); cubic term "
                f"~{cubic_1e1 * 1e-15:.1e}, far under the floor")

    # --- P1.3 : affine control, defect is IDENTICALLY zero -------------------
    aff = Chart(d, rng, affine=True)
    worst = 0.0
    for z, u in zip(Z, dirs):
        for s in all_scales:
            delta = s * u
            dfe = aff(z + delta) - aff(z) - aff.jac(z) @ delta
            worst = max(worst, float(np.max(np.abs(dfe))))
    out["P1_affine_defect_abs_max"] = worst
    out["P1_affine_hessian_abs_max"] = float(np.max(np.abs(aff.hess(Z[0]))))
    record("P1", "AFFINE control: defect == 0 exactly", worst, 1e-15,
           kind="abs", note="machine-zero; D2psi vanishes identically")

    # --- P1.4/5/6 : scaling in k, tau, anisotropy ---------------------------
    # DOMAIN-FREE: an "intervention" is an index into a set of primitive fields.
    # `zscale` sets the state scale; the small value keeps the chart's tanh out of
    # saturation so channel (c) does not contaminate channels (a)/(b).
    def sweep(k, tau, aniso, seed, n_chart=1, zscale=0.6,
              fixed_norm=False, random_frame=False):
        rel, dn, sat = [], [], []
        for c in range(n_chart):
            if n_chart == 1:
                ch, ZZ = chart, Z
            else:
                cr = np.random.default_rng(2000 + c)
                ch = Chart(d, cr, s=0.35)
                ZZ = zscale * cr.standard_normal((n_states, d))
            S = np.logspace(0.0, np.log10(aniso), d) if aniso > 1 else np.ones(d)
            r = np.random.default_rng(seed)
            for _ in range(n_rep):
                prims = [Primitive(d, r, amp=1.0) for _ in range(6)]
                Q = (np.linalg.qr(r.standard_normal((d, d)))[0]
                     if random_frame else np.eye(d))
                for z in ZZ:
                    if fixed_norm:
                        # anisotropic DIRECTION at fixed length: isolates
                        # D2psi[delta,delta] with ||delta|| held constant
                        zz = z
                        u = Q @ (r.standard_normal(d) * S)
                        delta = tau * u / np.linalg.norm(u)
                    else:
                        zz = S * z
                        delta = tau * sum(p(zz) for p in prims[:k])
                    dfe = ch(zz + delta) - ch(zz) - ch.jac(zz) @ delta
                    nrm = np.linalg.norm(delta)
                    rel.append(np.linalg.norm(dfe) / max(nrm, 1e-300))
                    dn.append(nrm)
                    sat.append(np.max(np.abs(tanh(zz @ ch.B.T + ch.e))))
        return (float(np.mean(rel)), float(np.mean(dn)), float(np.mean(sat)))

    ks = [1, 2, 3, 4, 5, 6]
    taus = [0.05, 0.1, 0.2, 0.4, 0.8]
    tau_fit = [0.05, 0.1, 0.2]
    anis = [1.0, 2.0, 5.0, 10.0, 20.0]
    NC, ZS = 12, 0.05

    out["P1_defect_vs_k"] = {str(k): sweep(k, 0.2, 1.0, 5)[0] for k in ks}
    out["P1_delta_norm_vs_k"] = {str(k): sweep(k, 0.2, 1.0, 5)[1] for k in ks}
    out["P1_defect_vs_tau"] = {f"{t:g}": sweep(3, t, 1.0, 5)[0] for t in taus}

    # (a) ||delta||-inflation channel, unsaturated, averaged over NC charts
    infl = {f"{a:g}": sweep(3, 0.2, a, 9, n_chart=NC, zscale=ZS) for a in anis}
    out["P1_aniso_inflation_rel_defect"] = {k: v[0] for k, v in infl.items()}
    out["P1_aniso_inflation_delta_norm"] = {k: v[1] for k, v in infl.items()}
    out["P1_aniso_inflation_saturation"] = {k: v[2] for k, v in infl.items()}
    out["P1_aniso_inflation_rel_over_delta"] = {
        k: v[0] / v[1] for k, v in infl.items()}

    # (b) direction-concentration at fixed ||delta||, RANDOM frame -> flat
    out["P1_aniso_fixed_norm_random_frame"] = {
        f"{a:g}": sweep(3, 0.05, a, 3, n_chart=NC, zscale=ZS,
                        fixed_norm=True, random_frame=True)[0] for a in anis}
    # ... and with the stretch AXIS-ALIGNED, whose apparent trend is chart-specific
    out["P1_aniso_fixed_norm_axis_aligned"] = {
        f"{a:g}": sweep(3, 0.05, a, 3, fixed_norm=True)[0] for a in anis}

    # (c) naive sweep at a LARGE state scale: non-monotone via saturation
    naive = {f"{a:g}": sweep(3, 0.2, a, 9, n_chart=NC, zscale=0.6) for a in anis}
    out["P1_defect_vs_aniso_naive"] = {k: v[0] for k, v in naive.items()}
    out["P1_aniso_naive_saturation"] = {k: v[2] for k, v in naive.items()}
    out["P1_hessian_norm_vs_aniso"] = {}
    for a in anis:
        S = np.logspace(0.0, np.log10(a), d) if a > 1 else np.ones(d)
        out["P1_hessian_norm_vs_aniso"][f"{a:g}"] = float(
            np.mean([np.linalg.norm(chart.hess(S * z)) for z in Z]))

    for nm, key, xs in (("k", "P1_defect_vs_k", ks),
                        ("tau", "P1_defect_vs_tau", taus),
                        ("anisotropy via ||d||-inflation",
                         "P1_aniso_inflation_rel_defect", anis)):
        ys = [out[key][f"{x:g}"] for x in xs]
        viol = sum(1 for a, b in zip(ys, ys[1:]) if b <= a)
        record("P1", f"relative defect increases with {nm}", viol, 0,
               kind="count", note=f"{ys[0]:.4e} -> {ys[-1]:.4e}")

    # anisotropy acts THROUGH ||delta||: rel/||delta|| stays ~constant
    rr = np.array(list(out["P1_aniso_inflation_rel_over_delta"].values()))
    out["P1_aniso_inflation_ratio_spread"] = float(
        (rr.max() - rr.min()) / rr.mean())
    record("P1", "anisotropy acts through ||d|| (rel/||d|| ~ const)",
           out["P1_aniso_inflation_ratio_spread"], 0.30,
           note=f"rel/||d|| in [{rr.min():.4f}, {rr.max():.4f}]")

    # NEGATIVE result, asserted as a POSITIVE check on flatness
    yf = np.array([out["P1_aniso_fixed_norm_random_frame"][f"{a:g}"] for a in anis])
    out["P1_aniso_fixed_norm_relative_range"] = float(
        (yf.max() - yf.min()) / yf.mean())
    record("P1", "NEGATIVE: fixed-||d|| anisotropy has NO effect (flat)",
           out["P1_aniso_fixed_norm_relative_range"], 2e-2,
           note=f"{yf[0]:.4e} -> {yf[-1]:.4e} over 20x stretch, 12 charts")

    # tau: the RELATIVE defect is ~linear in tau (defect ~ tau^2, ||delta|| ~ tau).
    # Fitted on the SMALL-tau window for the same reason as P1.1: at tau=0.8 the
    # O(tau^3) remainder bends the slope (recorded below).
    sl_tau = np.polyfit(np.log(tau_fit),
                        np.log([out["P1_defect_vs_tau"][f"{t:g}"] for t in tau_fit]),
                        1)[0]
    sl_tau_all = np.polyfit(np.log(taus),
                            np.log([out["P1_defect_vs_tau"][f"{t:g}"]
                                    for t in taus]), 1)[0]
    out["P1_rel_defect_tau_slope"] = float(sl_tau)
    out["P1_rel_defect_tau_slope_fit_window"] = [min(tau_fit), max(tau_fit)]
    out["P1_rel_defect_tau_slope_all_taus"] = float(sl_tau_all)
    record("P1", "relative defect ~ tau^1", abs(sl_tau - 1.0), 0.02,
           kind="abs", note=f"slope={sl_tau:.4f} on tau<=0.2; "
                            f"{sl_tau_all:.4f} incl. tau=0.8 (O(tau^3) bend)")

    # k: summing k random primitives grows ||delta|| like sqrt(k), and the
    # relative defect is proportional to ||delta||, hence ~ sqrt(k).
    yk = np.array([out["P1_defect_vs_k"][str(k)] for k in ks])
    rk = yk / np.sqrt(ks)
    out["P1_rel_defect_over_sqrt_k"] = {str(k): float(v) for k, v in zip(ks, rk)}
    out["P1_sqrt_k_spread"] = float((rk.max() - rk.min()) / rk.mean())
    record("P1", "k-growth follows sqrt(k) (||d|| ~ sqrt k)",
           out["P1_sqrt_k_spread"], 0.20,
           note=f"rel/sqrt(k) in [{rk.min():.5f}, {rk.max():.5f}]")

    # the naive anisotropy sweep at large state scale is non-monotone: record it
    # WITH the saturation diagnostic that causes the turnover
    yn = [out["P1_defect_vs_aniso_naive"][f"{a:g}"] for a in anis]
    out["P1_aniso_naive_is_monotone"] = bool(
        all(b > a for a, b in zip(yn, yn[1:])))
    out["P1_aniso_naive_turnover_at"] = (
        None if out["P1_aniso_naive_is_monotone"]
        else f"{anis[int(np.argmax(yn))]:g}")
    return out


# --------------------------------------------------------------------------- #
#  PROPOSITION 2 -- connection law and tensoriality of S + 2 Gamma            #
# --------------------------------------------------------------------------- #
def prop2(rng, d=4, n_states=6):
    """Gamma~[Dpsi u, Dpsi v] = Dpsi . Gamma[u,v] - D2psi[u,v],  and hence
    (S + 2 Gamma) transforms as a vector field while (S - 2 Gamma) does not.

    Checks
      P2.1  connection transformation law                    (expect ~1e-16)
      P2.2  unsymmetrised Hessian FAILS the law              (control, must be large)
      P2.3  coupling defect S' - Dpsi.S == 2 D2psi[X,Y]      (expect ~1e-16)
      P2.4  S + 2 Gamma is tensorial                         (expect ~1e-16)
      P2.5  S - 2 Gamma is NOT tensorial                     (control, must be large)
      P2.6  S alone is NOT tensorial                         (control, must be large)
      P2.7  AFFINE chart: all three variants exact           (fact D)
    """
    out = {}
    chart = Chart(d, rng, s=0.35)
    metric = Metric(d, rng)
    X = Primitive(d, rng, amp=1.0)
    Y = Primitive(d, rng, amp=0.8)
    Z = 0.6 * rng.standard_normal((n_states, d))
    out["chart_diffeo_margin_sigma_min"] = chart.margin(Z)

    # pushed-forward metric: G~(z') = K^T G(psi^-1 z') K,  K = Dpsi^{-1}
    def metric_t(zp):
        z = chart.inv(zp)
        K = np.linalg.inv(chart.jac(z))
        Gt = K.T @ metric(z) @ K
        return 0.5 * (Gt + Gt.T)

    Xt, Yt = X.pushforward(chart), Y.pushforward(chart)

    e_law, e_unsym, e_defect = [], [], []
    e_p2g, e_m2g, e_s = [], [], []
    conds = []
    for z in Z:
        J = chart.jac(z)
        H = chart.hess(z)
        zp = chart(z)
        u, v = X(z), Y(z)
        up, vp = J @ u, J @ v

        # -- P2.1 connection transformation law ---------------------------
        Gam_t = gamma_vec(metric_t, zp, up, vp)
        rhs = J @ gamma_vec(metric, z, u, v) - np.einsum('aij,i,j->a', H, u, v)
        e_law.append(relerr(Gam_t, rhs))
        conds.append(float(np.linalg.cond(metric(z))))

        # -- P2.2 control: an UNSYMMETRISED Hessian breaks the law ---------
        # Same analytic H, but with the (i,j) symmetry deliberately destroyed by
        # keeping only the lower-index-ordered half.  Fact B says this must fail.
        # H is indexed (a,i,j) and np.tril masks the LAST TWO axes, so this
        # triangularises exactly the (i,j) pair whose symmetry is at issue.
        H_un = np.tril(H) * 2.0
        rhs_un = J @ gamma_vec(metric, z, u, v) - np.einsum('aij,i,j->a', H_un, u, v)
        e_unsym.append(relerr(Gam_t, rhs_un))

        # -- P2.3 the coupling defect is exactly 2 D2psi[X, Y] -------------
        # S = DX[Y] + DY[X], computed by complex step in each chart
        S = cs_dir(X, z, v) + cs_dir(Y, z, u)
        St = cs_dir(Xt, zp, vp) + cs_dir(Yt, zp, up)
        e_defect.append(relerr(St - J @ S, 2.0 * np.einsum('aij,i,j->a', H, u, v)))

        # -- P2.4/5/6 tensoriality of the three candidate combinations -----
        Gm = gamma_vec(metric, z, u, v)
        e_p2g.append(relerr(St + 2.0 * Gam_t, J @ (S + 2.0 * Gm)))
        e_m2g.append(relerr(St - 2.0 * Gam_t, J @ (S - 2.0 * Gm)))
        e_s.append(relerr(St, J @ S))

    out["P2_metric_cond_max"] = float(max(conds))
    out["P2_connection_law_relerr"] = float(np.max(e_law))
    out["P2_unsymmetrised_hessian_relerr"] = float(np.max(e_unsym))
    out["P2_coupling_defect_vs_2D2psi_relerr"] = float(np.max(e_defect))
    out["P2_S_plus_2Gamma_relerr"] = float(np.max(e_p2g))
    out["P2_S_minus_2Gamma_relerr"] = float(np.max(e_m2g))
    out["P2_S_alone_relerr"] = float(np.max(e_s))

    record("P2", "connection law Gt = Dpsi.G - D2psi[u,v]",
           out["P2_connection_law_relerr"], 1e-12,
           note="fact B; Hessian symmetrised")
    record("P2", "CONTROL unsymmetrised Hessian must FAIL",
           1.0 / max(out["P2_unsymmetrised_hessian_relerr"], 1e-300), 1e3,
           kind="inv", note=f"relerr={out['P2_unsymmetrised_hessian_relerr']:.3e}")
    record("P2", "coupling defect == 2 D2psi[X,Y]",
           out["P2_coupling_defect_vs_2D2psi_relerr"], 1e-12, note="fact C")
    record("P2", "S + 2 Gamma is tensorial",
           out["P2_S_plus_2Gamma_relerr"], 1e-12, note="fact C, correct sign")
    record("P2", "CONTROL S - 2 Gamma must FAIL",
           1.0 / max(out["P2_S_minus_2Gamma_relerr"], 1e-300), 1e3,
           kind="inv", note=f"relerr={out['P2_S_minus_2Gamma_relerr']:.3e}")
    record("P2", "CONTROL S alone must FAIL",
           1.0 / max(out["P2_S_alone_relerr"], 1e-300), 1e3,
           kind="inv", note=f"relerr={out['P2_S_alone_relerr']:.3e}")

    # --- P2.7 : AFFINE chart -- every variant exact (fact D) ---------------
    aff = Chart(d, rng, affine=True)

    def metric_a(zp):
        z = aff.inv(zp)
        K = np.linalg.inv(aff.jac(z))
        Ga = K.T @ metric(z) @ K
        return 0.5 * (Ga + Ga.T)

    Xa, Ya = X.pushforward(aff), Y.pushforward(aff)
    a_p2g, a_m2g, a_s = [], [], []
    for z in Z:
        J = aff.jac(z)
        zp = aff(z)
        u, v = X(z), Y(z)
        up, vp = J @ u, J @ v
        S = cs_dir(X, z, v) + cs_dir(Y, z, u)
        St = cs_dir(Xa, zp, vp) + cs_dir(Ya, zp, up)
        Gm = gamma_vec(metric, z, u, v)
        Gt = gamma_vec(metric_a, zp, up, vp)
        a_p2g.append(relerr(St + 2.0 * Gt, J @ (S + 2.0 * Gm)))
        a_m2g.append(relerr(St - 2.0 * Gt, J @ (S - 2.0 * Gm)))
        a_s.append(relerr(St, J @ S))
    out["P2_affine_S_plus_2Gamma_relerr"] = float(np.max(a_p2g))
    out["P2_affine_S_minus_2Gamma_relerr"] = float(np.max(a_m2g))
    out["P2_affine_S_alone_relerr"] = float(np.max(a_s))
    record("P2", "AFFINE chart: all variants exact (fact D)",
           max(a_p2g + a_m2g + a_s), 1e-12,
           note="D2psi == 0, so the three variants are indistinguishable; "
                "PCA is therefore NOT a valid nonlinear-chart example")
    return out


# --------------------------------------------------------------------------- #
#  PROPOSITION 3 -- whole-field covariance of the pullback one-form field     #
# --------------------------------------------------------------------------- #
def prop3(rng, d=4, Dim=9, n_states=8, eps_list=(1e-6, 1e-4, 1e-2, 1e-1)):
    """With v = G^{-1} J^T b,  G = J^T W J:  under z' = psi(z) and D' = D o psi^-1,
    v' = Dpsi . v.  The ENTIRE field is a vector field.

    Checks
      P3.1  whole-field covariance on a RANDOM NONLINEAR D and NONLINEAR psi
      P3.2  chart-2 Jacobian by complex-step == J_D Dpsi^{-1} (chain rule not assumed)
      P3.3  G and alpha transform as (0,2) and (0,1) tensors
      P3.4  ABLATION: the raw Euclidean eps*I stabiliser BREAKS covariance
      P3.5  immersion / conditioning assumptions are measured, not assumed
    """
    out = {}
    chart = Chart(d, rng, s=0.35)
    dec = Decoder(d, Dim, rng)
    W = AmbientWeight(Dim, rng)
    b = AmbientOneForm(Dim, rng)
    Z = 0.6 * rng.standard_normal((n_states, d))
    dec_t = PulledBackDecoder(dec, chart)

    # --- P3.5 : assumptions, measured -------------------------------------
    out["P3_decoder_min_singular_value"] = dec.min_sigma(Z)
    out["P3_chart_diffeo_margin_sigma_min"] = chart.margin(Z)
    out["P3_newton_inverse_max_relerr"] = float(max(
        relerr(chart(chart.inv(chart(z))), chart(z)) for z in Z))
    # Z-SPACE round trip.  The psi-space residual above is satisfied by ANY
    # branch of the inverse, so on its own it cannot detect a wrong-branch
    # Newton solve; this one can, and it is the check that catches it.
    out["P3_newton_inverse_zspace_max_relerr"] = float(max(
        relerr(chart.inv(chart(z)), z) for z in Z))
    out["P3_chart_lipschitz_ratio"] = chart.lipschitz_ratio
    record("P3", "Z-space chart round trip psi^-1(psi(z)) == z",
           out["P3_newton_inverse_zspace_max_relerr"], 1e-13,
           note="detects wrong-branch Newton, which the psi-space check cannot")
    record("P3", "decoder is an immersion (sigma_min(J_D) > 0)",
           1.0 / max(out["P3_decoder_min_singular_value"], 1e-300), 1e3,
           kind="inv", note=f"sigma_min={out['P3_decoder_min_singular_value']:.4f}")
    record("P3", "Newton chart inverse converged",
           out["P3_newton_inverse_max_relerr"], 1e-14, note="psi(psi^-1(z'))==z'")

    e_v, e_G, e_a, e_jac, condsG = [], [], [], [], []
    for z in Z:
        zp = chart(z)
        Jp = chart.jac(z)

        v, G, al, cG = pullback_field(dec, W, b, z)
        condsG.append(cG)

        # -- P3.2 : chart-2 decoder Jacobian, differentiated NOT chain-ruled --
        Jd_t_cs = dec_t.jac_cs(zp)
        Jd_t_chain = dec.jac(z) @ np.linalg.inv(Jp)
        e_jac.append(relerr(Jd_t_cs, Jd_t_chain))

        # build the chart-2 field from the INDEPENDENTLY DIFFERENTIATED Jacobian
        x = dec_t(zp)
        Wx = W(x)
        Gt = Jd_t_cs.T @ Wx @ Jd_t_cs
        Gt = 0.5 * (Gt + Gt.T)
        alt = Jd_t_cs.T @ b(x)
        vt = np.linalg.solve(Gt, alt)

        # -- P3.1 : v' = Dpsi . v --------------------------------------------
        e_v.append(relerr(vt, Jp @ v))
        # -- P3.3 : G is (0,2), alpha is (0,1) -------------------------------
        K = np.linalg.inv(Jp)
        e_G.append(relerr(Gt, K.T @ G @ K))
        e_a.append(relerr(alt, K.T @ al))

    out["P3_G_condition_max"] = float(max(condsG))
    out["P3_wholefield_covariance_relerr"] = float(np.max(e_v))
    out["P3_wholefield_covariance_relerr_mean"] = float(np.mean(e_v))
    out["P3_decoder_jacobian_cs_vs_chainrule_relerr"] = float(np.max(e_jac))
    out["P3_metric_pullback_relerr"] = float(np.max(e_G))
    out["P3_oneform_pullback_relerr"] = float(np.max(e_a))

    record("P3", "WHOLE-FIELD covariance v' = Dpsi . v",
           out["P3_wholefield_covariance_relerr"], 1e-10,
           note="random nonlinear decoder AND random nonlinear chart")
    record("P3", "chart-2 J_D by complex step == chain rule",
           out["P3_decoder_jacobian_cs_vs_chainrule_relerr"], 1e-10,
           note="chain rule verified, not assumed")
    record("P3", "G transforms as a (0,2) tensor",
           out["P3_metric_pullback_relerr"], 1e-10)
    record("P3", "alpha transforms as a (0,1) tensor",
           out["P3_oneform_pullback_relerr"], 1e-10)

    # --- P3.4 : the Euclidean stabiliser ablation --------------------------
    # G -> G + eps I is not covariant: I is not a (0,2) tensor, so the stabilised
    # metric in chart 2 is K^T G K + eps I, not K^T (G + eps I) K.
    stab = {}
    for eps in eps_list:
        errs = []
        for z in Z:
            zp = chart(z)
            Jp = chart.jac(z)
            v, _, _, _ = pullback_field(dec, W, b, z, eps=eps, euclid_stab=True)
            Jd_t = dec_t.jac_cs(zp)
            x = dec_t(zp)
            Gt = Jd_t.T @ W(x) @ Jd_t
            Gt = 0.5 * (Gt + Gt.T) + eps * np.eye(d)
            vt = np.linalg.solve(Gt, Jd_t.T @ b(x))
            errs.append(relerr(vt, Jp @ v))
        stab[f"{eps:g}"] = float(np.max(errs))
    out["P3_euclidean_stabiliser_covariance_relerr"] = stab
    worst_eps = stab[f"{max(eps_list):g}"]
    record("P3", "ABLATION eps*I stabiliser must BREAK covariance",
           1.0 / max(worst_eps, 1e-300), 1e3, kind="inv",
           note=f"relerr {stab[f'{min(eps_list):g}']:.3e} (eps={min(eps_list):g}) "
                f"-> {worst_eps:.3e} (eps={max(eps_list):g})")
    return out


# --------------------------------------------------------------------------- #
#  DOUBLE-COUNTING DIAGNOSIS (fact E)                                         #
# --------------------------------------------------------------------------- #
def double_counting(rng, d=4, n_states=5, Ts=(0.05, 0.02, 0.01, 0.005), n=4000):
    """Generator composition ALREADY produces the symmetric cross term.

    For fields X, Y and integration time T,

        flow_{X+Y}^T(z) - [ (flow_X^T(z) - z) + (flow_Y^T(z) - z) + z ]
              = (T^2 / 2) ( DX[Y] + DY[X] ) + O(T^3)
              = (T^2 / 2) S + O(T^3).

    So the S direction is generated for FREE by integrating the summed field.
    Adding beta * S to the instantaneous velocity therefore inserts a SECOND
    copy of the same kinematic tensor (at order T rather than T^2), which is a
    derivation error, not a performance shortfall.

    Check: [flow(X+Y) - (dispX + dispY)] / (T^2/2) -> S with a relative error
    that is clean O(T), i.e. log-log slope ~ 1.
    """
    out = {}
    X = Primitive(d, rng, amp=1.0)
    Y = Primitive(d, rng, amp=0.8)
    Z = 0.6 * rng.standard_normal((n_states, d))

    def XY(z):
        return X(z) + Y(z)

    errs = {}
    for T in Ts:
        per = []
        for z in Z:
            joint = rk4_flow(XY, z, T, n)
            dx = rk4_flow(X, z, T, n) - z
            dy = rk4_flow(Y, z, T, n) - z
            emp = (joint - (z + dx + dy)) / (0.5 * T ** 2)
            S = X.jac(z) @ Y(z) + Y.jac(z) @ X(z)
            per.append(relerr(emp, S))
        errs[f"{T:g}"] = float(np.mean(per))
    out["E_joint_minus_additive_vs_S_relerr"] = errs
    slope = np.polyfit(np.log(list(Ts)), np.log([errs[f"{T:g}"] for T in Ts]), 1)[0]
    out["E_convergence_slope"] = float(slope)
    record("E", "joint-minus-additive == (T^2/2) S",
           errs[f"{min(Ts):g}"], 5e-3,
           note=f"relerr at T={min(Ts):g}; composition already yields S")
    record("E", "O(T) convergence: log-log slope vs 1", abs(slope - 1.0), 0.05,
           kind="abs", note=f"slope={slope:.4f}")

    # Quantify the second copy: the explicit coupling beta*S enters the
    # displacement at order T, composition's copy at order T^2, so their ratio
    # is beta * 2 / T -- i.e. at small T the hand-added term DOMINATES the very
    # effect it was introduced to model, and the two are collinear in S.
    beta = 1.0
    out["E_explicit_over_kinematic_ratio"] = {
        f"{T:g}": float(2.0 * beta / T) for T in Ts}
    # Collinearity: the two contributions point along the SAME tensor S.
    cosines = []
    for z in Z:
        S = X.jac(z) @ Y(z) + Y.jac(z) @ X(z)
        T = min(Ts)
        joint = rk4_flow(XY, z, T, n)
        dx = rk4_flow(X, z, T, n) - z
        dy = rk4_flow(Y, z, T, n) - z
        emp = (joint - (z + dx + dy)) / (0.5 * T ** 2)
        cosines.append(float(emp @ S / (np.linalg.norm(emp) * np.linalg.norm(S))))
    out["E_collinearity_cosine_min"] = float(min(cosines))
    record("E", "the two S copies are collinear (cos ~ 1)",
           1.0 - min(cosines), 1e-2, kind="abs",
           note=f"min cosine {min(cosines):.8f}: the hand-added term is not a "
                f"new direction")
    return out


# --------------------------------------------------------------------------- #
#  driver                                                                     #
# --------------------------------------------------------------------------- #
KIND_LABEL = {
    "rel": "rel.err <=",
    "abs": "abs <=",
    "count": "violations <=",
    "inv": "1/value <=",
}


def print_table():
    w = max(len(r["check"]) for r in ROWS) + 2
    line = "-" * (w + 46)
    print(line)
    print(f"{'PROP':<5} {'CHECK':<{w}} {'VALUE':>12} {'BOUND':>11}  OK")
    print(line)
    last = None
    for r in ROWS:
        if last is not None and r["prop"] != last:
            print(line)
        last = r["prop"]
        print(f"{r['prop']:<5} {r['check']:<{w}} {r['value']:>12.4e} "
              f"{r['threshold']:>11.1e}  {'PASS' if r['passed'] else 'FAIL'}")
    print(line)
    npass = sum(r["passed"] for r in ROWS)
    print(f"{npass}/{len(ROWS)} checks passed")
    print(line)


def seed_sweep(seeds=(1, 2, 3, 7, 101, 2718, 99991, 555, 31337), d=4):
    """Re-run the two exact-covariance propositions across independent seeds.

    A covariance identity that holds at one seed is worth little, so the headline
    residuals are reported as a MAXIMUM over seeds.  This sweep is also what
    exposed the wrong-branch chart failure: before the contraction bound was
    enforced in `Chart`, seeds 1 and 7 gave P2 law 2.46 and P3 field 6.3e-1.
    """
    saved, rows = list(ROWS), []
    ROWS.clear()
    for s in seeds:
        rng = np.random.default_rng(s)
        p2 = prop2(np.random.default_rng(rng.integers(1 << 32)), d=d)
        p3 = prop3(np.random.default_rng(rng.integers(1 << 32)), d=d)
        rows.append(dict(
            seed=int(s),
            p2_connection_law=p2["P2_connection_law_relerr"],
            p2_S_plus_2Gamma=p2["P2_S_plus_2Gamma_relerr"],
            p2_S_minus_2Gamma_control=p2["P2_S_minus_2Gamma_relerr"],
            p3_wholefield=p3["P3_wholefield_covariance_relerr"],
            p3_G_cond_max=p3["P3_G_condition_max"],
            n_failed=int(sum(1 for r in ROWS if not r["passed"])),
        ))
        ROWS.clear()
    ROWS.extend(saved)
    agg = {
        "n_seeds": len(seeds),
        "max_p2_connection_law": max(r["p2_connection_law"] for r in rows),
        "max_p2_S_plus_2Gamma": max(r["p2_S_plus_2Gamma"] for r in rows),
        "min_p2_S_minus_2Gamma_control": min(
            r["p2_S_minus_2Gamma_control"] for r in rows),
        "max_p3_wholefield": max(r["p3_wholefield"] for r in rows),
        "max_p3_G_cond": max(r["p3_G_cond_max"] for r in rows),
        "total_failed_checks": int(sum(r["n_failed"] for r in rows)),
        "per_seed": rows,
    }
    record("SEED", "P2 connection law over all seeds",
           agg["max_p2_connection_law"], 1e-12,
           note=f"max over {len(seeds)} seeds")
    record("SEED", "P2 S+2Gamma tensorial over all seeds",
           agg["max_p2_S_plus_2Gamma"], 1e-12, note="max over seeds")
    record("SEED", "P3 whole-field covariance over all seeds",
           agg["max_p3_wholefield"], 1e-10, note="max over seeds")
    record("SEED", "CONTROL S-2Gamma fails at EVERY seed",
           1.0 / max(agg["min_p2_S_minus_2Gamma_control"], 1e-300), 1e3,
           kind="inv",
           note=f"min control relerr {agg['min_p2_S_minus_2Gamma_control']:.3e}")
    return agg


def main(seed=20260907):
    t0 = time.time()
    rng = np.random.default_rng(seed)

    print("IHC-FM :: numerical verification of Propositions 1-3 "
          "+ double-counting diagnosis")
    print(f"float64 / complex128 throughout; complex-step h = {CSTEP_H:g}; "
          f"seed = {seed}\n")

    res = {"seed": seed, "dtype": "float64", "complex_step_h": CSTEP_H,
           "numpy": np.__version__}
    res["prop1"] = prop1(np.random.default_rng(rng.integers(1 << 32)))
    res["prop2"] = prop2(np.random.default_rng(rng.integers(1 << 32)))
    res["prop3"] = prop3(np.random.default_rng(rng.integers(1 << 32)))
    res["double_counting"] = double_counting(
        np.random.default_rng(rng.integers(1 << 32)))
    res["seed_sweep"] = seed_sweep()

    print_table()
    res["checks"] = ROWS
    res["n_checks"] = len(ROWS)
    res["n_passed"] = int(sum(r["passed"] for r in ROWS))
    res["all_passed"] = bool(res["n_passed"] == res["n_checks"])
    res["failed_checks"] = [r["check"] for r in ROWS if not r["passed"]]
    res["wall_clock_s"] = round(time.time() - t0, 3)

    os.makedirs("results", exist_ok=True)
    path = os.path.join("results", "propositions.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=2, sort_keys=True)
    print(f"wall clock {res['wall_clock_s']:.2f} s -> {path}")
    return res


if __name__ == "__main__":
    main()
