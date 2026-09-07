"""IHC-FM, track 2: the compositional hierarchy and the flow-matching objective.

DOMAIN-FREE. Nothing in this file refers to cells, genes, assets, doses-as-drugs or any
other application vocabulary. An "intervention" is an index p into a finite set; an
"exposure" is a nonnegative scalar tau_p; a "population" is an empirical measure on the
observed space. Biology and finance are validation case studies elsewhere and play no
part in the definitions below.

WHAT THIS FILE PROVIDES
-----------------------
The ambient one-form of IHC-FM, built as a permutation-invariant hierarchy over the
intervention set, and the time-dependent conditional-flow-matching objective and training
schedule that fit it. The geometry (decoder, ambient metric, pullback v = G^{-1} J^T b,
dose response, metric radial saturation) lives in `ihcfm_geometry.py` and is deliberately
NOT imported here: the orchestrator wires the two together. Everything geometric enters
this file as an INJECTED CALLABLE, so this module imports cleanly and self-tests on its
own with an identity pullback stand-in.

THE HIERARCHY
-------------
With a_p >= 0 the exposure amplitude of intervention p (a_p = 0 exactly when p is absent
or when tau_p = 0), and e_p in R^r a learned intervention embedding,

    m1 = sum_p     a_p e_p                       first  set moment   (B, r)
    m2 = sum_{p<q} a_p a_q (e_p . e_q)           second set moment   (B, r)   [elementwise]

    b(t, x, c) = A1(t, x, c, m1) . m1  +  C2(t, x, c, m1) . m2

where A1, C2 in R^{d x r} are state- and time-dependent field bases produced by two MLP
heads and c is a population code (possibly a zero-width vector). Three structural
properties follow from the FORM, not from training:

 (H1) PERMUTATION INVARIANCE. m1 and m2 are symmetric functions of the multiset
      {(a_p, e_p)}, so b is invariant to any relabelling of the intervention indices
      applied consistently to a and to the embedding table.

 (H2) SINGLETON VANISHING, BIT-EXACT. Every term of m2 carries a_p a_q with p != q, so if
      at most one a_p is nonzero then m2 == 0.0 exactly and the interaction contribution
      C2 . m2 is exactly the zero vector -- no 1-ulp residual. This is why the O(kr)
      identity is written as

          m2 = 0.5 * (m1 (.) m1 - sum_p u_p (.) u_p),   u_p = a_p e_p,

      with u_p FORMED FIRST. Writing the same quantity as 0.5*[(sum a_p e_p)^2 -
      sum a_p^2 e_p^2] rounds fl(a_p e_p)^2 and fl(a_p^2) fl(e_p^2) differently and leaves
      a ~1e-17 residual (this exact trap is documented in `synthetic.py`); grouping through
      u_p makes the singleton case a bit-identical cancellation u_p^2 - u_p^2 = 0.0.
      The same argument covers "any one exposure is zero" inside a larger set: that a_p is
      exactly 0.0, so every pair term containing it is exactly 0.0.

 (H3) ZERO-EXPOSURE IDENTITY. The main branch is also structurally multiplied by its
      moment (A1 . m1, not A1 alone), so a = 0 gives b == 0 exactly and the flow is the
      identity map. A model with an unmultiplied main head would have to LEARN that a
      null intervention does nothing; here it cannot fail to.

COST. m1 and m2 are both O(k r) per sample: no k^2 term is ever formed. The explicit
O(k^2) double loop is retained in `SetMoments.reference_moments` purely as a numerical
oracle for the self-test (the two agree to round-off, so the claim "interactions cost
O(kr), not O(k^2)" is about the same quantity computed more cheaply, not a different
approximation).

THE OBJECTIVE
-------------
`CFMObjective` is time-dependent conditional flow matching. For a coupling pi_c of the
observed pre/post populations we draw (x0, x1) ~ pi_c, t ~ U(0, 1), and regress the
network velocity at the interpolant point onto the conditional velocity:

    x_t = (1 - t) x0 + t x1 + sigma * eps,        u(t | x0, x1) = x1 - x0

(the sigma = 0 case is the linear/I-CFM path; sigma > 0 is the Gaussian-conditional
variant with constant width, for which the conditional velocity is unchanged). The
coupling is computed in OBSERVED SPACE -- on x, the coordinates the data is actually
given in -- and never from latent coordinates, because a coupling built in a learned
latent chart is not a property of the data and changes as the encoder moves. Three
couplings are selectable: 'independent' (product coupling), 'ot' (exact minibatch
assignment on squared distance, via linear_sum_assignment), and 'sinkhorn' (entropic OT,
log-domain, with an optional unbalanced/UOT relaxation).

THE SCHEDULE
------------
`train_hierarchical` runs three stages, for one reason: an interaction branch trained
jointly from scratch will happily absorb single-intervention effects, after which its
"interaction" is a relabelled main effect and recovery of the true interaction fails even
when the distributional fit looks good.

  Stage 1  main effects only, on singleton conditions only. The interaction branch is
           disabled (not merely penalised).
  Stage 2  interaction branch enabled and trained on combination conditions, with the
           stage-1 model frozen as an ANCHOR: an explicit penalty keeps the composed
           velocity on singleton conditions close to what stage 1 predicted there.
  Stage 3  joint fine-tune, singletons oversampled, plus an interaction-norm penalty
           lambda ||C2 . m2||^2 whose lambda is selected on held-out combinations.

Non-finite steps (either loss or gradient) are skipped and counted rather than silently
poisoning the parameters; the count is reported.
"""
from __future__ import annotations

import copy
import itertools
import json
import math
import pathlib
import time
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(2)

__all__ = [
    "InterventionEmbed",
    "SetMoments",
    "OneFormHierarchy",
    "PopulationEncoder",
    "CFMObjective",
    "ConditionalVelocity",
    "StandInDose",
    "identity_pullback",
    "train_hierarchical",
    "select_interaction_penalty",
]


# ----------------------------------------------------------------------------------
# small utilities
# ----------------------------------------------------------------------------------
def _mlp(n_in: int, hidden: int, n_out: int, depth: int = 2, ln: bool = True,
         act=nn.SiLU) -> nn.Sequential:
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


def _time_features(t: torch.Tensor, n_fourier: int = 2) -> torch.Tensor:
    """[t, sin(2^j pi t), cos(2^j pi t)] -- a smooth, bounded encoding of flow time.

    The objective is time-DEPENDENT (the conditional velocity of the interpolant is a
    function of t), so the field must be able to represent t-dependence; a single raw t
    input makes that unnecessarily hard for a small MLP.
    """
    t = t.reshape(-1, 1)
    feats = [t]
    for j in range(int(n_fourier)):
        w = math.pi * (2.0 ** j)
        feats += [torch.sin(w * t), torch.cos(w * t)]
    return torch.cat(feats, dim=1)


N_TIME_FEATURES = 1 + 2 * 2


# ----------------------------------------------------------------------------------
# intervention embeddings and set moments
# ----------------------------------------------------------------------------------
class InterventionEmbed(nn.Module):
    """A learned table e: {0..k-1} -> R^r of intervention embeddings.

    r is the FACTOR rank of the interaction family: the number of independent basis fields
    the second-moment branch can express (fact R of the project log -- the recoverable
    rank is the factor rank, NOT the numerical rank of the coefficient matrix
    off-diag(L L^T), which can be larger). r is therefore a genuine capacity knob and is
    reported with every result.

    Note on identifiability: m1 = sum_p a_p e_p determines the amplitude vector a only if
    the k embeddings are linearly independent, which needs r >= k. For r < k the main
    branch shares parameters across interventions by construction -- sometimes desirable,
    but it is a modelling restriction and not a free lunch, so the default is r = 4 and
    `r_at_least_k` is reported by `describe()`.
    """

    def __init__(self, k: int, r: int = 4, init_scale: float = 1.0,
                 normalise: bool = False, seed: int | None = None):
        super().__init__()
        self.k, self.r = int(k), int(r)
        self.normalise = bool(normalise)
        if seed is not None:
            g = torch.Generator().manual_seed(int(seed))
            w = torch.randn(self.k, self.r, generator=g)
        else:
            w = torch.randn(self.k, self.r)
        self.weight = nn.Parameter(float(init_scale) * w / math.sqrt(max(self.r, 1)))

    def table(self) -> torch.Tensor:
        """(k, r) embedding table, optionally row-normalised."""
        E = self.weight
        if self.normalise:
            E = E / (E.norm(dim=1, keepdim=True) + 1e-12)
        return E

    def forward(self, ids: torch.Tensor | Sequence[int] | None = None) -> torch.Tensor:
        E = self.table()
        if ids is None:
            return E
        if not torch.is_tensor(ids):
            ids = torch.as_tensor(list(ids), dtype=torch.long)
        return E.index_select(0, ids.reshape(-1))

    def describe(self) -> dict:
        with torch.no_grad():
            E = self.table()
            s = torch.linalg.svdvals(E)
            return dict(k=self.k, r=self.r, r_at_least_k=bool(self.r >= self.k),
                        singular_values=[float(x) for x in s],
                        rank_tol_1em8=int((s > 1e-8 * max(float(s[0]), 1e-300)).sum()))


class SetMoments(nn.Module):
    """Permutation-invariant first and second set moments in O(k r).

    forward(a, E) -> (m1, m2) with a of shape (B, k) and E of shape (k, r):

        u_p = a_p e_p                                   (B, k, r)   formed explicitly
        m1  = sum_p u_p                                 (B, r)      O(k r)
        m2  = 0.5 * (m1 (.) m1 - sum_p u_p (.) u_p)     (B, r)      O(k r)

    m2 is the elementwise second moment sum_{p<q} a_p a_q (e_p (.) e_q). The u_p grouping
    is load-bearing for bit-exact singleton vanishing (see module docstring, H2): it makes
    the singleton case the exact cancellation u_p^2 - u_p^2 rather than a difference of two
    differently-rounded products.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, a: torch.Tensor, E: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = a if a.dim() == 2 else a.reshape(1, -1)
        u = a.unsqueeze(-1) * E.unsqueeze(0)                 # (B, k, r)
        m1 = u.sum(dim=1)                                    # (B, r)
        sq = (u * u).sum(dim=1)                              # (B, r)
        m2 = 0.5 * (m1 * m1 - sq)                            # (B, r)
        return m1, m2

    @staticmethod
    def reference_moments(a: torch.Tensor, E: torch.Tensor
                          ) -> tuple[torch.Tensor, torch.Tensor]:
        """Explicit O(k^2) double loop over pairs. NUMERICAL ORACLE ONLY -- never used in
        training. Present so the self-test can show the O(kr) form computes the SAME
        quantity (agreement to round-off) rather than a cheaper approximation of it.
        """
        a = a if a.dim() == 2 else a.reshape(1, -1)
        B, k = a.shape
        r = E.shape[1]
        m1 = torch.zeros(B, r, dtype=a.dtype)
        for p in range(k):
            m1 = m1 + a[:, p:p + 1] * E[p].unsqueeze(0)
        m2 = torch.zeros(B, r, dtype=a.dtype)
        for p in range(k):
            for q in range(p + 1, k):
                m2 = m2 + (a[:, p:p + 1] * a[:, q:q + 1]) * (E[p] * E[q]).unsqueeze(0)
        return m1, m2


# ----------------------------------------------------------------------------------
# population encoder
# ----------------------------------------------------------------------------------
class PopulationEncoder(nn.Module):
    """DeepSets code for the observed PRE population mu0. Permutation-invariant.

    forward(mu0) with mu0 of shape (B, n, d) (or (n, d), treated as B = 1) returns
    (B, code_dim). code_dim = 0 means DISABLED and returns a genuine zero-width tensor
    (B, 0), so downstream concatenation is a no-op and the ablation "no population
    conditioning" is exact rather than approximate.

    The mean AND the second moment of the set are pooled: two populations with the same
    mean but different spread are different conditioning contexts, and a mean-only code
    cannot tell them apart.
    """

    def __init__(self, d: int, code_dim: int = 8, hidden: int = 64,
                 use_second_moment: bool = True, depth: int = 2):
        super().__init__()
        self.d, self.code_dim = int(d), int(code_dim)
        self.use_second_moment = bool(use_second_moment)
        self.enabled = self.code_dim > 0
        if self.enabled:
            n_in = self.d * (2 if self.use_second_moment else 1)
            self.phi = _mlp(n_in, hidden, hidden, depth=depth, ln=True)
            self.rho = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden),
                                     nn.SiLU(), nn.Linear(hidden, self.code_dim))
            self.norm = nn.LayerNorm(self.code_dim)

    def forward(self, mu0: torch.Tensor) -> torch.Tensor:
        if mu0.dim() == 2:
            mu0 = mu0.unsqueeze(0)
        B = mu0.shape[0]
        if not self.enabled:
            return mu0.new_zeros((B, 0))
        x = torch.cat([mu0, mu0 ** 2], dim=-1) if self.use_second_moment else mu0
        h = self.phi(x).mean(dim=1)
        return self.norm(self.rho(h))


# ----------------------------------------------------------------------------------
# the hierarchical one-form
# ----------------------------------------------------------------------------------
class OneFormHierarchy(nn.Module):
    """b(t, x, c) = A1(t, x, c, m1) . m1 + C2(t, x, c, m1) . m2.

    Both branches are STRUCTURALLY multiplied by their set moment, which is what buys
    zero-exposure identity (H3) and bit-exact singleton vanishing (H2). The interaction
    head sees m1 but never m2 -- it produces the basis, the moment supplies the gate --
    exactly as in the shared interface contract.

    Parameters
    ----------
    d_ambient   dimension of the ambient/observed space the one-form lives on
    r           factor rank (must match the InterventionEmbed)
    code_dim    width of the population code c (0 = disabled)
    hidden      width of both heads
    interaction_enabled
                if False the interaction branch is not evaluated at all (stage-1 and the
                'main-effects-only' ablation arm). This is a hard switch, not a penalty.
    """

    def __init__(self, d_ambient: int, r: int = 4, code_dim: int = 0, hidden: int = 64,
                 depth: int = 2, interaction_enabled: bool = True,
                 interaction_init_scale: float = 0.1, k: int | None = None,
                 embed: InterventionEmbed | None = None, embed_seed: int | None = 0,
                 interaction_sees_m1: bool = True):
        super().__init__()
        self.d, self.r, self.code_dim = int(d_ambient), int(r), int(code_dim)
        self.interaction_enabled = bool(interaction_enabled)
        # Owns the embedding table and the moment map, so the CONTRACT signature
        # .b(t, x, c, a, ids) is the primary entry point: the caller passes the amplitude
        # vector, not moments. `embed` may be shared with an enclosing module so that
        # there is exactly one table in the composed model.
        if embed is None:
            if k is None:
                k = int(r)
            embed = InterventionEmbed(int(k), self.r, seed=embed_seed)
        if embed.r != self.r:
            raise ValueError(f"embedding rank {embed.r} != one-form rank {self.r}")
        self.embed = embed
        self.k = embed.k
        self.moments = SetMoments()
        n_in = self.d + N_TIME_FEATURES + self.code_dim + self.r
        self.main_head = _mlp(n_in, hidden, self.d * self.r, depth=depth, ln=True)
        # ABLATION ARM, defaulting to the full contract form (interaction_sees_m1=True,
        # i.e. b_int(t, x, c, m1) . m2 exactly as specified). Setting it False removes m1
        # from the interaction head's inputs, leaving C2(t, x, c) . m2 -- a strictly
        # rank-r-shared interaction whose per-condition dependence can ONLY enter through
        # m2. The full form is more expressive but can also use m1 to IDENTIFY which
        # combination it is looking at and fit each training combination separately, which
        # is a leave-one-combination-out generalisation risk rather than a fitting one.
        # Both arms are measured; the default is not changed on the basis of the result.
        self.interaction_sees_m1 = bool(interaction_sees_m1)
        n_in_int = n_in if self.interaction_sees_m1 else n_in - self.r
        self.int_head = _mlp(n_in_int, hidden, self.d * self.r, depth=depth, ln=True)
        # small init on the interaction basis: the branch starts near-inert so stage 2
        # cannot lurch and destroy the stage-1 main effects on its first few steps.
        with torch.no_grad():
            self.int_head[-1].weight.mul_(float(interaction_init_scale))
            self.int_head[-1].bias.mul_(0.0)

    # -- features -----------------------------------------------------------
    def _features(self, t: torch.Tensor, x: torch.Tensor, c: torch.Tensor | None,
                  m1: torch.Tensor) -> torch.Tensor:
        tf = _time_features(t).to(dtype=x.dtype)
        if tf.shape[0] == 1 and x.shape[0] > 1:
            tf = tf.expand(x.shape[0], -1)
        parts = [x, tf, m1]
        if c is not None and c.shape[-1] > 0:
            if c.dim() == 1:
                c = c.unsqueeze(0)
            if c.shape[0] == 1 and x.shape[0] > 1:
                c = c.expand(x.shape[0], -1)
            parts.insert(2, c)
        return torch.cat(parts, dim=1)

    def _features_no_moment(self, t: torch.Tensor, x: torch.Tensor,
                            c: torch.Tensor | None) -> torch.Tensor:
        """Features WITHOUT the first moment -- the interaction_sees_m1=False arm."""
        tf = _time_features(t).to(dtype=x.dtype)
        if tf.shape[0] == 1 and x.shape[0] > 1:
            tf = tf.expand(x.shape[0], -1)
        parts = [x, tf]
        if c is not None and c.shape[-1] > 0:
            if c.dim() == 1:
                c = c.unsqueeze(0)
            if c.shape[0] == 1 and x.shape[0] > 1:
                c = c.expand(x.shape[0], -1)
            parts.append(c)
        return torch.cat(parts, dim=1)

    # -- amplitudes -> moments ---------------------------------------------
    def set_moments(self, a: torch.Tensor, ids: Sequence[int] | torch.Tensor | None = None
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(a, ids) -> (m1, m2). `ids` selects which embedding rows the columns of `a` mean.

        ids=None means `a` is the full (B, k) amplitude vector over all interventions,
        with exact zeros for absent ones. Passing ids is the sparse form: `a` is
        (B, len(ids)) and only those rows of the table participate. Both give identical
        moments because absent interventions contribute a_p == 0.0 exactly.
        """
        a = a if a.dim() == 2 else a.reshape(1, -1)
        if ids is None:
            E = self.embed.table()
            if a.shape[1] != E.shape[0]:
                raise ValueError(f"a has {a.shape[1]} columns but the table has "
                                 f"{E.shape[0]} interventions; pass ids for the sparse form")
        else:
            if not torch.is_tensor(ids):
                ids = torch.as_tensor([int(i) for i in ids], dtype=torch.long)
            E = self.embed(ids)
            if a.shape[1] != E.shape[0]:
                raise ValueError(f"a has {a.shape[1]} columns but {E.shape[0]} ids")
        return self.moments(a, E)

    # -- branches (CONTRACT signatures: (t, x, c, a, ids)) ------------------
    def b(self, t, x, c, a, ids: Sequence[int] | torch.Tensor | None = None) -> torch.Tensor:
        """The full ambient one-form b(t, x, c) for amplitude vector `a`. Contract entry."""
        m1, m2 = self.set_moments(a, ids)
        return self.b_from_moments(t, x, c, m1, m2)

    def interaction_only(self, t, x, c, a, ids: Sequence[int] | torch.Tensor | None = None
                         ) -> torch.Tensor:
        """The interaction contribution alone -- the object scored against ground truth.

        Contract entry point, same argument order as `b`. Returns exactly the zero vector
        (bit-exact) whenever the second moment vanishes, i.e. for every singleton and
        whenever any one exposure in the set is zero.
        """
        m1, m2 = self.set_moments(a, ids)
        return self.interaction_from_moments(t, x, c, m1, m2)

    def main_only(self, t, x, c, a, ids: Sequence[int] | torch.Tensor | None = None
                  ) -> torch.Tensor:
        """The main-effect contribution alone. Contract argument order."""
        m1, _ = self.set_moments(a, ids)
        return self.main_from_moments(t, x, c, m1)

    # -- moment-level path (used when the caller already has (m1, m2)) ------
    def main_from_moments(self, t, x, c, m1) -> torch.Tensor:
        f = self._features(t, x, c, m1)
        A1 = self.main_head(f).reshape(-1, self.d, self.r)
        A1 = torch.nan_to_num(A1, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.einsum('bdr,br->bd', A1, m1)

    def interaction_from_moments(self, t, x, c, m1, m2) -> torch.Tensor:
        """`nan_to_num` on the basis is what makes the vanishing unconditional: 0 * inf
        would be nan, so the basis is finitised before the moment gate multiplies it."""
        if not self.interaction_enabled:
            return torch.zeros_like(x)
        f = (self._features(t, x, c, m1) if self.interaction_sees_m1
             else self._features_no_moment(t, x, c))
        C2 = self.int_head(f).reshape(-1, self.d, self.r)
        C2 = torch.nan_to_num(C2, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.einsum('bdr,br->bd', C2, m2)

    def b_from_moments(self, t, x, c, m1, m2) -> torch.Tensor:
        return (self.main_from_moments(t, x, c, m1)
                + self.interaction_from_moments(t, x, c, m1, m2))

    # -- parameter groups for the schedule ----------------------------------
    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        return dict(main=list(self.main_head.parameters()),
                    interaction=list(self.int_head.parameters()))


# ----------------------------------------------------------------------------------
# stand-in dose response (test-only; geometry track owns the real HillDose)
# ----------------------------------------------------------------------------------
class StandInDose(nn.Module):
    """a_p(tau) = A_p (1 - exp(-tau / s_p)): bounded, monotone, a_p(0) = 0 EXACTLY.

    TEST-ONLY stand-in so this module is independently testable. The orchestrator injects
    `ihcfm_geometry.HillDose` instead; the only contract is forward(tau: (B,1)) -> (B, k)
    with a_p(0) == 0.0 exactly. Exactness at tau = 0 is not a tolerance: exp(0) is exactly
    1.0 in IEEE arithmetic, so 1 - exp(0) is exactly 0.0 and the product is exactly 0.0,
    for every parameter value. Zero-exposure identity therefore never depends on training.
    """

    def __init__(self, k: int, log_amp: float = 0.0, log_scale: float = 0.0):
        super().__init__()
        self.k = int(k)
        self.log_amp = nn.Parameter(torch.full((self.k,), float(log_amp)))
        self.log_scale = nn.Parameter(torch.full((self.k,), float(log_scale)))

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        """tau of shape (B, 1) -- one shared exposure -- or (B, k) -- per-intervention."""
        if tau.dim() == 1:
            tau = tau.reshape(-1, 1)
        if tau.shape[-1] not in (1, self.k):
            raise ValueError(f"tau must have last dim 1 or k={self.k}, got {tuple(tau.shape)}")
        amp = torch.nn.functional.softplus(self.log_amp).unsqueeze(0) + 1e-6
        scale = torch.nn.functional.softplus(self.log_scale).unsqueeze(0) + 1e-3
        return amp * (1.0 - torch.exp(-tau / scale))

    def params_table(self) -> dict:
        with torch.no_grad():
            return dict(amp=[float(x) for x in
                             torch.nn.functional.softplus(self.log_amp) + 1e-6],
                        scale=[float(x) for x in
                               torch.nn.functional.softplus(self.log_scale) + 1e-3])


def identity_pullback(t: torch.Tensor, z: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """v = b. The stand-in for `ihcfm_geometry.PullbackField` (which returns G^{-1} J^T b).

    Used so this module's self-tests exercise the hierarchy and the objective without
    depending on the geometry track. All structural claims (H1-H3) are unaffected by the
    choice of pullback, because G^{-1} J^T is linear in b: b == 0 exactly implies v == 0
    exactly for any invertible G.
    """
    return b


# ----------------------------------------------------------------------------------
# conditional flow matching
# ----------------------------------------------------------------------------------
def _sq_cost(X0: np.ndarray, X1: np.ndarray) -> np.ndarray:
    d2 = (np.sum(X0 ** 2, 1)[:, None] + np.sum(X1 ** 2, 1)[None, :]
          - 2.0 * X0 @ X1.T)
    return np.maximum(d2, 0.0)


def _sinkhorn_log(C: np.ndarray, eps: float = 0.05, iters: int = 200,
                  unbalanced_tau: float | None = None) -> np.ndarray:
    """Log-domain Sinkhorn on a normalised cost. Returns a coupling matrix (n0, n1).

    `unbalanced_tau` switches on the UOT relaxation: the marginal updates are damped by
    tau / (tau + eps), which softens the hard mass-conservation constraint. This matters
    when the two populations are not exact pushforwards of each other (outliers, unequal
    mass), which is the generic case for real observed populations.
    """
    n0, n1 = C.shape
    Cn = C / max(float(C.mean()), 1e-300)
    loga = np.full(n0, -math.log(n0))
    logb = np.full(n1, -math.log(n1))
    f = np.zeros(n0)
    g = np.zeros(n1)
    lam = 1.0 if unbalanced_tau is None else float(unbalanced_tau) / (float(unbalanced_tau) + eps)
    for _ in range(int(iters)):
        M = (f[:, None] + g[None, :] - Cn) / eps
        f = f + lam * eps * (loga - _logsumexp(M, axis=1))
        M = (f[:, None] + g[None, :] - Cn) / eps
        g = g + lam * eps * (logb - _logsumexp(M, axis=0))
    P = np.exp((f[:, None] + g[None, :] - Cn) / eps)
    s = P.sum()
    return P / max(s, 1e-300)


def _logsumexp(M: np.ndarray, axis: int) -> np.ndarray:
    mx = np.max(M, axis=axis, keepdims=True)
    out = mx + np.log(np.sum(np.exp(M - mx), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


class CFMObjective:
    """Time-dependent conditional flow matching with an OBSERVED-SPACE coupling.

    Path (linear / I-CFM, optionally Gaussian-conditional with constant width sigma):

        x_t = (1 - t) x0 + t x1 + sigma * eps,     eps ~ N(0, I)
        u(t | x0, x1) = x1 - x0

    The conditional velocity is independent of t for this path -- but the MARGINAL field
    it defines, v(t, x) = E[x1 - x0 | x_t = x], is genuinely t-dependent, which is why the
    network takes t as an input and why the self-test checks the learned field against the
    closed-form marginal velocity of a Gaussian-to-Gaussian problem rather than only
    against the regression target.

    COUPLING IN OBSERVED SPACE. `couple` consumes the observed pre/post arrays and returns
    index pairs. It is never handed latent coordinates: a coupling computed in a learned
    chart is a function of the current encoder, so it drifts during training and is not a
    property of the data. Options:
      'independent' product coupling (random pairing),
      'ot'          exact minibatch assignment minimising sum ||x0_i - x1_{pi(i)}||^2,
      'sinkhorn'    entropic OT, log-domain; `unbalanced_tau` gives the UOT relaxation,
      'paired'      index-aligned, for the case where the observations ARE paired.
    For 'sinkhorn' the pairing is drawn from the row-conditional of the coupling, which is
    the standard stochastic-pairing estimator of the entropic plan.

    MEASURED CAVEAT on 'ot'. Minibatch OT is NOT the paired coupling even when the target
    is a deterministic function of the source. On the affine test system of
    `_test_cfm_correctness` (x1 = A x0 + b0, n = 512) exact assignment agrees with the true
    pairing on only 79.5% of samples, and entropic OT on 22.9%. Each coupling therefore
    defines a DIFFERENT marginal velocity field, and a model fitted under one must not be
    scored against the closed-form marginal of another -- doing so looks like
    underfitting (cosine ~0.69, and it does not improve when the loss falls 14x under
    longer training) when it is in fact a coupling mismatch. This is why the closed-form
    marginal check uses 'paired' and why coupling choice is a reported, validated
    selection rather than a default.
    """

    def __init__(self, sigma: float = 0.0, coupling: str = "independent",
                 sinkhorn_eps: float = 0.05, sinkhorn_iters: int = 200,
                 unbalanced_tau: float | None = None, seed: int = 0,
                 max_coupling_n: int = 512):
        if coupling not in ("independent", "ot", "sinkhorn", "paired"):
            raise ValueError(f"unknown coupling {coupling!r}")
        self.sigma = float(sigma)
        self.coupling = str(coupling)
        self.sinkhorn_eps = float(sinkhorn_eps)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.unbalanced_tau = unbalanced_tau
        self.rng = np.random.default_rng(int(seed))
        self.max_coupling_n = int(max_coupling_n)

    # -- coupling -----------------------------------------------------------
    def couple(self, X0: np.ndarray, X1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Observed-space coupling -> (i0, i1) index arrays of equal length."""
        X0 = np.asarray(X0, dtype=np.float64)
        X1 = np.asarray(X1, dtype=np.float64)
        n0, n1 = X0.shape[0], X1.shape[0]
        if self.coupling == "paired":
            if n0 != n1:
                raise ValueError("'paired' coupling needs equal population sizes")
            i = np.arange(n0)
            return i, i
        if self.coupling == "independent":
            i0 = np.arange(n0)
            i1 = self.rng.integers(0, n1, size=n0)
            return i0, i1
        n = min(n0, n1, self.max_coupling_n)
        s0 = self.rng.permutation(n0)[:n]
        s1 = self.rng.permutation(n1)[:n]
        C = _sq_cost(X0[s0], X1[s1])
        if self.coupling == "ot":
            from scipy.optimize import linear_sum_assignment
            ri, ci = linear_sum_assignment(C)
            return s0[ri], s1[ci]
        P = _sinkhorn_log(C, eps=self.sinkhorn_eps, iters=self.sinkhorn_iters,
                          unbalanced_tau=self.unbalanced_tau)
        rows = P / np.maximum(P.sum(axis=1, keepdims=True), 1e-300)
        cum = np.cumsum(rows, axis=1)
        draw = self.rng.random((n, 1))
        pick = (draw > cum).sum(axis=1).clip(0, n - 1)
        return s0, s1[pick]

    # -- interpolant --------------------------------------------------------
    def sample_path(self, x0: torch.Tensor, x1: torch.Tensor,
                    t: torch.Tensor | None = None,
                    generator: torch.Generator | None = None) -> dict:
        """Draw t ~ U(0,1), build x_t, return the regression target u = x1 - x0."""
        B = x0.shape[0]
        if t is None:
            t = torch.rand((B, 1), generator=generator, dtype=x0.dtype)
        t = t.reshape(B, 1)
        noise = torch.zeros_like(x0)
        if self.sigma > 0.0:
            noise = self.sigma * torch.randn(x0.shape, generator=generator, dtype=x0.dtype)
        x_t = (1.0 - t) * x0 + t * x1 + noise
        u = x1 - x0
        return dict(t=t, x_t=x_t, target=u, noise=noise)

    def loss(self, v_pred: torch.Tensor, target: torch.Tensor,
             weights: torch.Tensor | None = None) -> torch.Tensor:
        se = ((v_pred - target) ** 2).sum(dim=1)
        if weights is None:
            return se.mean()
        w = weights.reshape(-1)
        return (se * w).sum() / w.sum().clamp_min(1e-12)

    def describe(self) -> dict:
        return dict(path="linear (I-CFM)", sigma=self.sigma, coupling=self.coupling,
                    sinkhorn_eps=self.sinkhorn_eps, sinkhorn_iters=self.sinkhorn_iters,
                    unbalanced_tau=self.unbalanced_tau,
                    conditional_velocity="x1 - x0 (t-independent for this path)")


# ----------------------------------------------------------------------------------
# wiring: the conditional velocity field
# ----------------------------------------------------------------------------------
class ConditionalVelocity(nn.Module):
    """v_theta(t, z | P, tau, mu0) -- the object the objective regresses.

    Composition:
        a       = mask (.) dose(tau)                       (B, k)   exact zeros off-set
        c       = PopulationEncoder(mu0)                   (B, code_dim) or (B, 0)
        b       = OneFormHierarchy.b(t, x, c, a)           (B, d)   contract entry point;
                  the one-form owns the embedding table and forms (m1, m2) internally
        v       = pullback(t, z, b)                        (B, d)

    `pullback` is INJECTED (default `identity_pullback`, which is what makes this module
    independently testable); the orchestrator passes a callable wrapping
    `ihcfm_geometry.PullbackField` so that v = G^{-1} J^T b. `dose` is likewise injected
    (default `StandInDose`; the geometry track's HillDose drops in with the same
    forward(tau) -> (B, k) contract). Multiplying by `mask` rather than indexing keeps
    non-member interventions at exactly 0.0, which is what singleton vanishing rests on.

    The float64 inference path (`.to_inference_dtype()`) exists because Test A of the
    chart harness is limited by the network's evaluation precision, not by the
    transformation: float32 inference stalls at ~1e-8 while the same weights in float64
    reach ~1e-14 (project fact D). It is a dtype cast for evaluation, not a retrain.
    """

    def __init__(self, d: int, k: int, r: int = 4, code_dim: int = 0, hidden: int = 64,
                 depth: int = 2, dose: nn.Module | None = None,
                 one_form: OneFormHierarchy | None = None,
                 pullback: Callable | None = None, embed_seed: int | None = 0,
                 interaction_enabled: bool = True, decoder: Callable | None = None,
                 interaction_sees_m1: bool = True):
        super().__init__()
        self.d, self.k, self.r = int(d), int(k), int(r)
        self.embed = InterventionEmbed(self.k, self.r, seed=embed_seed)
        self.moments = SetMoments()
        self.dose = dose if dose is not None else StandInDose(self.k)
        self.encoder = PopulationEncoder(self.d, code_dim=code_dim, hidden=hidden)
        # ONE embedding table in the composed model: the one-form owns the contract entry
        # point .b(t, x, c, a, ids), so it must see the same table this module doses.
        # A default one_form is constructed AROUND our table. An injected one_form brings
        # its own, and we do NOT silently adopt it: `dose` and the mask are already sized
        # to k, so a table with a different k would make mask (.) dose(tau) either raise
        # or (at k = 1) broadcast one amplitude onto every intervention. Fail loudly.
        self.one_form = one_form if one_form is not None else OneFormHierarchy(
            self.d, r=self.r, code_dim=int(code_dim), hidden=hidden, depth=depth,
            interaction_enabled=interaction_enabled, k=self.k, embed=self.embed,
            interaction_sees_m1=interaction_sees_m1)
        if self.one_form.embed is not self.embed:
            if self.one_form.embed.k != self.k or self.one_form.embed.r != self.r:
                raise ValueError(
                    f"injected one_form has an embedding table of (k={self.one_form.embed.k}, "
                    f"r={self.one_form.embed.r}) but this module is sized (k={self.k}, "
                    f"r={self.r}); the dose module and the intervention mask are built to k, "
                    "so adopting a different table would silently mis-broadcast amplitudes")
            self.embed = self.one_form.embed
        if self.one_form.d != self.d:
            raise ValueError(f"injected one_form is on d={self.one_form.d} but this module "
                             f"is on d={self.d}")
        if self.one_form.r != self.r:
            raise ValueError(f"injected one_form has r={self.one_form.r} but this module "
                             f"has r={self.r}")
        # the dose module must emit one amplitude per intervention
        with torch.no_grad():
            n_dose = int(self.dose(torch.zeros(2, 1)).shape[-1])
        if n_dose != self.k:
            raise ValueError(f"dose module emits {n_dose} amplitudes but k={self.k}")
        self.pullback = pullback if pullback is not None else identity_pullback
        self.decoder = decoder            # z -> x; None means observed space == latent
        self.code_dim = int(code_dim)

    # -- switches -----------------------------------------------------------
    def set_interaction_enabled(self, flag: bool) -> None:
        self.one_form.interaction_enabled = bool(flag)

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        g = self.one_form.parameter_groups()
        g["main"] = g["main"] + list(self.embed.parameters()) + list(self.dose.parameters())
        if self.encoder.enabled:
            g["encoder"] = list(self.encoder.parameters())
        return g

    # -- pieces -------------------------------------------------------------
    def amplitudes(self, tau: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """a = mask (.) dose(tau), with tau either (B, 1) shared or (B, k) per-intervention.

        The product with `mask` is elementwise and exact: a non-member intervention has
        a_p == 0.0 bit-exactly regardless of what the dose module returns, and an
        intervention at tau_p = 0 has dose 0.0 bit-exactly. Both routes to "this
        intervention is absent" therefore give the same exact zero, which is what the
        second moment's structural vanishing relies on.
        """
        return mask * self.dose(tau)

    def code(self, mu0: torch.Tensor | None) -> torch.Tensor | None:
        if mu0 is None or self.code_dim == 0:
            return None
        return self.encoder(mu0)

    def one_form_value(self, t, z, tau, mask, c=None) -> torch.Tensor:
        x = z if self.decoder is None else self.decoder(z)
        return self.one_form.b(t, x, c, self.amplitudes(tau, mask))

    def forward(self, t, z, tau, mask, c=None) -> torch.Tensor:
        b = self.one_form_value(t, z, tau, mask, c=c)
        return self.pullback(t, z, b)

    def interaction_only(self, t, z, tau, mask, c=None) -> torch.Tensor:
        x = z if self.decoder is None else self.decoder(z)
        b_int = self.one_form.interaction_only(t, x, c, self.amplitudes(tau, mask))
        return self.pullback(t, z, b_int)

    def main_only(self, t, z, tau, mask, c=None) -> torch.Tensor:
        x = z if self.decoder is None else self.decoder(z)
        b_main = self.one_form.main_only(t, x, c, self.amplitudes(tau, mask))
        return self.pullback(t, z, b_main)

    def to_inference_dtype(self, dtype=torch.float64) -> "ConditionalVelocity":
        """Cast a COPY of the trained weights for evaluation (project fact D)."""
        m = copy.deepcopy(self).to(dtype)
        m.eval()
        return m


# ----------------------------------------------------------------------------------
# the three-stage schedule
# ----------------------------------------------------------------------------------
def _pop_arrays(populations: Sequence, idx: Sequence[int], k: int) -> list[dict]:
    """Pack the fields this module needs out of whatever population objects it is given.

    Duck-typed on (.P, .tau, .pre, .post) so it accepts `composefm.synthetic.Population`
    without importing it -- this module stays free of any dependency on the benchmark.
    """
    out = []
    for i in idx:
        p = populations[i]
        mask = np.zeros(k, dtype=np.float64)
        for q in p.P:
            mask[int(q)] = 1.0
        out.append(dict(i=int(i), P=tuple(int(q) for q in p.P), tau=float(p.tau),
                        order=len(p.P), mask=mask,
                        pre=np.asarray(p.pre, dtype=np.float64),
                        post=np.asarray(p.post, dtype=np.float64)))
    return out


def _draw_batch(rec: dict, obj: CFMObjective, batch: int, rng: np.random.Generator,
                dtype=torch.float32) -> dict:
    """One CFM minibatch from one population: observed-space coupling, then interpolant."""
    i0, i1 = obj.couple(rec["pre"], rec["post"])
    n = i0.shape[0]
    take = rng.permutation(n)[:min(batch, n)]
    x0 = torch.as_tensor(rec["pre"][i0[take]], dtype=dtype)
    x1 = torch.as_tensor(rec["post"][i1[take]], dtype=dtype)
    path = obj.sample_path(x0, x1)
    m = x0.shape[0]
    path["mask"] = torch.as_tensor(np.tile(rec["mask"], (m, 1)), dtype=dtype)
    path["tau"] = torch.full((m, 1), rec["tau"], dtype=dtype)
    path["mu0"] = torch.as_tensor(rec["pre"][rng.permutation(rec["pre"].shape[0])[:64]],
                                  dtype=dtype)
    return path


def _cfm_loss_on(model: ConditionalVelocity, recs: Sequence[dict], obj: CFMObjective,
                 rng: np.random.Generator, batch: int = 256, reps: int = 4) -> float:
    """Mean CFM loss over the given populations. Used as the validation criterion."""
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for _ in range(int(reps)):
            for rec in recs:
                bt = _draw_batch(rec, obj, batch, rng)
                c = model.code(bt["mu0"].unsqueeze(0))
                v = model(bt["t"], bt["x_t"], bt["tau"], bt["mask"], c=c)
                tot += float(obj.loss(v, bt["target"]))
                cnt += 1
    return tot / max(cnt, 1)


def integrate_velocity(model: ConditionalVelocity, Z0: np.ndarray, P: Sequence[int],
                       tau: float, k: int, n_steps: int = 16,
                       mu0: np.ndarray | None = None, dtype=torch.float64) -> np.ndarray:
    """RK4 flow of v_theta over t in [0, 1]. float64 by default (project fact D)."""
    m = model if next(model.parameters()).dtype == dtype else model.to_inference_dtype(dtype)
    mask = np.zeros(k, dtype=np.float64)
    for q in P:
        mask[int(q)] = 1.0
    Z = np.array(Z0, dtype=np.float64, copy=True)
    n = Z.shape[0]
    mk = torch.as_tensor(np.tile(mask, (n, 1)), dtype=dtype)
    tv = torch.full((n, 1), float(tau), dtype=dtype)
    c = None
    if m.code_dim > 0:
        src = Z0 if mu0 is None else mu0
        c = m.code(torch.as_tensor(np.atleast_2d(src), dtype=dtype).unsqueeze(0))

    def f(Zc: np.ndarray, t: float) -> np.ndarray:
        with torch.no_grad():
            z = torch.as_tensor(Zc, dtype=dtype)
            tt = torch.full((Zc.shape[0], 1), float(t), dtype=dtype)
            return m(tt, z, tv, mk, c=c).numpy().astype(np.float64)

    h = 1.0 / int(n_steps)
    t = 0.0
    for _ in range(int(n_steps)):
        k1 = f(Z, t)
        k2 = f(Z + 0.5 * h * k1, t + 0.5 * h)
        k3 = f(Z + 0.5 * h * k2, t + 0.5 * h)
        k4 = f(Z + h * k3, t + h)
        Z = Z + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += h
    return Z


def _run_stage(model: ConditionalVelocity, recs: Sequence[dict], obj: CFMObjective,
               opt: torch.optim.Optimizer, n_steps: int, batch: int,
               rng: np.random.Generator, weights: np.ndarray | None = None,
               anchor: ConditionalVelocity | None = None,
               anchor_recs: Sequence[dict] | None = None, anchor_weight: float = 0.0,
               int_penalty: float = 0.0, grad_clip: float = 5.0) -> dict:
    """One optimisation stage. Non-finite steps are SKIPPED and counted, never applied."""
    if not recs:
        return dict(steps=0, skipped_nonfinite=0, final_loss=float("nan"),
                    mean_loss_last_decile=float("nan"), wall_clock_s=0.0)
    t0 = time.time()
    probs = None if weights is None else np.asarray(weights, dtype=np.float64) / float(
        np.sum(weights))
    skipped = 0
    hist: list[float] = []
    for step in range(int(n_steps)):
        j = int(rng.choice(len(recs), p=probs)) if probs is not None else int(
            rng.integers(0, len(recs)))
        bt = _draw_batch(recs[j], obj, batch, rng)
        c = model.code(bt["mu0"].unsqueeze(0))
        v = model(bt["t"], bt["x_t"], bt["tau"], bt["mask"], c=c)
        loss = obj.loss(v, bt["target"])
        total = loss
        if int_penalty > 0.0:
            vi = model.interaction_only(bt["t"], bt["x_t"], bt["tau"], bt["mask"], c=c)
            total = total + float(int_penalty) * (vi ** 2).sum(dim=1).mean()
        if anchor is not None and anchor_weight > 0.0 and anchor_recs:
            ja = int(rng.integers(0, len(anchor_recs)))
            ba = _draw_batch(anchor_recs[ja], obj, batch, rng)
            ca = model.code(ba["mu0"].unsqueeze(0))
            va = model(ba["t"], ba["x_t"], ba["tau"], ba["mask"], c=ca)
            with torch.no_grad():
                ca0 = anchor.code(ba["mu0"].unsqueeze(0))
                v0 = anchor(ba["t"], ba["x_t"], ba["tau"], ba["mask"], c=ca0)
            total = total + float(anchor_weight) * ((va - v0) ** 2).sum(dim=1).mean()
        if not torch.isfinite(total):
            skipped += 1
            opt.zero_grad(set_to_none=True)
            continue
        opt.zero_grad(set_to_none=True)
        total.backward()
        gn = torch.nn.utils.clip_grad_norm_([p for gp in opt.param_groups
                                             for p in gp["params"]], grad_clip)
        if not torch.isfinite(gn):
            skipped += 1
            opt.zero_grad(set_to_none=True)
            continue
        opt.step()
        hist.append(float(loss))
    tail = max(1, len(hist) // 10)
    return dict(steps=int(n_steps), applied=len(hist), skipped_nonfinite=skipped,
                final_loss=(hist[-1] if hist else float("nan")),
                mean_loss_first_decile=(float(np.mean(hist[:tail])) if hist else float("nan")),
                mean_loss_last_decile=(float(np.mean(hist[-tail:])) if hist else float("nan")),
                wall_clock_s=time.time() - t0)


def train_hierarchical(model: ConditionalVelocity, populations: Sequence,
                       train_idx: Sequence[int], objective: CFMObjective | None = None,
                       val_idx: Sequence[int] | None = None,
                       steps: tuple[int, int, int] = (300, 300, 200),
                       batch: int = 128, lr: float = 3e-3, lr_stage3: float = 1e-3,
                       anchor_weight: float = 1.0, int_penalty: float = 0.0,
                       singleton_oversample: float = 3.0, seed: int = 0,
                       freeze_main_stage2: bool = False,
                       eval_states: np.ndarray | None = None,
                       eval_tau: float = 1.0, verbose: bool = False) -> dict:
    """Three-stage hierarchical fit. Returns a report; the model is trained in place.

    Stage 1  singleton conditions only, interaction branch HARD-DISABLED.
    Stage 2  combination conditions, interaction branch enabled, stage-1 model frozen as
             an anchor and a penalty applied to singleton-condition velocity drift.
    Stage 3  all conditions jointly, singletons oversampled by `singleton_oversample`,
             plus `int_penalty` * mean ||v_int||^2. Use `select_interaction_penalty` to
             pick that coefficient on held-out combinations rather than by hand.

    `val_idx` (held-out populations) is only ever READ, for reporting and selection.
    """
    obj = objective if objective is not None else CFMObjective(seed=seed)
    rng = np.random.default_rng(int(seed))
    k = model.k
    recs = _pop_arrays(populations, train_idx, k)
    singles = [r for r in recs if r["order"] == 1]
    combos = [r for r in recs if r["order"] >= 2]
    val_recs = _pop_arrays(populations, val_idx, k) if val_idx else []
    # Held-out COMBINATIONS specifically. The interaction-norm penalty is a statement
    # about combination conditions, so it must be selected on those and not on an average
    # that singleton held-out populations can dominate.
    val_combos = [r for r in val_recs if r["order"] >= 2]
    if not singles:
        raise ValueError("stage 1 needs singleton conditions in the training index")
    report: dict = dict(config=dict(steps=list(steps), batch=batch, lr=lr,
                                   lr_stage3=lr_stage3, anchor_weight=anchor_weight,
                                   int_penalty=int_penalty,
                                   singleton_oversample=singleton_oversample,
                                   n_singletons=len(singles), n_combos=len(combos),
                                   n_val=len(val_recs), objective=obj.describe()))
    if eval_states is None:
        eval_states = np.concatenate([r["pre"] for r in singles], axis=0)[:128]
    Zev = np.asarray(eval_states, dtype=np.float64)

    def _singleton_velocities(m: ConditionalVelocity) -> np.ndarray:
        """v on every singleton condition at a fixed t-grid -- the anchoring probe."""
        rows = []
        mdl = m.to_inference_dtype(torch.float64)
        with torch.no_grad():
            for p in range(k):
                mask = np.zeros(k)
                mask[p] = 1.0
                mk = torch.as_tensor(np.tile(mask, (Zev.shape[0], 1)), dtype=torch.float64)
                tv = torch.full((Zev.shape[0], 1), float(eval_tau), dtype=torch.float64)
                z = torch.as_tensor(Zev, dtype=torch.float64)
                c = mdl.code(z.unsqueeze(0)) if mdl.code_dim > 0 else None
                for tt in (0.0, 0.25, 0.5, 0.75, 1.0):
                    t = torch.full((Zev.shape[0], 1), tt, dtype=torch.float64)
                    rows.append(mdl(t, z, tv, mk, c=c).numpy().ravel())
        return np.concatenate(rows)

    # ---------------- stage 1: main effects on singletons ----------------
    model.set_interaction_enabled(False)
    groups = model.parameter_groups()
    p1 = groups["main"] + groups.get("encoder", [])
    opt1 = torch.optim.Adam(p1, lr=lr)
    report["stage1"] = _run_stage(model, singles, obj, opt1, steps[0], batch, rng)
    v_stage1 = _singleton_velocities(model)
    anchor = copy.deepcopy(model)
    for prm in anchor.parameters():
        prm.requires_grad_(False)
    anchor.eval()
    report["stage1"]["val_cfm_loss"] = (_cfm_loss_on(model, val_recs, obj, rng)
                                        if val_recs else float("nan"))
    report["stage1"]["val_combo_cfm_loss"] = (_cfm_loss_on(model, val_combos, obj, rng)
                                              if val_combos else float("nan"))
    report["stage1"]["train_combo_cfm_loss"] = (_cfm_loss_on(model, combos, obj, rng)
                                                if combos else float("nan"))

    # ---------------- stage 2: interaction on combinations, singletons anchored ------
    model.set_interaction_enabled(True)
    if combos:
        # Stage-2 main-branch policy. The soft anchor penalises singleton-velocity DRIFT
        # but does not stop combination-only gradients from reshaping the main branch,
        # which lets interaction signal leak into main effects (measured: see
        # docs/GATE_LOG.md, stage-2 diagnosis). freeze_main_stage2=True optimises ONLY
        # the interaction group in stage 2, making the anchor structural rather than a
        # penalty. Both are training-schedule options; neither changes the architecture.
        stage2_params = (list(groups["interaction"]) if freeze_main_stage2
                         else list(groups["interaction"]) + list(p1))
        report.setdefault("config", {})["freeze_main_stage2"] = bool(freeze_main_stage2)
        opt2 = torch.optim.Adam(stage2_params, lr=lr)
        report["stage2"] = _run_stage(model, combos, obj, opt2, steps[1], batch, rng,
                                      anchor=anchor, anchor_recs=singles,
                                      anchor_weight=anchor_weight)
    else:
        report["stage2"] = dict(steps=0, skipped_nonfinite=0, note="no combination conditions")
    v_stage2 = _singleton_velocities(model)
    drift = float(np.linalg.norm(v_stage2 - v_stage1)
                  / max(np.linalg.norm(v_stage1), 1e-300))
    report["stage2"]["singleton_velocity_drift_rel"] = drift
    report["stage2"]["val_cfm_loss"] = (_cfm_loss_on(model, val_recs, obj, rng)
                                        if val_recs else float("nan"))
    report["stage2"]["val_combo_cfm_loss"] = (_cfm_loss_on(model, val_combos, obj, rng)
                                              if val_combos else float("nan"))
    report["stage2"]["train_combo_cfm_loss"] = (_cfm_loss_on(model, combos, obj, rng)
                                                if combos else float("nan"))

    # ---------------- stage 3: joint fine-tune -------------------------------------
    all_recs = singles + combos
    w = np.array([float(singleton_oversample) if r["order"] == 1 else 1.0
                  for r in all_recs])
    opt3 = torch.optim.Adam(list(model.parameters()), lr=lr_stage3)
    report["stage3"] = _run_stage(model, all_recs, obj, opt3, steps[2], batch, rng,
                                  weights=w, int_penalty=int_penalty,
                                  anchor=anchor, anchor_recs=singles,
                                  anchor_weight=0.25 * anchor_weight)
    v_stage3 = _singleton_velocities(model)
    report["stage3"]["singleton_velocity_drift_rel"] = float(
        np.linalg.norm(v_stage3 - v_stage1) / max(np.linalg.norm(v_stage1), 1e-300))
    report["stage3"]["val_cfm_loss"] = (_cfm_loss_on(model, val_recs, obj, rng)
                                        if val_recs else float("nan"))
    report["stage3"]["val_combo_cfm_loss"] = (_cfm_loss_on(model, val_combos, obj, rng)
                                              if val_combos else float("nan"))
    report["stage3"]["train_combo_cfm_loss"] = (_cfm_loss_on(model, combos, obj, rng)
                                                if combos else float("nan"))

    report["skipped_nonfinite_total"] = int(sum(
        int(report[s].get("skipped_nonfinite", 0)) for s in ("stage1", "stage2", "stage3")))
    report["wall_clock_s"] = float(sum(
        float(report[s].get("wall_clock_s", 0.0)) for s in ("stage1", "stage2", "stage3")))
    if verbose:
        print(json.dumps({s: report[s] for s in ("stage1", "stage2", "stage3")}, indent=2))
    return report


def select_interaction_penalty(model_factory: Callable[[], ConditionalVelocity],
                               populations: Sequence, train_idx: Sequence[int],
                               val_idx: Sequence[int],
                               lambdas: Sequence[float] = (0.0, 1e-3, 1e-2),
                               objective_kw: dict | None = None,
                               seed: int = 0, **train_kw) -> dict:
    """Select the stage-3 interaction-norm coefficient on HELD-OUT combinations.

    Returns the per-lambda validation losses, the selected lambda and the winning model.
    Selection is on held-out combination CFM loss -- never on the training conditions and
    never on the interaction-recovery score (that would leak the ground truth the recovery
    metric is supposed to test).
    """
    rows = []
    best = None
    for lam in lambdas:
        m = model_factory()
        obj = CFMObjective(seed=seed, **(objective_kw or {}))
        rep = train_hierarchical(m, populations, train_idx, objective=obj,
                                 val_idx=val_idx, int_penalty=float(lam), seed=seed,
                                 **train_kw)
        vl = float(rep["stage3"]["val_combo_cfm_loss"])
        if not math.isfinite(vl):
            raise ValueError("no held-out COMBINATION populations in val_idx: the "
                             "interaction-norm penalty cannot be selected without them")
        rows.append(dict(int_penalty=float(lam), val_combo_cfm_loss=vl,
                         val_cfm_loss_all_heldout=float(rep["stage3"]["val_cfm_loss"]),
                         singleton_drift=float(rep["stage3"]["singleton_velocity_drift_rel"])))
        if best is None or vl < best[0]:
            best = (vl, float(lam), m, rep)
    return dict(rows=rows, selected_int_penalty=best[1],
                selected_val_combo_loss=best[0],
                selection_criterion="held-out COMBINATION CFM loss (order >= 2 only)",
                model=best[2], report=best[3])


# ==================================================================================
# self-test
# ==================================================================================
def _energy_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Two-sample energy distance. Scale-free comparison of two empirical measures."""
    def pd(A, B):
        return np.sqrt(np.maximum(
            np.sum(A ** 2, 1)[:, None] + np.sum(B ** 2, 1)[None, :] - 2 * A @ B.T, 0.0))
    return float(2 * pd(X, Y).mean() - pd(X, X).mean() - pd(Y, Y).mean())


def _test_permutation_invariance(seed: int = 0) -> dict:
    """H1: relabelling the interventions consistently leaves (m1, m2) and b unchanged."""
    torch.manual_seed(seed)
    k, r, d, B = 6, 3, 5, 7
    embed = InterventionEmbed(k, r, seed=seed)
    of = OneFormHierarchy(d, r=r, code_dim=0, hidden=32, k=k, embed=embed)
    a = torch.rand(B, k, dtype=torch.float64)
    a[:, 2] = 0.0                                   # one absent intervention, on purpose
    x = torch.randn(B, d, dtype=torch.float64)
    t = torch.rand(B, 1, dtype=torch.float64)
    of = of.double()
    perm = torch.as_tensor(np.random.default_rng(seed).permutation(k).copy())
    with torch.no_grad():
        m1, m2 = of.set_moments(a)
        b0 = of.b(t, x, None, a)
    with torch.no_grad():
        of.embed.weight.copy_(of.embed.weight.index_select(0, perm))
    a_p = a.index_select(1, perm)
    with torch.no_grad():
        m1p, m2p = of.set_moments(a_p)
        b1 = of.b(t, x, None, a_p)
    d_m1 = float((m1 - m1p).abs().max())
    d_m2 = float((m2 - m2p).abs().max())
    d_b = float((b0 - b1).abs().max())
    # Sparse form: pass only the present ids. It must agree with the dense form to
    # round-off (not bit-exactly: dropping a term changes torch's reduction tree, so the
    # sum is reassociated -- measured ~2e-16). It IS part of the pass criterion at 1e-14.
    with torch.no_grad():
        of.embed.weight.copy_(of.embed.weight.index_select(
            0, torch.argsort(perm)))                # restore
    ids = [j for j in range(k) if float(a[0, j]) != 0.0]
    with torch.no_grad():
        m1s, m2s = of.set_moments(a[:, ids], ids=ids)
    d_s1 = float((m1 - m1s).abs().max())
    d_s2 = float((m2 - m2s).abs().max())
    return dict(max_abs_dev_m1=d_m1, max_abs_dev_m2=d_m2, max_abs_dev_b=d_b,
                sparse_vs_dense_m1=d_s1, sparse_vs_dense_m2=d_s2,
                passed=bool(max(d_m1, d_m2, d_b, d_s1, d_s2) <= 1e-14))


def _test_singleton_vanishing(seed: int = 0) -> dict:
    """H2: interaction_only is BIT-EXACTLY zero for singletons and for any zero exposure."""
    torch.manual_seed(seed)
    k, r, d, B = 5, 4, 6, 9
    embed = InterventionEmbed(k, r, seed=seed)
    of = OneFormHierarchy(d, r=r, code_dim=3, hidden=32, k=k, embed=embed).double()
    x = torch.randn(B, d, dtype=torch.float64)
    t = torch.rand(B, 1, dtype=torch.float64)
    c = torch.randn(B, 3, dtype=torch.float64)
    worst_single, worst_zero, worst_m2 = 0.0, 0.0, 0.0
    n_exact = 0
    n_cases = 0
    for p in range(k):
        for amp in (0.3, 1.0, 7.0):
            a = torch.zeros(B, k, dtype=torch.float64)
            a[:, p] = amp
            _, m2 = of.set_moments(a)
            vi = of.interaction_only(t, x, c, a)
            worst_single = max(worst_single, float(vi.abs().max()))
            worst_m2 = max(worst_m2, float(m2.abs().max()))
            n_exact += int(bool((vi == 0.0).all()))
            n_cases += 1
    # A full set in which ONE exposure is exactly zero: every pair term through that
    # member must die. The BIT-EXACT probe for this is embedding-invariance: if u_p =
    # 0.0 * e_p is exactly zero then m2 cannot depend on e_p at all, so perturbing that
    # row of the table must leave m2 bit-identical (the reduction tree is unchanged, so
    # there is no reassociation to hide behind).
    worst_zero_bitexact = 0.0
    worst_contrib, worst_ordered = 0.0, 0.0
    n_zero_exact, n_zero_cases = 0, 0
    rngz = np.random.default_rng(seed + 5)
    for p in range(k):
        a = torch.rand(B, k, dtype=torch.float64) + 0.5
        a[:, p] = 0.0
        _, m2_full = of.set_moments(a)
        vi_full = of.interaction_only(t, x, c, a)
        saved = of.embed.weight.detach().clone()
        with torch.no_grad():
            of.embed.weight[p] = torch.as_tensor(
                rngz.normal(size=of.embed.r) * 13.0, dtype=of.embed.weight.dtype)
        _, m2_pert = of.set_moments(a)
        vi_pert = of.interaction_only(t, x, c, a)
        with torch.no_grad():
            of.embed.weight.copy_(saved)
        dev = max(float((m2_full - m2_pert).abs().max()),
                  float((vi_full - vi_pert).abs().max()))
        worst_zero_bitexact = max(worst_zero_bitexact, dev)
        n_zero_exact += int(dev == 0.0)
        n_zero_cases += 1
        # Reported separately and NOT part of the pass criterion: comparing the dense
        # k-member moment against the sparse (k-1)-member one is a REASSOCIATION of a
        # floating-point sum (torch's blocked reduction tree changes shape when a term is
        # dropped), so it is ~1e-16 rather than 0.0. That is a property of summation
        # order, not of the construction, and folding it into the structural claim would
        # misattribute it.
        ids = [j for j in range(k) if j != p]
        _, m2_drop = of.set_moments(a[:, ids], ids=ids)
        worst_zero = max(worst_zero, float((m2_full - m2_drop).abs().max()))
        # ATTRIBUTION, measured rather than asserted. Two facts pin the ~1e-15 on
        # summation order: (i) the dropped member's own contributions u_p and u_p^2 are
        # exactly 0.0, so it adds nothing to either accumulator; (ii) recomputing both
        # moments with an IDENTICAL sequential reduction order (zero member accumulated
        # last) gives a difference of exactly 0.0. Under torch's default blocked
        # reduction the same comparison gives ~1e-15.
        u = a.unsqueeze(-1) * of.embed.table().unsqueeze(0)
        worst_contrib = max(worst_contrib, float(u[:, p].abs().max()),
                            float((u[:, p] * u[:, p]).abs().max()))

        def _ordered_m2(a_, E_, order):
            uu = a_.unsqueeze(-1) * E_.unsqueeze(0)
            m1_ = torch.zeros(a_.shape[0], E_.shape[1], dtype=a_.dtype)
            sq_ = torch.zeros_like(m1_)
            for j in order:
                m1_ = m1_ + uu[:, j]
                sq_ = sq_ + uu[:, j] * uu[:, j]
            return 0.5 * (m1_ * m1_ - sq_)

        with torch.no_grad():
            Etab = of.embed.table()
            mo_full = _ordered_m2(a, Etab, ids + [p])
            mo_drop = _ordered_m2(a[:, ids], Etab[ids], range(k - 1))
        worst_ordered = max(worst_ordered, float((mo_full - mo_drop).abs().max()))
    # and the doubly structural case: the branch also respects the hard disable switch
    of.interaction_enabled = False
    a = torch.rand(B, k, dtype=torch.float64) + 0.5
    disabled = float(of.interaction_only(t, x, c, a).abs().max())
    of.interaction_enabled = True
    return dict(max_abs_interaction_on_singletons=worst_single,
                max_abs_m2_on_singletons=worst_m2,
                bit_exact_singleton_cases=f"{n_exact}/{n_cases}",
                max_dev_perturbing_zero_exposure_embedding=worst_zero_bitexact,
                bit_exact_zero_exposure_cases=f"{n_zero_exact}/{n_zero_cases}",
                max_dev_dense_vs_sparse_default_reduction=worst_zero,
                max_dev_dense_vs_sparse_forced_same_order=worst_ordered,
                max_abs_dropped_member_contribution=worst_contrib,
                reassociation_note=("dense-vs-sparse moment agreement is ~1e-15 under "
                                    "torch's default blocked reduction and EXACTLY 0.0 "
                                    "when the reduction order is forced to match, while "
                                    "the dropped member's own contributions are exactly "
                                    "0.0 -- so the residual is summation order, not the "
                                    "construction. Both numbers are reported and the "
                                    "forced-order one IS in the pass criterion"),
                max_abs_when_branch_disabled=disabled,
                passed=bool(worst_single == 0.0 and worst_m2 == 0.0
                            and n_exact == n_cases
                            and worst_zero_bitexact == 0.0
                            and n_zero_exact == n_zero_cases
                            and worst_contrib == 0.0
                            and worst_ordered == 0.0
                            and disabled == 0.0))


def _test_zero_dose_identity(seed: int = 0) -> dict:
    """H3: all tau = 0 gives a == 0, b == 0 and hence the identity flow."""
    torch.manual_seed(seed)
    k, r, d, n = 4, 3, 5, 64
    m = ConditionalVelocity(d, k, r=r, code_dim=4, hidden=32).double()
    Z = np.random.default_rng(seed).normal(size=(n, d))
    z = torch.as_tensor(Z, dtype=torch.float64)
    mask = torch.ones(n, k, dtype=torch.float64)
    tau0 = torch.zeros(n, 1, dtype=torch.float64)
    c = m.code(z.unsqueeze(0))
    a0 = float(m.amplitudes(tau0, mask).abs().max())
    worst_b, worst_v = 0.0, 0.0
    for tt in (0.0, 0.37, 1.0):
        t = torch.full((n, 1), tt, dtype=torch.float64)
        worst_b = max(worst_b, float(m.one_form_value(t, z, tau0, mask, c=c).abs().max()))
        worst_v = max(worst_v, float(m(t, z, tau0, mask, c=c).abs().max()))
    Zf = integrate_velocity(m, Z, tuple(range(k)), 0.0, k, n_steps=8)
    ident = float(np.max(np.abs(Zf - Z)))
    return dict(max_abs_amplitude_at_tau0=a0, max_abs_b_at_tau0=worst_b,
                max_abs_v_at_tau0=worst_v, max_abs_flow_identity_defect=ident,
                passed=bool(a0 == 0.0 and worst_b == 0.0 and worst_v == 0.0
                            and ident == 0.0))


def _test_moment_cost(seed: int = 0, ks: Sequence[int] = (2, 4, 8, 16, 32),
                      r: int = 4, B: int = 256, reps: int = 30) -> dict:
    """O(kr) moments vs the explicit O(k^2) pair loop: same value, growing cost ratio."""
    rows = []
    for k in ks:
        embed = InterventionEmbed(k, r, seed=seed)
        E = embed.table().detach().double()
        sm = SetMoments()
        a = torch.rand(B, k, dtype=torch.float64)
        with torch.no_grad():
            m1f, m2f = sm(a, E)
            t0 = time.perf_counter()
            for _ in range(reps):
                sm(a, E)
            t_fast = (time.perf_counter() - t0) / reps
            m1s, m2s = SetMoments.reference_moments(a, E)
            t0 = time.perf_counter()
            for _ in range(reps):
                SetMoments.reference_moments(a, E)
            t_slow = (time.perf_counter() - t0) / reps
        scale = max(float(m2s.abs().max()), 1e-300)
        rows.append(dict(k=int(k), r=int(r),
                         t_okr_s=t_fast, t_ok2_s=t_slow,
                         ratio_slow_over_fast=t_slow / max(t_fast, 1e-12),
                         max_abs_dev_m1=float((m1f - m1s).abs().max()),
                         rel_dev_m2=float((m2f - m2s).abs().max()) / scale))
    ratios = [x["ratio_slow_over_fast"] for x in rows]
    max_rel = max(x["rel_dev_m2"] for x in rows)
    max_abs_m1 = max(x["max_abs_dev_m1"] for x in rows)
    # Slope of measured wall-clock in log k. The pair loop must show a super-linear slope
    # (that IS the O(k^2) claim and it is in the pass criterion); the O(kr) form must show
    # a strictly SMALLER slope. Its own slope is NOT asserted to be ~1 -- see the honest
    # reading below.
    lk = np.log(np.array([x["k"] for x in rows], dtype=float))
    sl_fast = float(np.polyfit(lk, np.log([x["t_okr_s"] for x in rows]), 1)[0])
    sl_slow = float(np.polyfit(lk, np.log([x["t_ok2_s"] for x in rows]), 1)[0])
    # HONEST READING of the slopes: the O(kr) form is dominated by fixed per-call torch
    # overhead at these sizes, so its measured log-log slope is well BELOW 1 and is not
    # evidence of linear scaling -- it is evidence that the k-dependent work is small
    # compared to dispatch. The paper-relevant number is the pair loop's slope (~2) and
    # the growing ratio between the two.
    return dict(per_k=rows, ratio_first=ratios[0], ratio_last=ratios[-1],
                ratio_monotone_increasing=bool(all(np.diff(ratios) > 0)),
                loglog_slope_okr=sl_fast, loglog_slope_ok2=sl_slow,
                max_rel_dev_m2=max_rel, max_abs_dev_m1=max_abs_m1,
                arithmetic_claim="main effects O(k); interactions O(k*r), not O(k^2)",
                slope_ok2_superlinear=bool(sl_slow > 1.5),
                slope_okr_below_ok2=bool(sl_fast < sl_slow),
                passed=bool(max_rel <= 1e-12 and max_abs_m1 <= 1e-12
                            and ratios[-1] > ratios[0]
                            and sl_slow > 1.5 and sl_fast < sl_slow))


def _test_cfm_correctness(seed: int = 0) -> dict:
    """CFM target vs the closed-form conditional velocity, and a loss-reduction check.

    Closed-form system: x1 = A x0 + b0 deterministically, so for the linear interpolant
    the conditional velocity is EXACTLY u(t | x0, x1) = x1 - x0 = (A - I) x0 + b0,
    independent of t. That is the analytic quantity the regression target must equal.
    A separate check confirms the t-dependence is real: the MARGINAL velocity
    E[x1 - x0 | x_t = x] of a Gaussian source under an affine map is available in closed
    form, and the trained field is compared against it.
    """
    rng = np.random.default_rng(seed)
    d, n = 4, 512
    A = np.eye(d) + 0.35 * rng.normal(size=(d, d)) / math.sqrt(d)
    b0 = 0.4 * rng.normal(size=d)
    X0 = rng.normal(size=(n, d))
    X1 = X0 @ A.T + b0
    out: dict = {}
    for sigma in (0.0, 0.05):
        obj = CFMObjective(sigma=sigma, coupling="independent", seed=seed)
        x0 = torch.as_tensor(X0, dtype=torch.float64)
        x1 = torch.as_tensor(X1, dtype=torch.float64)
        g = torch.Generator().manual_seed(seed)
        path = obj.sample_path(x0, x1, generator=g)
        analytic = torch.as_tensor(X0 @ (A - np.eye(d)).T + b0, dtype=torch.float64)
        dev = float((path["target"] - analytic).abs().max())
        # the interpolant itself must reproduce (1-t) x0 + t x1 (+ sigma eps)
        recon = float(((1.0 - path["t"]) * x0 + path["t"] * x1 + path["noise"]
                       - path["x_t"]).abs().max())
        out[f"sigma_{sigma}"] = dict(
            max_abs_target_minus_analytic_conditional_velocity=dev,
            max_abs_interpolant_reconstruction=recon,
            sigma=sigma, path="linear interpolant x_t=(1-t)x0+t x1+sigma*eps",
            conditional_velocity="x1-x0")
    # ---- training reduces the loss, and approaches the analytic MARGINAL field ----
    torch.manual_seed(seed)
    k = 1
    model = ConditionalVelocity(d, k, r=2, code_dim=0, hidden=48)
    obj = CFMObjective(sigma=0.0, coupling="independent", seed=seed)
    pops = [type("P", (), dict(P=(0,), tau=1.0, pre=X0, post=X1))()]
    recs = _pop_arrays(pops, [0], k)
    rngt = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    l0 = _cfm_loss_on(model, recs, obj, rngt, batch=256, reps=8)
    stg = _run_stage(model, recs, obj, opt, 400, 256, rngt)
    l1 = _cfm_loss_on(model, recs, obj, rngt, batch=256, reps=8)
    # Analytic MARGINAL velocity. Under the paired coupling x1 is a deterministic function
    # of x0, so with x_t = M_t x0 + t b0 and M_t = (1-t) I + t A,
    #     E[x1 - x0 | x_t] = (A - I) M_t^{-1} (x_t - t b0) + b0.
    # The trailing + b0 is load-bearing: dropping it (an earlier version of this test did)
    # leaves a term of norm ~0.42 out of ~19 and makes a correctly-fitted model look like
    # it scores rel_l2 0.79 while its CFM loss is 4e-05 -- the two readings are
    # inconsistent, which is how the error was caught. The identity below cross-checks the
    # formula against x1 - x0 directly and is asserted to machine precision.
    objp = CFMObjective(sigma=0.0, coupling="paired", seed=seed)
    modelp = ConditionalVelocity(d, k, r=2, code_dim=0, hidden=64)
    optp = torch.optim.Adam(modelp.parameters(), lr=3e-3)
    stgp = _run_stage(modelp, recs, objp, optp, 1500, 256, np.random.default_rng(seed))
    lp = _cfm_loss_on(modelp, recs, objp, np.random.default_rng(seed), batch=256, reps=6)
    mdlp = modelp.to_inference_dtype(torch.float64)
    cos_rows = []
    with torch.no_grad():
        for tt in (0.1, 0.5, 0.9):
            Mt = (1.0 - tt) * np.eye(d) + tt * A
            Xt = X0 @ Mt.T + tt * b0
            u_true = (Xt - tt * b0) @ np.linalg.inv(Mt).T @ (A - np.eye(d)).T + b0
            formula_dev = float(np.max(np.abs(u_true - (X1 - X0))))
            zz = torch.as_tensor(Xt, dtype=torch.float64)
            mk = torch.ones(n, k, dtype=torch.float64)
            tv = torch.ones(n, 1, dtype=torch.float64)
            t = torch.full((n, 1), tt, dtype=torch.float64)
            up = mdlp(t, zz, tv, mk).numpy()
            cos_rows.append(dict(
                t=tt, closed_form_vs_x1_minus_x0=formula_dev,
                cosine=float((u_true.ravel() @ up.ravel())
                             / max(np.linalg.norm(u_true) * np.linalg.norm(up), 1e-300)),
                rel_l2=float(np.linalg.norm(u_true - up)
                             / max(np.linalg.norm(u_true), 1e-300))))
    # How far each coupling is from the true pairing -- the measurement behind the caveat
    # in CFMObjective's docstring, and the reason this check uses 'paired'.
    pair_frac = {}
    for cp, kw in (("paired", {}), ("ot", {}),
                   ("sinkhorn", dict(sinkhorn_eps=0.05, sinkhorn_iters=200)),
                   ("independent", {})):
        i0, i1 = CFMObjective(coupling=cp, seed=seed, max_coupling_n=n, **kw).couple(X0, X1)
        pair_frac[cp] = float(np.mean(i0 == i1))
    out["training"] = dict(
        loss_before=l0, loss_after=l1, loss_reduction_factor=l0 / max(l1, 1e-300),
        stage=stg, paired_stage=stgp, paired_final_cfm_loss=lp,
        marginal_velocity_vs_closed_form=cos_rows,
        identity_pair_fraction_by_coupling=pair_frac,
        note=("the closed-form marginal E[x1-x0 | x_t] belongs to the PAIRED coupling, so "
              "the model scored against it is trained with coupling='paired'. Minibatch OT "
              "is NOT the paired coupling even here -- see identity_pair_fraction_by_"
              "coupling -- so an OT-trained field is fitting a DIFFERENT marginal and must "
              "not be scored against this one"))
    dev_max = max(out[f"sigma_{s}"]["max_abs_target_minus_analytic_conditional_velocity"]
                  for s in (0.0, 0.05))
    rec_max = max(out[f"sigma_{s}"]["max_abs_interpolant_reconstruction"]
                  for s in (0.0, 0.05))
    fdev = max(r["closed_form_vs_x1_minus_x0"] for r in cos_rows)
    marg_rel = max(r["rel_l2"] for r in cos_rows)
    out["max_abs_closed_form_formula_check"] = fdev
    out["max_rel_l2_vs_closed_form_marginal"] = marg_rel
    out["passed"] = bool(dev_max <= 1e-12 and rec_max <= 1e-12 and l1 < l0
                         and fdev <= 1e-12 and marg_rel <= 0.05)
    out["max_abs_target_dev"] = dev_max
    return out


def _test_coupling_sensitivity(system, pops, train_idx, val_idx, seed: int = 0,
                               steps=(120, 120, 80)) -> dict:
    """Independent vs OT vs entropic-UOT coupling, selected on held-out combinations."""
    d, k = system.d, system.k
    rows = []
    for name, kw in (("independent", dict(coupling="independent")),
                     ("ot", dict(coupling="ot")),
                     ("sinkhorn_uot", dict(coupling="sinkhorn", sinkhorn_eps=0.05,
                                           sinkhorn_iters=25, unbalanced_tau=1.0))):
        torch.manual_seed(seed)
        m = ConditionalVelocity(d, k, r=2, code_dim=0, hidden=48)
        obj = CFMObjective(sigma=0.0, seed=seed, max_coupling_n=128, **kw)
        t0 = time.time()
        rep = train_hierarchical(m, pops, train_idx, objective=obj, val_idx=val_idx,
                                 steps=steps, batch=96, seed=seed)
        Zv = pops[val_idx[0]].pre
        pred = integrate_velocity(m, Zv, pops[val_idx[0]].P, pops[val_idx[0]].tau,
                                  k, n_steps=12)
        ed = _energy_distance(pred, pops[val_idx[0]].post)
        ed_null = _energy_distance(Zv, pops[val_idx[0]].post)
        rows.append(dict(coupling=name, coupling_kw=dict(kw),
                         val_combo_cfm_loss=float(rep["stage3"]["val_combo_cfm_loss"]),
                         val_cfm_loss_all_heldout=float(rep["stage3"]["val_cfm_loss"]),
                         heldout_energy_distance=ed,
                         heldout_energy_distance_null=ed_null,
                         normalised_energy_distance=ed / max(ed_null, 1e-300),
                         wall_clock_s=time.time() - t0))
    best = min(rows, key=lambda r: r["val_combo_cfm_loss"])
    losses = [r["val_combo_cfm_loss"] for r in rows]
    return dict(rows=rows, selected=best["coupling"],
                selection_criterion="held-out COMBINATION CFM loss (order >= 2 only)",
                spread_max_over_min=max(losses) / max(min(losses), 1e-300))


def _test_hierarchical_schedule(seed: int = 0, steps=(300, 300, 200)) -> dict:
    """Full schedule on the controlled benchmark, scored against the KNOWN interaction."""
    from composefm.synthetic import SyntheticSystem     # benchmark, not part of the method
    system = SyntheticSystem(d=6, k=4, r=2, eps=0.30, noise=0.02, seed=0)
    pops = system.sample_populations(n_cells=120, n_replicates=1, taus=(1.0,), n_triples=1,
                                     seed=11)
    train_idx, val_idx = SyntheticSystem.loco_split(pops, held_out=[(0, 1)])
    d, k = system.d, system.k
    torch.manual_seed(seed)
    model = ConditionalVelocity(d, k, r=2, code_dim=0, hidden=48)
    obj = CFMObjective(sigma=0.0, coupling="ot", seed=seed, max_coupling_n=128)
    Zev = system.control(128, seed=777)
    rep = train_hierarchical(model, pops, train_idx, objective=obj, val_idx=val_idx,
                             steps=steps, batch=96, lr=3e-3, anchor_weight=1.0,
                             int_penalty=0.0, singleton_oversample=3.0, seed=seed,
                             eval_states=Zev)

    # ---- (a) stage 1 fits singletons: flow error vs truth on singleton conditions ----
    def _flow_err(m, conds):
        errs = []
        for P in conds:
            Z0 = system.control(96, seed=4242)
            truth = system.flow(P, Z0, 1.0)
            pred = integrate_velocity(m, Z0, P, 1.0, k, n_steps=12)
            scale = float(np.mean(np.linalg.norm(truth - Z0, axis=1))) + 1e-300
            errs.append(float(np.mean(np.linalg.norm(pred - truth, axis=1))) / scale)
        return float(np.mean(errs))

    singles = [(p,) for p in range(k)]
    out: dict = dict(stages={s: rep[s] for s in ("stage1", "stage2", "stage3")})
    out["singleton_flow_err_final"] = _flow_err(model, singles)
    out["heldout_combination_flow_err_final"] = _flow_err(model, [(0, 1)])
    out["val_combo_cfm_loss_by_stage"] = {s: float(rep[s]["val_combo_cfm_loss"])
                                          for s in ("stage1", "stage2", "stage3")}
    out["val_cfm_loss_all_heldout_by_stage"] = {s: float(rep[s]["val_cfm_loss"])
                                                for s in ("stage1", "stage2", "stage3")}
    out["singleton_drift_stage2"] = float(rep["stage2"]["singleton_velocity_drift_rel"])
    out["singleton_drift_stage3"] = float(rep["stage3"]["singleton_velocity_drift_rel"])
    # Stage 2's job is to fit combinations that stage 1 structurally cannot (stage 1 has
    # no interaction branch at all). Two DIFFERENT questions, reported separately and
    # never conflated:
    #   in-distribution : does it improve the TRAINING combinations it is fitted on?
    #   LOCO            : does it improve a combination held out entirely?
    out["stage2_improves_train_combos"] = bool(
        rep["stage2"]["train_combo_cfm_loss"] < rep["stage1"]["train_combo_cfm_loss"])
    out["stage2_improves_heldout_combos"] = bool(
        rep["stage2"]["val_combo_cfm_loss"] < rep["stage1"]["val_combo_cfm_loss"])
    out["train_combo_cfm_loss_by_stage"] = {st: float(rep[st]["train_combo_cfm_loss"])
                                            for st in ("stage1", "stage2", "stage3")}

    # ---- (c) recovery of the TRUE interaction field --------------------------------
    # The model's interaction branch is a velocity of the CFM interpolant path, whereas
    # the benchmark's truth is the GENERATOR of an ODE flow. These differ at O(||d||^2)
    # (project fact P1), so a scale mismatch is expected and `optimal_scale` is reported;
    # cosine is the scale-free number to read.
    mdl = model.to_inference_dtype(torch.float64)
    t_grid = (0.125, 0.375, 0.625, 0.875)

    def pred_interaction(P, Z, tau):
        Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
        n = Z.shape[0]
        mask = np.zeros(k)
        for p in P:
            mask[int(p)] = 1.0
        mk = torch.as_tensor(np.tile(mask, (n, 1)), dtype=torch.float64)
        tv = torch.full((n, 1), float(tau), dtype=torch.float64)
        z = torch.as_tensor(Z, dtype=torch.float64)
        acc = np.zeros_like(Z)
        with torch.no_grad():
            for tt in t_grid:
                t = torch.full((n, 1), tt, dtype=torch.float64)
                acc += mdl.interaction_only(t, z, tv, mk).numpy()
        return acc / len(t_grid)

    # ---- ORACLE CEILING and the KINEMATIC CONFOUND, both measured -----------------
    # Two numbers are needed before the recovery score can be read at all.
    #
    # (1) CEILING. The benchmark's truth is the GENERATOR of an ODE flow; the model's
    #     interaction branch is a velocity along the straight CFM interpolant. These are
    #     different objects, so even an ORACLE that reads the interaction off as
    #     (composed displacement - additive-flow displacement) does not score 1.0. It is
    #     measured here and is the ceiling any CFM-based readout can reach.
    #
    # (2) CONFOUND. A singleton-anchored main branch predicts, at best, the SUM of
    #     singleton displacements. The additive-flow displacement differs from that sum by
    #     the kinematic displacement-composition defect (project fact P1, O(||d||^2)).
    #     Whatever the main branch cannot express, the m2-gated branch absorbs. So the
    #     defect's size RELATIVE to the true interaction, and its ANGLE to it, bound how
    #     well the interaction branch can possibly align with the truth.
    from composefm.synthetic import rk4 as _rk4
    pair_conds = [c for c in system.conditions() if len(c) == 2]

    def _oracle_interaction(P, Z, tau):
        Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
        full = system.flow(P, Z, tau, n_steps=60)
        add = _rk4(lambda Q: system.additive_field(P, Q, tau), Z, T=1.0, n_steps=60)
        return full - add

    out["oracle_ceiling_cfm_readout"] = system.score_interaction_recovery(
        _oracle_interaction, Zev, conditions=[c for c in system.conditions()
                                              if len(c) >= 2], tau=1.0)
    kin = []
    for P in pair_conds:
        add_disp = _rk4(lambda Q: system.additive_field(P, Q, 1.0), Zev, T=1.0,
                        n_steps=60) - Zev
        ssum = sum(system.flow((p,), Zev, 1.0, n_steps=60) - Zev for p in P)
        defect = add_disp - ssum
        Ig = system.interaction_field(P, Zev, 1.0)
        nd, ni = np.linalg.norm(defect), np.linalg.norm(Ig)
        kin.append(dict(condition="+".join(f"X{p}" for p in P),
                        defect_norm=float(nd), true_interaction_norm=float(ni),
                        defect_over_interaction=float(nd / max(ni, 1e-300)),
                        cosine_defect_vs_true_interaction=float(
                            defect.ravel() @ Ig.ravel() / max(nd * ni, 1e-300))))
    out["kinematic_composition_confound"] = dict(
        per_pair=kin,
        ratio_min=float(min(x["defect_over_interaction"] for x in kin)),
        ratio_max=float(max(x["defect_over_interaction"] for x in kin)),
        abs_cosine_max=float(max(abs(x["cosine_defect_vs_true_interaction"])
                                 for x in kin)),
        reading=("the displacement-composition defect is comparable to or larger than the "
                 "true interaction and is nearly ORTHOGONAL to it, so an m2-gated branch "
                 "sitting on top of a singleton-anchored main branch must absorb both; "
                 "the recovered interaction is therefore a SUM of the true interaction and "
                 "a kinematic term, which caps its cosine well below the oracle ceiling"))

    conds_all = [c for c in system.conditions() if len(c) >= 2]
    conds_train = [c for c in conds_all if tuple(sorted(c)) != (0, 1)]
    out["interaction_recovery_all_combos"] = system.score_interaction_recovery(
        pred_interaction, Zev, conditions=conds_all, tau=1.0)
    out["interaction_recovery_train_combos"] = system.score_interaction_recovery(
        pred_interaction, Zev, conditions=conds_train, tau=1.0)
    out["interaction_recovery_heldout_pair"] = system.score_interaction_recovery(
        pred_interaction, Zev, conditions=[(0, 1)], tau=1.0)
    # ---- WHAT EACH BRANCH ACTUALLY FITTED ----------------------------------------
    def _branch_avg(fn, P, Z, tau):
        Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
        n = Z.shape[0]
        mask = np.zeros(k)
        for p in P:
            mask[int(p)] = 1.0
        acc = np.zeros_like(Z)
        with torch.no_grad():
            for tt in t_grid:
                acc += fn(torch.full((n, 1), tt, dtype=torch.float64),
                          torch.as_tensor(Z, dtype=torch.float64),
                          torch.full((n, 1), float(tau), dtype=torch.float64),
                          torch.as_tensor(np.tile(mask, (n, 1)),
                                          dtype=torch.float64)).numpy()
        return acc / len(t_grid)

    def _cos(a, b):
        return float(a.ravel() @ b.ravel()
                     / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-300))

    battr = []
    for P in pair_conds:
        mb = _branch_avg(mdl.main_only, P, Zev, 1.0)
        ib = _branch_avg(mdl.interaction_only, P, Zev, 1.0)
        add_disp = _rk4(lambda Q: system.additive_field(P, Q, 1.0), Zev, T=1.0,
                        n_steps=60) - Zev
        ssum = sum(system.flow((p,), Zev, 1.0, n_steps=60) - Zev for p in P)
        resid = (system.flow(P, Zev, 1.0, n_steps=60) - Zev) - ssum
        Ig = system.interaction_field(P, Zev, 1.0)
        battr.append(dict(
            condition="+".join(f"X{p}" for p in P),
            held_out=bool(tuple(sorted(P)) == (0, 1)),
            cos_main_vs_additive_flow_displacement=_cos(mb, add_disp),
            main_over_additive_norm=float(np.linalg.norm(mb)
                                          / max(np.linalg.norm(add_disp), 1e-300)),
            cos_main_vs_sum_singleton_main=_cos(mb, sum(
                _branch_avg(mdl.main_only, (p,), Zev, 1.0) for p in P)),
            cos_interaction_vs_true_interaction=_cos(ib, Ig),
            cos_interaction_vs_full_minus_singletons=_cos(ib, resid),
            interaction_over_residual_norm=float(np.linalg.norm(ib)
                                                 / max(np.linalg.norm(resid), 1e-300))))
    out["branch_attribution"] = dict(per_pair=battr, t_grid=list(t_grid))

    out["interaction_recovery_singletons_must_be_zero"] = float(np.max(
        [np.max(np.abs(pred_interaction(P, Zev, 1.0))) for P in singles]))
    out["skipped_nonfinite_total"] = int(rep["skipped_nonfinite_total"])
    out["wall_clock_s"] = float(rep["wall_clock_s"])
    out["system"] = dict(d=d, k=k, r_true=system.r, eps=system.eps,
                         n_populations=len(pops), n_train=len(train_idx),
                         n_val=len(val_idx),
                         heldout=sorted({pops[i].label for i in val_idx}))
    out["coupling_sensitivity"] = _test_coupling_sensitivity(
        system, pops, train_idx, val_idx, seed=seed)
    return out


def main() -> dict:
    t_all = time.time()
    report: dict = dict(module="ihcfm_hierarchy", torch=torch.__version__)
    report["permutation_invariance"] = _test_permutation_invariance()
    report["singleton_vanishing"] = _test_singleton_vanishing()
    report["zero_dose_identity"] = _test_zero_dose_identity()
    report["moment_cost"] = _test_moment_cost()
    report["cfm_correctness"] = _test_cfm_correctness()
    report["hierarchical_schedule"] = _test_hierarchical_schedule()
    report["signatures"] = [
        "InterventionEmbed(k, r=4, init_scale=1.0, normalise=False, seed=None)",
        "SetMoments()  # forward(a, E) -> (m1, m2); .reference_moments(a, E) oracle",
        ("OneFormHierarchy(d_ambient, r=4, code_dim=0, hidden=64, depth=2, "
         "interaction_enabled=True, interaction_init_scale=0.1, k=None, embed=None, "
         "embed_seed=0)"),
        "PopulationEncoder(d, code_dim=8, hidden=64, use_second_moment=True, depth=2)",
        ("CFMObjective(sigma=0.0, coupling='independent', sinkhorn_eps=0.05, "
         "sinkhorn_iters=200, unbalanced_tau=None, seed=0, max_coupling_n=512)"),
        ("ConditionalVelocity(d, k, r=4, code_dim=0, hidden=64, depth=2, dose=None, "
         "one_form=None, pullback=None, embed_seed=0, interaction_enabled=True, "
         "decoder=None)"),
        ("train_hierarchical(model, populations, train_idx, objective=None, val_idx=None, "
         "steps=(300,300,200), batch=128, lr=3e-3, lr_stage3=1e-3, anchor_weight=1.0, "
         "int_penalty=0.0, singleton_oversample=3.0, seed=0, eval_states=None, "
         "eval_tau=1.0, verbose=False)"),
    ]
    checks = {
        "permutation_invariance": report["permutation_invariance"]["passed"],
        "singleton_vanishing_bit_exact": report["singleton_vanishing"]["passed"],
        "zero_dose_identity": report["zero_dose_identity"]["passed"],
        "moment_cost_okr": report["moment_cost"]["passed"],
        "cfm_target_matches_analytic": report["cfm_correctness"]["passed"],
        "stage2_improves_train_combos":
            report["hierarchical_schedule"]["stage2_improves_train_combos"],
        "singleton_anchoring_holds":
            report["hierarchical_schedule"]["singleton_drift_stage2"] < 0.5,
        "interaction_pred_zero_on_singletons": (
            report["hierarchical_schedule"]["interaction_recovery_singletons_must_be_zero"]
            == 0.0),
    }
    report["checks"] = checks
    _hs = report["hierarchical_schedule"]
    report["honest_negatives"] = []
    if not _hs["stage2_improves_heldout_combos"]:
        report["honest_negatives"].append(dict(
            claim_NOT_supported="the interaction branch generalises to an unseen combination",
            measured=dict(
                heldout_combo_cfm_loss_by_stage=_hs["val_combo_cfm_loss_by_stage"],
                train_combo_cfm_loss_by_stage=_hs["train_combo_cfm_loss_by_stage"],
                heldout_pair_recovery=_hs["interaction_recovery_heldout_pair"],
                heldout_pair_main_branch=[b for b in _hs["branch_attribution"]["per_pair"]
                                          if b["held_out"]]),
            diagnosed_cause=(
                "stage 2 does improve the TRAINING combinations it is fitted on, so the "
                "branch has capacity and the schedule optimises. It does NOT improve the "
                "leave-one-combination-out pair. Two measured causes, neither of which is "
                "fixed by simplifying the model: (a) on the held-out pair the MAIN branch "
                "itself is wrong -- its cosine against the additive-flow displacement is "
                "NEGATIVE and its norm is ~0.12 of the additive scale -- so the failure "
                "starts in the main effects, before the interaction branch is reached; "
                "(b) the kinematic displacement-composition defect is 0.09-1.98x the true "
                "interaction and near-orthogonal to it, so the m2-gated branch necessarily "
                "absorbs a large non-interaction term (interaction/residual norm ratio "
                "~3.8 on the held-out pair)."),
            implication_for_the_method=(
                "the hierarchy's STRUCTURAL properties (permutation invariance, bit-exact "
                "singleton and zero-exposure vanishing, zero-dose identity, O(kr) cost) "
                "are verified and independent of this. What is NOT supported is a claim "
                "that a straight-line observed-space interpolant separates the generator "
                "interaction from the kinematic composition term. That separation is what "
                "the geometry track's field composition exists to provide (project facts "
                "P1/E), so the negative is evidence about the INTERPOLANT, and it should be "
                "re-measured once the geometry pullback replaces identity_pullback.")))
    if _hs["interaction_recovery_all_combos"]["cosine"] < 0.7 * _hs[
            "oracle_ceiling_cfm_readout"]["cosine"]:
        report["honest_negatives"].append(dict(
            claim_NOT_supported="the recovered interaction field matches the ground truth",
            measured=dict(
                model_cosine=_hs["interaction_recovery_all_combos"]["cosine"],
                model_rel_l2=_hs["interaction_recovery_all_combos"]["rel_l2"],
                oracle_ceiling_cosine=_hs["oracle_ceiling_cfm_readout"]["cosine"],
                oracle_ceiling_rel_l2=_hs["oracle_ceiling_cfm_readout"]["rel_l2"]),
            diagnosed_cause=("the oracle ceiling for ANY CFM-interpolant readout of this "
                             "benchmark is cosine 0.939 / rel_l2 0.407, not 1.0, because "
                             "the truth is an ODE generator and the readout is an "
                             "interpolant velocity. The model sits well below even that "
                             "ceiling, and the kinematic confound above is the measured "
                             "reason. Reporting rel_l2 > 1 as 'interaction not recovered' "
                             "is correct; attributing it to the hierarchy's parameterisation "
                             "would not be, since the ablation with the interaction head "
                             "blind to m1 moves the cosine only 0.449 -> 0.524."),
            implication_for_the_method=(
                "recovery of the TRUE interaction field is the point of the component, so "
                "this is a failure of the current end-to-end configuration and must be "
                "reported as such. It is NOT authorisation to simplify the hierarchy: the "
                "diagnosed cause is upstream of it.")))
    report["wall_clock_s"] = time.time() - t_all
    out = pathlib.Path(__file__).resolve().parents[2] / "results" / "ihcfm_hierarchy_selftest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out, "w"), indent=2, default=float)
    hs = report["hierarchical_schedule"]
    print(f"perm inv: m1 {report['permutation_invariance']['max_abs_dev_m1']:.2e} "
          f"m2 {report['permutation_invariance']['max_abs_dev_m2']:.2e} "
          f"b {report['permutation_invariance']['max_abs_dev_b']:.2e}")
    print(f"singleton |I| {report['singleton_vanishing']['max_abs_interaction_on_singletons']!r} "
          f"bit-exact singletons {report['singleton_vanishing']['bit_exact_singleton_cases']}"
          f" zero-exposure {report['singleton_vanishing']['bit_exact_zero_exposure_cases']}")
    print(f"tau=0: |b| {report['zero_dose_identity']['max_abs_b_at_tau0']!r} "
          f"identity defect {report['zero_dose_identity']['max_abs_flow_identity_defect']!r}")
    print(f"moments: ratio k=2 {report['moment_cost']['ratio_first']:.2f} -> "
          f"k=32 {report['moment_cost']['ratio_last']:.2f}, "
          f"slopes okr {report['moment_cost']['loglog_slope_okr']:.2f} / "
          f"ok2 {report['moment_cost']['loglog_slope_ok2']:.2f}, "
          f"agree {report['moment_cost']['max_rel_dev_m2']:.2e}")
    print(f"CFM target dev {report['cfm_correctness']['max_abs_target_dev']:.2e}, "
          f"loss {report['cfm_correctness']['training']['loss_before']:.4f} -> "
          f"{report['cfm_correctness']['training']['loss_after']:.4f}")
    print(f"schedule: held-out combo loss {hs['val_combo_cfm_loss_by_stage']}, "
          f"singleton drift s2 {hs['singleton_drift_stage2']:.4f}")
    print(f"interaction recovery (all combos): "
          f"cos {hs['interaction_recovery_all_combos']['cosine']:.4f} "
          f"rel_l2 {hs['interaction_recovery_all_combos']['rel_l2']:.4f}")
    print(f"oracle ceiling for a CFM readout: "
          f"cos {hs['oracle_ceiling_cfm_readout']['cosine']:.4f} "
          f"rel_l2 {hs['oracle_ceiling_cfm_readout']['rel_l2']:.4f}")
    print(f"kinematic confound: defect/interaction "
          f"{hs['kinematic_composition_confound']['ratio_min']:.3f}-"
          f"{hs['kinematic_composition_confound']['ratio_max']:.3f}, "
          f"|cos| <= {hs['kinematic_composition_confound']['abs_cosine_max']:.3f}")
    print(f"stage2: train combos improve {hs['stage2_improves_train_combos']}, "
          f"held-out combos improve {hs['stage2_improves_heldout_combos']} "
          f"(HONEST NEGATIVE if False)")
    print(f"coupling selected: {hs['coupling_sensitivity']['selected']} "
          f"(spread {hs['coupling_sensitivity']['spread_max_over_min']:.2f}x)")
    print(f"checks {checks}")
    print(f"wrote {out} in {report['wall_clock_s']:.1f}s")
    return report


if __name__ == "__main__":
    main()

