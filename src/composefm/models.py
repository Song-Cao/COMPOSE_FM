"""COMPOSE-FM and its baselines.

The flow-matching innovation, stated precisely
----------------------------------------------
Standard conditional flow matching learns a *time-dependent* velocity v(z, t, cond) whose
time argument is an artifact of the interpolation path. Here we learn an **autonomous
generator** X(z; e_p) per intervention and define the perturbed population as the
time-`tau` flow of that generator, so:

  (i)  exposure (dose / knockdown efficiency / executed quantity) *is* integration time,
       and the semigroup Psi^{tau+tau'} = Psi^{tau'} o Psi^{tau} holds by construction;
  (ii) several interventions compose by combining *generators*, not displacements.

(ii) is the load-bearing point. Adding latent displacements -- what CPA/GEARS-class models
do -- is not a covariant operation: it gives different predictions for the same combination
under two equally valid latent charts. Adding *vector fields* is covariant, and so is
integrating the combined field.

Status of that claim (do not overstate it): measured in experiments/gate_a_chart.py on the
toy ground truth. The mechanism reproduces robustly -- displacement-addition chart mismatch
8.8e-3 to 3.4e-2 depending on re-chart strength, versus generator-composition mismatch
3.4e-13 to 2.1e-12, a ratio of ~1.6e10 to 2.6e10. But the gate as written has NOT returned
PASS: under v1 criteria the weakest re-chart missed an absolute threshold, and under v2
criteria the displacement mismatch sits at or below the between-seed prediction noise floor
(~1.2e-2 to 1.3e-2). Read gate_a.json for the current verdict rather than assuming this
docstring. The qualitative asymmetry is not in doubt; its practical materiality is, and that
is the open question the gate is meant to settle.

COMPOSE-FM's composition operator:

    X_P(z) = gamma(z, P) * sum_{p in P} X_p(z)  +  (1/2) sum_{p<q} beta_pq(z) S_pq(z)

  * gamma  -- a contraction gate in (0,1], because real composition is sub-additive;
  * S_pq   -- symmetric coupling, optionally connection-corrected (see couple.py);
  * beta   -- sparse interaction gate.

Trained by OT-coupled flow matching: minibatch optimal transport pairs control cells to
perturbed cells, and the loss asks the time-tau flow of X_P to carry z0 to its OT partner.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------- OT coupling
def ot_pairs(Z0: np.ndarray, Z1: np.ndarray, eps: float = 0.05, iters: int = 200):
    """Entropic-OT barycentric partner for each row of Z0 (Sinkhorn, numpy)."""
    C = ((Z0[:, None, :] - Z1[None, :, :]) ** 2).sum(-1)
    C = C / max(C.mean(), 1e-12)
    K = np.exp(-C / eps)
    n, m = K.shape
    u = np.ones(n) / n
    v = np.ones(m) / m
    a = np.ones(n) / n
    b = np.ones(m) / m
    for _ in range(iters):
        u = a / np.maximum(K @ v, 1e-300)
        v = b / np.maximum(K.T @ u, 1e-300)
    P = u[:, None] * K * v[None, :]
    P = P / np.maximum(P.sum(1, keepdims=True), 1e-300)
    return P @ Z1


def mlp(din, dout, hidden=128, depth=3, act=nn.SiLU, final=None):
    layers, d = [], din
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), act()]
        d = hidden
    layers += [nn.Linear(d, dout)]
    if final is not None:
        layers += [final()]
    return nn.Sequential(*layers)


# ---------------------------------------------------------------- generators
class Generators(nn.Module):
    """Autonomous per-intervention fields X_p(z) = net(z, e_p)."""

    def __init__(self, d, n_pert, emb=16, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(n_pert, emb)
        self.net = mlp(d + emb, d, hidden=hidden)
        self.d = d

    def field_single(self, Z, p_idx):
        e = self.emb(p_idx).expand(Z.shape[0], -1) if p_idx.dim() == 1 and p_idx.numel() == 1 \
            else self.emb(p_idx)
        return self.net(torch.cat([Z, e], -1))

    def forward(self, Z, p_idx):
        return self.field_single(Z, p_idx)


class ContractionGate(nn.Module):
    """gamma(z, P) in (0, 1]: predicted from state and set-summary features.

    Features are permutation-invariant in P (sum/max pooled), and include the norm of the
    additive drive and the mean pairwise alignment -- the two quantities that predicted the
    real saturation scalar on Norman (Spearman +0.388 and +0.335).
    """

    def __init__(self, d, emb=16, hidden=96):
        super().__init__()
        self.net = mlp(d + emb + 3, 1, hidden=hidden, depth=2)

    def forward(self, Z, e_sum, add_norm, align, k):
        f = torch.cat([Z, e_sum,
                       add_norm.unsqueeze(-1),
                       align.unsqueeze(-1),
                       torch.full_like(add_norm.unsqueeze(-1), float(k))], -1)
        return torch.sigmoid(self.net(f))          # (N,1) in (0,1)


class InteractionGate(nn.Module):
    def __init__(self, d, emb=16, hidden=64):
        super().__init__()
        self.net = mlp(d + 2 * emb, 1, hidden=hidden, depth=2)

    def forward(self, Z, e_p, e_q):
        return torch.tanh(self.net(torch.cat([Z, e_p, e_q], -1)))


class ComposeFM(nn.Module):
    """The full model. `mode` selects the composition operator for ablations."""

    def __init__(self, d, n_pert, emb=16, hidden=128, mode="full", use_coupling=True):
        super().__init__()
        self.gen = Generators(d, n_pert, emb=emb, hidden=hidden)
        self.gate = ContractionGate(d, emb=emb)
        self.inter = InteractionGate(d, emb=emb)
        self.mode = mode
        self.use_coupling = use_coupling
        self.d = d
        self.register_buffer("const_gate", torch.tensor(1.0))

    # -- composed field ----------------------------------------------------
    def field(self, Z, P):
        """P: list of int perturbation indices. Returns X_P(Z)."""
        dev = Z.device
        idx = [torch.tensor([p], device=dev) for p in P]
        Xs = [self.gen.field_single(Z, i) for i in idx]
        add = torch.stack(Xs, 0).sum(0)
        if self.mode == "additive":
            return add
        if self.mode == "const":
            return self.const_gate * add

        es = torch.stack([self.gen.emb(i).expand(Z.shape[0], -1) for i in idx], 0)
        e_sum = es.sum(0)
        add_norm = add.norm(dim=-1)
        if len(P) > 1:
            n = 0.0
            cnt = 0
            for a in range(len(P)):
                for b in range(a + 1, len(P)):
                    ca = torch.sum(Xs[a] * Xs[b], -1) / (
                        Xs[a].norm(dim=-1) * Xs[b].norm(dim=-1) + 1e-8)
                    n = n + ca
                    cnt += 1
            align = n / max(cnt, 1)
        else:
            align = torch.zeros_like(add_norm)

        g = self.gate(Z, e_sum, add_norm, align, len(P))
        out = g * add

        if self.use_coupling and len(P) > 1 and self.mode == "full":
            S = torch.zeros_like(add)
            for a in range(len(P)):
                for b in range(a + 1, len(P)):
                    Za = Z.detach().requires_grad_(True)
                    # symmetric coupling  DX_b X_a + DX_a X_b  via JVP
                    def fa(zz): return self.gen.field_single(zz, idx[a])
                    def fb(zz): return self.gen.field_single(zz, idx[b])
                    _, jvp_ba = torch.func.jvp(fb, (Z,), (Xs[a],))
                    _, jvp_ab = torch.func.jvp(fa, (Z,), (Xs[b],))
                    bcoef = self.inter(Z, self.gen.emb(idx[a]).expand(Z.shape[0], -1),
                                       self.gen.emb(idx[b]).expand(Z.shape[0], -1))
                    S = S + bcoef * (jvp_ba + jvp_ab)
            out = out + 0.5 * S
        return out

    # -- flow --------------------------------------------------------------
    def flow(self, Z, P, tau=1.0, n_steps=8):
        h = tau / n_steps
        for _ in range(n_steps):
            k1 = self.field(Z, P)
            k2 = self.field(Z + h * k1 / 2, P)
            k3 = self.field(Z + h * k2 / 2, P)
            k4 = self.field(Z + h * k3, P)
            Z = Z + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return Z


class DisplacementCPA(nn.Module):
    """Baseline: CPA/GEARS-style additive latent DISPLACEMENT (not a field).

    This is the class of model the covariance argument indicts: it predicts a one-shot
    displacement per perturbation and adds them. No integration, no autonomous generator,
    so no semigroup and no chart-equivariance.
    """

    def __init__(self, d, n_pert, emb=16, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(n_pert, emb)
        self.net = mlp(d + emb, d, hidden=hidden)

    def disp(self, Z, p):
        e = self.emb(torch.tensor([p], device=Z.device)).expand(Z.shape[0], -1)
        return self.net(torch.cat([Z, e], -1))

    def predict(self, Z, P, tau=1.0):
        return Z + tau * sum(self.disp(Z, p) for p in P)


# ---------------------------------------------------------------- losses
def energy_distance(X, Y):
    """Unbiased-ish energy distance between two point clouds (torch)."""
    def pdist(A, B):
        return torch.cdist(A, B, p=2)
    n, m = len(X), len(Y)
    xy = pdist(X, Y).mean()
    xx = pdist(X, X).sum() / max(n * (n - 1), 1)
    yy = pdist(Y, Y).sum() / max(m * (m - 1), 1)
    return 2 * xy - xx - yy


def energy_distance_np(X, Y):
    X = np.asarray(X); Y = np.asarray(Y)
    def pd(A, B):
        return np.sqrt(np.maximum(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1), 0))
    n, m = len(X), len(Y)
    return float(2 * pd(X, Y).mean()
                 - pd(X, X).sum() / max(n * (n - 1), 1)
                 - pd(Y, Y).sum() / max(m * (m - 1), 1))
