"""GATE A2 -- does the covariance defect MATTER, or is it below the noise floor?

Gate A established the qualitative asymmetry (displacement-addition is not covariant,
generator-composition is, ratio ~1e10) but the ABSOLUTE displacement error (0.9-3.4%) sat
at or below the between-seed prediction noise floor (~1.2-1.3%). A defect smaller than a
model's own sampling noise is not a defect a practitioner should care about.

So the question is not "is it nonzero" but "what makes it large". Three axes, each with a
mechanistic reason to expect growth:

  (1) NUMBER of composed interventions k. Displacement addition commits a first-order error
      per pair; the number of pairs grows as k(k-1)/2, so the error should grow
      super-linearly in k. Combinatorial perturbation screens go to k=2..5.
  (2) EXPOSURE tau. Displacement addition is a one-shot linearisation; its error is O(tau^2)
      while the covariant flow is exact at all tau. Higher dose => bigger error.
  (3) CHART ANISOTROPY. A generic random diffeomorphism is nearly conformal on average. Real
      chart changes (PCA basis vs VAE latent) are strongly anisotropic and curvature-carrying.

Pass condition (registered before running):
  the displacement mismatch exceeds the noise floor by >= 3x in a regime that is
  EMPIRICALLY REACHABLE (k <= 5, tau <= 3, anisotropy within the range spanned by real
  encoder pairs), while generator composition stays < 1e-8.

If it passes, the covariance claim has practical teeth and names the regime where it bites.
If it fails on every axis, the claim is a conceptual nicety and the architecture must earn
its place on the composition law alone -- which would be reported as such.
"""
import json, sys, pathlib, itertools
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from composefm.toy import ToySystem

OUT = pathlib.Path(__file__).resolve().parents[1] / "results"


def anisotropic_chart(d, strength=0.35, aniso=1.0, seed=7):
    """phi(z) = D (z + c A sin(z)) with D a diagonal anisotropy of condition `aniso`.

    aniso=1 is the near-conformal generic case; larger values stretch axes unequally, as a
    change between two real encoders does.
    """
    g = np.random.default_rng(seed)
    A = g.normal(size=(d, d)) * 0.6
    scales = np.exp(np.linspace(-np.log(aniso) / 2, np.log(aniso) / 2, d))
    D = np.diag(scales)

    def phi(Z):
        Z = np.atleast_2d(Z)
        return (Z + strength * (np.sin(Z) @ A.T)) @ D.T

    def dphi(Z):
        Z = np.atleast_2d(Z)
        return D[None] @ (np.eye(d)[None] + strength * (A[None] * np.cos(Z)[:, None, :]))

    def phi_inv(W, iters=80):
        Z = np.atleast_2d(W).astype(float).copy()
        for _ in range(iters):
            r = phi(Z) - W
            Z = Z - np.linalg.solve(dphi(Z), r[:, :, None])[:, :, 0]
        return Z
    return phi, dphi, phi_inv


def rk4(f, Z0, T, n=80):
    h = T / n
    Z = np.array(Z0, dtype=float)
    for _ in range(n):
        k1 = f(Z); k2 = f(Z + h * k1 / 2)
        k3 = f(Z + h * k2 / 2); k4 = f(Z + h * k3)
        Z = Z + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return Z


def one_setting(d, k, tau, aniso, strength, seed, n_states=300):
    sysm = ToySystem(d=d, n_pert=8, seed=seed)
    phi, dphi, phi_inv = anisotropic_chart(d, strength=strength, aniso=aniso, seed=100 + seed)
    Z = sysm.control(n_states, seed=10 + seed)
    P = list(range(k))
    W = phi(Z)

    # rule 1: add displacements in chart 1 vs chart 2
    disps_z = [sysm.flow([p], Z, tau) - Z for p in P]
    pred_z = Z + sum(disps_z)
    lhs = phi(pred_z)
    disps_w = [phi(Z + dz) - W for dz in disps_z]
    rhs = W + sum(disps_w)
    scale = np.linalg.norm(lhs - W, axis=1).mean() + 1e-12
    err_disp = np.linalg.norm(lhs - rhs, axis=1).mean() / scale

    # rule 2: compose generators, integrate
    def Xz(Zq):
        return sum(sysm.field([p], Zq) for p in P)

    def Xw(Wq):
        Zq = phi_inv(Wq)
        return np.einsum('nij,nj->ni', dphi(Zq), Xz(Zq))
    fz = rk4(Xz, Z, tau); fw = rk4(Xw, W, tau)
    scale2 = np.linalg.norm(fw - W, axis=1).mean() + 1e-12
    err_gen = np.linalg.norm(phi(fz) - fw, axis=1).mean() / scale2

    # noise floor: resample the control population, same pipeline
    Z2 = sysm.control(n_states, seed=500 + seed)
    d2 = [sysm.flow([p], Z2, tau) - Z2 for p in P]
    pred2 = Z2 + sum(d2)
    floor = abs(np.linalg.norm(pred2 - Z2, axis=1).mean()
                - np.linalg.norm(pred_z - Z, axis=1).mean()) / scale
    return err_disp, err_gen, floor


if __name__ == "__main__":
    seeds = (0, 1, 2, 3)
    rows = []
    # axis 1: k, at modest tau and generic chart
    for k in (2, 3, 4, 5):
        v = [one_setting(6, k, 1.0, 1.0, 0.35, s) for s in seeds]
        rows.append(dict(axis="k", k=k, tau=1.0, aniso=1.0,
                         disp=float(np.mean([x[0] for x in v])),
                         gen=float(np.mean([x[1] for x in v])),
                         floor=float(np.mean([x[2] for x in v]))))
    # axis 2: tau, at k=2
    for tau in (0.5, 1.0, 2.0, 3.0):
        v = [one_setting(6, 2, tau, 1.0, 0.35, s) for s in seeds]
        rows.append(dict(axis="tau", k=2, tau=tau, aniso=1.0,
                         disp=float(np.mean([x[0] for x in v])),
                         gen=float(np.mean([x[1] for x in v])),
                         floor=float(np.mean([x[2] for x in v]))))
    # axis 3: anisotropy, at k=2 tau=1
    for a in (1.0, 3.0, 10.0, 30.0):
        v = [one_setting(6, 2, 1.0, a, 0.35, s) for s in seeds]
        rows.append(dict(axis="aniso", k=2, tau=1.0, aniso=a,
                         disp=float(np.mean([x[0] for x in v])),
                         gen=float(np.mean([x[1] for x in v])),
                         floor=float(np.mean([x[2] for x in v]))))
    # combined reachable regime: k=4, tau=2, aniso=10
    v = [one_setting(6, 4, 2.0, 10.0, 0.35, s) for s in seeds]
    rows.append(dict(axis="combined", k=4, tau=2.0, aniso=10.0,
                     disp=float(np.mean([x[0] for x in v])),
                     gen=float(np.mean([x[1] for x in v])),
                     floor=float(np.mean([x[2] for x in v]))))

    for r in rows:
        ratio = r["disp"] / max(r["floor"], 1e-12)
        print(f"{r['axis']:9s} k={r['k']} tau={r['tau']:.1f} aniso={r['aniso']:5.1f} | "
              f"disp {r['disp']:.3e} | gen {r['gen']:.2e} | floor {r['floor']:.3e} | "
              f"disp/floor {ratio:6.2f}")

    best = max(rows, key=lambda r: r["disp"] / max(r["floor"], 1e-12))
    ratio = best["disp"] / max(best["floor"], 1e-12)
    reachable = best["k"] <= 5 and best["tau"] <= 3.0 and best["aniso"] <= 30.0
    gen_ok = all(r["gen"] < 1e-8 for r in rows)
    verdict = bool(ratio >= 3.0 and reachable and gen_ok)
    print(f"\nbest regime: k={best['k']} tau={best['tau']} aniso={best['aniso']} "
          f"-> disp/floor {ratio:.2f} | generator covariant everywhere: {gen_ok}")
    print("GATE A2:", "PASS" if verdict else "FAIL")
    OUT.mkdir(exist_ok=True)
    json.dump(dict(rows=rows, best=best, best_ratio=ratio, gen_ok=gen_ok,
                   verdict="PASS" if verdict else "FAIL"),
              open(OUT / "gate_a2.json", "w"), indent=2)
