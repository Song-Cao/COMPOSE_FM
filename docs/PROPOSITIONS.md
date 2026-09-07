# IHC-FM: Correctness Propositions

Formal foundation for the composition operator. Three propositions, each with a
precise statement, explicit assumptions, a coordinate proof suitable for a paper
appendix, and the measured numerical verification. A fourth section records the
**double-counting diagnosis** that motivates replacing the connection coupling.

Everything here is stated in **domain-free** language: a latent chart
`Z = R^d`, a chart change `psi`, a decoder into an ambient `R^Dim`, and vector
fields on `Z`. An "intervention" is an index into a finite set of primitive
vector fields; "exposure" `tau` is a scalar. No biological or financial content
enters any definition, statement, or proof — those are validation case studies
elsewhere.

Verification script: `experiments/verify_propositions.py`
Measured residuals: `results/propositions.json`
All arithmetic in `float64`; all derivatives by **complex-step differentiation**
(`h = 1e-20`), which has no truncation error and no subtractive cancellation, so
residuals are limited only by float64 round-off (~1e-16) rather than by a
finite-difference floor (~1e-8).

**Status: 33/33 checks pass.** Headline residuals are reported as a maximum over
9 independent seeds, not a single lucky draw.

## Notation

| symbol | meaning |
|---|---|
| `z in Z = R^d` | latent state in chart 1 |
| `psi: Z -> Z`, `z' = psi(z)` | chart change (a diffeomorphism) |
| `J = Dpsi(z)`, `J^a_i = d psi^a / d z^i` | its Jacobian |
| `D2psi(z)^a_ij` | its second derivative, symmetric in `(i,j)` |
| `X, Y` | vector fields on `Z` |
| `S = DX[Y] + DY[X]` | symmetric kinematic cross term |
| `G(z)`, `Gamma^k_ij` | metric and its Levi-Civita connection |
| `D: Z -> R^Dim`, `J_D` | decoder and its Jacobian |
| `W(x)` | ambient positive-definite weight |
| `b(x)` | ambient one-form |

Latin indices `i,j,k,l` are chart-1 latent indices; `a,b,c` chart-2 latent
indices; repeated indices are summed. A tilde marks an object read in chart 2.

---

## Proposition 1 — the displacement defect

### Statement

Let `psi: Z -> Z` be `C^2` on a neighbourhood of `z`. For a displacement
`delta in R^d`,

```
    psi(z + delta) - psi(z)  =  Dpsi(z) delta  +  R(z, delta),
    R(z, delta) = (1/2) D2psi(z)[delta, delta] + O(||delta||^3).
```

Consequently:

1. **Fields are covariant, finite displacements are not.** Composing vector
   *fields* and then integrating is chart-covariant, because a field transforms
   linearly, `X~(psi(z)) = Dpsi(z) X(z)`, with no remainder. Adding finite
   *displacements* incurs the defect `R`, which is second order in `||delta||`
   and generally nonzero.
2. **Affine control.** If `psi` is affine, `psi(z) = A z + c`, then
   `D2psi == 0` identically and `R == 0` *exactly* — not approximately. A
   displacement-additive model is therefore exactly covariant under affine
   reparameterisation.
3. **Scaling.** The *relative* defect `||R|| / ||delta||` is, to leading order,
   proportional to `||delta||`. Hence it grows with any mechanism that grows the
   displacement: exposure `tau` (linearly), and intervention count `k` (as
   `sqrt(k)` for independent primitives).

### Assumptions

* `psi` is `C^2` on a neighbourhood containing the segment `[z, z + delta]`
  (Taylor with the integral remainder needs no more).
* `psi` is a diffeomorphism there — used for the statement to be about a chart
  change at all, not for the expansion itself.
* The `O(||delta||^3)` term is uniform on the sampled region; guaranteed here
  because the test chart's nonlinearity is a bounded analytic `tanh` block.
* Nothing is assumed about `delta`'s origin. It may be a learned displacement,
  a sum of per-intervention displacements, or an integrated flow.

### Proof (coordinates)

Fix `z` and `delta`, and let `g(t) = psi(z + t delta)` for `t in [0,1]`. Since
`psi` is `C^2`, `g` is `C^2` with

```
    g'(t)^a  = d_i psi^a (z + t delta) delta^i,
    g''(t)^a = d_i d_j psi^a (z + t delta) delta^i delta^j.
```

Taylor with integral remainder at `t = 0`, evaluated at `t = 1`:

```
    psi(z + delta)^a = g(1)^a
                     = g(0)^a + g'(0)^a + int_0^1 (1 - t) g''(t)^a dt
                     = psi(z)^a + d_i psi^a(z) delta^i
                       + int_0^1 (1-t) d_i d_j psi^a(z + t delta) delta^i delta^j dt.
```

The first two terms are `psi(z) + Dpsi(z) delta`. The remainder is the defect:

```
    R(z, delta)^a = int_0^1 (1 - t) d_i d_j psi^a(z + t delta) delta^i delta^j dt.     (1)
```

Bounding the integrand by `M = sup ||D2psi||` on the segment and using
`int_0^1 (1-t) dt = 1/2` gives `||R|| <= (M/2) ||delta||^2`, i.e.
`R = O(||delta||^2)`. Expanding `d_i d_j psi^a` about `t = 0` inside (1) and
integrating term by term isolates the leading coefficient:

```
    R(z, delta)^a = (1/2) d_i d_j psi^a(z) delta^i delta^j + O(||delta||^3),
```

which is claim (1). Only the part of `D2psi` symmetric in `(i,j)` contributes,
since it is contracted against `delta^i delta^j`; by Clairaut's theorem `D2psi`
is symmetric anyway for `C^2` `psi`.

For claim (2), if `psi(z) = A z + c` then `d_i d_j psi^a == 0` for all `z`, so the
integrand in (1) vanishes identically and `R == 0` — an exact statement, valid
for arbitrarily large `delta`, not an asymptotic one.

For claim (3), divide the leading term by `||delta||`:

```
    ||R|| / ||delta|| = (1/2) | D2psi[u, u] | ||delta|| + O(||delta||^2),
    u = delta / ||delta||,
```

so the relative defect is proportional to `||delta||` with a chart-dependent
constant. If `delta = tau sum_{p=1}^{k} X_p(z)` then `||delta||` scales like
`tau` and, for primitives with independent random orientations,
`E||sum_p X_p||` scales like `sqrt(k)` — giving the `tau^1` and `sqrt(k)` laws.

### Verification (measured)

Random nonlinear chart, `d = 4`, 8 states, `float64`, analytic `Dpsi` and
`D2psi`, complex-step cross-check.

| quantity | measured | expected |
|---|---|---|
| defect norm log-log slope vs `\|\|delta\|\|` (window `<= 1e-2`) | **1.999142** (dev. 8.576e-4) | 2 |
| `relerr(defect, ½D2psi[δ,δ]) / \|\|delta\|\|`, constancy spread on `[1e-4, 1e-2]` | **2.138e-3** | constant |
| leading-coefficient limit `relerr/\|\|delta\|\|` | **0.684** | finite constant |
| **affine chart: defect, absolute max** | **2.459e-16** | 0 (machine zero) |
| affine chart: `\|\|D2psi\|\|_max` | **0.0** | identically 0 |
| relative defect vs `tau` (slope, `tau <= 0.2`) | **0.99215** (dev. 7.845e-3) | 1 |
| relative defect vs `tau`, all `tau <= 0.8` | 0.947 | bends by `O(tau^3)` |
| `k`-growth: `rel/sqrt(k)` spread, `k = 1..6` | **0.1222** | ~constant |
| anisotropy via `\|\|delta\|\|`-inflation | **monotone**, 0 violations | increases |
| anisotropy at fixed `\|\|delta\|\|`, random frame | **flat**, rel. range 8.05e-3 | see below |

**Two honest caveats, both diagnosed rather than tuned away.**

*Measurement cancellation floor.* The defect is formed as a difference of `O(1)`
quantities whose true size is `O(||delta||^2)`. Below `||delta|| ~ 1e-4` the
float64 round-off of those `O(1)` terms (~1e-16 absolute) is no longer negligible
against the signal, and the measured relative error stops falling and *rises*
again (6.825e-5 at `1e-4`, 7.136e-5 at `1e-5`, 7.665e-3 at `1e-6`, against the
O(`||delta||`) law that predicts continued decrease). This is subtractive
cancellation in
the *measurement*, not a failure of the proposition, and the claim is
falsifiable. Two hypotheses make opposite predictions about the **absolute**
residual:

* *round-off*: the expansion is correct and we are seeing float64 error in the
  `O(1)` terms. Then the absolute residual is pinned near
  `eps_mach * ||psi(z)||`, independent of `||delta||`, and only the *relative*
  error rises, because the signal shrinks beneath a fixed floor.
* *a missing term*: the expansion omits something larger than the cubic term.
  Then the absolute residual would be **larger** than that floor.

Measured at `||delta|| = 1e-6`: absolute residual **1.039e-16** against the
predicted floor `eps_mach ||psi||` = **1.510e-16**, a ratio of **0.688** — within
a factor of 1.5 of the round-off prediction, and not above it. For scale, the
genuine cubic term is invisible here: calibrated from the measured remainder at
`||delta|| = 1e-1` (absolute **1.044e-5**) and scaled by `||delta||^3`, it is
**1.044e-20** at `||delta|| = 1e-6` — four decades *below* the floor. The floor,
not the cubic term, is what sets the observed residual. The coefficient is
therefore fitted on the window `[1e-4, 1e-2]`.

*Anisotropy is a partially negative result.* Latent anisotropy raises the defect
through **one** mechanism only, and we separate three candidate channels:

* **`||delta||`-inflation — CONFIRMED.** Anisotropic geometry makes the same
  field amplitudes produce a longer displacement; since the relative defect is
  proportional to `||delta||`, it grows. Monotone over 12 chart draws in the
  unsaturated regime (2.258e-4 -> 2.027e-3 across a 20x stretch), and it acts
  *through* `||delta||`: the ratio `rel/||delta||` stays constant to 21%.
  **This is the only anisotropy claim the paper should make.**
* **Direction-concentration at fixed `||delta||` — NOT SUPPORTED.** Averaged over
  a random anisotropy frame and 12 charts, the relative defect is flat across a
  20x stretch (5.582e-4 -> 5.605e-4, relative range 8.05e-3, no trend). With the
  stretch axes aligned to the coordinate axes a trend appears, but its **sign is
  not stable across chart draws** (decreasing here, increasing in an independent
  probe at a larger state scale). That sweep measures alignment between the
  stretch axes and the eigendirections of `D2psi` — a property of the particular
  chart, not an effect of anisotropy.
* **Saturation turnover.** At large state scale the naive sweep is
  *non-monotone*: the test chart's nonlinearity is a bounded `tanh`, so
  stretching `z` drives its argument into saturation where `D2psi -> 0`
  (measured `||D2psi||` falls 8.901e-2 -> 2.112e-2 as anisotropy goes 1 -> 20,
  with mean `|tanh|` saturation rising 0.7148 -> 0.9943). Recorded with the
  diagnostic that explains it.

---

## Proposition 2 — connection law and the tensoriality of `S + 2 Gamma`

### Statement

Let `X, Y` be vector fields on `Z`, let `G` be a `C^2` Riemannian metric with
Levi-Civita connection `Gamma`, and let `psi` be a `C^3` diffeomorphism with
`J = Dpsi(z)`. Write `S = DX[Y] + DY[X]` for the symmetric kinematic cross term.
Then, with all objects on the left read in chart 2 at `z' = psi(z)`:

**(a) The cross term is not a tensor; its defect is exactly `2 D2psi`:**

```
    S~  =  J . S  +  2 D2psi(z)[X, Y].                                     (2)
```

**(b) The connection is not a tensor; it has the opposite defect:**

```
    Gamma~[J X, J Y]  =  J . Gamma[X, Y]  -  D2psi(z)[X, Y].               (3)
```

**(c) Hence the combination `S + 2 Gamma` *is* a vector field:**

```
    S~ + 2 Gamma~  =  J . ( S + 2 Gamma ).                                 (4)
```

The sign is fixed and not a convention: `S - 2 Gamma` and `S` alone both fail.

**(d) Affine degeneracy.** If `psi` is affine, `D2psi == 0` and all three
combinations `S + 2Gamma`, `S - 2Gamma`, `S` transform correctly. An affine map
therefore **cannot** distinguish them.

### Assumptions

* `psi` is a `C^3` diffeomorphism on a neighbourhood of `z` (so `D2psi` exists
  and is continuous, and `Gamma~` is defined).
* `G` is `C^2` and positive definite, so `G^{-1}` exists and `Gamma` is the
  unique torsion-free metric connection.
* `D2psi` is used **symmetrised in its two lower indices**. This is automatic for
  `C^2` maps by Clairaut, but is a real implementation requirement: an
  unsymmetrised second-derivative array gives wrong answers (verified below).
* `X, Y` are `C^1` and are transported as *fields* (`X~ = J X` at the
  corresponding point), not as displacement vectors at a fixed point.
* Chart 2's metric is the push-forward `G~ = K^T G K` with `K = J^{-1}`; that is,
  the same geometry read in new coordinates, not an independently learned metric.
* **(d) is a warning, not a caveat:** PCA and any whitening are affine, so they
  are invalid as examples of a nonlinear reparameterisation.

### Proof (index notation)

Throughout, `a,b,c` are chart-2 indices, `i,j,k,l` chart-1 indices,
`J^a_i = d_i psi^a`, `K^i_a = (J^{-1})^i_a`, and `d_a = K^i_a d_i` is the
chart-2 partial derivative. All objects are evaluated at corresponding points
(`z` in chart 1, `z' = psi(z)` in chart 2).

**(a) The cross-term defect.** A vector field transforms as
`X~^a(z') = J^a_i X^i(z)`. Differentiate with respect to `z'^b`, using the chain
rule `d_b = K^j_b d_j`:

```
    d_b X~^a = K^j_b d_j ( J^a_i X^i )
             = K^j_b ( J^a_i d_j X^i  +  X^i d_j J^a_i )
             = K^j_b ( J^a_i d_j X^i  +  X^i d_j d_i psi^a ).
```

Contract with `Y~^b = J^b_l Y^l` and use `K^j_b J^b_l = delta^j_l`:

```
    (DX~[Y~])^a = J^a_i (d_l X^i) Y^l  +  (d_i d_l psi^a) X^i Y^l
                = J^a_i (DX[Y])^i      +  D2psi^a[X, Y].                   (5)
```

Swapping the roles of `X` and `Y` gives the same second term, since
`D2psi^a_il X^i Y^l` is symmetric under `(X,i) <-> (Y,l)` when `D2psi` is
symmetric in its lower indices. Adding the two copies:

```
    S~^a = (DX~[Y~] + DY~[X~])^a = J^a_i S^i + 2 D2psi^a[X, Y],
```

which is (2). Note the factor **2** arises from the two summands of `S`, each
contributing one identical copy of `D2psi[X,Y]` — this is the origin of the `2`
in `S + 2 Gamma`.

**(b) The connection law.** The Levi-Civita coefficients are

```
    Gamma^k_ij = (1/2) G^{kl} ( d_i G_lj + d_j G_li - d_l G_ij ).
```

The standard transformation law of an affine connection under `z' = psi(z)` is

```
    Gamma~^c_ab = J^c_k K^i_a K^j_b Gamma^k_ij  +  K^i_a K^j_b d_i d_j psi^c
                                                  ... (with the sign convention
                                                  fixed by the derivation below).
```

Rather than quote it, derive the contracted form we actually need. Covariant
differentiation of a vector field is chart-independent:
`(nabla_Y X)~^a = J^a_i (nabla_Y X)^i`, where in coordinates

```
    (nabla_Y X)^k = (d_i X^k) Y^i + Gamma^k_ij X^i Y^j.
```

Write this in chart 2 and substitute (5):

```
    (nabla_{Y~} X~)^a = (DX~[Y~])^a + Gamma~^a[X~, Y~]
                      = J^a_i (DX[Y])^i + D2psi^a[X,Y] + Gamma~^a[J X, J Y].
```

Chart-independence forces this to equal
`J^a_i (nabla_Y X)^i = J^a_i (DX[Y])^i + J^a_i Gamma^i[X,Y]`. Cancelling the
common `J^a_i (DX[Y])^i`:

```
    Gamma~^a[J X, J Y] = J^a_i Gamma^i[X, Y] - D2psi^a[X, Y],
```

which is (3). The defect of the connection is exactly the negative of the defect
each summand of `S` picks up.

**(c) The tensorial combination.** Add (2) and twice (3):

```
    S~ + 2 Gamma~[J X, J Y]
        = ( J S + 2 D2psi[X,Y] ) + 2 ( J Gamma[X,Y] - D2psi[X,Y] )
        = J ( S + 2 Gamma[X,Y] )  +  (2 - 2) D2psi[X,Y]
        = J ( S + 2 Gamma[X,Y] ),
```

which is (4). The `D2psi` terms cancel identically. **The coefficient is forced:**
for `S + c Gamma` the defect is `(2 - c) D2psi`, which vanishes only at `c = 2`.
In particular `S - 2 Gamma` carries defect `4 D2psi` — twice as large as `S`
alone — so the wrong sign is *worse* than no correction at all.

**(d) Affine degeneracy.** For `psi(z) = A z + c` we have `D2psi == 0`, so (2)
reduces to `S~ = J S` and (3) to `Gamma~ = J Gamma`: every combination
`S + c Gamma` transforms correctly for every `c`. The coefficient is
unidentifiable from affine maps.

### Verification (measured)

Random nonlinear chart, learned metric in Cholesky form
`G = L L^T` (mirroring `src/composefm/compose.py::Connection.metric`), `d = 4`,
6 states, `float64`, complex-step derivatives, connection symmetrised in `(i,j)`.

| check | measured | expected |
|---|---|---|
| connection law (3), `Gamma~` vs `J Gamma - D2psi` | **4.585e-15** | ~1e-16 |
| coupling defect (2) vs `2 D2psi[X,Y]` | **2.594e-14** | ~1e-16 |
| **`S + 2 Gamma` is tensorial (4)** | **2.971e-15** | ~1e-16 |
| CONTROL: `S - 2 Gamma` | **5.335e-2** | must FAIL |
| CONTROL: `S` alone | **3.996e-2** | must FAIL |
| CONTROL: unsymmetrised `D2psi` in the law | **8.341e-2** | must FAIL |
| affine chart: all of `S+2G`, `S-2G`, `S` | **3.102e-15** | all exact (fact D) |
| metric condition number (max) | 25.82 | well-conditioned |

Across **9 independent seeds**: connection law `<= 7.541e-15`, `S + 2 Gamma`
`<= 3.160e-15`, and the `S - 2 Gamma` control fails at **every** seed (minimum
relative error 4.954e-2). The correct sign is therefore not a single-seed
artefact, and the incorrect one is never accidentally satisfied.

> **Implementation note.** The codebase previously carried the wrong sign; it is
> fixed at `src/composefm/compose.py` (~line 427), which now applies
> `S = S + 2.0 * self.conn.gamma(...)`.

> **Why PCA must not be used as the nonlinear-chart example.** Row 7 above is the
> whole reason: under an affine change of chart the three candidate couplings are
> numerically indistinguishable (all ~1e-15). Any covariance experiment run on
> PCA-whitened coordinates would report success for a model with the wrong sign.

---

## Proposition 3 — whole-field covariance of the pullback one-form construction

This is the load-bearing proposition. Proposition 2 makes *one hand-built term*
tensorial; Proposition 3 says the **entire learned field** is a vector field, by
construction, with no term-by-term repair.

### Statement

Let `D: Z -> R^Dim` be a `C^1` immersion with Jacobian `J_D(z)` (`Dim x d`,
rank `d`), let `W(x)` be a positive-definite `C^0` weight on the ambient space,
and let `b(x)` be an ambient one-form. Define, in the latent chart,

```
    G(z)     = J_D(z)^T W(D(z)) J_D(z)        (pullback metric, d x d, PD)
    alpha(z) = J_D(z)^T b(D(z))               (intrinsic lift, a one-form)
    v(z)     = G(z)^{-1} alpha(z)             (the field)
```

Let `psi` be a diffeomorphism, `z' = psi(z)`, `J = Dpsi(z)`, and read the *same*
decoder in the new chart, `D' = D o psi^{-1}` (so that `D'(z') = D(z)`: the same
latent point maps to the same ambient point). Then

```
    G'(z')     = J^{-T} G(z) J^{-1}        i.e. G is a (0,2) tensor,
    alpha'(z') = J^{-T} alpha(z)           i.e. alpha is a (0,1) tensor,
    v'(z')     = J v(z)                    i.e. v is a VECTOR FIELD.           (6)
```

Every component of the construction transforms with the correct valence, so `v`
transforms correctly *as a whole* — for arbitrary `W`, arbitrary `b`, arbitrary
immersion `D`, and arbitrary `psi`.

### Assumptions (all of them, and why each is needed)

1. **`D` is an immersion on the data region:** `rank J_D(z) = d`, equivalently
   `sigma_min(J_D) > 0`. This is what makes `G = J_D^T W J_D` positive
   *definite* rather than merely semi-definite, hence invertible. Measured, not
   assumed: `sigma_min(J_D) = 0.4609` on the test sample.
2. **`W(x)` positive definite.** Needed for `G` to be PD. `W` must be a function
   of the **ambient** point `x`, not of the latent coordinates: `W(D(z))` is then
   automatically the same object in both charts, which is what makes the
   pullback work. A `W` that depended on `z` directly would not be chart-free.
3. **`G` well-conditioned** on the region where `v` is evaluated. Covariance is
   exact in exact arithmetic for any invertible `G`, but the *numerical*
   residual grows with `cond(G)`; measured `cond(G) <= 40.0` here (`<= 57.2`
   across seeds).
4. **`psi` is a diffeomorphism** with invertible `J` on the region. Enforced by
   construction in the verification (see the branch caveat below).
5. **NO raw Euclidean `eps * I` stabiliser.** The identity matrix is **not** a
   `(0,2)` tensor: `I != J^{-T} I J^{-1}` unless `J` is orthogonal. Adding
   `eps I` to `G` therefore **breaks arbitrary-chart covariance**, because the
   stabilised metric in chart 2 is `J^{-T} G J^{-1} + eps I`, not
   `J^{-T} (G + eps I) J^{-1}`. This is stated as a hard constraint on the
   method, and its cost is measured in the ablation below. If numerical
   stabilisation is required it must be applied **tensorially** — e.g. a
   multiple of `G` itself (`G -> (1 + eps) G`, which is covariant and merely
   rescales `v`), or an ambient `W -> W + eps_ambient I_Dim` (covariant in
   `Z` because it is applied in the fixed ambient frame before pullback,
   affecting the *geometry* rather than the chart-dependent identity in `Z`).

### Proof (coordinates)

Write `x = D(z)` and `x' = D'(z')`. Because `D' = D o psi^{-1}` and
`z' = psi(z)`, we have `D'(z') = D(psi^{-1}(psi(z))) = D(z) = x`, so

```
    x' = x :  the ambient point is the same in both charts.                 (7)
```

This is the crux: `W` and `b` are evaluated at the *same* ambient point in both
charts, so they contribute **no** transformation factors at all.

**Step 1 — the decoder Jacobian.** Differentiate `D' = D o psi^{-1}` at `z'` by
the chain rule, with `d psi^{-1} / d z' = J^{-1}`:

```
    J_{D'}(z')^A_a = d D'^A / d z'^a
                   = (d D^A / d z^i) (d (psi^{-1})^i / d z'^a)
                   = J_D(z)^A_i (J^{-1})^i_a.
```

In matrix form, with `K = J^{-1}`:

```
    J_{D'} = J_D K.                                                         (8)
```

So `J_D` carries one *upper* ambient index (untouched) and one *lower* latent
index (transformed by `K`) — it is a `(0,1)`-tensor-valued ambient vector.

**Step 2 — the metric is a `(0,2)` tensor.** Using (8) and (7):

```
    G'(z') = J_{D'}^T W(x') J_{D'}
           = (J_D K)^T W(x) (J_D K)
           = K^T ( J_D^T W(x) J_D ) K
           = K^T G(z) K  =  J^{-T} G(z) J^{-1}.                             (9)
```

In indices, `G'_ab = K^i_a K^j_b G_ij` — exactly the transformation law of a
`(0,2)` tensor. No assumption on `W` beyond it being evaluated ambiently.

**Step 3 — the lift is a `(0,1)` tensor.** Same substitution:

```
    alpha'(z') = J_{D'}^T b(x') = (J_D K)^T b(x) = K^T ( J_D^T b(x) )
               = K^T alpha(z)  =  J^{-T} alpha(z).                        (10)
```

In indices, `alpha'_a = K^i_a alpha_i`. Note this is the **pullback of a
one-form**, which needs no inverse of `J_D` — essential, since `J_D` is
`Dim x d` and not square. Pulling back a one-form is always defined for a smooth
map; pushing forward a *vector* would not be.

**Step 4 — the field is a vector field.** Combine (9) and (10):

```
    v'(z') = G'(z')^{-1} alpha'(z')
           = ( K^T G K )^{-1} ( K^T alpha )
           = K^{-1} G^{-1} K^{-T} K^T alpha
           = K^{-1} G^{-1} alpha
           = J v(z),
```

using `K^{-1} = J` and the cancellation `K^{-T} K^T = I`. In indices,
`v'^a = J^a_i v^i`, which is (6).

**Why the inverse metric is what makes this work.** The construction is
*index-balanced*: the two lower indices of `G^{-1}` (raised to upper) exactly
consume the one lower index of `alpha` and leave one upper index. Concretely,
`alpha` transforms with `K^T` and `G^{-1}` with `K^{-1} (·) K^{-T}`; the
`K^{-T} K^T` pair annihilates and a single `K^{-1} = J` survives. Any
construction that broke this pairing — adding a non-tensorial term to `G`,
or using a Euclidean inner product in place of `W` — would leave a residual
factor of `J` and destroy (6).

**Contrast with Proposition 2.** Proposition 2 repairs a *single* term by adding
a compensating connection. Proposition 3 needs no repair: covariance follows from
the *valences* of the objects, so it holds for the whole learned field
simultaneously and for arbitrary learned `W`, `b`, `D`. This is why the method's
covariance does not depend on any learned quantity being correct.

### Verification (measured)

**Random nonlinear decoder and random nonlinear chart**, as required:
`D(z) = W2 tanh(W1 z + b1) + W3 z + b2` with `d = 4`, `Dim = 9`, random
Gaussian weights; `psi` a random `tanh` chart; `W(x)` a random state-dependent PD
ambient weight; `b(x)` a random nonlinear ambient one-form; 8 states; `float64`.

The chart-2 decoder Jacobian is obtained by **complex-step differentiating the
actually composed map** `z' -> D(psi^{-1}(z'))`, with the Newton inverse carried
through in complex arithmetic — *not* by substituting `J_D K`. Substituting (8)
would make the test a tautology; instead (8) is itself checked (row 2).

| check | measured | expected |
|---|---|---|
| **WHOLE-FIELD covariance `v' = Dpsi . v`** (max) | **1.629e-15** | `<= 1e-10` |
| whole-field covariance (mean) | **8.831e-16** | — |
| chart-2 `J_D` by complex step vs chain rule (8) | **4.587e-16** | ~1e-16 |
| `G` transforms as a `(0,2)` tensor (9) | **9.191e-16** | ~1e-16 |
| `alpha` transforms as a `(0,1)` tensor (10) | **7.430e-16** | ~1e-16 |
| decoder immersion `sigma_min(J_D)` | **0.4609** | `> 0` |
| `cond(G)` max | **40.02** | well-conditioned |
| chart round trip `psi^{-1}(psi(z)) = z` | **3.077e-16** | ~1e-16 |

**Achieved relative residual: 1.63e-15** — four to five orders of magnitude
better than the 1e-10 bar, and consistent with pure float64 round-off at
`cond(G) ~ 40`. Across **9 independent seeds** the maximum is **3.904e-15**
(worst-case `cond(G) = 57.2`).

#### Ablation: the Euclidean `eps I` stabiliser breaks covariance

Assumption 5 is not decorative. Adding `G -> G + eps I` and repeating the test:

| `eps` | covariance relative error |
|---|---|
| `1e-6` | 2.123e-6 |
| `1e-4` | 2.123e-4 |
| `1e-2` | 2.089e-2 |
| `1e-1` | 1.828e-1 |

The error is **linear in `eps`** and utterly dominates the 1.63e-15 residual of
the unstabilised construction — at `eps = 1e-2`, a routine choice, covariance is
lost at the 2% level, i.e. 13 orders of magnitude worse. `eps I` is therefore
**forbidden in the method definition**, and any required conditioning must be
tensorial (assumption 5).

#### A real failure found and fixed: the chart-inverse branch

Reported because it is a genuine trap. A first pass drew `psi` with an
unconstrained nonlinearity, and at 2 of 9 seeds the covariance checks failed
catastrophically — **P2 connection law 2.46, P3 whole-field 6.30e-1** — while the
`psi`-space residual `psi(psi^{-1}(z')) - z'` remained at 3e-16.

Diagnosis: the drawn nonlinearity violated the contraction bound
`s ||C||_2 ||B||_2 < sigma_min(A)`, so `Dpsi` became near-singular on the sample
(`sigma_min` down to **2.0e-3**), `psi` was not injective, and Newton converged
to a **different branch** of `psi^{-1}`. A `psi`-space residual check cannot
detect this — *any* branch satisfies it — whereas the `Z`-space round trip does:
`||psi^{-1}(psi(z)) - z||` was **8.2e-1**.

Fix (both retained): the chart now **rescales its nonlinear block at
construction** to enforce `s ||C||_2 ||B||_2 <= 0.6 sigma_min(A)`, which makes
`psi` a global diffeomorphism for every draw while keeping it genuinely nonlinear
(`||D2psi||` stays 0.06–0.16, and the `S - 2 Gamma` control still fails); and a
`Z`-space round-trip check is now asserted at 1e-13. After the fix, **0 failed
checks across 9 seeds**.

The lesson generalises to the method: every covariance guarantee here is a
statement about a *single* chart, so an experiment that reparameterises data must
verify the reparameterisation is injective on the data region — in `Z` space, not
merely in image space.

---

## The double-counting diagnosis — why the connection coupling is replaced

This section records a **derivation error**, not a performance shortfall. It is
the motivation for the method's interaction term and belongs in the paper as
such.

### The claim

For vector fields `X, Y` on `Z` and integration time `T`, let
`flow_F^T(z)` denote the time-`T` flow of `F`. Then

```
    flow_{X+Y}^T(z) - [ z + (flow_X^T(z) - z) + (flow_Y^T(z) - z) ]
        = (T^2 / 2) ( DX[Y] + DY[X] )  +  O(T^3)
        = (T^2 / 2) S  +  O(T^3).                                          (11)
```

**Integrating the summed field already produces the symmetric cross term `S`,
automatically and for free.** The difference between the joint flow and the sum
of the individual displacements *is* `S` at leading order.

### Why this invalidates the old justification

The previous coupling term was justified by the argument: *"`S` is the leading
term of joint-minus-additive behaviour, so a model that must capture interaction
should carry `beta * S` explicitly."* Equation (11) shows the premise proves the
opposite of the conclusion. The quantity `S` is what **generator composition
already yields**; a model that composes fields and integrates gets it without
any coupling term. Adding `beta * S` to the instantaneous velocity therefore
inserts a **second copy** of the same kinematic tensor.

Worse, the two copies enter at different orders in `T` and are collinear:

* composition's copy appears in the *displacement* at order `T^2`;
* the hand-added term is in the *velocity*, so it appears in the displacement at
  order `T`.

Their ratio is `2 beta / T`, which **diverges as `T -> 0`**: measured 40x at
`T = 0.05` and 400x at `T = 0.005` (`beta = 1`). At small exposure the
hand-added term does not refine the effect it was introduced to model — it
overwhelms it by two orders of magnitude, in the same direction. Measured
collinearity between the empirical joint-minus-additive residual and `S`:
cosine **0.99999790**. The added term supplies no new direction.

### What replaces it

Since generator composition already covers the kinematic cross term, the only
structure a model can *add* is what lies **outside** the kinematic span. The
replacement interaction residual is therefore required to be:

* **non-kinematic** — not in the span of `{DX_p[X_q] + DX_q[X_p]}` nor of the
  primitives themselves (measured against the known ground truth by
  `src/composefm/synthetic.py::nonkinematic_fraction`);
* **singleton-vanishing** — exactly zero when only one intervention is active,
  so interaction is unidentifiable from singletons and leave-one-combination-out
  is a real test;
* **low-rank** — factoring through `r` basis fields.

The old connection-coupling form is **retained as an ablation arm**, not deleted:
it is the correct control for the claim that the non-kinematic residual is doing
work that the kinematic term cannot.

### Proof sketch of (11)

Let `F = X + Y` and expand each flow to second order in `T`. For a `C^1` field
`F`, `d/dt flow_F^t = F(flow_F^t)`, so

```
    flow_F^T(z) = z + T F(z) + (T^2/2) DF(z)[F(z)] + O(T^3).
```

Apply this to `F = X + Y`, `F = X`, and `F = Y`:

```
    flow_{X+Y}^T(z) = z + T(X+Y) + (T^2/2) ( D(X+Y)[X+Y] ) + O(T^3),
    flow_X^T(z) - z = T X + (T^2/2) DX[X] + O(T^3),
    flow_Y^T(z) - z = T Y + (T^2/2) DY[Y] + O(T^3).
```

Expanding the bilinear middle term,
`D(X+Y)[X+Y] = DX[X] + DX[Y] + DY[X] + DY[Y]`. Subtracting the two individual
displacements from the joint flow, the `O(T)` terms cancel exactly, as do the
self-terms `DX[X]` and `DY[Y]`, leaving

```
    (T^2/2) ( DX[Y] + DY[X] ) + O(T^3) = (T^2/2) S + O(T^3),
```

which is (11). Dividing by `T^2/2`, the relative error against `S` is `O(T)`.

### Verification (measured)

`d = 4`, 5 states, primitives `X_p(z) = M_p tanh(z)` (the family of
`src/composefm/synthetic.py`), `float64`, RK4 with `n = 4000` steps so the
integrator error is far below the measured `O(T)` effect.

| `T` | `relerr( [flow(X+Y) - (dispX + dispY)] / (T²/2), S )` | explicit/kinematic ratio |
|---|---|---|
| 0.05 | **1.6085e-2** | 40x |
| 0.02 | **6.4366e-3** | 100x |
| 0.01 | **3.2189e-3** | 200x |
| 0.005 | **1.6096e-3** | 400x |

Convergence is clean first order: log-log slope **0.99971** (expected 1), i.e.
the residual is exactly the predicted `O(T)`, confirming that the `(T²/2) S`
term is not merely close but is the correct leading term.

Collinearity of the two copies: minimum cosine **0.99999790**.

> **Relation to the previously recorded values.** An earlier measurement of the
> same effect (RK2, `n = 4000`, a different primitive draw) gave 1.538e-2,
> 6.128e-3, 3.060e-3, 1.529e-3 at the same four `T`. The values here agree in
> order of magnitude and reproduce the same clean `O(T)` law, differing in the
> third significant digit because both the field draw and the integrator differ.
> The conclusion is identical and is robust to both choices; the numbers are not
> interchangeable and should not be quoted as though from one run.

---

## Summary of measured residuals

| # | claim | measured | bar |
|---|---|---|---|
| P1 | quadratic defect rate (slope) | 1.999142 | 2 |
| P1 | affine chart defect (exact zero) | 2.459e-16 | machine zero |
| P1 | relative defect `~ tau^1` (slope) | 0.99215 | 1 |
| P2 | connection law `Gamma~ = Dpsi.Gamma - D2psi` | 4.585e-15 | 1e-12 |
| P2 | coupling defect `= 2 D2psi[X,Y]` | 2.594e-14 | 1e-12 |
| P2 | **`S + 2 Gamma` tensorial** | 2.971e-15 | 1e-12 |
| P2 | control `S - 2 Gamma` (must fail) | 5.335e-2 | — |
| P2 | control `S` alone (must fail) | 3.996e-2 | — |
| P2 | affine chart: all variants exact | 3.102e-15 | 1e-12 |
| P3 | **whole-field covariance `v' = Dpsi.v`** | **1.629e-15** | 1e-10 |
| P3 | `G` as `(0,2)` tensor | 9.191e-16 | 1e-10 |
| P3 | `alpha` as `(0,1)` tensor | 7.430e-16 | 1e-10 |
| P3 | ablation `eps I` at `eps = 1e-2` (must fail) | 2.089e-2 | — |
| E | joint-minus-additive `= (T²/2) S` at `T = 0.005` | 1.610e-3 | 5e-3 |
| E | `O(T)` convergence slope | 0.99971 | 1 |

Seed robustness (9 seeds): P2 law `<= 7.541e-15`, `S + 2 Gamma` `<= 3.160e-15`,
P3 whole-field `<= 3.904e-15`, `S - 2 Gamma` control fails at every seed
(min 4.954e-2). Total failed checks: **0**.

Reproduce with:

```
python experiments/verify_propositions.py     # ~21 s, local CPU, float64, no GPU
```
