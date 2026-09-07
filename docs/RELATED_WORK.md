# IHC-FM: annotated related work

Every entry below was verified against Crossref (`api.crossref.org/works/<doi>`) or the
arXiv API (`export.arxiv.org/api/query?id_list=<id>`) on 7 September 2026. Titles, author
lists, volumes and pages are copied from those responses, not from memory: author names come
from the complete `<name>` list (arXiv) or the complete `author` array (Crossref), which is
also the source of the author *counts* and last-author names given for the two long lists
(CPA, 19 authors; Trellis, 13 authors). Nothing in this file is a guessed identifier: if a
DOI or arXiv id is written here, the metadata line under it came back from the corresponding
API. Venues are given only for the Crossref rows, which return them; see §F.

Each entry ends with **Difference** — one line stating what IHC-FM does that this work does
not, or what IHC-FM takes from it and therefore does not claim.

---

## A. Flow matching: objectives, couplings, paths

### A1. Flow Matching for Generative Modeling
`arXiv:2210.02747` — Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel,
Matt Le. First release 6 October 2022.

Introduces the conditional flow matching objective: regress a learned velocity field onto
the velocity of a tractable conditional probability path, which yields an unbiased gradient
for the intractable marginal-path objective.

**Difference.** IHC-FM uses this objective unchanged as its training loss. The contribution
is *what is parameterised* — a chart-covariant pullback field carrying an intervention-set
hierarchy — not a new objective. IHC-FM claims no improvement to the CFM estimator itself.

### A2. Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow
`arXiv:2209.03003` — Xingchao Liu, Chengyue Gong, Qiang Liu. First release 7 September 2022.

Rectified flow: learn a velocity field along straight-line interpolants between coupled
endpoints, then optionally reflow to straighten trajectories.

**Difference.** IHC-FM's interpolant is the same straight line
`x_t = (1 - t) x_0 + t x_1`. It does not reflow, and it makes no claim about trajectory
straightness or few-step sampling. The straight-line path is a *deliberate* baseline
choice here, because a straight observed-space interpolant is exactly what our
double-counting diagnosis identifies as unable to separate a generator interaction from the
kinematic composition term (see §D and `docs/PROPOSITIONS.md`).

### A3. Improving and generalizing flow-based generative models with minibatch optimal transport
`arXiv:2302.00482` — Alexander Tong, Kilian Fatras, Nikolay Malkin, Guillaume Huguet,
Yanlei Zhang, Jarrid Rector-Brooks, Guy Wolf, Yoshua Bengio. First release 1 February 2023.

The I-CFM / OT-CFM family. Generalises CFM to arbitrary couplings of the source and target
marginals, and shows that replacing the independent (product) coupling with a minibatch
optimal-transport coupling reduces the variance of the regression target and straightens
the learned flow.

**Difference.** IHC-FM adopts the coupling generalisation and selects among
`independent` / exact minibatch assignment / entropic (Sinkhorn, optionally unbalanced)
couplings on a *held-out distributional* score. Our added methodological point is an
evaluation-validity constraint that this line does not state: the CFM loss value is **not
comparable across couplings**, because it tracks the identity-pair fraction the coupling
produces (measured: paired 1.0, exact assignment 0.795, entropic 0.229, independent
0.0039). On our measurements the coupling ranking *inverts* between CFM loss and normalised
energy distance, and the exact-assignment coupling (1.1188) is worse than predicting no
change. Ranking or selecting on CFM loss across couplings is therefore invalid, and we
report the concrete inversion rather than asserting the principle.

### A4. Meta Flow Matching: Integrating Vector Fields on the Wasserstein Manifold
`arXiv:2408.14608` — Lazar Atanackovic, Xi Zhang, Brandon Amos, Mathieu Blanchette,
Leo J. Lee, Yoshua Bengio, Alexander Tong, Kirill Neklyudov. First release 26 August 2024.

Amortises flow matching over *populations*: the velocity field is conditioned on an
embedding of the initial empirical measure, so a single model generalises across
distributions rather than being fitted per distribution.

**Difference.** IHC-FM takes population conditioning directly from this work and claims no
credit for it: our `PopulationEncoder` is exactly that mechanism, and it is **off by
default** because we measured it to be harmful where it matters. Under a leave-replicate-out
split it is neutral overall (paired diff +0.027, 95% CI [-0.026, +0.085], n = 87,
p = 0.086) but significantly *harmful on combinations only* (-0.059, [-0.113, -0.011],
n = 57, p = 0.016). What MFM does not do is compose several interventions: it conditions one
velocity field on a population, whereas IHC-FM's field is indexed by an intervention *set*
with structural singleton-vanishing interaction terms, so unseen combinations are the
object of prediction.

### A5. Metric Flow Matching for Smooth Interpolations on the Data Manifold
`arXiv:2405.14780` — Kacper Kapuśniak, Peter Potaptchik, Teodora Reu, Leo Zhang,
Alexander Tong, Michael Bronstein, Avishek Joey Bose, Francesco Di Giovanni. First release
23 May 2024.

Replaces the straight-line interpolant with geodesics of a learned data-dependent metric,
so the conditional path stays on the data manifold rather than cutting through empty space.

**Difference.** Both works learn a metric; they use it for different objects. MFM curves the
**interpolating path** used to build the regression target. IHC-FM leaves the path straight
and uses the metric in the **field itself**: the velocity is `v = G^{-1} J^T b` with
`G = J^T W(Dec(z)) J` a pullback metric, which is what makes the whole learned field a
vector field under latent reparameterisation (Proposition 3, whole-field covariance
residual 1.63e-15). A metric used to bend a path does not confer that property on the field.

### A6. Energy Guided Geometric Flow Matching
`arXiv:2509.25230` — Aaron Zweig, Mingxuan Zhang, Elham Azizi, David Knowles. First release
25 September 2025.

Uses a learned energy to define the geometry along which flow-matching interpolants are
constructed, guiding paths through high-density regions of the data manifold.

**Difference.** Again the metric/geometry acts on the interpolant rather than on the
parameterisation of the field. IHC-FM's geometric object is a **pullback of an ambient
one-form**, which is what buys covariance by valence rather than by repair, and which
constrains the *stabilisation* we are allowed to apply: adding `eps I` to the latent metric
is forbidden in our method definition because the identity is not a `(0,2)` tensor, and the
measured cost is linear in `eps` (2.12e-6 at `eps = 1e-6`, 2.09e-2 at `eps = 1e-2` on the
analytic construction; 2.60e-3 at `eps = 1e-2` on the implemented field).

---

## B. Perturbation modelling: additive displacements and their evaluation

### B1. Predicting cellular responses to complex perturbations in high-throughput screens (CPA)
`doi:10.15252/msb.202211517` — Mohammad Lotfollahi, Anna Klimovskaia Susmelj,
Carlo De Donno, Leon Hetzel *et al.* (19 authors; last author Fabian J. Theis),
*Molecular Systems Biology* **19**(6), 8 May 2023.

The compositional perturbation autoencoder. Encodes each perturbation and each covariate as
a learned latent **displacement vector**, and predicts a combination by *adding* those
displacements in the autoencoder's latent space, with a dose-response scaling per
perturbation.

**Difference.** This is the modelling pattern IHC-FM is built against, and the difference is
geometric rather than architectural. A latent space learned by an autoencoder is defined
only up to a diffeomorphism `psi`; a sum of *displacements* is not covariant under `psi`
(the defect is `(1/2) D2psi[delta, delta] + O(||delta||^3)`, measured log-log slope
1.999142 against the predicted 2), while a sum of *vector fields* is exactly covariant.
IHC-FM therefore composes generators and integrates, and reports the *measured*
consequence: the chart-dependence of the *trained outcome* is real but modest
(correlation between chart curvature and cross-chart discrepancy +0.92, with the discrepancy
reaching the seed-noise floor only at the strongest charts). We do not claim CPA-class
models fail because of this defect alone.

### B2. Predicting transcriptional outcomes of novel multigene perturbations with GEARS
`doi:10.1038/s41587-023-01905-6` — Yusuf Roohani, Kexin Huang, Jure Leskovec,
*Nature Biotechnology* **42**(6):927–935, 17 August 2023.

Predicts multi-gene perturbation outcomes by combining per-perturbation effect
representations on a gene-relation graph, with a combination handled by pooling the
constituent perturbation embeddings — an additive-displacement composition rule with
learned, graph-informed per-perturbation terms.

**Difference.** The relevant class property is the composition rule, not the graph prior.
IHC-FM's competitor in Table 1 is a *correct* implementation of that class,
`FactoredAdditiveCFM` — bit-exactly additive (defect 0.0), zero-exposure-vanishing (0.0)
and permutation-invariant (0.0), parameter-matched at 21,991 vs IHC-FM's 22,135 (0.993x)
under an identical training protocol — so the comparison is against a working additive model
rather than a weakened one. IHC-FM's addition is an interaction branch that is *structurally
zero on every singleton*, which is what makes leave-one-combination-out a real test of it.

### B3. Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines
`doi:10.1038/s41592-025-02772-6` — Constantin Ahlmann-Eltze, Wolfgang Huber,
Simon Anders, *Nature Methods* **22**(8):1657–1661, August 2025.

Benchmarks deep perturbation-effect predictors against deliberately simple baselines —
including predicting the training-set mean shift and predicting no change — and finds the
deep models do not beat them on held-out perturbations.

**Difference.** IHC-FM adopts this critique as a *protocol requirement* rather than
rebutting it. Every normalised figure we report carries its no-change denominator, and the
no-change control is a first-class row. This is also where our headline honest negative
comes from: on the organoid leave-one-combination-out screen IHC-FM **loses to no-change**
(paired diff +0.688, 95% CI [+0.351, +1.058], n = 136 populations, better on only 72/136,
sign test p = 0.549), even while beating the additive baseline decisively (-1.577,
[-2.100, -1.101], 113/136, sign test p = 1.8e-15). We report both.

### B4. Deep Learning-Based Genetic Perturbation Models *Do* Outperform Uninformative Baselines on Well-Calibrated Metrics
`doi:10.1101/2025.10.20.683304` — Henry E. Miller, Gabriel M. Mejia, Francis J. A. Leblanc,
Bo Wang, Brendan Swain, Lucas Paulo de Lima Camillo, bioRxiv preprint, 21 October 2025.

The rebuttal to B3: argues that the negative result is driven by metric calibration, and
that on appropriately calibrated metrics the deep models do separate from uninformative
baselines.

**Difference.** IHC-FM does not take a side in this dispute; it takes the *methodological*
lesson that both sides share — the choice of metric can reverse a ranking — and makes that
falsifiable in our own results. We measure a concrete reversal: ranking the baseline ladder
by CFM loss and by normalised energy distance gives **different orders**, and on the
coupling comparison the exact-assignment coupling wins on CFM loss (0.8600) while being
*worse than no change* on the distributional metric (1.1188). We therefore fix the
distributional metric as the only ranking key in code, and report the divergence rather
than asserting a preference. This is a stronger position than either paper's, because it is
measured on the same models rather than argued.

---

## C. Learned intervention fields with Lie-geometric structure

### C1. Latent Confounded Causal Discovery via Lie Bracket Geometry
`arXiv:2606.19610` — Sridhar Mahadevan. First release 17 June 2026.

Two algorithms for causal discovery under latent confounding. **BRIDGE** combines a
density-ratio or transport engine with a geometric screen based on bracket residuals and
passes retained arrows to a downstream discovery method; **SKFM** (Spectral Kernel Flow
Matching) amortises intervention *response fields* and summarises residual nonclosure via a
spectral subspace. The paper's own stated scope is single-node intervention targets, and it
frames the geometry as a diagnostic and candidate generator rather than a full
identification result.

**Difference — stated carefully, because the priority here is not ours.** Learned
intervention response fields with Lie-geometric structure are **this line's contribution,
not ours**; IHC-FM does not introduce learned Lie-geometric intervention fields and does not
claim to. Three concrete separations: (i) *task* — BRIDGE/SKFM recover graph structure under
latent confounding, whereas IHC-FM predicts the distributional response to an unseen
*intervention set* and never estimates a graph; (ii) *arity* — the cited scope is
single-node interventions, while IHC-FM's entire object is the `k >= 2` composition, with an
interaction branch that vanishes bit-exactly on singletons; (iii) *symmetric vs
antisymmetric* — bracket-based geometry uses the antisymmetric part `[X, Y]`, which is
already a tensor and never encounters the transformation defect, whereas the object carrying
the *magnitude* of a composition is the symmetric part `S = DX[Y] + DY[X]`, which is not a
tensor (its defect is exactly `2 D2psi[X, Y]`, measured 2.594e-14 against the predicted
identity). Our Proposition 2 concerns that symmetric part and the forced coefficient in
`S + 2 Gamma`.

---

## D. What IHC-FM retracts from its own earlier draft

Recorded here because the related-work positioning changed as a result, and because the
retraction is part of the paper's argument rather than an erratum.

An earlier draft of this project (working title *"Compose the Generators, Not the
Displacements"*) centred a **connection-corrected pairwise coupling term** `beta * S^{grad}`
added to the instantaneous velocity, justified by the argument that `S` is the leading term
of joint-minus-additive behaviour. Our own diagnosis shows the premise proves the opposite
of the conclusion: integrating the *summed* field already produces `S` for free,

```
flow_{X+Y}^T(z) - [ z + (flow_X^T - z) + (flow_Y^T - z) ] = (T^2/2) S + O(T^3),
```

measured relative error 1.610e-3 at `T = 0.005` with log-log convergence slope 0.99971. A
hand-added `beta S` in the velocity is therefore a **second copy** of the same kinematic
tensor, entering one order lower in `T`: the ratio of the two copies is `2 beta / T`, i.e.
40x at `T = 0.05` and 400x at `T = 0.005`, and their directions are collinear (minimum
cosine 0.99999790). The term supplies no new direction and dominates at small exposure.

Consequently IHC-FM replaces it with a **non-kinematic, singleton-vanishing, low-rank**
interaction branch, and retains the connection coupling as an **ablation arm** — it is the
correct control for the claim that the new branch does work the kinematic term cannot. No
external citation is affected; the change is internal, and Proposition 2 survives as a
statement about the tensoriality of `S + 2 Gamma` (residual 2.971e-15, with the wrong-sign
control failing at 5.335e-2 at every one of 9 seeds).

---

## E. Data sources (validation case studies only)

### E1. Trellis tree-based analysis reveals stromal regulation of patient-derived organoid drug responses
`doi:10.1016/j.cell.2023.11.005` — María Ramos Zapatero, Alexander Tong,
James W. Opzoomer, Rhianna O'Sullivan *et al.* (13 authors; last author Christopher J.
Tape), *Cell* **186**(25):5606–5619.e24, December 2023.

Mass-cytometry screen of patient-derived colorectal organoids under single agents and drug
combinations, with matched controls per condition — the source of our organoid
leave-one-combination-out case study (666 populations, 5 folds, PCA fitted on 60,000
control cells only, 8 components, cumulative EVR 0.690).

**Difference.** This is a data source, not a method comparison. IHC-FM makes no claim about
the Trellis analysis; we use the screen because it contains a genuine combination ladder
with a held-out triple.

### E2. Order-flow data

Publicly downloadable Binance `BTCUSDT` aggregated-trade archives for 2–5 June 2025
(`BTCUSDT-aggTrades-2025-06-{02,03,04,05}.zip`), aggregated into 34,545 ten-second windows
across 48 blocks. No paper is cited for this because none is used: the data are raw exchange
archives, and the state features (log return, realised volatility, flow imbalance, log
intensity, size dispersion, sign autocorrelation) are defined in
`src/composefm/data_finance.py`.

**Framing constraint, stated in the results file itself.** This case study is
**observational order-flow conditioning, not causal intervention estimation**. Order flow is
not assigned; it is observed. We therefore make no causal claim on this axis, and the
exposure-extrapolation arm is reported as a diagnosed distribution shift (oracle in-split vs
transferred ratio 1.799 / 1.803 / 1.868 / 2.802 across the four splits) rather than as an
extrapolation test.

---

## F. Citation ledger

Every identifier that appears in `paper/main.tex`, with the API that confirmed it.

| key | identifier | confirmed via | first release (from API) | venue |
|---|---|---|---|---|
| `lipman2022fm` | arXiv:2210.02747 | arXiv API | 2022-10-06 | not verified here |
| `liu2022rectified` | arXiv:2209.03003 | arXiv API | 2022-09-07 | not verified here |
| `tong2023otcfm` | arXiv:2302.00482 | arXiv API | 2023-02-01 | not verified here |
| `atanackovic2024mfm` | arXiv:2408.14608 | arXiv API | 2024-08-26 | not verified here |
| `kapusniak2024metricfm` | arXiv:2405.14780 | arXiv API | 2024-05-23 | not verified here |
| `zweig2025egfm` | arXiv:2509.25230 | arXiv API | 2025-09-25 | not verified here |
| `mahadevan2026bridge` | arXiv:2606.19610 | arXiv API | 2026-06-17 | not verified here |
| `lotfollahi2023cpa` | doi:10.15252/msb.202211517 | Crossref | 2023-05-08 | Mol. Syst. Biol. **19**(6) |
| `roohani2023gears` | doi:10.1038/s41587-023-01905-6 | Crossref | 2023-08-17 | Nat. Biotechnol. **42**(6):927–935 |
| `ahlmanneltze2025linear` | doi:10.1038/s41592-025-02772-6 | Crossref | 2025-08 | Nat. Methods **22**(8):1657–1661 |
| `miller2025rebuttal` | doi:10.1101/2025.10.20.683304 | Crossref | 2025-10-21 | bioRxiv preprint |
| `ramoszapatero2023trellis` | doi:10.1016/j.cell.2023.11.005 | Crossref | 2023-12 | Cell **186**(25):5606–5619.e24 |

**On the empty venue cells.** The arXiv API returns identifier, title, authors and release
date; it does not return the peer-reviewed venue. Several of these arXiv entries are widely
cited with conference venues, but that attribution was *not* confirmed by any API call in
this session, so it is not asserted here and `paper/main.tex` cites those seven works by
arXiv identifier and release year only. Anyone preparing camera-ready copy should fill the
venue field from the publisher record; do not copy a venue from memory into the bibliography.
The Crossref rows carry full journal metadata because Crossref returned it.
