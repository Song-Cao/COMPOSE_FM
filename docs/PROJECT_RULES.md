# PROJECT RULES — standing reference, read at the start of every work session

Authored from the user's instructions (2026-09-05). These are **strict rules for the whole
life-cycle**. Do not relax them. If a decision seems to require breaking one, the rule wins.

---

## 1. Central purposes — IN NO CONDITION DEVIATE

1. **NeurIPS/ICLR-main-level significance** of the idea.
2. **NeurIPS workshop scope** for the execution (short paper).
3. **Methodological flow-matching innovation.** A novel FM method is *absolutely
   necessary*. Never drop or forget this.
4. Applications in **both** (a) one biological task and (b) one quantitative-finance task.

## 2. Pipeline

1. Method prototype testing / proof of concept.
2. **If the gate fails → adjust the proposal with methodological creativity and reasoning,
   then repeat step 1.** The gate definition may itself change as the method changes.
   Iterate until a gate passes.
3. Generate the full evaluation plan: datasets (≥1 bio + ≥1 quant finance), SOTA
   comparisons, ablations.
4. Run the full evaluation plan; adjust model/plan mid-flight if needed; collect everything
   the paper needs.
5. Paper drafting — LaTeX first; format, structure and tone must follow NeurIPS workshop
   convention (read the official guideline and a few real workshop papers first).

## 3. The failure mode to avoid — THE most important rule

> Propose a significant innovation → set a gate → gate fails → **limit the claim or scope
> without seeking alternative innovations** → after a few iterations it has become a pure
> statistical / analytical / failure-mode study → **boring.**

**This has already happened once in this project** (GLIF → COUPLE → covariance diagnostic →
"analysis paper"), and the user caught it. Guard actively against it.

**The ONLY correct strategy:**

> Propose a significant innovation → set a gate (**not only a statistical test — also run a
> prototype model on a toy dataset, quickly**) → gate fails → **seek alternative solutions
> creatively**: reasonable mathematical constructions, architecture tweaks, bridging
> solutions from a different but related field, or even random creativity → don't stop
> iterating until the idea works or time is up.

Rigour is good, but **never let rigour shrink the project into something safe and
uninteresting.** A negative result is not a deliverable here.

## 4. Deadline

- **Steps 1–2 (proposal formation): 12 hours total**, including thinking, searching, CPU and
  GPU time.
- **Step 4 (full evaluation on GPU): 12 hours.**
- **Hard stop: 09:00 on 6 September 2026, London time.**
- If the idea cannot be made to work by the deadline: **say so and stop.** If a gate passes
  at any time: proceed to full plan → evaluation → paper.

## 5. Communication

- Run the whole cycle **in one go**. Ask for permissions/decisions **at the very beginning**,
  then run without stopping.
- If something needs a decision mid-flight and it is **not** "abort the project", **decide it
  yourself** from research and insight, and move on.

## 6. Compute and storage

- Local project dir: `/Users/songcao/workdir/FM_project` — all code, data, models, results,
  organised as a normal ML research repo.
- GitHub repo created and kept in sync with the local dir.
- **GPU only when GPU is needed**, and only on host **`runpod-fm`** (never another RunPod host).
- **Total GPU usage must never exceed 25 GPU-hours.**
- RunPod working dir: `/workspace/FM_project` on the network volume. Use it for cached
  results, checkpoints and data transfer — the pod will eventually be deleted.
- Verified runpod-fm facts: 1× A100-SXM4-80GB, driver 580.159.04, CUDA 13.0 driver-side,
  **no `nvcc` on PATH**, system python 3.12.3 with torch 2.8.0+cu128, `/workspace` = 202 T
  free MooseFS volume, **`/` is only 30 G** so keep envs/caches off it.

## 7. Active reference

Keep the research proposal and implementation plan up to date as things change
(`docs/PROPOSAL.md`, `docs/PLAN.md`). Record gate outcomes in `docs/GATE_LOG.md` with dates,
numbers, and the decision taken.

## 8. Creative-thinking reference

`docs/refs/FM_30_papers_digest.txt` — 30 SOTA FM method/architecture papers (Jan 2025 –
Sep 2026) with challenge → solution chains. Use for both creative-thinking patterns and
concrete inspiration. Directly relevant precedents already identified:

- **#03 "How to build a consistency model"** (arXiv:2505.18825) — Lagrangian / Eulerian /
  **semigroup**-based objectives for two-time flow maps, self-distillation without a teacher.
- **#18 Space Group Conditional Flow Matching** (arXiv:2509.23822) — require the **velocity
  field to commute with symmetry operations**; group-averaging symmetrisation.
- **#24 Multitask Learning with Stochastic Interpolants** (arXiv:2508.04605) — replace the
  scalar clock with **operator-valued** interpolants controlling different subspaces.
- **#17 Energy Guided Geometric Flow Matching** (arXiv:2509.25230) — user-supplied, read in
  full. Its conformal energy metric `G = γ + clip(λ exp(E_K))` **was implemented and tested
  earlier in this project, and did NOT clear its gate.** A KDE energy on Norman control
  cells (energy mean 7.38, sd 2.34; conformal factor f mean 2.378, range [2.000, 3.718])
  gave the closed-form Levi-Civita contraction
  `2Γ(X_p,X_q) = (∇log f·X_q)X_p + (∇log f·X_p)X_q − (X_p·X_q)∇log f`. Results:
  the covariant index beat the naive one (mean ρ 0.535 vs 0.472, Wilcoxon p=0.032) and beat
  a trivial `‖z‖` baseline (0.455), but **lost to the constant-generator ablation (0.570)**
  — and the sign test was null (56 improved / 50 worsened / 25 tied, p=0.63),
  so the gain was in magnitude, not frequency. **Gate 1 did not pass.** Artifacts:
  `covariance_sweep.csv`, `covariant_summary.csv`, `covariant_test.csv`.
  → Treat the metric as *available machinery whose value is unproven on real data*, NOT as a
  working component of the current design. It is optional in `couple.py` and must earn its
  place in an ablation before any paper claim rests on it.
- **#14 Lagrangian Flow Matching** (arXiv:2605.15419) — least-action path design.
- **#25 Pareto Optimal FM with Physics Constraints** (arXiv:2506.08604) — conflict-free
  multi-objective gradients when two losses fight.

## 9. Non-negotiables carried from earlier in the project

- Never claim to introduce learned Lie-geometric intervention fields — BRIDGE
  (arXiv:2606.19610v2) has them. Its scope limit (single-node interventions) is our lane:
  **composition**.
- Leakage-aware splits mandatory; never report all-genes Pearson.
- Linear/additive composition is the primary baseline, not a straw man.
