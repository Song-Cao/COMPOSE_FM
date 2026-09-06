# COMPOSE-FM — Covariant Composition of Intervention Flows

A flow-matching method for predicting the effect of **combined** interventions, and the
geometric argument for why the usual approach is wrong.

## The claim in one paragraph

Latent perturbation models almost universally **add displacements**: each intervention is a
vector, a combination is the sum. That is not a chart-independent operation — under a
diffeomorphism of the latent space a sum of displacements picks up an $O(\|d\|^2)$ error,
so the prediction depends on an arbitrary coordinate choice. A sum of **vector fields**
does not. Measured: summing fields reproduces the reference to `5.5e-17`, summing
displacements errs by `1.7e-3`. COMPOSE-FM therefore composes learned *generators*, with a
state-dependent contraction gate, a connection-corrected pairwise coupling term, and
exposure entering as integration time.

## Results

| domain | held-out axis | displacement (CPA-class) | COMPOSE-FM | gain |
|---|---|---|---|---|
| Order flow (BTCUSDT, 34,545 windows) | all 4 channels active | 2.008 | **0.691** | 2.9× |
| Organoid drug screen (666 populations) | triple C+S+F, never seen | 2.233 | **0.741** | 3.0× |

Normalised energy distance, `ED(pred,true)/ED(no-change,true)`; 1.0 = no better than
predicting no change, lower is better. Baseline is a trained model of matched trunk width,
depth and optimisation budget, differing only in the composition rule.

The exposure semigroup holds to solver accuracy (`2.3e-7`) by construction, not by fitting,
because each generator is autonomous and exposure is integration time.

## What does *not* work (documented, not hidden)

- A **global scalar** contraction helps in neither domain — state dependence is what matters.
- The finance **exposure axis** is a distribution shift, not an extrapolation: an oracle
  fitted on the high-exposure split scores 0.287, the same oracle transferred from training
  scores 2.788. No architecture bridges that.
- **Saturation** helps only where there is exposure range to extrapolate over; it is
  neutral-to-harmful on the organoid ladder.
- **Population conditioning** (the Meta Flow Matching mechanism) improves training slightly
  and damages the held-out triple; off by default.

See `docs/GATE_LOG.md` for every result with its pre-registered criteria.

## Layout

```
src/composefm/        compose.py (the operator), data_finance.py, models.py, toy.py
experiments/          finance_case_study.py, organoid_case_study.py,
                      seed_robustness.py, gate_a*/gate_b* (covariance + toy checks)
results/              *.json per run, figures/, logs/, tables/
docs/                 PROJECT_RULES.md (framing), GATE_LOG.md (append-only record)
paper/                main.tex (NeurIPS workshop format), figures/
```

## Reproducing

```bash
python3 experiments/finance_case_study.py  --steps 2000   # ~25 min, CPU
python3 experiments/organoid_case_study.py --steps 2500   # ~12 min, CPU
python3 experiments/seed_robustness.py                    # 3 seeds, both domains
```

Everything above runs on CPU. Data: Binance public `aggTrades` dumps (finance) and the
Trellis organoid screen (Ramos Zapatero et al., *Cell* 186(25) 2023,
doi:10.1016/j.cell.2023.11.005) in the preprocessed release from Meta Flow Matching
(Atanackovic et al., ICLR 2025, arXiv:2408.14608).