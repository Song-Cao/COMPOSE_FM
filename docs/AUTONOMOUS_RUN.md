# Autonomous run plan — started 2026-09-05 ~23:30 London

User asleep ~8 h. Standing authority: run steps 3-5 without stopping. Decisions already
taken by the user, do not re-litigate:

* **Framing (Rule 0)**: flow-matching METHOD paper; finance + bio are case studies.
  Method first, no domain terms in the model definition.
* **Two datasets**: finance FIRST (significance), organoid drug-screen SECOND.
* **Organoid fallback**: up to ~3 h of effort permitted on the 25.7 GB archive if it
  misbehaves; **total budget extended to < 15 h** (so ~14:00 London 6 Sep, not 09:00).
* **GPU**: only if needed, only `runpod-fm`, <= 25 GPU-h, and SAVE COST — prefer the
  smallest run that supports the claim. CPU is fine for d<=16 latents.
* **No more stringent toy gates.** Toy runs are sanity checks now.
* **Model detail matters** — make the architecture more expressive where defensible.
* GitHub: user is creating an empty `COMPOSE-FM` repo; detect and push.
* Finance backups if needed: FI-2010 (etsin.fairdata.fi, granted), NASDAQ, Qlib CSI300.

## Order of work

1. Finance case study end-to-end (data ready: 4 symbol-days Binance aggTrades, ~3.5 M
   trades; channels = side x size-class; exposure = executed quantity).
2. Organoid case study (composition ladder: 6 singles, 4 pairs, 1 triple + doses).
3. Shared evaluation: same operator, same ablations, both domains.
4. LaTeX draft in NeurIPS workshop format, method-first.

## Hard rules for autonomous operation

* Never fabricate a number. If a run fails, report the failure.
* Every claim in the draft traces to a file in `results/`.
* Keep committing; the log is the audit trail.
