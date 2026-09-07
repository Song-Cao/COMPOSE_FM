"""Consolidated metric suite over the SAVED organoid predictions. NOTHING IS RETRAINED.

`results/loco_screen_predictions.npz` holds the (pred, true, ctrl) clouds for every scored
population of the organoid screen, for three arms (IHC-FM, PerturbedMean, NoChange) across
the five leave-one-combination-out folds and the three leave-replicate-out folds at
code_dim 0 and 16. This script re-scores those stored clouds through the FULL metric suite
in `src/composefm/metrics.py` -- absolute and normalised energy distance with the
denominator, MMD, sliced Wasserstein, Pearson/Spearman/R^2/cosine on the mean shift,
top-marker overlap, the residualised metric, and grouped-bootstrap CIs.

WHY RE-SCORE STORED PREDICTIONS RATHER THAN RE-RUN. The screen already reported normalised
ED; the other metrics were never computed on it. Re-fitting to obtain them would change the
numbers for reasons unrelated to the metric (neural arms have measured across-seed std
0.27-0.37), so the saved clouds are the only way to add metrics while keeping every
existing number in the paper exactly reproducible. Every metric here is a pure function of
the stored arrays.

METRIC POLICY. The suite is REPORTED, not ranked on wholesale: the project's selection
metric is normalised ED with its no-change denominator always alongside. The extra metrics
exist to expose failure modes that a single distance hides -- a good cosine with collapsed
magnitude (`l2_ratio`), a good Pearson with the wrong magnitude (`r2_delta` punishes it),
and above all the RESIDUALISED metric, which asks whether the model beats "predict the
average observed response" rather than merely beating no change. That last one is the
question a composition paper must actually answer.

The independent unit is the replicate record (`unit` in the stored metadata), and every CI
here resamples units, never rows. Rows within a unit share a control draw and a plate, so
resampling rows would treat correlated reads as independent tests.

USAGE
    PYTHONPATH=src python experiments/metrics_full.py           # full
    PYTHONPATH=src python experiments/metrics_full.py --smoke   # loco folds only
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np

import sys

if "src" not in sys.path:
    sys.path.insert(0, "src")

from composefm import metrics as M                                     # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "results")
NPZ = os.path.join(RESULTS, "loco_screen_predictions.npz")

# The metrics summarised per unit. Every one is already implemented in metrics.py.
METRICS = ("ed_absolute", "ed_normalised", "ed_denominator",
           "sw_pred", "sw_denominator", "sw_normalised",
           "mmd_pred", "mmd_denominator",
           "pearson_delta", "spearman_delta", "r2_delta", "cosine_delta",
           "top_overlap", "l2_ratio")

ARMS = ("ihcfm", "perturbed_mean", "no_change")
N_BOOT = 2000
EVAL_MAX_N = 400          # the cloud cap the screen itself used
N_SLICES = 64             # sliced-Wasserstein directions, as in the screen


def load_records(path: str = NPZ) -> dict:
    """Group the stored arrays into metrics-ready records, keyed by split then arm.

    Key layout in the npz is "<split>|<arm>|<i>|<field>" with fields pred/true/ctrl/meta,
    and meta is (unit, condition, dose, order, pop_index). Records are rebuilt by that
    explicit key -- never by array order -- so a record's arm, split and unit are read
    from the file rather than inferred.
    """
    z = np.load(path, allow_pickle=True)
    grouped: dict = defaultdict(lambda: defaultdict(dict))
    for key in z.keys():
        split, arm, i, field = key.split("|")
        grouped[split][arm].setdefault(int(i), {})[field] = key
    out: dict = {}
    for split, arms in grouped.items():
        out[split] = {}
        for arm, items in arms.items():
            recs = []
            for i in sorted(items):
                sl = items[i]
                if not {"pred", "true", "ctrl", "meta"} <= set(sl):
                    continue
                meta = z[sl["meta"]]
                recs.append(dict(
                    unit=str(meta[0]), condition=str(meta[1]),
                    dose=int(meta[2]), order=int(meta[3]), pop_index=int(meta[4]),
                    pred=np.asarray(z[sl["pred"]], dtype=np.float64),
                    true=np.asarray(z[sl["true"]], dtype=np.float64),
                    ctrl=np.asarray(z[sl["ctrl"]], dtype=np.float64)))
            out[split][arm] = recs
    return out


def score_split(recs_by_arm: dict, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Full suite for one split: per-arm summaries, paired contrasts, residualised report."""
    out: dict = {"arms": {}}
    rows_by_arm = {}
    for arm, recs in recs_by_arm.items():
        if not recs:
            continue
        rows = M.score_records(recs, seed=seed, max_n=EVAL_MAX_N, n_slices=N_SLICES)
        rows_by_arm[arm] = rows
        summ = M.summarise_per_unit(rows, metrics=list(METRICS), n_boot=n_boot, seed=seed)
        cell = dict(summary=summ, n_records=len(rows),
                    n_units=int(len({r["unit"] for r in rows})),
                    denominator_check=M.flag_denominator_collapse(rows))
        if len(recs) >= 2:
            res = M.residualised_report(recs, max_n=EVAL_MAX_N, seed=seed)
            dl = res.pop("distribution_level", [])
            vals = np.array([r["ed_vs_mean_response"] for r in dl], dtype=np.float64)
            grp = np.array([str(r["unit"]) for r in dl])
            ok = np.isfinite(vals)
            cell["residualised"] = dict(
                shift_level=res["shift_level"],
                mean_response_shift=res["mean_response_shift"],
                # "beats predicting the AVERAGE observed response" -- a strictly harder
                # bar than beating no change, and the one a composition claim needs.
                ed_vs_mean_response=(M.grouped_bootstrap_ci(vals[ok], grp[ok],
                                                            n_boot=n_boot, seed=seed)
                                     if ok.sum() else None),
                interpretation=("ed_vs_mean_response < 1 means the arm beats predicting "
                                "the mean observed response; the ordinary no-change ratio "
                                "can be well below 1 while this is above 1"))
            if cell["residualised"]["ed_vs_mean_response"]:
                for kk in ("group_means", "groups"):
                    cell["residualised"]["ed_vs_mean_response"].pop(kk, None)
        out["arms"][arm] = cell

    # ---- paired contrasts, aligned by (unit, condition, dose) -----------------------
    out["paired"] = {}
    for a, b in (("ihcfm", "perturbed_mean"), ("ihcfm", "no_change"),
                 ("perturbed_mean", "no_change")):
        if a not in rows_by_arm or b not in rows_by_arm:
            continue
        ka = {(r["unit"], r["condition"], r.get("dose"), r.get("pop_index")): r
              for r in rows_by_arm[a]}
        kb = {(r["unit"], r["condition"], r.get("dose"), r.get("pop_index")): r
              for r in rows_by_arm[b]}
        shared = sorted(set(ka) & set(kb))
        if len(shared) < 2:
            continue
        block = {}
        for m in ("ed_normalised", "ed_absolute", "sw_normalised", "cosine_delta",
                  "r2_delta"):
            va = np.array([ka[s][m] for s in shared], dtype=np.float64)
            vb = np.array([kb[s][m] for s in shared], dtype=np.float64)
            gg = np.array([str(s[0]) for s in shared])
            ok = np.isfinite(va) & np.isfinite(vb)
            if ok.sum() < 2:
                continue
            pd = M.paired_group_difference(va[ok], vb[ok], gg[ok], n_boot=n_boot,
                                           seed=seed, name_a=a, name_b=b)
            pd.pop("per_unit_diff", None)
            pd.pop("units", None)
            block[m] = pd
        out["paired"][f"{a}_vs_{b}"] = dict(
            n_paired_records=len(shared), metrics=block,
            sign_convention="mean_diff = a - b; NEGATIVE means a is better on a loss metric",
            alignment="paired by (unit, condition, dose, pop_index) identity, not list order")
    return out


def pool_loco(scored: dict, records: dict, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Pool the five leave-one-combination-out folds into one paired comparison.

    Pooling is over POPULATIONS across folds; each population was scored by a model that
    never saw its intervention combination, so the pooled set is a legitimate held-out
    sample. The unit is still the replicate record.
    """
    loco = [s for s in records if s.startswith("loco_")]
    pooled_rows: dict = {a: [] for a in ARMS}
    for split in loco:
        for arm in ARMS:
            recs = records[split].get(arm, [])
            if not recs:
                continue
            rows = M.score_records(recs, seed=seed, max_n=EVAL_MAX_N, n_slices=N_SLICES)
            for r in rows:
                r["split"] = split
            pooled_rows[arm] += rows
    out = dict(folds=sorted(loco), arms={})
    for arm, rows in pooled_rows.items():
        if not rows:
            continue
        out["arms"][arm] = dict(
            summary=M.summarise_per_unit(rows, metrics=list(METRICS), n_boot=n_boot,
                                         seed=seed),
            n_records=len(rows),
            n_units=int(len({(r["split"], r["unit"]) for r in rows})))
    out["paired"] = {}
    for a, b in (("ihcfm", "perturbed_mean"), ("ihcfm", "no_change")):
        ka = {(r["split"], r["unit"], r["condition"], r.get("dose")): r
              for r in pooled_rows[a]}
        kb = {(r["split"], r["unit"], r["condition"], r.get("dose")): r
              for r in pooled_rows[b]}
        shared = sorted(set(ka) & set(kb))
        if len(shared) < 2:
            continue
        block = {}
        for m in ("ed_normalised", "ed_absolute", "sw_normalised", "mmd_pred",
                  "cosine_delta", "r2_delta", "pearson_delta", "spearman_delta",
                  "top_overlap", "l2_ratio"):
            va = np.array([ka[s][m] for s in shared], dtype=np.float64)
            vb = np.array([kb[s][m] for s in shared], dtype=np.float64)
            # unit = (fold, replicate record): a replicate id repeats across folds and
            # those are NOT the same independent unit
            gg = np.array([f"{s[0]}::{s[1]}" for s in shared])
            ok = np.isfinite(va) & np.isfinite(vb)
            if ok.sum() < 2:
                continue
            pd = M.paired_group_difference(va[ok], vb[ok], gg[ok], n_boot=n_boot,
                                           seed=seed, name_a=a, name_b=b)
            pd.pop("per_unit_diff", None)
            pd.pop("units", None)
            block[m] = pd
        out["paired"][f"{a}_vs_{b}"] = dict(n_paired_records=len(shared), metrics=block)
    out["note"] = ("pooled over held-out combination folds; unit = (fold, replicate "
                   "record) because a replicate id recurs across folds and those are not "
                   "the same independent unit")
    return out


def main(smoke: bool = False, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    t0 = time.time()
    records = load_records()
    splits = sorted(records)
    if smoke:
        splits = [s for s in splits if s.startswith("loco_")][:2]
        n_boot = 200
    print(f"=== metrics_full: {len(splits)} splits, arms "
          f"{sorted({a for s in splits for a in records[s]})} ===", flush=True)

    per_split = {}
    for s in splits:
        per_split[s] = score_split(records[s], n_boot=n_boot, seed=seed)
        n = {a: len(records[s][a]) for a in records[s]}
        print(f"  {s}: {n}", flush=True)

    out = dict(
        meta=dict(
            source=os.path.relpath(NPZ, HERE),
            note="re-scored from SAVED predictions; no model was retrained",
            metrics=list(METRICS),
            n_boot=n_boot, eval_max_n=EVAL_MAX_N, n_slices=N_SLICES, seed=seed,
            unit_definition=("the replicate record; every CI resamples units, never rows, "
                             "because rows within a unit share a control draw and a plate"),
            selection_policy=("normalised ED with its no-change denominator is the "
                              "project's selection metric; the rest of the suite is "
                              "reported to expose failure modes a single distance hides"),
            residualised_note=("ed_vs_mean_response is the harder bar: it asks whether the "
                               "arm beats predicting the AVERAGE observed response, not "
                               "merely no change"),
            smoke=bool(smoke)),
        per_split=per_split)
    if not smoke:
        out["pooled_loco"] = pool_loco(per_split, records, n_boot=n_boot, seed=seed)
    out["meta"]["wall_clock_s"] = time.time() - t0

    def _j(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, dict):
            return {str(k): _j(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_j(v) for v in o]
        return o

    path = os.path.join(RESULTS, f"metrics_full{'_smoke' if smoke else ''}.json")
    with open(path, "w") as fh:
        json.dump(_j(out), fh, indent=1)
    print(f"\nwrote {path}  ({time.time() - t0:.0f}s)", flush=True)
    if "pooled_loco" in out:
        for arm in ARMS:
            s = out["pooled_loco"]["arms"].get(arm, {}).get("summary", {})
            e = s.get("ed_normalised")
            if e:
                print(f"  pooled {arm:<16} nED {e['point']:.4f} "
                      f"[{e['lo']:.4f}, {e['hi']:.4f}]  n_units={e['n_groups']}",
                      flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    main(smoke=a.smoke)
