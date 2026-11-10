"""Is `selected8 > random8` an effect of selection, or of an unequal redraw?

`picks_ablation` gives the two arms different redraw schedules. `random8` is
drawn afresh for every evaluation seed, so its reported spread contains its own
sampling noise. `selected8` is selected ONCE per (dataset, K) cell and reused
across all 20 seeds, so the variance of selection never enters. The contrast is
therefore between a noisy arm and a frozen one, and the frozen one is ours.

Two defects are tangled in that setup and this script separates them:

  pub    ranking supports are seeds 0..4 -- the SAME draws the evaluation uses,
         so the selected streams have already seen the support they are scored
         on. Selection runs once per cell.        (reproduces the published run)
  disj   ranking supports moved to SEL_SEED_OFFSET+0..4, disjoint from the
         evaluation draws. Still once per cell.   (pub -> disj = the leak)
  sym    selection repeated for every evaluation seed, on its own ranking
         supports AND its own K-shot val subsample, so `selected8` carries the
         same per-seed redraw `random8` has.      (disj -> sym = the asymmetry)

All three share one evaluation pass: base/last1/fixed8/random8 are fitted once
per seed and the three selected8 variants differ only in which streams they
read. A difference among them is attributable to the selection schedule alone.

Ranking uses the batched solver (`fastprobe.stream_scores`), which is the same
convex objective `layer_probe.pick_streams` fits sequentially -- `--verify`
checks the `pub` picks against the stored picks of a previous run before
spending anything.

    python -m icd_pauc.picks_symmetric --datasets ham --Ks 16 --n_seeds 3
    python -m icd_pauc.picks_symmetric --all --tag sym10 --n_seeds 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from icd_pauc.model_zoo import load_zoo, DATASETS
from icd_pauc.deep_cache import POOLS, N_BLOCKS
from icd_pauc.subsample import build_support
from icd_pauc.train_icd_pauc import fit_anchor, set_seed
from icd_pauc.metrics_pauc import evaluate_all
from icd_pauc.maf_head import zscore
from icd_pauc.layer_probe import load_split
from icd_pauc.fastprobe import stream_rows, stream_scores
from icd_pauc.deepfuse_v2 import SEL_SEED_OFFSET, VAL_SUBSAMPLE_SEED, all_stats
from icd_pauc.picks_ablation import (ALL10, FIXED8, LAST1, ALL_STREAMS,
                                     random8, fuse_arm)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS = os.path.join(_HERE, "..", "..", "results")

NAMES = [(p, b) for p in POOLS for b in range(1, N_BLOCKS + 1)]
SCHEDULES = ("pub", "disj", "sym")
ARMS = ("base", "last1", "fixed8", "random8",
        "selected8_pub", "selected8_disj", "selected8_sym")


def select(dtr, ytr, dva, yva, st_tr, st_va, K, C, mal,
           sup_seeds, val_seed, topn=8, rank_metric="bacc"):
    """Rank the 96 streams on a K-shot val subsample and return the top-n."""
    sel_sup = [build_support(ytr, "kshot", K, seed=s) for s in sup_seeds]
    ysup = torch.stack([ytr[si] for si in sel_sup])
    va_idx = build_support(yva, "kshot", K, seed=val_seed)
    Zsup = [torch.stack([stream_rows(dtr, p, b, si, *st_tr[(p, b)])
                         for si in sel_sup]) for p, b in NAMES]
    Zev = [stream_rows(dva, p, b, va_idx, *st_va[(p, b)]) for p, b in NAMES]
    sc, _, _ = stream_scores(Zsup, Zev, ysup, yva[va_idx], C, mal,
                             rank_metric, chunk_streams=16)
    del Zsup, Zev
    return [NAMES[i] for i in np.argsort(-sc)[:topn]]


def schedule_seeds(schedule, sd, sel_seeds):
    """(ranking support seeds, val subsample seed) for one evaluation seed.

    pub/disj do not depend on sd -- their selection is computed once per cell.
    """
    if schedule == "pub":
        return list(range(sel_seeds)), VAL_SUBSAMPLE_SEED
    if schedule == "disj":
        return [SEL_SEED_OFFSET + s for s in range(sel_seeds)], VAL_SUBSAMPLE_SEED
    base = SEL_SEED_OFFSET + sd * sel_seeds
    return [base + s for s in range(sel_seeds)], VAL_SUBSAMPLE_SEED + sd


def run(datasets, Ks, n_seeds, sel_seeds, tag, verify_against=None):
    seeds = list(range(n_seeds))
    rows, picks_log = [], []
    t0 = time.time()
    out = os.path.join(RESULTS, f"picks_symmetric_{tag}.json")

    for ds in datasets:
        cfg = DATASETS[ds]
        C, mal = cfg["nb_classes"], cfg["malignant"]
        dtr, ytr = load_split(ds, "train")
        dva, yva = load_split(ds, "val")
        dte, yte = load_split(ds, "test")
        yte_np = yte.numpy()
        st_tr, st_va = all_stats(dtr), all_stats(dva)
        base = zscore(load_zoo(ds, "panderm_lp", "train")["embeddings"])
        base_te = zscore(load_zoo(ds, "panderm_lp", "test")["embeddings"])

        for K in Ks:
            # the two once-per-cell schedules
            picks = {}
            for sch in ("pub", "disj"):
                sup_seeds, val_seed = schedule_seeds(sch, 0, sel_seeds)
                picks[sch] = select(dtr, ytr, dva, yva, st_tr, st_va,
                                    K, C, mal, sup_seeds, val_seed)
            if verify_against:
                _verify(verify_against, ds, K, picks["pub"])

            acc = {a: {"AUROC_bin": [], "bacc": []} for a in ARMS}
            for sd in seeds:
                si = build_support(ytr, "kshot", K, seed=sd)
                vi = build_support(yva, "kshot", K, seed=sd + 1000)

                # selection redrawn for THIS evaluation seed
                sup_seeds, val_seed = schedule_seeds("sym", sd, sel_seeds)
                picks["sym"] = select(dtr, ytr, dva, yva, st_tr, st_va,
                                      K, C, mal, sup_seeds, val_seed)
                picks_log.append({"dataset": ds, "K": K, "seed": sd,
                                  "picks": [f"{p}{b}" for p, b in picks["sym"]]})

                set_seed(sd)
                h = fit_anchor(base[si].to(DEVICE), ytr[si].to(DEVICE), C,
                               DEVICE, feat_dim=base.shape[1])
                with torch.no_grad():
                    p = F.softmax(h(base_te.to(DEVICE)), 1).cpu().numpy()
                m = evaluate_all(p, yte_np, mal)
                for k in acc["base"]:
                    acc["base"][k].append(m[k])

                for arm, pk in (("last1", LAST1),
                                ("fixed8", FIXED8),
                                ("random8", random8(sd)),
                                ("selected8_pub", picks["pub"]),
                                ("selected8_disj", picks["disj"]),
                                ("selected8_sym", picks["sym"])):
                    m = fuse_arm(pk, dtr, dva, dte, ytr, yva, si, vi,
                                 C, mal, yte_np, sd)
                    for k in acc[arm]:
                        acc[arm][k].append(m[k])

            for arm in ARMS:
                rows.append({
                    "dataset": ds, "K": K, "arm": arm, "n_seeds": n_seeds,
                    "sel_seeds": sel_seeds,
                    "picks": ([f"{p}{b}" for p, b in picks[arm.split("_")[1]]]
                              if arm.startswith("selected8_") and
                              arm != "selected8_sym" else
                              "per-seed" if arm in ("random8", "selected8_sym")
                              else None),
                    "bacc": float(np.mean(acc[arm]["bacc"])),
                    "AUROC_bin": float(np.mean(acc[arm]["AUROC_bin"])),
                    "bacc_seeds": [float(v) for v in acc[arm]["bacc"]],
                    "AUROC_bin_seeds": [float(v) for v in acc[arm]["AUROC_bin"]],
                })

            b = {a: np.mean(acc[a]["bacc"]) for a in ARMS}
            print(f"[{time.time()-t0:7.1f}s] {ds:<14} K={K:<3} bacc  "
                  f"base {b['base']:.4f}  fixed8 {b['fixed8']:.4f}  "
                  f"random8 {b['random8']:.4f}  "
                  f"sel/pub {b['selected8_pub']:.4f}  "
                  f"sel/disj {b['selected8_disj']:.4f}  "
                  f"sel/sym {b['selected8_sym']:.4f}", flush=True)

            os.makedirs(RESULTS, exist_ok=True)
            with open(out, "w") as f:
                json.dump(rows, f, indent=1)
            with open(out.replace(".json", "_picks.json"), "w") as f:
                json.dump(picks_log, f, indent=1)

    print(f"wrote {out}  ({len(rows)} rows, {time.time()-t0:.0f}s)")
    return rows


def _verify(path, ds, K, got):
    """The batched ranker must reproduce the sequential one it replaces."""
    with open(path) as f:
        ref = json.load(f)
    for r in ref:
        if r["dataset"] == ds and r["K"] == K and r["arm"] == "selected8":
            want = [tuple(x) for x in r["picks"]]
            same = len(set(want) & set(map(tuple, got)))
            flag = "OK" if same == len(want) else f"DIFF {same}/{len(want)}"
            print(f"  [verify] {ds} K={K} pub picks vs stored: {flag}", flush=True)
            return
    print(f"  [verify] {ds} K={K}: no stored reference", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ham"])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--Ks", nargs="+", type=int, default=[1, 4, 16])
    ap.add_argument("--n_seeds", type=int, default=20)
    ap.add_argument("--sel_seeds", type=int, default=5)
    ap.add_argument("--tag", default="smoke")
    ap.add_argument("--verify", default=None,
                    help="path to a picks_ablation_*.json to check pub picks against")
    a = ap.parse_args()
    run(ALL10 if a.all else a.datasets, a.Ks, a.n_seeds, a.sel_seeds,
        a.tag, a.verify)
