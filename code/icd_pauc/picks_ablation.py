"""Tier-0 ablation: does the SELECTION step earn its keep?

The locked main line claims "adaptive selection beats a fixed configuration".
`rank_fix_smoke.json` (3 datasets, smoke) hinted the opposite on balanced
accuracy: a hand-picked fixed set of 8 late streams was statistically
indistinguishable from the ranked top-8. That file compared against `rank1`
(single-draw ranking), not the 5-draw averaging the paper actually uses, and
covered 3 datasets. This script settles it at the paper's own scope.

Five arms. FOUR of them form a clean family; base does not:

  last1       cls pooling, block 24 only           1 stream, same head
  fixed8      cls pooling, blocks 17-24            a priori, zero selection cost
  random8     8 of 96 drawn uniformly, per seed    zero information, zero cost
  selected8   K-shot-budget ranked top-8           this paper's method
  base        last-layer CLS + LINEAR PROBE        (official default protocol)

Within {last1, fixed8, random8, selected8} everything downstream of the stream
set is shared: same support sets, same seeds, same LDCAdapterHead(
learn_alpha=False), same fit_adam(val_select=False), same metrics. A difference
among those four is attributable to the stream set and nothing else.

base is NOT in that family. It is a linear probe on one stream, so it differs in
head architecture and parameter count too. Any contrast against base therefore
confounds the stream set with the head, and CANNOT be read as "what fusion
contributes" -- an earlier version of this docstring and of the manuscript said
it could. That is why last1 exists: it holds the stream set at the one stream
the default protocol reads and changes only the head, so fixed8 vs last1
isolates fusion and last1 vs base isolates the head.

Reads the questions in order:
  head alone?            last1    vs base          <- stream set held at 1
  fusion works?          fixed8   vs last1         <- head held fixed: CLEAN
  selection works?       selected8 vs fixed8       <- the one that decides §4
  better than chance?    selected8 vs random8
  (vs the field)         selected8 vs base         <- confounded, but it is the
                                                      protocol actually in use

Per-seed values are stored so paired tests and patient-level bootstrap can be
run afterwards without re-fitting.

    python -m icd_pauc.picks_ablation --datasets ham --Ks 16 --n_seeds 3
    python -m icd_pauc.picks_ablation --all --tag all10
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
from icd_pauc.deep_cache import load_deep, POOLS, N_BLOCKS
from icd_pauc.subsample import build_support
from icd_pauc.train_icd_pauc import fit_anchor, set_seed
from icd_pauc.metrics_pauc import evaluate_all
from icd_pauc.maf_head import LDCAdapterHead, fit_adam, zscore
from icd_pauc.layer_probe import pick_streams, load_split, stream, stream_stats

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS = os.path.join(_HERE, "..", "..", "results")

ALL10 = ["ham", "pad", "dermnet", "bcn", "hiba", "mskcc", "ddi",
         "patch16", "derm7pt_derm", "derm7pt_clin"]

# The a priori fixed set. cls is the pooling the default protocol already uses,
# blocks 17-24 are simply "the last eight" -- the widening a practitioner would
# reach for without looking at any data. Deliberately NOT tuned.
FIXED8 = [("cls", b) for b in range(N_BLOCKS - 7, N_BLOCKS + 1)]
ALL_STREAMS = [(p, b) for p in POOLS for b in range(1, N_BLOCKS + 1)]

# The arm that makes the fusion question answerable: the SAME single stream the
# default protocol reads, put through the SAME adapter head as the 8-stream
# arms. fixed8 vs last1 then varies the stream count and nothing else, and
# last1 vs base varies the head and nothing else. Without this arm, fixed8 vs
# base moves both at once and cannot be attributed -- which is what the earlier
# version of this script and of the manuscript got wrong.
LAST1 = [("cls", N_BLOCKS)]

# random8 must be drawn from the same 96 candidates the ranker sees, and must
# be redrawn per seed -- a single lucky draw is not a chance baseline.
RANDOM_SEED_OFFSET = 10_000


def random8(seed, n=8):
    rs = np.random.RandomState(RANDOM_SEED_OFFSET + seed)
    return [ALL_STREAMS[i] for i in rs.choice(len(ALL_STREAMS), n, replace=False)]


def fuse_arm(picks, dtr, dva, dte, ytr, yva, si, vi, C, mal, yte_np, sd,
             inductive=False):
    """Train the shared head on one stream set and score the test split.

    inductive=True standardises val and test with TRAIN statistics, so no split
    contributes to its own normalisation."""
    st = [stream_stats(dtr, p, b) if inductive else None for p, b in picks]
    Str = torch.stack([stream(dtr, p, b, t) for (p, b), t in zip(picks, st)], 1)
    Sva = torch.stack([stream(dva, p, b, t) for (p, b), t in zip(picks, st)], 1)
    Ste = torch.stack([stream(dte, p, b, t) for (p, b), t in zip(picks, st)], 1)
    set_seed(sd)
    head = fit_adam(LDCAdapterHead(len(picks), Str.shape[-1], C,
                                   learn_alpha=False),
                    Str[si].to(DEVICE), ytr[si].to(DEVICE),
                    Sva[vi].to(DEVICE), yva[vi].to(DEVICE), C, mal,
                    val_select=False)
    with torch.no_grad():
        p = F.softmax(head(Ste.to(DEVICE)), 1).cpu().numpy()
    return evaluate_all(p, yte_np, mal)


def run(datasets, Ks, n_seeds, sel_seeds, tag, inductive=False, test_mask=None):
    seeds = list(range(n_seeds))
    rows = []
    t0 = time.time()

    for ds in datasets:
        cfg = DATASETS[ds]
        C, mal = cfg["nb_classes"], cfg["malignant"]
        dtr, ytr = load_split(ds, "train")
        dva, yva = load_split(ds, "val")
        dte, yte = load_split(ds, "test")
        if test_mask is not None:
            # De-duplicated evaluation: drop the test rows whose image also
            # appears in train/val, or whose filename carries contradictory
            # labels. Train is untouched, so the heads are unchanged and only
            # the evaluation set shrinks.
            m = torch.as_tensor(np.load(test_mask))
            n0 = len(yte)
            dte = {k: (v[m] if torch.is_tensor(v) and v.shape[0] == n0 else v)
                   for k, v in dte.items()}
            yte = yte[m]
            print(f"  [test mask] {ds}: {n0} -> {len(yte)} rows", flush=True)
        yte_np = yte.numpy()
        raw_tr = load_zoo(ds, "panderm_lp", "train")["embeddings"]
        raw_te = load_zoo(ds, "panderm_lp", "test")["embeddings"]
        if test_mask is not None:
            raw_te = raw_te[torch.as_tensor(np.load(test_mask))]
        if inductive:
            mu = raw_tr.mean(0, keepdim=True)
            sd_ = raw_tr.std(0, keepdim=True)
            base = (raw_tr - mu) / (sd_ + 1e-6)
            base_te = (raw_te - mu) / (sd_ + 1e-6)
        else:
            base, base_te = zscore(raw_tr), zscore(raw_te)

        for K in Ks:
            # the paper's own selection: K-shot budget, ranked on bacc
            picks_sel, _ = pick_streams(ds, K, 8, "val", list(range(sel_seeds)),
                                        C, mal, rank_metric="bacc",
                                        sel_budget="kshot", inductive=inductive)
            acc = {a: {"AUROC_bin": [], "bacc": []}
                   for a in ("base", "last1", "fixed8", "random8", "selected8")}

            for sd in seeds:
                si = build_support(ytr, "kshot", K, seed=sd)
                vi = build_support(yva, "kshot", K, seed=sd + 1000)

                # base: the official default protocol, last-layer CLS + probe
                set_seed(sd)
                h = fit_anchor(base[si].to(DEVICE), ytr[si].to(DEVICE), C,
                               DEVICE, feat_dim=base.shape[1])
                with torch.no_grad():
                    p = F.softmax(h(base_te.to(DEVICE)), 1).cpu().numpy()
                m = evaluate_all(p, yte_np, mal)
                for k in acc["base"]:
                    acc["base"][k].append(m[k])

                for arm, picks in (("last1", LAST1),
                                   ("fixed8", FIXED8),
                                   ("random8", random8(sd)),
                                   ("selected8", picks_sel)):
                    m = fuse_arm(picks, dtr, dva, dte, ytr, yva, si, vi,
                                 C, mal, yte_np, sd, inductive)
                    for k in acc[arm]:
                        acc[arm][k].append(m[k])

            for arm in acc:
                rows.append({
                    "dataset": ds, "K": K, "arm": arm, "n_seeds": n_seeds,
                    "picks": (picks_sel if arm == "selected8" else
                              FIXED8 if arm == "fixed8" else
                              LAST1 if arm == "last1" else
                              "per-seed" if arm == "random8" else None),
                    "bacc": float(np.mean(acc[arm]["bacc"])),
                    "AUROC_bin": float(np.mean(acc[arm]["AUROC_bin"])),
                    "bacc_seeds": [float(v) for v in acc[arm]["bacc"]],
                    "AUROC_bin_seeds": [float(v) for v in acc[arm]["AUROC_bin"]],
                })

            b = {a: np.mean(acc[a]["bacc"]) for a in acc}
            print(f"[{time.time()-t0:7.1f}s] {ds:<14} K={K:<3} bacc  "
                  f"base {b['base']:.4f}  last1 {b['last1']:.4f}  "
                  f"fixed8 {b['fixed8']:.4f}  "
                  f"random8 {b['random8']:.4f}  selected8 {b['selected8']:.4f}",
                  flush=True)

            os.makedirs(RESULTS, exist_ok=True)
            out = os.path.join(RESULTS, f"picks_ablation_{tag}.json")
            with open(out, "w") as f:
                json.dump(rows, f, indent=1)

    print(f"wrote {out}  ({len(rows)} rows, {time.time()-t0:.0f}s)")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ham"])
    ap.add_argument("--all", action="store_true", help="all 10 datasets")
    ap.add_argument("--Ks", nargs="+", type=int, default=[1, 4, 16])
    ap.add_argument("--n_seeds", type=int, default=20)
    ap.add_argument("--sel_seeds", type=int, default=5,
                    help="draws averaged when ranking; layer_probe default is 5")
    ap.add_argument("--tag", default="smoke")
    ap.add_argument("--inductive", action="store_true",
                    help="standardise val/test with training statistics (inductive); default is transductive")
    ap.add_argument("--test_mask", default=None,
                    help="boolean .npy mask that keeps test rows (used for the de-duplicated DermNet evaluation)")
    a = ap.parse_args()
    run(ALL10 if a.all else a.datasets, a.Ks, a.n_seeds, a.sel_seeds, a.tag,
        a.inductive, a.test_mask)
