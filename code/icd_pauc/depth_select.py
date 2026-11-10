"""Depth profile scored on validation AND test, for any backbone with a deep cache.

The test-scored depth profile picks the best block on the same test
split it scores, so the oracle gap is an upper bound. This script fits every
(pool, block) probe once per evaluation seed and scores the SAME fitted probes on

  test       -> the published oracle profile (reproduces layer_profile_*.json)
  val_full   -> selection on the complete validation split
  val_kshot  -> selection on a K-per-class validation subsample (seed 2000), the
                budget the paper's capped selection spends

so a block chosen on validation can be scored on test without refitting.

The same profile on another backbone (DINOv2-L, CLIP-L) only
needs a deep cache in the same format, `deep_<ds>_<backbone>_<split>.pt`.

    python -m icd_pauc.depth_select --backbone panderm --all --tag panderm
    python -m icd_pauc.depth_select --backbone dinov2_l --all --tag dinov2 --norm inductive
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from icd_pauc.model_zoo import DATASETS, CACHE_DIR
from icd_pauc.deep_cache import POOLS
from icd_pauc.subsample import build_support
from icd_pauc.fastprobe import batched_probe
from icd_pauc.metrics_pauc import to_binary, malignant_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS = os.path.join(_HERE, "..", "..", "results")
ALL10 = ["ham", "pad", "dermnet", "bcn", "hiba", "mskcc", "ddi",
         "patch16", "derm7pt_derm", "derm7pt_clin"]
VAL_SUBSAMPLE_SEED = 2000
SPLITS = ("train", "val", "test")


def load_backbone_split(ds, backbone, split):
    path = os.path.join(CACHE_DIR, f"deep_{ds}_{backbone}_{split}.pt")
    return torch.load(path, weights_only=False)


def split_stats(d, pool):
    """Per-(block, dim) mean/sd of one pooling over a whole split: [1,L,D] each."""
    z = d[pool].float()
    return z.mean(0, keepdim=True), z.std(0, keepdim=True) + 1e-6


def bacc_batch(pred, y, C):
    """Balanced accuracy for each row of pred [S,N] against y [N], on device.

    Mean recall over the classes present in y -- sklearn's definition."""
    present = torch.bincount(y, minlength=C) > 0
    out = torch.empty(pred.shape[0], device=pred.device)
    for s in range(pred.shape[0]):
        hit = torch.bincount(y[pred[s] == y], minlength=C).float()
        tot = torch.bincount(y, minlength=C).float()
        out[s] = (hit[present] / tot[present]).mean()
    return out


def score_split(Z, W, b, y, C, mal, with_auc):
    """Z [N,D] (device), W [S,D,C], b [S,C] -> per-seed bacc (+ AUROC_bin)."""
    logits = torch.einsum("nd,sdc->snc", Z, W) + b[:, None, :]
    pred = logits.argmax(-1)
    bacc = bacc_batch(pred, y, C).cpu().numpy()
    auc = None
    if with_auc:
        probs = torch.softmax(logits, -1).cpu().numpy()
        yb = to_binary(y.cpu().numpy(), mal)
        auc = (np.array([roc_auc_score(yb, malignant_score(p, mal)) for p in probs])
               if len(np.unique(yb)) > 1 else np.full(len(probs), np.nan))
    return bacc, auc


def run_cell(data, stats, ds, K, seeds, C, mal):
    ytr, yva, yte = (data[s]["labels"].long() for s in SPLITS)
    sup = [build_support(ytr.numpy(), "kshot", K, seed=sd) for sd in seeds]
    vk = build_support(yva.numpy(), "kshot", K, seed=VAL_SUBSAMPLE_SEED)
    ysup = torch.stack([ytr[si] for si in sup]).to(DEVICE)
    yva_d, yte_d = yva.to(DEVICE), yte.to(DEVICE)
    yvk_d = yva_d[torch.as_tensor(vk, device=DEVICE)]
    n_blocks = data["train"][POOLS[0]].shape[1]
    rows = []
    for pool in POOLS:
        for blk in range(1, n_blocks + 1):
            def rows_of(split, idx=None):
                mu, sd = stats[split][pool]
                z = data[split][pool][:, blk - 1] if idx is None else \
                    data[split][pool][idx, blk - 1]
                return ((z.float() - mu[:, blk - 1]) / sd[:, blk - 1]).to(DEVICE)

            Zs = torch.stack([rows_of("train", si) for si in sup])
            W, b, info = batched_probe(Zs, ysup, C)
            Zva = rows_of("val")
            bt, at = score_split(rows_of("test"), W, b, yte_d, C, mal, True)
            bv, _ = score_split(Zva, W, b, yva_d, C, mal, False)
            bk, _ = score_split(Zva[torch.as_tensor(vk, device=DEVICE)], W, b,
                                yvk_d, C, mal, False)
            rows.append({
                "dataset": ds, "K": K, "pool": pool, "block": blk,
                "test_bacc": float(bt.mean()), "test_AUROC_bin": float(np.nanmean(at)),
                "val_full_bacc": float(bv.mean()), "val_kshot_bacc": float(bk.mean()),
                "val_full_bacc_seeds": [float(v) for v in bv],
                "val_kshot_bacc_seeds": [float(v) for v in bk],
                "test_bacc_seeds": [float(v) for v in bt],
                "test_AUROC_bin_seeds": [float(v) for v in at],
                "n_unconverged": info["n_unconverged"],
            })
            del Zs, W, b, Zva
    return rows


def run(datasets, Ks, n_seeds, backbone, norm, tag):
    seeds = list(range(n_seeds))
    out = os.path.join(RESULTS, f"depth_select_{tag}.json")
    rows, t0 = [], time.time()
    for ds in datasets:
        cfg = DATASETS[ds]
        C, mal = cfg["nb_classes"], cfg["malignant"]
        data = {s: load_backbone_split(ds, backbone, s) for s in SPLITS}
        stats = {}
        for s in SPLITS:
            src = "train" if norm == "inductive" else s
            stats[s] = {p: split_stats(data[src], p) for p in POOLS}
        for K in Ks:
            cell = run_cell(data, stats, ds, K, seeds, C, mal)
            rows += cell
            best = max(cell, key=lambda r: r["test_bacc"])
            last = max((r for r in cell if r["block"] == cell[-1]["block"]),
                       key=lambda r: r["test_bacc"])
            print(f"[{time.time()-t0:7.1f}s] {ds:<14} K={K:<3} oracle "
                  f"{best['pool']}{best['block']} {best['test_bacc']:.4f}  "
                  f"last {last['test_bacc']:.4f}  unconv "
                  f"{sum(r['n_unconverged'] for r in cell)}", flush=True)
            os.makedirs(RESULTS, exist_ok=True)
            with open(out, "w") as f:
                json.dump({"backbone": backbone, "norm": norm, "n_seeds": n_seeds,
                           "val_subsample_seed": VAL_SUBSAMPLE_SEED, "rows": rows}, f)
        del data, stats
    print(f"wrote {out}  ({len(rows)} rows, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ham"])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--Ks", nargs="+", type=int, default=[1, 4, 16])
    ap.add_argument("--n_seeds", type=int, default=20)
    ap.add_argument("--backbone", default="panderm")
    ap.add_argument("--norm", choices=["transductive", "inductive"],
                    default="transductive")
    ap.add_argument("--tag", default="smoke")
    a = ap.parse_args()
    run(ALL10 if a.all else a.datasets, a.Ks, a.n_seeds, a.backbone, a.norm, a.tag)
