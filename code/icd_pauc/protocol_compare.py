"""The same method, the same data, evaluated under two protocols.

LDC (and the CoOp/Tip-Adapter line it descends from) subsamples only the *train*
split to K shots and then uses the *entire* validation split to pick a
hyperparameter -- for LDC, a 51-point grid search over the fusion weight beta
(`ldc_train.py:265`). Its validation splits are not small: EuroSAT's is 5400
labelled images against a 1-shot budget of 10.

Nothing about that leaks test data. What it does is spend labels that the
"K-shot" claim does not account for. This script measures what that costs, by
running one method under both accounting rules:

  ldc     streams are ranked on the ENTIRE validation split
          labels touched = K*C (support) + |val|
  honest  streams are ranked on a validation split capped to K per class
          labels touched = 2*K*C

Support sets, adapter, seeds and test split are identical across the two arms,
so the difference is attributable to the selection budget alone.

Accuracy is reported first because that is the metric LDC uses; bacc and AUROC
follow, because plain accuracy is misleading on these class-imbalanced datasets.

Usage:
  python -m icd_pauc.protocol_compare --datasets ham pad --Ks 1 2 4 8 16
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import ttest_rel

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from icd_pauc.model_zoo import DATASETS, load_zoo
from icd_pauc.layer_probe import load_split, stream, pick_streams
from icd_pauc.subsample import build_support
from icd_pauc.train_icd_pauc import fit_anchor, set_seed
from icd_pauc.metrics_pauc import evaluate_all
from icd_pauc.maf_head import LDCAdapterHead, fit_adam, zscore

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS = os.path.join(_HERE, "..", "..", "results")
PROTOCOLS = ("ldc", "honest")
_SEL_BUDGET = {"ldc": "full", "honest": "kshot"}
METRICS = ("acc", "bacc", "AUROC_bin")


def labels_touched(protocol, K, C, n_val):
    """Total labelled images the run actually consumed."""
    return K * C + (n_val if protocol == "ldc" else K * C)


def run(datasets, Ks, topn, n_seeds, sel_seeds, rank_metric):
    seeds = list(range(n_seeds))
    rows = []
    for ds in datasets:
        cfg = DATASETS[ds]
        C, mal = cfg["nb_classes"], cfg["malignant"]
        dtr, ytr = load_split(ds, "train")
        dva, yva = load_split(ds, "val")
        dte, yte = load_split(ds, "test")
        yte_np = yte.numpy()
        base = zscore(load_zoo(ds, "panderm_lp", "train")["embeddings"])
        base_te = zscore(load_zoo(ds, "panderm_lp", "test")["embeddings"])

        print(f"\n{'='*96}\n== {ds} (C={C}, |val|={len(yva)}, {n_seeds} seeds)\n{'='*96}")
        for K in Ks:
            per = {}
            for proto in PROTOCOLS:
                picks, _ = pick_streams(ds, K, topn, "val", list(range(sel_seeds)),
                                        C, mal, rank_metric=rank_metric,
                                        sel_budget=_SEL_BUDGET[proto])
                Str = torch.stack([stream(dtr, p, b) for p, b in picks], 1)
                Sva = torch.stack([stream(dva, p, b) for p, b in picks], 1)
                Ste = torch.stack([stream(dte, p, b) for p, b in picks], 1)

                acc = {m: [] for m in METRICS}
                for sd in seeds:
                    si = build_support(ytr, "kshot", K, seed=sd)
                    vi = build_support(yva, "kshot", K, seed=sd + 1000)
                    set_seed(sd)
                    head = fit_adam(LDCAdapterHead(len(picks), Str.shape[-1], C,
                                                   learn_alpha=False),
                                    Str[si].to(DEVICE), ytr[si].to(DEVICE),
                                    Sva[vi].to(DEVICE), yva[vi].to(DEVICE), C, mal,
                                    val_select=False)
                    with torch.no_grad():
                        p = F.softmax(head(Ste.to(DEVICE)), 1).cpu().numpy()
                    sc = evaluate_all(p, yte_np, mal)
                    for m in METRICS:
                        acc[m].append(sc[m])
                per[proto] = acc
                rows.append({"dataset": ds, "K": K, "protocol": proto,
                             "labels_touched": labels_touched(proto, K, C, len(yva)),
                             "picks": [f"blk{b}/{p}" for p, b in picks],
                             **{m: float(np.nanmean(acc[m])) for m in METRICS},
                             **{f"{m}_per_seed": list(map(float, acc[m])) for m in METRICS}})

            # last-layer probe, identical under both protocols (it selects nothing)
            lp = {m: [] for m in METRICS}
            for sd in seeds:
                si = build_support(ytr, "kshot", K, seed=sd)
                set_seed(sd)
                h = fit_anchor(base[si].to(DEVICE), ytr[si].to(DEVICE), C, DEVICE,
                               feat_dim=base.shape[1])
                with torch.no_grad():
                    p = F.softmax(h(base_te.to(DEVICE)), 1).cpu().numpy()
                sc = evaluate_all(p, yte_np, mal)
                for m in METRICS:
                    lp[m].append(sc[m])
            rows.append({"dataset": ds, "K": K, "protocol": "lp_baseline",
                         "labels_touched": K * C,
                         **{m: float(np.nanmean(lp[m])) for m in METRICS},
                         **{f"{m}_per_seed": list(map(float, lp[m])) for m in METRICS}})

            nl = labels_touched("ldc", K, C, len(yva))
            nh = labels_touched("honest", K, C, len(yva))
            print(f"\n  K={K:<3} labels used: LDC protocol {nl}  vs  honest protocol {nh}"
                  f"  ({nl/nh:.1f}x)")
            for m in METRICS:
                a = np.array(per["ldc"][m], float)
                b = np.array(per["honest"][m], float)
                base_m = float(np.nanmean(lp[m]))
                ga, gb = float(np.nanmean(a)) - base_m, float(np.nanmean(b)) - base_m
                _, pv = ttest_rel(a, b)
                star = ("***" if pv < .001 else "**" if pv < .01
                        else "*" if pv < .05 else "ns")
                # "gain retained" is only meaningful when the LDC-protocol arm
                # actually gained something over the baseline.
                keep = f"{100 * gb / ga:.0f}%" if ga > 1e-6 else "n/a"
                print(f"       {m:<10} baseline {base_m:.4f} | LDC {np.nanmean(a):.4f} "
                      f"({ga:+.4f}) | honest {np.nanmean(b):.4f} ({gb:+.4f}) | "
                      f"protocol gap {np.nanmean(a - b):+.4f} {star} | gain kept {keep}",
                      flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--Ks", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--topn", type=int, default=8)
    ap.add_argument("--n_seeds", type=int, default=20)
    ap.add_argument("--sel_seeds", type=int, default=5)
    ap.add_argument("--rank_metric", default="bacc", choices=["AUROC_bin", "bacc"])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    rows = run(a.datasets, a.Ks, a.topn, a.n_seeds, a.sel_seeds, a.rank_metric)
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, f"protocol_compare{a.tag}.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
