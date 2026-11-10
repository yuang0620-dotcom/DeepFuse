"""Tables 2 and 3 for the revision, under one standardisation per call.

Table 2  per dataset: default protocol -> DeepFuse, balanced accuracy and AUROC.
Table 4  every comparator averaged over the primary datasets, with the new
         Li et al. (maf_ldc) and Head2Toe rows, mean difference to DeepFuse over
         the primary cells, and the per-cell envelope under dataset-clustered
         inference (Derm7pt as two clusters and as one).

    python -m icd_pauc.revision_tables --tag ind
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from icd_pauc.revision_stats_clusters import clustered

RESULTS = os.path.join(_HERE, "..", "..", "results")
PRIMARY = ["ham", "pad", "dermnet", "mskcc", "ddi", "patch16", "derm7pt_derm", "derm7pt_clin"]
ORDER = ["ham", "patch16", "derm7pt_derm", "pad", "hiba", "mskcc", "derm7pt_clin",
         "bcn", "dermnet", "ddi"]
KS = (4, 8, 16)
LABEL = {"rfs": "RFS/SKD", "baseline_pp": "Baseline++", "r2d2": "R2D2", "base": "Default linear",
         "negative_margin": "Neg.-Margin", "baseline": "Baseline", "tip": "Tip-Adapter",
         "matchingnet": "MatchingNet", "protonet": "ProtoNet", "metabaseline": "Meta-Baseline",
         "dsn": "DSN", "maf_ldc": "Multi-level adapter fusion (Li et al.)",
         "head2toe": "Head2Toe", "deepfuse": "DeepFuse", "deepfuse_full": "DeepFuse (full val)"}


def load(tag):
    v = {}
    for r in json.load(open(os.path.join(RESULTS, f"revision_main_{tag}.json"))):
        for m in ("bacc", "AUROC_bin"):
            v[(r["dataset"], r["K"], m, r["arm"])] = 100 * r[m]
    fp = os.path.join(RESULTS, f"fsl_heads_rev_{tag}.json")
    if os.path.exists(fp):
        for r in json.load(open(fp)):
            if not r["arm"].endswith("@ref"):
                v[(r["dataset"], r["K"], r["metric"], r["arm"])] = 100 * r["mean"]
    return v


def table2(v):
    print("\nTable 2: default protocol -> DeepFuse, columns K = 4, 8, 16")
    for m in ("bacc", "AUROC_bin"):
        print(f"  ({m})")
        for ds in ORDER:
            if m == "AUROC_bin" and ds == "dermnet":
                continue
            cells = []
            for K in KS:
                a, b = v.get((ds, K, m, "base")), v.get((ds, K, m, "deepfuse"))
                cells.append("   n/a      " if a is None or b is None else f"{a:5.1f} -> {b:5.1f}")
            print(f"    {ds:<14} " + "   ".join(cells))


def table3(v):
    arms = sorted({k[3] for k in v} - {"deepfuse", "deepfuse_full"})
    rows = []
    for m in ("bacc", "AUROC_bin"):
        ds_m = [d for d in PRIMARY if not (m == "AUROC_bin" and d == "dermnet")]
        ours = {(d, K): v.get((d, K, m, "deepfuse")) for d in ds_m for K in KS}
        for a in arms + ["deepfuse"]:
            per_K = [np.mean([v[(d, K, m, a)] for d in ds_m if (d, K, m, a) in v] or [np.nan])
                     for K in KS]
            diffs = [ours[(d, K)] - v[(d, K, m, a)] for d in ds_m for K in KS
                     if (d, K, m, a) in v and ours[(d, K)] is not None]
            rows.append({"metric": m, "arm": a, "K4": per_K[0], "K8": per_K[1],
                         "K16": per_K[2], "delta": float(np.mean(diffs)) if diffs else None,
                         "n_cells": len(diffs)})
    for m in ("bacc", "AUROC_bin"):
        print(f"\nTable 4 ({m}), mean over primary datasets"
              f"{' (DermNet excluded)' if m == 'AUROC_bin' else ''}")
        sub = sorted([r for r in rows if r["metric"] == m and r["arm"] != "deepfuse"],
                     key=lambda r: r["delta"] if r["delta"] is not None else 99)
        for r in sub + [r for r in rows if r["metric"] == m and r["arm"] == "deepfuse"]:
            d = f"{r['delta']:+.2f}" if r["delta"] is not None and r["arm"] != "deepfuse" else "  ---"
            print(f"  {LABEL.get(r['arm'], r['arm']):<40} {r['K4']:5.1f} {r['K8']:5.1f} "
                  f"{r['K16']:5.1f}  Δ {d}")
    return rows


def envelope(v):
    out = []
    comps = sorted({k[3] for k in v} - {"deepfuse", "deepfuse_full"})
    for m in ("bacc", "AUROC_bin"):
        ds_m = [d for d in PRIMARY if not (m == "AUROC_bin" and d == "dermnet")]
        per_cell = []
        for d in ds_m:
            for K in KS:
                vals = [v[(d, K, m, a)] for a in comps if (d, K, m, a) in v]
                o = v.get((d, K, m, "deepfuse"))
                if vals and o is not None:
                    per_cell.append((d, K, o - max(vals)))
        for merged in (False, True):
            r = clustered(per_cell, merged)
            r.update(metric=m, derm7pt_one_cluster=merged, n_cells=len(per_cell),
                     n_comparators=len(comps))
            out.append(r)
            print(f"  envelope {m:<10} {'7 clusters' if merged else '8 clusters'}: "
                  f"Δ={r['delta']:+.2f} CI=[{r['ci_lo']:+.2f},{r['ci_hi']:+.2f}] "
                  f"p={r['p_exact']:.4f}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", nargs="+", default=["ind"])
    for tag in ap.parse_args().tag:
        print(f"\n######## {tag} ########")
        v = load(tag)
        table2(v)
        rows = table3(v)
        env = envelope(v)
        json.dump({"table3": rows, "envelope": env},
                  open(os.path.join(RESULTS, f"revision_tables_{tag}.json"), "w"), indent=1)
