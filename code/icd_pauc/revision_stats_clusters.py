"""Count the two Derm7pt views as ONE cluster in every clustered test.

The published tests cluster on the logical dataset, so Derm7pt-derm and
Derm7pt-clin -- two photographs of the same lesions -- enter as independent
clusters. Here the cluster key maps both to `derm7pt`; a cluster's value is the
mean over all its cells, exactly as before. With 7 primary clusters the exact
sign-flip floor rises from 2/256 = 0.0078 to 2/128 = 0.0156.

Recomputes, under both cluster definitions:
  Table 4   four ablation contrasts x {as run, inductive, DermNet dedup, symmetric}
  envelope  DeepFuse vs the best comparator per cell (Table 3 text)
  budget    full- vs capped-validation selection per K

    python -m icd_pauc.revision_stats_clusters
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from icd_pauc import revision_stats as rs

RESULTS = rs.RESULTS
PRIMARY = rs.CLEAN8
MERGE = {"derm7pt_derm": "derm7pt", "derm7pt_clin": "derm7pt"}
CONTRASTS = [("last1 - base", "last1", "base"),
             ("fixed8 - last1", "fixed8", "last1"),
             ("selected8 - fixed8", "selected8", "fixed8"),
             ("selected8 - random8", "selected8", "random8")]


def clustered(per_cell, merged):
    key = (lambda d: MERGE.get(d, d)) if merged else (lambda d: d)
    return rs.cluster_test([(key(ds), K, d) for ds, K, d in per_cell])


def ablation_rows(tag, rename=None, replace=None):
    rows = json.load(open(os.path.join(RESULTS, f"picks_ablation_{tag}.json")))
    if replace:
        extra = json.load(open(os.path.join(RESULTS, f"picks_ablation_{replace}.json")))
        drop = {r["dataset"] for r in extra}
        rows = [r for r in rows if r["dataset"] not in drop] + extra
    by = {}
    for r in rows:
        by[(r["dataset"], r["K"], (rename or {}).get(r["arm"], r["arm"]))] = r
    return by


def table4(merged, Kmin=4, metric="bacc"):
    columns = {
        "as_run": ablation_rows("all10_5arm"),
        "inductive": ablation_rows("all10_ind"),
        "dermnet_dedup": ablation_rows("all10_5arm", replace="dermnet_dedup"),
    }
    sym = json.load(open(os.path.join(RESULTS, "picks_symmetric_sym10.json")))
    columns["symmetric"] = {(r["dataset"], r["K"],
                             "selected8" if r["arm"] == "selected8_sym" else r["arm"]): r
                            for r in sym if r["arm"] != "selected8_pub"
                            and r["arm"] != "selected8_disj"}
    out = []
    for col, by in columns.items():
        for label, hi, lo in CONTRASTS:
            per_cell = []
            for (ds, K, arm), r in by.items():
                if arm != hi or ds not in PRIMARY or K < Kmin:
                    continue
                b = by.get((ds, K, lo))
                if b is None:
                    continue
                per_cell.append((ds, K, (np.mean(r[f"{metric}_seeds"]) -
                                         np.mean(b[f"{metric}_seeds"])) * 100))
            r = clustered(per_cell, merged)
            r.update(column=col, contrast=label, metric=metric, n_cells=len(per_cell))
            out.append(r)
    return out


def envelope(merged, tag="_libfewshot_audited", Ks=(4, 8, 16)):
    H = {(r["dataset"], r["K"], r["metric"], r["arm"]): r["mean"]
         for r in json.load(open(os.path.join(RESULTS, f"fsl_heads{tag}.json")))}
    rows = json.load(open(os.path.join(RESULTS, "protocol_compare.json")))
    ours = {(r["dataset"], r["K"], m, r["protocol"]): r[m]
            for r in rows for m in ("bacc", "AUROC_bin")}
    arms = sorted({k[3] for k in H if not k[3].endswith("@ref")})
    out = []
    for metric in ("bacc", "AUROC_bin"):
        for scope, keep in (("primary8", PRIMARY), ("all10", None)):
            per_cell = []
            for ds in sorted({k[0] for k in H}):
                if keep is not None and ds not in keep:
                    continue
                for K in Ks:
                    vals = [H.get((ds, K, metric, h), np.nan) for h in arms]
                    vals.append(ours.get((ds, K, metric, "lp_baseline"), np.nan))
                    o = ours.get((ds, K, metric, "honest"), np.nan)
                    per_cell.append((ds, K, (o - np.nanmax(vals)) * 100))
            r = clustered(per_cell, merged)
            r.update(metric=metric, scope=scope)
            out.append(r)
    return out


def budget(merged):
    rows = json.load(open(os.path.join(RESULTS, "protocol_compare.json")))
    by = {(r["dataset"], r["K"], r["protocol"]): r for r in rows}
    out = []
    for scope, keep in (("primary8", PRIMARY), ("all10", None)):
        for K in sorted({r["K"] for r in rows}):
            per_cell = [(ds, K, (by[(ds, K, "ldc")]["bacc"] - by[(ds, K, "honest")]["bacc"]) * 100)
                        for ds in sorted({r["dataset"] for r in rows})
                        if (keep is None or ds in keep) and (ds, K, "ldc") in by]
            r = clustered(per_cell, merged)
            r.update(scope=scope, K=K)
            out.append(r)
    return out


def main():
    res = {}
    for merged in (False, True):
        name = "derm7pt_one_cluster" if merged else "published_clusters"
        res[name] = {"table4": table4(merged), "envelope": envelope(merged),
                     "budget": budget(merged)}
        print(f"\n===== {name} =====")
        for r in res[name]["table4"]:
            print(f"  T4 {r['column']:<14} {r['contrast']:<20} Δ={r['delta']:+.2f} "
                  f"p={r['p_exact']:.4f} (n={r['n_clusters']}, floor {r['p_floor']:.4f})")
        for r in res[name]["envelope"]:
            print(f"  ENV {r['metric']:<10} {r['scope']:<9} Δ={r['delta']:+.2f} "
                  f"CI=[{r['ci_lo']:+.2f},{r['ci_hi']:+.2f}] p={r['p_exact']:.4f}")
        for r in res[name]["budget"]:
            print(f"  BUD {r['scope']:<9} K={r['K']:<3} Δ={r['delta']:+.2f} p={r['p_exact']:.4f}")
    out = os.path.join(RESULTS, "revision_stats_clusters.json")
    json.dump(res, open(out, "w"), indent=1)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
