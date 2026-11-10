"""Everything the manuscript's [AUTHOR ACTION] markers ask for on the stats side.

Four things, all computed from results already on disk -- no model is refitted:

  A  cluster sign-flip by EXACT ENUMERATION instead of 10,000 Monte Carlo draws.
     With 8 clusters there are only 2^8 = 256 sign patterns, so the Monte Carlo
     approximation was estimating something it could have enumerated. It also
     puts a floor under the p value that has to be stated: the observed pattern
     and its total flip both attain |mean| >= |obs|, so no 8-cluster sign-flip
     test can return less than 2/256 = 0.0078.

  B  Benjamini-Hochberg across the per-cell bootstrap verdicts behind Figure 4.
     Section 3.6 already promises this correction; the figure was still drawn
     from unadjusted intervals.

  C  full-validation vs capped-validation selection under dataset-clustered
     inference. The withdrawn p values treated the five budgets of one dataset
     as five independent observations of the same test images.

  D  the modality family: the two p values the text left as "p > 0.05", and a
     declared family with its BH correction.

    python -m icd_pauc.revision_stats
    python -m icd_pauc.revision_stats --json results/revision_stats.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(_HERE, "..", "..", "results")

# Main aggregate: drop the two datasets whose official splits overlap by >75% (see the paper); DermNet is kept.
CLEAN8 = ["ham", "pad", "dermnet", "mskcc", "ddi", "patch16",
          "derm7pt_derm", "derm7pt_clin"]

QUESTIONS = [
    ("Q0 head alone",      "last1",     "base"),
    ("Q1 fusion works",    "fixed8",    "last1"),
    ("Q2 selection works", "selected8", "fixed8"),
    ("Q3 beats chance",    "selected8", "random8"),
]


# --------------------------------------------------------------- primitives
def exact_signflip(cluster_means):
    """Two-sided cluster sign-flip p by enumerating all 2^n sign patterns.

    Returns (p, n_patterns, floor) where floor = 2/2^n is the smallest p the
    test can return -- worth reporting, because a result at the floor is not
    evidence of a small p value, only of the test running out of resolution.
    """
    cm = np.asarray(cluster_means, dtype=float)
    n = len(cm)
    obs = abs(cm.mean())
    signs = np.array(list(itertools.product([-1, 1], repeat=n)))   # [2^n, n]
    hits = int((np.abs(signs @ cm / n) >= obs - 1e-12).sum())
    return hits / len(signs), len(signs), 2.0 / len(signs)


def cluster_bootstrap_ci(per_ds, n_boot=10000, seed=0):
    """Percentile CI resampling whole datasets, carrying all their cells."""
    rng = np.random.default_rng(seed)
    keys = sorted(per_ds)
    arrs = [np.asarray(per_ds[k], dtype=float) for k in keys]
    n = len(keys)
    if n < 2:
        return float("nan"), float("nan")
    boot = [np.concatenate([arrs[i] for i in rng.integers(0, n, n)]).mean()
            for _ in range(n_boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def cluster_test(per_cell):
    """per_cell: list of (dataset, K, delta) -> clustered CI + exact p."""
    per_ds = {}
    for ds, _K, d in per_cell:
        per_ds.setdefault(ds, []).append(d)
    cm = [float(np.mean(v)) for _, v in sorted(per_ds.items())]
    lo, hi = cluster_bootstrap_ci(per_ds)
    p, n_pat, floor = exact_signflip(cm)
    return dict(delta=float(np.mean(np.concatenate(
                    [np.asarray(v) for _, v in sorted(per_ds.items())]))),
                cluster_mean=float(np.mean(cm)), n_clusters=len(cm),
                ci_lo=lo, ci_hi=hi, p_exact=p, n_patterns=n_pat, p_floor=floor)


def bh(pvals, q=0.05):
    """Benjamini-Hochberg. Returns (rejected mask, adjusted p, threshold)."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    crit = q * np.arange(1, n + 1) / n
    below = np.where(ranked <= crit)[0]
    k = below.max() + 1 if len(below) else 0
    rej = np.zeros(n, bool)
    if k:
        rej[order[:k]] = True
    adj = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    adj_out = np.empty(n)
    adj_out[order] = np.minimum(adj, 1.0)
    return rej, adj_out, (ranked[k - 1] if k else 0.0)


def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"


# ------------------------------------------------------------------ A: arms
def load_ablation(tag):
    with open(os.path.join(RESULTS, f"picks_ablation_{tag}.json")) as f:
        rows = json.load(f)
    by = {(r["dataset"], r["K"], r["arm"]): r for r in rows}
    cells = sorted({(r["dataset"], r["K"]) for r in rows})
    return by, cells


def section_a(tags, metrics=("bacc", "AUROC_bin")):
    out = []
    print("\n" + "=" * 96)
    print("A. cluster sign-flip, EXACT enumeration (was 10,000 Monte Carlo draws)")
    print("=" * 96)
    for tag in tags:
        try:
            by, cells = load_ablation(tag)
        except FileNotFoundError:
            print(f"  [skip] picks_ablation_{tag}.json not found")
            continue
        for scope, keep in (("primary8", CLEAN8), ("all10", None)):
            cs = [c for c in cells if keep is None or c[0] in keep]
            for metric in metrics:
                print(f"\n  {tag} · {scope} ({len(cs)} cells) · {metric}")
                for label, hi, lo in QUESTIONS:
                    per_cell = []
                    for ds, K in cs:
                        a, b = by.get((ds, K, hi)), by.get((ds, K, lo))
                        if a is None or b is None:
                            continue
                        per_cell.append((ds, K,
                                         float(np.mean(a[f"{metric}_seeds"])) -
                                         float(np.mean(b[f"{metric}_seeds"]))))
                    if len(per_cell) < 2:
                        continue
                    r = cluster_test(per_cell)
                    r.update(tag=tag, scope=scope, metric=metric,
                             question=label, hi=hi, lo=lo)
                    out.append(r)
                    at_floor = " (AT FLOOR)" if abs(r["p_exact"] - r["p_floor"]) < 1e-12 else ""
                    print(f"    {label:<20} Δ={r['delta']:+.4f}  "
                          f"CI=[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]  "
                          f"p_exact={r['p_exact']:.4f} {stars(r['p_exact']):<3}"
                          f" (floor {r['p_floor']:.4f}, {r['n_clusters']} clusters)"
                          f"{at_floor}")
    return out


# -------------------------------------------------------------- B: BH cells
def section_b(q=0.05):
    out = []
    print("\n" + "=" * 96)
    print(f"B. Benjamini-Hochberg across per-cell bootstrap verdicts (Figure 4), q={q}")
    print("=" * 96)
    for metric, fn in (("bacc", "bootstrap_fsl_bacc_kshot.json"),
                       ("AUROC_bin", "bootstrap_fsl_AUROC_bin_kshot.json")):
        path = os.path.join(RESULTS, fn)
        if not os.path.exists(path):
            print(f"  [skip] {fn} not found")
            continue
        with open(path) as f:
            rows = json.load(f)
        for scope, keep in (("primary8", CLEAN8), ("all10", None)):
            sub = [r for r in rows if keep is None or r["dataset"] in keep]
            p = [r["p_both"] for r in sub]
            rej, adj, thr = bh(p, q)
            uncorr = sum(1 for r in sub
                         if r["ci_both"][0] > 0 or r["ci_both"][1] < 0)
            print(f"\n  {metric} · {scope}: {len(sub)} cells · "
                  f"unadjusted CI excludes 0 in {uncorr} · "
                  f"BH survivors {int(rej.sum())} (threshold p<={thr:.4f})")
            for r, pr, a, rj in zip(sub, p, adj, rej):
                out.append({"metric": metric, "scope": scope,
                            "dataset": r["dataset"], "K": r["K"],
                            "delta": r["delta"], "baseline_arm": r["baseline_arm"],
                            "ci_both": r["ci_both"], "p_both": pr,
                            "p_bh": float(a), "bh_reject": bool(rj)})
                if rj or pr < 0.05:
                    print(f"      {r['dataset']:<14} K={r['K']:<3} "
                          f"Δ={r['delta']:+.4f} vs {r['baseline_arm']:<9} "
                          f"p={pr:.3f} p_BH={a:.3f} "
                          f"{'SURVIVES' if rj else 'dropped by BH'}")
    return out


# ------------------------------------------------- C: full vs capped budget
def section_c():
    print("\n" + "=" * 96)
    print("C. full-validation vs capped-validation selection, dataset-clustered")
    print("=" * 96)
    path = os.path.join(RESULTS, "protocol_compare.json")
    if not os.path.exists(path):
        print("  [skip] protocol_compare.json not found")
        return []
    with open(path) as f:
        rows = json.load(f)
    by = {(r["dataset"], r["K"], r["protocol"]): r for r in rows}
    Ks = sorted({r["K"] for r in rows})
    out = []
    for metric in ("bacc", "AUROC_bin"):
        for scope, keep in (("all10", None), ("primary8", CLEAN8)):
            print(f"\n  {metric} · {scope}")
            for K in Ks:
                per_cell, ratios = [], []
                for ds in sorted({r["dataset"] for r in rows}):
                    if keep is not None and ds not in keep:
                        continue
                    a, b = by.get((ds, K, "ldc")), by.get((ds, K, "honest"))
                    if a is None or b is None:
                        continue
                    per_cell.append((ds, K, (a[metric] - b[metric]) * 100))
                    if b.get("labels_touched"):
                        ratios.append(a["labels_touched"] / b["labels_touched"])
                if len(per_cell) < 2:
                    continue
                r = cluster_test(per_cell)
                r.update(metric=metric, scope=scope, K=K,
                         cost_ratio_median=float(np.median(ratios)) if ratios else None)
                out.append(r)
                print(f"    K={K:<3} Δ={r['delta']:+.3f} pts  "
                      f"CI=[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]  "
                      f"p_exact={r['p_exact']:.4f} {stars(r['p_exact']):<3} "
                      f"({r['n_clusters']} clusters, floor {r['p_floor']:.4f})  "
                      f"label cost x{r['cost_ratio_median']:.1f}"
                      if r["cost_ratio_median"] else "")
            # BH over the K family within this metric/scope
            fam = [x for x in out if x["metric"] == metric and x["scope"] == scope]
            rej, adj, _ = bh([x["p_exact"] for x in fam])
            for x, a, rj in zip(fam, adj, rej):
                x["p_bh"] = float(a)
                x["bh_reject"] = bool(rj)
            print(f"    BH over {len(fam)} budgets: "
                  f"{int(rej.sum())} survive at q=0.05")
    return out


# ------------------------------------------------------------ D: modality
def section_d(q=0.05):
    print("\n" + "=" * 96)
    print("D. modality family: exact p values and a declared family with BH")
    print("=" * 96)
    path = os.path.join(RESULTS, "derm7pt_paired.json")
    if not os.path.exists(path):
        print("  [skip] derm7pt_paired.json not found")
        return []
    with open(path) as f:
        rows = json.load(f)

    prim = [{"estimator": "prereg_bootstrap", "metric": r["metric"], "K": r["K"],
             "delta": r["prereg_delta"], "p": r["bootstrap_p"],
             "ci": r.get("bootstrap_ci95")} for r in rows]
    seed = [{"estimator": "per_seed_wilcoxon", "metric": r["metric"], "K": r["K"],
             "delta": r["delta_mean"], "p": r["p_value"], "n": r["n_seeds"],
             "p_floor": 2.0 / 2 ** r["n_seeds"]} for r in rows]

    out = []
    for name, fam in (("primary (bootstrap over lesions)", prim),
                      ("robustness (paired Wilcoxon over seeds)", seed)):
        rej, adj, _ = bh([x["p"] for x in fam], q)
        print(f"\n  family: {name} -- {len(fam)} tests, "
              f"{int(rej.sum())} survive BH at q={q}")
        for x, a, rj in zip(fam, adj, rej):
            x["p_bh"] = float(a)
            x["bh_reject"] = bool(rj)
            out.append(x)
            floor = x.get("p_floor")
            note = f" floor {floor:.4f}" if floor else ""
            print(f"    {x['metric']:<10} K={x['K']:<3} Δ={x['delta']:+.2f} pts  "
                  f"p={x['p']:.4f}{note}  p_BH={a:.4f}  "
                  f"{'survives' if rj else 'not significant after BH'}")
    return out


# ------------------------------------------- E: per-cell baseline envelope
def section_e(tag="_libfewshot_audited", Ks=(4, 8, 16)):
    """The abstract's envelope contrast, with an exact cluster sign-flip.

    The reported p of 0.236 is not a multiple of 1/256 and so cannot have come
    from an exact 8-cluster test; it was Monte Carlo. The envelope itself is a
    post-hoc per-cell maximum over every baseline, which makes it conservative
    by construction -- that is the point of quoting it.
    """
    print("\n" + "=" * 96)
    print("E. per-cell baseline envelope, exact cluster sign-flip")
    print("=" * 96)
    hp = os.path.join(RESULTS, f"fsl_heads{tag}.json")
    pp = os.path.join(RESULTS, "protocol_compare.json")
    if not (os.path.exists(hp) and os.path.exists(pp)):
        print("  [skip] baseline or protocol results missing")
        return []
    with open(hp) as f:
        H = {(r["dataset"], r["K"], r["metric"], r["arm"]): r["mean"]
             for r in json.load(f)}
    with open(pp) as f:
        rows = json.load(f)
    ours = {(r["dataset"], r["K"], m, r["protocol"]): r[m]
            for r in rows for m in ("bacc", "AUROC_bin")}
    # `sel` arms only. The @ref column is the same head at LibFewShot's own
    # constants and is reported for auditability; the headline comparison is
    # against each baseline re-tuned on our own budget, which is the setting
    # generous to the baseline.
    arms = sorted({k[3] for k in H if not k[3].endswith("@ref")})
    out = []
    for metric in ("bacc", "AUROC_bin"):
        for scope, keep in (("primary8", CLEAN8), ("all10", None)):
            per_cell = []
            for ds in sorted({k[0] for k in H}):
                if keep is not None and ds not in keep:
                    continue
                for K in Ks:
                    vals = [H.get((ds, K, metric, h), np.nan) for h in arms]
                    vals.append(ours.get((ds, K, metric, "lp_baseline"), np.nan))
                    o = ours.get((ds, K, metric, "honest"), np.nan)
                    if np.isnan(o) or all(np.isnan(v) for v in vals):
                        continue
                    per_cell.append((ds, K, (o - np.nanmax(vals)) * 100))
            if len(per_cell) < 2:
                continue
            r = cluster_test(per_cell)
            r.update(metric=metric, scope=scope, n_cells=len(per_cell))
            out.append(r)
            print(f"  {metric:<10} {scope:<9} {len(per_cell):>2} cells  "
                  f"Δ={r['delta']:+.2f} pts  "
                  f"CI=[{r['ci_lo']:+.2f},{r['ci_hi']:+.2f}]  "
                  f"p_exact={r['p_exact']:.4f} {stars(r['p_exact'])}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+",
                    default=["all10_5arm", "all10_ind"])
    ap.add_argument("--q", type=float, default=0.05)
    ap.add_argument("--json", default=os.path.join(RESULTS, "revision_stats.json"))
    a = ap.parse_args()

    res = {"A_signflip_exact": section_a(a.tags),
           "B_bh_cells": section_b(a.q),
           "C_full_vs_capped": section_c(),
           "D_modality": section_d(a.q),
           "E_envelope": section_e()}
    with open(a.json, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\n-> {a.json}")


if __name__ == "__main__":
    main()
