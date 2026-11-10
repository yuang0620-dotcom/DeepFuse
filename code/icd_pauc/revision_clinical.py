"""Clinical operating-point metrics and McNemar tests.

Reads the per-image test predictions written by revision_main.py
(results/revision_preds_<tag>/) and fsl_heads_run.py --save_preds
(results/fsl_preds_rev_<tag>/). Nothing is refitted.

Clinical metrics. The malignant decision is the argmax class falling in the
dataset's malignant set (model_zoo.DATASETS[ds]['malignant']), so the operating
point is the classifier's own decision, not a threshold tuned on test.
Sensitivity, specificity, PPV, NPV and F1 of that decision are averaged over the
20 seeds. DermNet is excluded: only one of its 23 classes is unambiguously
malignant.

McNemar. For DeepFuse against a comparator, each seed gives a 2x2 table of
per-image multiclass correctness; the exact two-sided McNemar p is the binomial
test on the discordant pairs. The 20 seeds share their test images, so they are
not pooled; the cell-level p is the median over seeds, and Benjamini-Hochberg is
applied over the 24 primary cells of each comparison.

    python -m icd_pauc.revision_clinical --tag ind
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import binomtest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from icd_pauc.model_zoo import DATASETS
from icd_pauc.revision_stats import bh

RESULTS = os.path.join(_HERE, "..", "..", "results")
PRIMARY = ["ham", "pad", "dermnet", "mskcc", "ddi", "patch16", "derm7pt_derm", "derm7pt_clin"]
CLINICAL_SETS = [d for d in PRIMARY if d != "dermnet"]
KS = (4, 8, 16)
OURS = "deepfuse"


def load_cell(tag, ds, K):
    out = {}
    for sub in (f"revision_preds_{tag}", f"fsl_preds_rev_{tag}"):
        p = os.path.join(RESULTS, sub, f"{ds}_K{K}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p)
        y = z["y_test"]
        for k in z.files:
            if k.endswith("__pred"):
                out[k[:-6]] = z[k]
    return y, out


def clinical(pred, y, mal):
    """pred [S,N] class ids -> seed-mean sens/spec/ppv/npv/f1 of the malignant call."""
    yt = np.isin(y, mal)
    rows = []
    for p in pred:
        yp = np.isin(p, mal)
        tp, fp = np.sum(yp & yt), np.sum(yp & ~yt)
        fn, tn = np.sum(~yp & yt), np.sum(~yp & ~yt)
        sens = tp / (tp + fn) if tp + fn else np.nan
        spec = tn / (tn + fp) if tn + fp else np.nan
        ppv = tp / (tp + fp) if tp + fp else np.nan
        npv = tn / (tn + fn) if tn + fn else np.nan
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        rows.append((sens, spec, ppv, npv, f1))
    a = np.array(rows, float)
    return dict(zip(("sens", "spec", "ppv", "npv", "f1"), np.nanmean(a, 0))), \
        int(np.isnan(a[:, 2]).sum())


def mcnemar(pa, pb, y, return_diffs=False):
    """Per-seed exact McNemar between arms a and b -> (median p, mean b-c, all p)."""
    ps, diffs = [], []
    for a, b in zip(pa, pb):
        ca, cb = a == y, b == y
        n01, n10 = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))
        n = n01 + n10
        ps.append(binomtest(n01, n, 0.5).pvalue if n else 1.0)
        diffs.append(n01 - n10)
    if return_diffs:
        return float(np.median(ps)), float(np.mean(diffs)), ps, diffs
    return float(np.median(ps)), float(np.mean(diffs)), ps


_SIGNS = {}


def signflip_p(diffs):
    """Exact two-sided sign-flip p over seeds for the per-seed discordant difference.

    The median McNemar p is blind to direction: a cell whose seeds are each
    significant but split both ways gets a tiny median p. This tests whether the
    seed-mean difference is consistently non-zero across support draws.
    Floor is 2/2^n (1.9e-6 at 20 seeds).
    """
    d = np.asarray(diffs, dtype=np.float64)
    n = len(d)
    if n not in _SIGNS:
        bits = (np.arange(2 ** n, dtype=np.int64)[:, None] >> np.arange(n)) & 1
        _SIGNS[n] = (1 - 2 * bits).astype(np.float32)
    stat = np.abs(_SIGNS[n] @ d.astype(np.float32))
    return float(np.mean(stat >= abs(d.sum()) - 1e-6))


def bacc_of(pred, y):
    cls = np.unique(y)
    return float(np.mean([np.mean([np.mean(p[y == c] == c) for c in cls]) for p in pred]))


def run(tag, cell_test="median"):
    clin_rows, mc_rows = [], []
    for ds in PRIMARY:
        mal = DATASETS[ds]["malignant"]
        for K in KS:
            y, arms = load_cell(tag, ds, K)
            if OURS not in arms:
                continue
            comps = {a: v for a, v in arms.items()
                     if a not in (OURS, "deepfuse_full")}
            best = max(comps, key=lambda a: bacc_of(comps[a], y))
            for target, label in (("base", "default protocol"), (best, "best comparator")):
                p, d, allp, diffs = mcnemar(arms[OURS], comps[target], y,
                                            return_diffs=True)
                extra = ({"signflip_p": signflip_p(diffs)}
                         if cell_test == "signflip" else {})
                mc_rows.append({"dataset": ds, "K": K, "vs": label, "arm": target,
                                "median_p": p, **extra, "mean_discordant_diff": d,
                                "n_seeds_p05": int(sum(x < 0.05 for x in allp)),
                                "n_seeds_ours_better": int(sum(
                                    1 for a, b in zip(arms[OURS], comps[target])
                                    if np.sum((a == y) & (b != y)) > np.sum((a != y) & (b == y))))})
            if ds in CLINICAL_SETS:
                for a, pred in arms.items():
                    m, n_undef = clinical(pred, y, mal)
                    clin_rows.append({"dataset": ds, "K": K, "arm": a, **m,
                                      "n_seeds_no_positive_call": n_undef})
    p_key = "signflip_p" if cell_test == "signflip" else "median_p"
    for label in ("default protocol", "best comparator"):
        fam = [r for r in mc_rows if r["vs"] == label]
        rej, adj, _ = bh([r[p_key] for r in fam])
        for r, a, j in zip(fam, adj, rej):
            r["p_bh"], r["bh_reject"] = float(a), bool(j)
    suffix = "" if cell_test == "median" else f"_{cell_test}"
    out = os.path.join(RESULTS, f"revision_clinical_{tag}{suffix}.json")
    json.dump({"clinical": clin_rows, "mcnemar": mc_rows}, open(out, "w"), indent=1)
    print(f"-> {out}")
    print_summary(clin_rows, mc_rows)


def print_summary(clin_rows, mc_rows):
    arms = sorted({r["arm"] for r in clin_rows})
    print(f"\nClinical metrics (malignant call), mean over {len(CLINICAL_SETS)} datasets x K∈{KS}")
    print(f"{'arm':<18}" + "".join(f"{m:>8}" for m in ("sens", "spec", "ppv", "npv", "f1")))
    for a in arms:
        rs = [r for r in clin_rows if r["arm"] == a]
        print(f"{a:<18}" + "".join(f"{100*np.nanmean([r[m] for r in rs]):8.1f}"
                                   for m in ("sens", "spec", "ppv", "npv", "f1")))
    for label in ("default protocol", "best comparator"):
        fam = [r for r in mc_rows if r["vs"] == label]
        pos = sum(r["bh_reject"] and r["mean_discordant_diff"] > 0 for r in fam)
        neg = sum(r["bh_reject"] and r["mean_discordant_diff"] < 0 for r in fam)
        print(f"\nMcNemar vs {label}: {len(fam)} cells, BH-significant "
              f"{pos} in favour of DeepFuse, {neg} against")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", nargs="+", default=["ind"])
    ap.add_argument("--cell_test", choices=["median", "signflip"], default="median",
                    help="median: published median-over-seeds McNemar p (default); "
                         "signflip: direction-aware exact sign-flip over per-seed "
                         "discordant differences, written to *_signflip.json")
    a = ap.parse_args()
    for t in a.tag:
        run(t, a.cell_test)
