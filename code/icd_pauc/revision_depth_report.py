"""Depth-diagnosis tables for the revision, from depth_select_<tag>.json.

Per (dataset, K) cell, following the published Figure 3 definitions:
  block score   best of the four poolings at that block, on TEST
  oracle gap    max block score - block-24 score              (published)
  val-selected  the (block, pooling) with the highest VALIDATION balanced
                accuracy, scored on test; gain = its test score - block-24 score
  recovered     gain / oracle gap, aggregated as sum(gain) / sum(gap)

Two validation budgets: the full validation split, and K images per class.

    python -m icd_pauc.revision_depth_report --tag panderm_trans
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(_HERE, "..", "..", "results")
PRIMARY = ["ham", "pad", "dermnet", "mskcc", "ddi", "patch16", "derm7pt_derm", "derm7pt_clin"]


def cells(tag):
    d = json.load(open(os.path.join(RESULTS, f"depth_select_{tag}.json")))
    by = {}
    for r in d["rows"]:
        by.setdefault((r["dataset"], r["K"]), []).append(r)
    return d, by


def summarise_cell(rows):
    L = max(r["block"] for r in rows)
    block = {b: max((r for r in rows if r["block"] == b), key=lambda r: r["test_bacc"])
             for b in range(1, L + 1)}
    scores = np.array([block[b]["test_bacc"] for b in range(1, L + 1)]) * 100
    last = scores[-1]
    out = {"oracle_block": int(scores.argmax()) + 1, "oracle": float(scores.max()),
           "last": float(last), "gap": float(scores.max() - last),
           "last_cls": float(next(r["test_bacc"] for r in rows
                                  if r["block"] == L and r["pool"] == "cls") * 100)}
    for budget in ("val_full", "val_kshot"):
        sel = max(rows, key=lambda r: r[f"{budget}_bacc"])
        out[f"{budget}_avgsel_pick"] = f"{sel['pool']}{sel['block']}"
        out[f"{budget}_avgsel_gain"] = sel["test_bacc"] * 100 - last
        test_s, picks = per_seed_selection(rows, budget)
        out[f"{budget}_picks"] = picks
        out[f"{budget}_test"] = float(np.mean(test_s)) * 100
        out[f"{budget}_gain"] = float(np.mean(test_s)) * 100 - last
    return out


def per_seed_selection(rows, budget):
    """Each seed picks the stream with its own highest validation score.

    Small validation budgets tie often (two validation images per class give
    only a handful of possible balanced accuracies), so a tie is resolved by
    averaging the test scores of all tied streams: the expected outcome of
    breaking the tie at random, without favouring list order."""
    val = np.array([r[f"{budget}_bacc_seeds"] for r in rows])     # [streams, seeds]
    test = np.array([r["test_bacc_seeds"] for r in rows])
    test_s, picks = [], []
    for s in range(val.shape[1]):
        tied = np.flatnonzero(np.isclose(val[:, s], val[:, s].max()))
        test_s.append(test[tied, s].mean())
        picks.append(f"{rows[tied[0]]['pool']}{rows[tied[0]]['block']}"
                     + (f"+{len(tied) - 1}tied" if len(tied) > 1 else ""))
    return test_s, picks


def report(tag):
    meta, by = cells(tag)
    res = {k: summarise_cell(v) for k, v in by.items()}
    Ks = sorted({k[1] for k in res})
    print(f"\n{tag}: backbone={meta['backbone']} norm={meta['norm']} "
          f"seeds={meta['n_seeds']}")
    print(f"{'dataset':<14}" + "".join(f"   K={K}: best  gap  valF  valK" for K in Ks))
    for ds in sorted({k[0] for k in res}):
        line = f"{ds:<14}"
        for K in Ks:
            c = res[(ds, K)]
            line += (f"   {c['oracle_block']:>9} {c['gap']:5.2f} "
                     f"{c['val_full_gain']:+5.2f} {c['val_kshot_gain']:+5.2f}")
        print(line)
    agg = []
    for scope, keep in (("all10", None), ("primary8", PRIMARY)):
        for K in Ks:
            cs = [res[(ds, K)] for ds, KK in res if KK == K and (keep is None or ds in keep)]
            gap = np.array([c["gap"] for c in cs])
            row = {"scope": scope, "K": K, "n": len(cs),
                   "mean_gap": float(gap.mean()), "max_gap": float(gap.max()),
                   "blocks": sorted(c["oracle_block"] for c in cs),
                   "n_last_best": int(sum(c["oracle_block"] == max(
                       r["block"] for r in by[(d, K)]) for (d, KK), c in res.items()
                       if KK == K and (keep is None or d in keep)))}
            for budget in ("val_full", "val_kshot"):
                g = np.array([c[f"{budget}_gain"] for c in cs])
                row[f"{budget}_mean_gain"] = float(g.mean())
                row[f"{budget}_recovered"] = float(g.sum() / gap.sum()) if gap.sum() > 0 else float("nan")
                row[f"{budget}_n_worse"] = int((g < 0).sum())
                ga = np.array([c[f"{budget}_avgsel_gain"] for c in cs])
                row[f"{budget}_avgsel_mean_gain"] = float(ga.mean())
                row[f"{budget}_avgsel_recovered"] = float(ga.sum() / gap.sum()) if gap.sum() > 0 else float("nan")
                row[f"{budget}_avgsel_n_worse"] = int((ga < 0).sum())
            agg.append(row)
            print(f"  {scope:<9} K={K:<3} gap mean {row['mean_gap']:.2f} max {row['max_gap']:.2f} | "
                  f"val-full gain {row['val_full_mean_gain']:+.2f} "
                  f"({100*row['val_full_recovered']:.0f}% recovered, {row['val_full_n_worse']} worse) | "
                  f"val-K gain {row['val_kshot_mean_gain']:+.2f} "
                  f"({100*row['val_kshot_recovered']:.0f}%, {row['val_kshot_n_worse']} worse) | "
                  f"best blocks {row['blocks']}")
    out = os.path.join(RESULTS, f"revision_depth_report_{tag}.json")
    json.dump({"meta": {k: meta[k] for k in ("backbone", "norm", "n_seeds")},
               "cells": {f"{d}|{K}": v for (d, K), v in res.items()},
               "aggregate": agg}, open(out, "w"), indent=1)
    print(f"-> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", nargs="+", default=["panderm_trans"])
    for t in ap.parse_args().tag:
        report(t)
