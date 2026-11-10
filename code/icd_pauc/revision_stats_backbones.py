"""Table 3(b) ablation contrasts on three backbones from ONE pipeline.

All three backbones are read from `revision_main` runs (selection supports drawn
from seeds 1000-1004, disjoint from evaluation seeds 0-19), so their rows are
directly comparable. The published PanDerm Table 4 came from `picks_ablation`
(selection seeds 0-4); its last1/fixed8/random8/base arms reproduce bit-for-bit
here, only selected8 differs.

Per backbone x standardisation x cluster definition: the four Table 4 contrasts
plus DeepFuse - base, using revision_stats.cluster_test (exact sign-flip,
cluster bootstrap CI). Between-backbone tests: for each contrast, the
per-cluster difference of that contrast between two backbones, sign-flipped.

    python -m icd_pauc.revision_stats_backbones
"""
from __future__ import annotations

import glob
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
KS = (4, 16)
BACKBONES = {"PanDerm": "panderm_abl", "DINOv2-L": "dinov2_abl", "CLIP-L": "clip_abl"}
CONTRASTS = [("last1 - base", "last1", "base"),
             ("fixed8 - last1", "fixed8", "last1"),
             ("selected8 - fixed8", "deepfuse", "fixed8"),
             ("selected8 - random8", "deepfuse", "random8"),
             ("selected8 - base", "deepfuse", "base")]


def load(prefix):
    """Main file plus per-dataset gap-fill files; later files win."""
    by = {}
    for f in sorted(glob.glob(os.path.join(RESULTS, f"revision_main_{prefix}*.json")),
                    key=os.path.getmtime):
        for r in json.load(open(f)):
            by[(r["dataset"], r["K"], r["arm"])] = r
    missing = [(d, K, a) for d in PRIMARY for K in KS
               for a in ("base", "last1", "fixed8", "random8", "deepfuse")
               if (d, K, a) not in by]
    if missing:
        raise SystemExit(f"{prefix}: missing {missing[:5]} ...")
    return by


def per_cell(by, hi, lo, metric="bacc"):
    return [(d, K, (np.mean(by[(d, K, hi)][f"{metric}_seeds"]) -
                    np.mean(by[(d, K, lo)][f"{metric}_seeds"])) * 100)
            for d in PRIMARY for K in KS]


def keyed(cells, merged):
    return [(MERGE.get(d, d) if merged else d, K, v) for d, K, v in cells]


def main():
    out = {"within": [], "between": []}
    for norm in ("trans", "ind"):
        data = {n: load(f"{p}_{norm}") for n, p in BACKBONES.items()}
        for merged in (False, True):
            for name, by in data.items():
                for label, hi, lo in CONTRASTS:
                    r = rs.cluster_test(keyed(per_cell(by, hi, lo), merged))
                    r.update(backbone=name, norm=norm, merged=merged, contrast=label,
                             n_cells=len(PRIMARY) * len(KS))
                    out["within"].append(r)
            names = list(data)
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a, b = names[i], names[j]
                    for label, hi, lo in CONTRASTS:
                        ca = per_cell(data[a], hi, lo)
                        cb = per_cell(data[b], hi, lo)
                        diff = [(d, K, x - y) for (d, K, x), (_, _, y) in zip(ca, cb)]
                        r = rs.cluster_test(keyed(diff, merged))
                        r.update(pair=f"{a} - {b}", norm=norm, merged=merged, contrast=label)
                        out["between"].append(r)
    path = os.path.join(RESULTS, "revision_stats_backbones.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    for r in out["within"]:
        if r["merged"]:
            print(f"{r['norm']:5s} {r['backbone']:8s} {r['contrast']:20s} "
                  f"{r['delta']:+.2f} [{r['ci_lo']:+.2f},{r['ci_hi']:+.2f}] p={r['p_exact']:.4f}")
    print("-- between backbones (merged clusters)")
    for r in out["between"]:
        if r["merged"]:
            print(f"{r['norm']:5s} {r['pair']:20s} {r['contrast']:20s} "
                  f"{r['delta']:+.2f} p={r['p_exact']:.4f}")
    print("wrote", path)


if __name__ == "__main__":
    main()
