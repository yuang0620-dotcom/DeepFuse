"""Spearman correlation between class count and the test-selected (oracle) block.

Reads results/revision_depth_report_<backbone>_<norm>.json and reports, for each
budget K in {1, 4, 16}, the rank correlation over the ten datasets between the
number of classes and the oracle block. The paper (Section 4.1) quotes the six
PanDerm tests (three budgets x two standardisations).

Usage: python class_count_depth.py [results_dir]
"""
import json
import os
import sys

from scipy.stats import spearmanr

N_CLASSES = {"ham": 7, "patch16": 16, "derm7pt_derm": 2, "pad": 6, "hiba": 9,
             "mskcc": 2, "derm7pt_clin": 2, "bcn": 9, "dermnet": 23, "ddi": 2}


def main(results_dir):
    for backbone in ("panderm", "dinov2", "clip"):
        for norm in ("trans", "ind"):
            path = os.path.join(results_dir, f"revision_depth_report_{backbone}_{norm}.json")
            with open(path) as f:
                cells = json.load(f)["cells"]
            for k in (1, 4, 16):
                pairs = [(N_CLASSES[ds], cells[f"{ds}|{k}"]["oracle_block"])
                         for ds in sorted(N_CLASSES) if f"{ds}|{k}" in cells]
                rho, p = spearmanr(*zip(*pairs))
                print(f"{backbone:8s} {norm:5s} K={k:<2d} n={len(pairs)} rho={rho:+.3f} p={p:.3f}")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "..", "results"))
