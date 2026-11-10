"""Compare the `./run.sh check` outputs with the shipped result files.

The check run evaluates DDI at K=4 with seeds 0 and 1. Every per-seed number it
produces must equal the first two seeds of the matching shipped result. The check
outputs are deleted afterwards, so results/ is left as shipped.
"""
import glob
import json
import os
import shutil
import sys

import numpy as np

PAIRS = [  # check output -> shipped file it must agree with
    ("revision_main_check_trans", "revision_main_trans"),
    ("revision_main_check_ind", "revision_main_ind"),
    ("fsl_heads_rev_check", "fsl_heads_rev_trans"),
    ("revision_main_check_abl", "revision_main_dinov2_abl_trans_ddi"),
    ("picks_symmetric_check", "picks_symmetric_sym10"),
    ("picks_ablation_check", "picks_ablation_all10_5arm"),
    ("protocol_compare_check", "protocol_compare"),
]
KEY_FIELDS = ("dataset", "K", "arm", "metric", "protocol", "encoder")


def key(row: dict) -> tuple:
    return tuple((k, str(row[k])) for k in KEY_FIELDS if k in row)


def seed_arrays(row: dict) -> dict:
    return {k: np.asarray(v, float) for k, v in row.items()
            if "seed" in k and isinstance(v, list) and v and isinstance(v[0], (int, float))}


def compare_rows(new_rows: list, old_rows: list) -> tuple:
    old = {key(r): r for r in old_rows}
    n = same = 0
    for r in new_rows:
        ref = old.get(key(r))
        if ref is None:
            continue
        ref_arrays = seed_arrays(ref)
        for k, v in seed_arrays(r).items():
            if k in ref_arrays:
                n += 1
                same += bool(np.array_equal(v, ref_arrays[k][: len(v)]))
    return n, same


def compare_depth(new: dict, old: dict) -> tuple:
    old_rows = {(r["dataset"], r["K"], r["pool"], r["block"]): r for r in old["rows"]}
    n = same = 0
    for r in new["rows"]:
        ref = old_rows[(r["dataset"], r["K"], r["pool"], r["block"])]
        for k in ("val_full_bacc_seeds", "val_kshot_bacc_seeds"):
            n += 1
            same += bool(np.array_equal(np.asarray(r[k]), np.asarray(ref[k])[: len(r[k])]))
    return n, same


def main(results: str) -> int:
    total = total_same = 0
    for new, old in PAIRS + [("depth_select_check_panderm", "depth_select_panderm_trans")]:
        a = json.load(open(os.path.join(results, f"{new}.json")))
        b = json.load(open(os.path.join(results, f"{old}.json")))
        n, same = compare_depth(a, b) if new.startswith("depth_select") else compare_rows(a, b)
        total, total_same = total + n, total_same + same
        print(f"{new:30s} {same}/{n} per-seed arrays identical to {old}.json")
    for p in glob.glob(os.path.join(results, "*check*")):
        shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    ok = total > 0 and total == total_same
    print(f"\n{'PASS' if ok else 'FAIL'}: {total_same}/{total} per-seed arrays identical")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
