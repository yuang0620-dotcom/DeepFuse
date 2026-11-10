"""Build the split files released with the DeepFuse paper.

Inputs are the index files in splits/index/ and the ISIC Archive metadata; nothing here
uses model outputs or images. The metadata path is set with --isic_meta.

Outputs (in --out):
  derm7pt_case_split.csv            case-level 65/15/20 split shared by both Derm7pt views
                                    (the split used in the paper)
  dermnet_test_contamination.csv    per official test row: in train, in val, contradictory
                                    label, masked (449 rows masked in the paper)
  bcn20000_test_lesion_overlap.csv  per official test image: lesion_id also in train
  hiba_test_patient_overlap.csv     per official test image: patient_id also in train
  bcn20000_lesion_disjoint_split.csv, hiba_patient_disjoint_split.csv
                                    group-disjoint resplits at the official proportions
                                    (seed 0). Provided for future work; not evaluated in
                                    the paper.
"""
import argparse, os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def stem(v):
    return os.path.splitext(os.path.basename(str(v)))[0]


def derm7pt(root, out):
    d = pd.read_csv(os.path.join(root, "derm7pt_derm.csv"))
    c = pd.read_csv(os.path.join(root, "derm7pt_clin.csv"))
    m = d.merge(c, on="case", suffixes=("_derm", "_clin"))
    assert len(m) == len(d) == len(c), "views must share every case"
    assert (m.split_derm == m.split_clin).all() and (m.label_derm == m.label_clin).all()
    m = m.rename(columns={"split_derm": "split", "label_derm": "label"})
    m[["case", "split", "label", "image_derm", "image_clin"]].to_csv(
        os.path.join(out, "derm7pt_case_split.csv"), index=False)
    return m.split.value_counts().to_dict()


def dermnet(root, out):
    r = pd.read_csv(os.path.join(root, "dermnet.csv"))
    r["name"] = r["image"].map(lambda v: os.path.basename(str(v)))
    tr, va = set(r.name[r.split == "train"]), set(r.name[r.split == "val"])
    nlab = r.groupby("name")["label"].nunique()
    t = r[r.split == "test"].reset_index(drop=True)
    t["test_row"] = np.arange(len(t))
    t["in_train"] = t.name.isin(tr)
    t["in_val"] = t.name.isin(va)
    t["contradictory_label"] = t.name.map(nlab) > 1
    t["masked"] = t.in_train | t.in_val | t.contradictory_label
    t[["test_row", "image", "label", "in_train", "in_val", "contradictory_label",
       "masked"]].to_csv(os.path.join(out, "dermnet_test_contamination.csv"), index=False)
    return int(t.masked.sum()), len(t)


def isic_group(root, meta_path, out, ds, csvname, key, tag):
    df = pd.read_csv(os.path.join(root, csvname))
    meta = pd.read_csv(meta_path, usecols=["isic_id", key], low_memory=False)
    g = dict(zip(meta.isic_id, meta[key]))
    df["isic_id"] = df["image"].map(stem)
    df[key] = df.isic_id.map(g)
    assert df[key].notna().mean() > 0.99, f"{ds}: {key} missing for many images"
    df[key] = df[key].fillna(df.isic_id)           # a missing key is its own group
    trkeys = set(df[key][df.split == "train"])
    te = df[df.split == "test"].copy()
    te["overlaps_train"] = te[key].isin(trkeys)
    te[["image", key, "overlaps_train"]].to_csv(
        os.path.join(out, f"{tag}_test_{key.split('_')[0]}_overlap.csv"), index=False)
    # group-disjoint resplit at the official proportions
    n = df.split.value_counts()
    f_test, f_val = n["test"] / len(df), n["val"] / (n["train"] + n["val"])
    a, b = next(GroupShuffleSplit(1, test_size=f_test, random_state=0).split(df, groups=df[key]))
    rest = df.iloc[a]
    c, d = next(GroupShuffleSplit(1, test_size=f_val, random_state=0).split(rest, groups=rest[key]))
    new = pd.Series("train", index=df.index)
    new.iloc[b] = "test"
    new.loc[rest.index[d]] = "val"
    df["split"] = new
    for s1 in ("train", "val", "test"):
        for s2 in ("train", "val", "test"):
            if s1 < s2:
                assert not set(df[key][df.split == s1]) & set(df[key][df.split == s2])
    df[["image", "label", key, "split"]].to_csv(
        os.path.join(out, f"{tag}_{key.split('_')[0]}_disjoint_split.csv"), index=False)
    return float(te.overlaps_train.mean()), df.split.value_counts().to_dict()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=os.path.join(os.path.dirname(__file__), "..", "splits", "index"))
    ap.add_argument("--isic_meta", default="/path/to/isic_archive/metadata.csv")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "splits"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    print("derm7pt", derm7pt(a.index, a.out))
    print("dermnet masked/test", dermnet(a.index, a.out))
    print("bcn20000", isic_group(a.index, a.isic_meta, a.out, "bcn",
                                 "bcn.csv", "lesion_id", "bcn20000"))
    print("hiba", isic_group(a.index, a.isic_meta, a.out, "hiba",
                             "hiba.csv", "patient_id", "hiba"))
