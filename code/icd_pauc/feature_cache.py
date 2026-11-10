"""Extract & cache frozen PanDerm ViT-L features for the ICD+pAUC method.

Runs once per dataset/split. Backbone is fully frozen; we cache z[N,1024] + labels
so stage-1/2 head training reads cached tensors and is second-scale.

Usage:
    python -m icd_pauc.feature_cache --dataset ham
"""
import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# make repo root importable (this file lives in classification/icd_pauc/)
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import icd_pauc  # loads config.env and the PanDerm code path
icd_pauc.require_panderm()
from models.builder import get_encoder
from datasets.derm_data import Derm_Dataset
from panderm_model.downstream.extract_features import extract_features_from_dataloader

CACHE_DIR = os.path.join(_HERE, "cache")

# Paths. Index CSVs (image, label, split) ship with the release in splits/index/.
# Images are read from $DEEPFUSE_DATA/<dataset folder>/, see README "Dataset layout".
_REPO = os.path.dirname(os.path.dirname(_HERE))
INDEX_DIR = os.path.join(_REPO, "splits", "index")
DATA_ROOT = os.environ.get("DEEPFUSE_DATA", "/path/to/PanDerm_data")

DATASETS = {
    "ham": {
        "csv": os.path.join(INDEX_DIR, "ham.csv"),
        "root": os.path.join(DATA_ROOT, "HAM10000_clean/ISIC2018/"),
        "nb_classes": 7,
        # malignant diagnosis indices: 0=akiec, 1=bcc, 4=mel
        "malignant": [0, 1, 4],
    },
}

CKPT = os.environ.get("PANDERM_CKPT", "/path/to/panderm_ll_data6_checkpoint-499.pth")


class _Args:
    def __init__(self, ckpt):
        self.pretrained_checkpoint = ckpt


def cache_path(dataset, split):
    return os.path.join(CACHE_DIR, f"{dataset}_{split}.pt")


def build_backbone():
    args = _Args(CKPT)
    model, transform = get_encoder(args, "PanDerm_Large_LP")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, transform, args


def extract_split(dataset, split, model, transform, args, batch_size=64, num_workers=8):
    cfg = DATASETS[dataset]
    df = pd.read_csv(cfg["csv"])
    flags = {"train": dict(train=True), "val": dict(val=True), "test": dict(test=True)}[split]
    ds = Derm_Dataset(df, root=cfg["root"], transforms=transform,
                      binary=False, data_percent=1, **flags)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    asset = extract_features_from_dataloader(args, model, loader)
    return {
        "embeddings": torch.from_numpy(asset["embeddings"]).float(),  # [N,1024]
        "labels": torch.from_numpy(np.asarray(asset["labels"])).long(),  # [N] multiclass
        "filenames": asset["filenames"],
    }


def load_cache(dataset, split):
    return torch.load(cache_path(dataset, split), weights_only=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ham", choices=list(DATASETS.keys()))
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="re-extract even if cache exists")
    args = ap.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)
    splits = ["train", "val", "test"]
    need = [s for s in splits if args.force or not os.path.exists(cache_path(args.dataset, s))]
    if not need:
        print("all splits cached:", [cache_path(args.dataset, s) for s in splits])
        return

    model, transform, enc_args = build_backbone()
    for split in need:
        print(f"\n=== extracting {args.dataset}/{split} ===")
        out = extract_split(args.dataset, split, model, transform, enc_args,
                            batch_size=args.batch_size, num_workers=args.num_workers)
        torch.save(out, cache_path(args.dataset, split))
        print(f"{split}: z={tuple(out['embeddings'].shape)} "
              f"labels={tuple(out['labels'].shape)} "
              f"class_counts={torch.bincount(out['labels']).tolist()}")
        print("saved ->", cache_path(args.dataset, split))


if __name__ == "__main__":
    main()
