"""Full-depth frozen-feature cache for a timm ViT, in the `deep_cache.py` format.

To test whether the depth diagnosis transfers beyond PanDerm, this taps
every transformer block of a general-domain ViT exactly as `deep_cache.py` taps
PanDerm: block outputs BEFORE the final LayerNorm, pooled four ways over the patch
tokens (prefix tokens excluded), stored fp16.

Output: deep_<dataset>_<backbone>_<split>.pt
        {'blocks':[1..L], 'cls'|'mean'|'max'|'std': [N,L,D] fp16, 'labels':[N]}

    python -m icd_pauc.deep_cache_timm --backbone dinov2_l --all
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import timm
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import icd_pauc  # loads config.env and the PanDerm code path
icd_pauc.require_panderm()
from datasets.derm_data import Derm_Dataset
from icd_pauc.model_zoo import DATASETS, TIMM_MODELS, CACHE_DIR, _geom
from icd_pauc.deep_cache import POOLS

ALL10 = ["ham", "pad", "dermnet", "bcn", "hiba", "mskcc", "ddi",
         "patch16", "derm7pt_derm", "derm7pt_clin"]


def build(backbone, device):
    kw = TIMM_MODELS[backbone]
    model = timm.create_model(kw["model_name"], pretrained=True, num_classes=0,
                              **{k: v for k, v in kw.items() if k != "model_name"})
    cfg = timm.data.resolve_model_data_config(model)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, _geom(cfg["mean"], cfg["std"])


def cache_path(dataset, backbone, split):
    return os.path.join(CACHE_DIR, f"deep_{dataset}_{backbone}_{split}.pt")


@torch.no_grad()
def extract(model, tf, dataset, split, device, batch_size, num_workers):
    cfg = DATASETS[dataset]
    df = pd.read_csv(cfg["csv"])
    flag = {"train": dict(train=True), "val": dict(val=True), "test": dict(test=True)}[split]
    ds = Derm_Dataset(df, root=cfg["root"], transforms=tf, binary=False,
                      data_percent=1, **flag)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    n_prefix = model.num_prefix_tokens
    n_blocks = len(model.blocks)
    taps = {}
    handles = [blk.register_forward_hook(
        (lambda b: lambda _m, _i, out: taps.__setitem__(b, out.detach()))(b))
        for b, blk in enumerate(model.blocks)]

    acc = {p: [] for p in POOLS}
    ys = []
    for x, y, _ in tqdm(loader, desc=f"deep/{dataset}/{split}", leave=False):
        taps.clear()
        model.forward_features(x.to(device, non_blocking=True))
        per = {p: [] for p in POOLS}
        for b in range(n_blocks):
            t = taps[b].float()
            pt = t[:, n_prefix:]
            per["cls"].append(t[:, 0])
            per["mean"].append(pt.mean(1))
            per["max"].append(pt.max(1).values)
            per["std"].append(pt.std(1))
        for p in POOLS:
            acc[p].append(torch.stack(per[p], 1).half().cpu())
        ys.append(torch.as_tensor(np.asarray(y)))
    for h in handles:
        h.remove()
    out = {p: torch.cat(acc[p]) for p in POOLS}
    out["blocks"] = list(range(1, n_blocks + 1))
    out["labels"] = torch.cat(ys)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="dinov2_l", choices=list(TIMM_MODELS))
    ap.add_argument("--datasets", nargs="+", default=["ham"])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tf = build(a.backbone, device)
    print(f"{a.backbone}: {len(model.blocks)} blocks, "
          f"{model.num_prefix_tokens} prefix token(s)", flush=True)
    for ds in (ALL10 if a.all else a.datasets):
        for split in ("train", "val", "test"):
            p = cache_path(ds, a.backbone, split)
            if os.path.exists(p) and not a.force:
                print(f"  {ds}/{split}: exists, skip", flush=True)
                continue
            out = extract(model, tf, ds, split, device, a.batch_size, a.num_workers)
            torch.save(out, p)
            print(f"  {ds}/{split}: {tuple(out['cls'].shape)}  "
                  f"{os.path.getsize(p)/1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
