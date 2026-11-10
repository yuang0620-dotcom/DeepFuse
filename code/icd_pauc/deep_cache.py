"""Full-depth frozen-feature extraction: ALL 24 PanDerm blocks x 4 poolings.

`maf_cache.py` sampled only blocks {6,12,18,24} and 2 poolings. That was enough to
show mid-depth beats the last layer on dermoscopy (block 12 gave +0.066 AUROC at
K=4), but it cannot say WHERE the optimum actually is, nor whether the pooling
choice matters. This extracts the whole depth axis so the profile can be measured.

Poolings, per block output x [B,197,D] (token 0 = CLS, 1: = 196 patch tokens):
  cls   x[:,0]                 the usual choice
  mean  x[:,1:].mean(1)        global appearance
  max   x[:,1:].max(1)         strongest local response (focal lesion cues)
  std   x[:,1:].std(1)         spatial heterogeneity across patches -- the closest
                               linear analogue of dermoscopic "irregularity"
                               (pigment-network disorder, asymmetry of structure)

Stored fp16 to keep the cache ~1GB/dataset; probes cast back to fp32.

Output: deep_<dataset>_panderm_<split>.pt
        {'blocks':[1..24], 'cls'|'mean'|'max'|'std': [N,24,1024] fp16, 'labels':[N]}

Usage:
  python -m icd_pauc.deep_cache --dataset ham
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
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
from icd_pauc.model_zoo import DATASETS
from icd_pauc.maf_cache import build_panderm, TF, CACHE_DIR

POOLS = ("cls", "mean", "max", "std")
N_BLOCKS = 24


def cache_path(dataset, split):
    return os.path.join(CACHE_DIR, f"deep_{dataset}_panderm_{split}.pt")


def load_deep(dataset, split):
    return torch.load(cache_path(dataset, split), weights_only=False)


@torch.no_grad()
def extract(model, dataset, split, device, batch_size=32, num_workers=8):
    cfg = DATASETS[dataset]
    df = pd.read_csv(cfg["csv"])
    flag = {"train": dict(train=True), "val": dict(val=True), "test": dict(test=True)}[split]
    ds = Derm_Dataset(df, root=cfg["root"], transforms=TF, binary=False,
                      data_percent=1, **flag)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    taps = {}
    handles = []
    for b in range(N_BLOCKS):
        def mk(b):
            def hook(_m, _i, out):
                taps[b] = out.detach()
            return hook
        handles.append(model.blocks[b].register_forward_hook(mk(b)))

    acc = {p: [] for p in POOLS}
    YS = []
    for x, y, _ in tqdm(loader, desc=f"deep/{dataset}/{split}", leave=False):
        taps.clear()
        model.forward_features(x.to(device), is_train=False)
        per = {p: [] for p in POOLS}
        for b in range(N_BLOCKS):
            t = taps[b]
            pt = t[:, 1:]
            per["cls"].append(t[:, 0])
            per["mean"].append(pt.mean(1))
            per["max"].append(pt.max(1).values)
            per["std"].append(pt.std(1))
        for p in POOLS:
            acc[p].append(torch.stack(per[p], 1).half().cpu())   # [B,24,D] fp16
        YS.append(torch.as_tensor(np.asarray(y)))

    for h in handles:
        h.remove()
    out = {p: torch.cat(acc[p]) for p in POOLS}
    out["blocks"] = list(range(1, N_BLOCKS + 1))
    out["labels"] = torch.cat(YS)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ham", choices=list(DATASETS.keys()))
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_panderm(device)
    for split in ("train", "val", "test"):
        p = cache_path(args.dataset, split)
        if os.path.exists(p) and not args.force:
            print(f"  {split}: exists, skip", flush=True); continue
        out = extract(model, args.dataset, split, device, args.batch_size, args.num_workers)
        torch.save(out, p)
        print(f"  {split}: {tuple(out['cls'].shape)} x{len(POOLS)} poolings  "
              f"{os.path.getsize(p)/1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
