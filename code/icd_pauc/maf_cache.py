"""Multi-level frozen-feature extraction from PanDerm ViT-L (LDC's MAF, ported).

LDC taps 4 depths of the CLIP visual encoder and fuses them; we ported only the
final layer, which the oracle diagnostic showed is the space where the headroom
actually is (feature-space ceiling bacc 0.99 vs logits-space 0.72).

PanDerm-L has 24 blocks. We hook blocks {6,12,18,24} (1-indexed) and pool each
two ways -- CLS token and patch-token mean -- since a ViT's intermediate CLS is
much weaker than its patch statistics at shallow depth.

Output cache: maf_<dataset>_panderm.pt with
    {'levels': [6,12,18,24], 'cls': [N,L,1024], 'mean': [N,L,1024], 'labels': [N]}

Usage:
  python -m icd_pauc.maf_cache --dataset ham
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import icd_pauc  # loads config.env and the PanDerm code path
icd_pauc.require_panderm()
from datasets.derm_data import Derm_Dataset
from models.modeling_finetune import panderm_large_patch16_224
from icd_pauc.model_zoo import DATASETS, PANDERM_CKPT

CACHE_DIR = os.path.join(_HERE, "cache")
LEVELS = [6, 12, 18, 24]                      # 1-indexed block outputs to tap
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
TF = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD),
])


def build_panderm(device):
    model = panderm_large_patch16_224()
    sd = torch.load(PANDERM_CKPT, map_location="cpu", weights_only=True)
    sd = {k.replace("encoder.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def extract(model, dataset, split, device, batch_size=32, num_workers=8):
    cfg = DATASETS[dataset]
    df = pd.read_csv(cfg["csv"])
    flag = {"train": dict(train=True), "val": dict(val=True), "test": dict(test=True)}[split]
    ds = Derm_Dataset(df, root=cfg["root"], transforms=TF, binary=False, data_percent=1, **flag)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    taps = {}
    handles = []
    for lv in LEVELS:                          # block index lv-1 is the lv-th block
        def mk(lv):
            def hook(_m, _i, out):
                taps[lv] = out.detach()
                return None
            return hook
        handles.append(model.blocks[lv - 1].register_forward_hook(mk(lv)))

    CLS, AVG, YS = [], [], []
    for x, y, _ in tqdm(loader, desc=f"maf/{dataset}/{split}", leave=False):
        taps.clear()
        model.forward_features(x.to(device), is_train=False)
        cls = torch.stack([taps[lv][:, 0] for lv in LEVELS], 1)         # [B,L,D]
        avg = torch.stack([taps[lv][:, 1:].mean(1) for lv in LEVELS], 1)  # [B,L,D]
        CLS.append(cls.float().cpu()); AVG.append(avg.float().cpu())
        YS.append(torch.as_tensor(np.asarray(y)))

    for h in handles:
        h.remove()
    return {"levels": LEVELS, "cls": torch.cat(CLS), "mean": torch.cat(AVG),
            "labels": torch.cat(YS)}


def cache_path(dataset, split):
    return os.path.join(CACHE_DIR, f"maf_{dataset}_panderm_{split}.pt")


def load_maf(dataset, split):
    return torch.load(cache_path(dataset, split), weights_only=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ham", choices=list(DATASETS.keys()))
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_panderm(device)
    for split in ["train", "val", "test"]:
        p = cache_path(args.dataset, split)
        if os.path.exists(p):
            print(f"  {split}: exists, skip"); continue
        out = extract(model, args.dataset, split, device, args.batch_size, args.num_workers)
        torch.save(out, p)
        print(f"  {split}: cls={tuple(out['cls'].shape)} mean={tuple(out['mean'].shape)}")


if __name__ == "__main__":
    main()
