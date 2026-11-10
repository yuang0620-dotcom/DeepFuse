"""Multi-encoder frozen-feature extraction for the ranking-mismatch study.

A registry of ~8 diverse frozen encoders (domain FM / supervised / SSL / MIM / CLIP,
ViT & CNN). Each is probed independently downstream, so differing feature dims are fine.
Common geometry (Resize256->CenterCrop224) + each model's own normalization; all at 224
for speed. Features cached to cache/zoo_<dataset>_<model>_<split>.pt.

Usage:
    python -m icd_pauc.model_zoo --dataset ham
    python -m icd_pauc.model_zoo --dataset pad
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import timm
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

CACHE_DIR = os.path.join(_HERE, "cache")
PANDERM_CKPT = os.environ.get(
    "PANDERM_CKPT",
    os.path.join(os.path.dirname(os.path.dirname(_HERE)), "weights",
                 "panderm_ll_data6_checkpoint-499.pth"))

# Paths. Index CSVs (image, label, split) ship with the release in splits/index/.
# Images are read from $DEEPFUSE_DATA/<dataset folder>/, see README "Dataset layout".
_REPO = os.path.dirname(os.path.dirname(_HERE))
INDEX_DIR = os.path.join(_REPO, "splits", "index")
DATA_ROOT = os.environ.get("DEEPFUSE_DATA", "/path/to/PanDerm_data")

DATASETS = {
    "ham": {
        "csv": os.path.join(INDEX_DIR, "ham.csv"),
        "root": os.path.join(DATA_ROOT, "HAM10000_clean/ISIC2018/"),
        "nb_classes": 7, "malignant": [0, 1, 4],
    },
    "pad": {
        "csv": os.path.join(INDEX_DIR, "pad.csv"),
        "root": os.path.join(DATA_ROOT, "pad-ufes/images/"),
        "nb_classes": 6, "malignant": [1, 4, 5],
    },
    # 23-class clinical photos. Added to test whether ICD's gain scales with the
    # number of classes (LDC's own ablation: +4.9pp on 102-class flowers, -0.0pp on
    # 10-class eurosat). Binary screening metrics are NOT meaningful here: only
    # class 13 is cleanly malignant, class 8 mixes melanoma with benign nevi.
    "dermnet": {
        "csv": os.path.join(INDEX_DIR, "dermnet.csv"),
        "root": os.path.join(DATA_ROOT, "Dermnet/images/"),
        "nb_classes": 23, "malignant": [13],
    },

    # ---- the remaining six PanDerm public datasets, added 2026-07-25 ----------
    # The modality claim previously rested on one dermoscopy dataset (ham) against
    # two clinical-photo ones (pad, dermnet). These widen both arms and add a third
    # modality (histopathology). Labels and splits are in splits/index/, derived
    # from the CSVs of PanDerm's pre-processed data release; images are untouched. Malignancy follows ham's convention of counting actinic
    # keratosis (akiec = AK / Bowen's / in-situ SCC) as malignant.

    # Dermoscopy, 9 classes: 0 mel, 1 scc, 2 nevus, 3 bcc, 4 sk, 5 ak, 6 solar
    # lentigo, 7 dermatofibroma, 8 vascular. License CC-BY-NC.
    "bcn": {
        "csv": os.path.join(INDEX_DIR, "bcn.csv"),
        "root": os.path.join(DATA_ROOT, "BCN20000/images/"),
        "nb_classes": 9, "malignant": [0, 1, 3, 5],
    },
    # Dermoscopy, same 9-class space as bcn (relabelled from its `diagnosis`
    # column, 2 lichenoid-keratosis rows dropped for lack of a bcn counterpart).
    # PanDerm ships only a melanoma-vs-rest binary label for HIBA; the 9-class
    # version is strictly more informative and reproduces it exactly as class 0.
    # License CC-BY.
    "hiba": {
        "csv": os.path.join(INDEX_DIR, "hiba.csv"),
        "root": os.path.join(DATA_ROOT, "HIBA/images/"),
        "nb_classes": 9, "malignant": [0, 1, 3, 5],
    },
    # Dermoscopy, malignant-vs-benign. Kept binary because aligning its 27
    # diagnoses to the bcn 9-class space would discard 14% of the images.
    # License CC-0.
    "mskcc": {
        "csv": os.path.join(INDEX_DIR, "mskcc.csv"),
        "root": os.path.join(DATA_ROOT, "MSKCC/images/"),
        "nb_classes": 2, "malignant": [1],
    },
    # Clinical photos, malignant-vs-benign, biopsy-confirmed. Small (647) but the
    # only dataset with a balanced Fitzpatrick skin-tone axis (203/239/205 for
    # I-II / III-IV / V-VI), kept in the `fst_grouped` column for subgroup work.
    "ddi": {
        "csv": os.path.join(INDEX_DIR, "ddi.csv"),
        "root": os.path.join(DATA_ROOT, "DDI/images/"),
        "nb_classes": 2, "malignant": [1],
    },
    # Histopathology WSI patches, 16 classes -- a third imaging modality, well
    # outside the photographic ones. Classes 0-11 are non-tumour tissue types,
    # 12 bcc / 13 sqcc / 14 melanoma are malignant, 15 naevus is a benign tumour.
    "patch16": {
        "csv": os.path.join(INDEX_DIR, "patch16.csv"),
        "root": os.path.join(DATA_ROOT, "PATCH16/images/"),
        "nb_classes": 16, "malignant": [12, 13, 14],
    },
    # Derm7pt: 834 lesions imaged BOTH dermoscopically and clinically. The two
    # entries share one case order, one label vector and one split, so a paired
    # test across them isolates modality with every other factor held fixed --
    # patient, lesion, label space, class balance. PanDerm's shipped splits are
    # drawn independently per modality (only 44.9% agree, so a lesion can sit in
    # dermoscopy-train and clinical-test); prepare_datasets.py replaces them with
    # a single case-level stratified 65/15/20 split applied to both arms.
    "derm7pt_derm": {
        "csv": os.path.join(INDEX_DIR, "derm7pt_derm.csv"),
        "root": os.path.join(DATA_ROOT, "Derm7pt/images/"),
        "nb_classes": 2, "malignant": [1],
    },
    "derm7pt_clin": {
        "csv": os.path.join(INDEX_DIR, "derm7pt_clin.csv"),
        "root": os.path.join(DATA_ROOT, "Derm7pt/images/"),
        "nb_classes": 2, "malignant": [1],
    },
}

# Imaging modality per dataset -- the grouping variable for the modality tests.
MODALITY = {
    "ham": "dermoscopy", "bcn": "dermoscopy", "hiba": "dermoscopy",
    "mskcc": "dermoscopy", "derm7pt_derm": "dermoscopy",
    "pad": "clinical", "dermnet": "clinical", "ddi": "clinical",
    "derm7pt_clin": "clinical",
    "patch16": "histopathology",
}

# timm encoders: name -> create_model kwargs. Diverse inductive biases.
TIMM_MODELS = {
    "vit_l_in21k":   dict(model_name="vit_large_patch16_224.augreg_in21k"),
    "dinov2_l":      dict(model_name="vit_large_patch14_dinov2.lvd142m", img_size=224),
    "eva02_b":       dict(model_name="eva02_base_patch14_224.mim_in22k"),
    "clip_l":        dict(model_name="vit_large_patch14_clip_224.laion2b"),
    "clip_b":        dict(model_name="vit_base_patch16_clip_224.laion2b"),
    "convnext_l":    dict(model_name="convnext_large.fb_in22k"),
    "resnet50":      dict(model_name="resnet50.a1_in1k"),
}
MODEL_NAMES = ["panderm_lp"] + list(TIMM_MODELS.keys())


def _geom(mean, std):
    return transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(mean=mean, std=std),
    ])


def build_model(name, device):
    """Returns (model, transform, feat_fn) with model frozen & on device."""
    if name == "panderm_lp":
        model = panderm_large_patch16_224()
        sd = torch.load(PANDERM_CKPT, map_location="cpu", weights_only=True)
        sd = {k.replace("encoder.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        tf = _geom((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        def feat_fn(m, x):
            f = m.forward_features(x, is_train=False)
            return f[:, 0] if f.ndim == 3 else f
    else:
        kw = TIMM_MODELS[name]
        model = timm.create_model(kw["model_name"], pretrained=True, num_classes=0,
                                  **{k: v for k, v in kw.items() if k != "model_name"})
        cfg = timm.data.resolve_model_data_config(model)
        tf = _geom(cfg["mean"], cfg["std"])

        def feat_fn(m, x):
            return m(x)  # num_classes=0 -> pooled features [B,D]

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, tf, feat_fn


@torch.no_grad()
def extract(name, dataset, split, device, batch_size=32, num_workers=8):
    cfg = DATASETS[dataset]
    df = pd.read_csv(cfg["csv"])
    model, tf, feat_fn = build_model(name, device)
    flag = {"train": dict(train=True), "val": dict(val=True), "test": dict(test=True)}[split]
    ds = Derm_Dataset(df, root=cfg["root"], transforms=tf, binary=False, data_percent=1, **flag)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    embs, labels = [], []
    for x, y, _ in tqdm(loader, desc=f"{name}/{dataset}/{split}", leave=False):
        x = x.to(device, non_blocking=True)
        f = feat_fn(model, x).float().cpu()
        embs.append(f)
        labels.append(torch.as_tensor(np.asarray(y)).long())
    return {"embeddings": torch.cat(embs), "labels": torch.cat(labels)}


def cache_path(dataset, name, split):
    return os.path.join(CACHE_DIR, f"zoo_{dataset}_{name}_{split}.pt")


def load_zoo(dataset, name, split):
    return torch.load(cache_path(dataset, name, split), weights_only=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ham", choices=list(DATASETS.keys()))
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--models", nargs="*", default=MODEL_NAMES)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for name in args.models:
        splits = ["train", "val", "test"]
        todo = [s for s in splits if args.force or not os.path.exists(cache_path(args.dataset, name, s))]
        if not todo:
            print(f"[{name}] cached, skip")
            continue
        print(f"\n=== {name} on {args.dataset} ===")
        for split in todo:
            out = extract(name, args.dataset, split, device,
                          batch_size=args.batch_size, num_workers=args.num_workers)
            torch.save(out, cache_path(args.dataset, name, split))
            pos = int(np.isin(out["labels"].numpy(), DATASETS[args.dataset]["malignant"]).sum())
            print(f"  {split}: z={tuple(out['embeddings'].shape)} n={len(out['labels'])} mal={pos}")


if __name__ == "__main__":
    main()
