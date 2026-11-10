"""Frozen PanDerm + ICD de-confusion + pAUC operating-point training.

Two-stage, feature-cached (backbone never touched here -- see feature_cache.py):
  stage-1: fit CE(+L2) linear probe base_head on the K-shot support -> freeze (anchor)
  stage-2 (arm=full): train ICD residual head with pAUC surrogate + CE + L1 anchor,
           class-balanced batches, early-stop on val pAUC@80.
  arm=anchor: skip stage-2, evaluate the frozen anchor == PanDerm-LP baseline.

Usage:
  python -m icd_pauc.train_icd_pauc --dataset ham --mode kshot --value 16 --arm full
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from icd_pauc.feature_cache import load_cache, DATASETS
from icd_pauc.model_zoo import load_zoo, MODEL_NAMES
from icd_pauc.subsample import build_support
from icd_pauc.heads import BaseHead, ICDHead, num_trainable_params
from icd_pauc.losses import composite_loss
from icd_pauc.metrics_pauc import evaluate_all, to_binary

RESULTS_DIR = os.path.join(_HERE, "..", "..", "results")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict_probs(base_head, icd_head, z):
    """Returns softmax probs [N,C]. If icd_head is None -> anchor probs."""
    base_logits = base_head(z)
    logits = base_logits if icd_head is None else icd_head(base_logits, z)
    return F.softmax(logits, dim=1).cpu().numpy()


def fit_anchor(z_sup, y_sup, num_classes, device, feat_dim=1024,
               max_iter=1000, lr=1.0, class_weight=None):
    """Stage-1 CE(+L2) linear probe via full-batch LBFGS (mirrors PanDerm LP).

    class_weight: per-class CE weights, mean-normalised to 1 (see
    full_data_eval.class_weights). Under the K-shot protocol the support is exactly
    K per class, so all weights equal 1 and this is bit-for-bit inert -- it is a
    full-data-regime intervention only.
    """
    head = BaseHead(feat_dim, num_classes).to(device)
    # L2 strength mirrors panderm logistic_regression.py: C = feat_dim*num_classes/100
    C_reg = feat_dim * num_classes / 100.0
    opt = torch.optim.LBFGS(head.parameters(), lr=lr, max_iter=max_iter,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logits = head(z_sup)
        loss = F.cross_entropy(logits, y_sup, weight=class_weight)
        l2 = sum((p ** 2).sum() for p in head.parameters())
        loss = loss + (1.0 / C_reg) * 0.5 * l2
        loss.backward()
        return loss

    opt.step(closure)
    for p in head.parameters():
        p.requires_grad = False
    head.eval()
    return head


def train_icd(base_head, z_sup, y_sup, z_val, y_val, num_classes, malignant_idx,
              device, args):
    icd = ICDHead(num_classes, in_dim=z_sup.shape[1], d=args.d).to(device)
    opt = torch.optim.Adam(icd.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # class-balanced sampler on binary malignant label -> batches contain positives
    y_bin = to_binary(y_sup.cpu().numpy(), malignant_idx)
    class_count = np.bincount(y_bin, minlength=2).astype(float)
    w_per_class = 1.0 / np.maximum(class_count, 1.0)
    sample_w = w_per_class[y_bin]
    sampler = WeightedRandomSampler(torch.as_tensor(sample_w, dtype=torch.double),
                                    num_samples=len(y_sup), replacement=True)
    ds = TensorDataset(z_sup, y_sup)
    loader = DataLoader(ds, batch_size=min(args.batch_size, len(y_sup)), sampler=sampler)

    mal = list(malignant_idx)
    best_pauc, best_state, best_epoch = -1.0, None, -1
    patience, bad = args.patience, 0

    for epoch in range(args.epochs):
        icd.train()
        last = {}
        for zb, yb in loader:
            zb, yb = zb.to(device), yb.to(device)
            with torch.no_grad():
                base_logits = base_head(zb)
            final_logits = icd(base_logits, zb)
            loss, parts, valid = composite_loss(
                final_logits, base_logits, yb, mal,
                lambda1=args.lambda1, lambda2=args.lambda2, margin=args.margin,
                pauc_weight=args.pauc_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = parts

        # early stop on val pAUC@80
        icd.eval()
        probs_val = predict_probs(base_head, icd, z_val)
        val_m = evaluate_all(probs_val, y_val.cpu().numpy(), mal)
        vp = val_m["pAUC@80"]
        if not np.isnan(vp) and vp > best_pauc:
            best_pauc, best_epoch = vp, epoch
            best_state = {k: v.detach().clone() for k, v in icd.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if epoch % 20 == 0 or epoch == args.epochs - 1:
            print(f"  ep{epoch:3d} loss={last.get('total', 0):.4f} "
                  f"pauc={last.get('pauc', 0):.4f} ce={last.get('ce', 0):.4f} "
                  f"l1={last.get('l1', 0):.4f} | val pAUC@80={vp:.4f} (best {best_pauc:.4f}@{best_epoch})")
        if bad >= patience:
            print(f"  early stop at ep{epoch} (no val improve for {patience})")
            break

    # val_select=False keeps the FINAL weights instead of the best-val checkpoint.
    # Under an honest K-shot budget the val split is tiny (7 samples at K=1), so
    # val-pAUC selection is very noisy; this switch separates "the head is useless"
    # from "model selection on a 7-sample val set is useless".
    if best_state is not None and getattr(args, "val_select", True):
        icd.load_state_dict(best_state)
    icd.eval()
    return icd, best_pauc, best_epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ham", choices=list(DATASETS.keys()))
    ap.add_argument("--encoder", default="panderm_lp", choices=MODEL_NAMES,
                    help="frozen encoder whose cached features to probe")
    ap.add_argument("--val_budget", default="kshot", choices=["kshot", "full"],
                    help="kshot: val is subsampled to the same per-class budget as the "
                         "support (honest FSL, total labels = 2*K*C). full: use all 575 "
                         "val samples for early stopping (leaks labels; old behaviour).")
    ap.add_argument("--mode", default="kshot", choices=["kshot", "percent"])
    ap.add_argument("--value", type=float, default=16, help="K (kshot) or fraction (percent)")
    ap.add_argument("--arm", default="full", choices=["anchor", "full"])
    ap.add_argument("--d", type=int, default=512, help="ICD hidden dim")
    ap.add_argument("--lambda1", type=float, default=0.1, help="L1 anchor weight")
    ap.add_argument("--lambda2", type=float, default=1.0, help="CE weight")
    ap.add_argument("--margin", type=float, default=1.0, help="pAUC hinge margin")
    ap.add_argument("--pauc_weight", type=float, default=1.0,
                    help="weight on pAUC surrogate; 0 = +ICD(CE-only) ablation (arm 3)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = DATASETS[args.dataset]
    C = cfg["nb_classes"]
    mal = cfg["malignant"]

    tr = load_zoo(args.dataset, args.encoder, "train")
    va = load_zoo(args.dataset, args.encoder, "val")
    te = load_zoo(args.dataset, args.encoder, "test")

    value = int(args.value) if args.mode == "kshot" else float(args.value)
    sup_idx = build_support(tr["labels"], args.mode, value, seed=args.seed)
    z_sup = tr["embeddings"][sup_idx].to(device)
    y_sup = tr["labels"][sup_idx].to(device)

    # Validation budget must scale with the support, otherwise early stopping on the
    # full 575-sample val split silently spends far more labels than the K-shot claim.
    if args.val_budget == "kshot":
        val_idx = build_support(va["labels"], args.mode, value, seed=args.seed + 1000)
        z_val = va["embeddings"][val_idx].to(device)
        y_val = va["labels"][val_idx].to(device)
    else:
        z_val, y_val = va["embeddings"].to(device), va["labels"].to(device)
    z_te, y_te = te["embeddings"].to(device), te["labels"].to(device)

    n_pos_sup = int(to_binary(y_sup.cpu().numpy(), mal).sum())
    label_budget = int(len(sup_idx) + len(y_val))
    print(f"[{args.dataset}/{args.encoder}] mode={args.mode} value={value} arm={args.arm} | "
          f"support={len(sup_idx)} (pos={n_pos_sup}) val={len(y_val)} "
          f"[budget={label_budget}] test={len(y_te)} C={C}")

    # stage-1 anchor
    base_head = fit_anchor(z_sup, y_sup, C, device, feat_dim=z_sup.shape[1])
    assert all(not p.requires_grad for p in base_head.parameters()), "anchor must be frozen"

    arm = args.arm
    icd = None
    best_val_pauc, best_epoch = float("nan"), -1
    fallback = False
    if arm == "full":
        if n_pos_sup < 2:
            print("  !! too few malignant positives in support -> fall back to anchor")
            fallback = True
        else:
            icd, best_val_pauc, best_epoch = train_icd(
                base_head, z_sup, y_sup, z_val, y_val, C, mal, device, args)
            print(f"  ICD trainable params = {num_trainable_params(icd):,}")

    # test evaluation
    probs_te = predict_probs(base_head, icd, z_te)
    test_m = evaluate_all(probs_te, y_te.cpu().numpy(), mal)

    result = {
        "dataset": args.dataset, "encoder": args.encoder,
        "mode": args.mode, "value": value,
        "arm": ("anchor" if (arm == "anchor" or fallback) else "full"),
        "requested_arm": arm, "fallback": fallback,
        "support_size": int(len(sup_idx)), "support_pos": n_pos_sup,
        "val_budget": args.val_budget, "val_size": int(len(y_val)),
        "label_budget": label_budget,
        "d": args.d, "lambda1": args.lambda1, "lambda2": args.lambda2,
        "seed": args.seed, "best_val_pAUC@80": best_val_pauc, "best_epoch": best_epoch,
        "test": test_m,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tag = (f"{args.dataset}_{args.encoder}_{args.mode}{value}_"
           f"{result['arm']}_val{args.val_budget}_seed{args.seed}")
    out_path = os.path.join(RESULTS_DIR, f"{tag}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print("  TEST:", " ".join(f"{k}={test_m[k]:.4f}" if isinstance(test_m[k], float)
                              else f"{k}={test_m[k]}" for k in
                              ["pAUC@80", "pAUC@90", "pAUC@95", "AUROC_bin",
                               "NNB@80", "NNB@90", "acc", "bacc", "wF1"]))
    print("  saved ->", out_path)


if __name__ == "__main__":
    main()
