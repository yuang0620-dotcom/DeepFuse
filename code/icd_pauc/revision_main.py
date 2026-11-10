"""Main-table run: one pass, every arm, per-image predictions.

What changes against the published Table 2 run (`protocol_compare.py`):

  * --norm inductive standardises val/test with TRAIN statistics (inductive);
    --norm transductive reproduces the published standardisation.
  * stream selection draws its ranking supports from seeds 1000-1004, disjoint
    from the evaluation seeds 0-19, which is what the manuscript states. The
    published run used seeds 0-4 (see picks_symmetric.py, schedule `pub`).
  * two further comparators:
      maf_ldc   Li et al.'s multi-level adapter fusion as released (LDC
                clip_ldc/model.py, fuse_type=2): blocks 6/12/18/24, per-level
                Linear-LayerNorm-ReLU adapter, fixed weights 0.1/0.2/0.3/0.4, then
                a Linear-BN-ReLU-Linear head. LDC adapts every token and then
                averages; our cache stores the token mean, so the adapter here
                acts on the mean-pooled block output.
      head2toe  Evci et al.: all 96 pooled streams concatenated, a group-lasso
                linear head ranks feature dimensions, the top F are kept and a
                plain L2 probe is refitted. F is chosen on the same K-shot
                validation budget the other comparators get.
  * every arm's test predictions are saved (predicted class, malignant score), so
    McNemar tests and sensitivity/specificity/PPV/NPV/F1 can be computed later
    without refitting.

    python -m icd_pauc.revision_main --datasets ham --Ks 16 --n_seeds 2 --tag smoke
    python -m icd_pauc.revision_main --all --norm inductive --tag ind
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from icd_pauc.model_zoo import DATASETS, load_zoo
from icd_pauc.deep_cache import POOLS
from icd_pauc.subsample import build_support
from icd_pauc.train_icd_pauc import fit_anchor, set_seed
from icd_pauc.metrics_pauc import evaluate_all, malignant_score
from icd_pauc.maf_head import LDCAdapterHead, fit_adam
from icd_pauc.fastprobe import batched_probe
from icd_pauc.depth_select import load_backbone_split, split_stats, ALL10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS = os.path.join(_HERE, "..", "..", "results")
SPLITS = ("train", "val", "test")
SEL_SEED_OFFSET = 1000
VAL_SUBSAMPLE_SEED = 2000
TOPN = 8
MAF_LEVELS = (6, 12, 18, 24)
MAF_WEIGHTS = (0.1, 0.2, 0.3, 0.4)
H2T_FEATURES = (1024, 2048, 4096, 8192, 16384, 32768)
H2T_LAMBDA_FRAC = 0.1
H2T_STEPS = 300
METRICS = ("acc", "bacc", "AUROC_bin")


# ------------------------------------------------------------------ features
class Features:
    """Standardised access to a deep cache: columns, streams, the post-LN CLS."""

    def __init__(self, ds, backbone, norm):
        self.data = {s: load_backbone_split(ds, backbone, s) for s in SPLITS}
        self.L = self.data["train"][POOLS[0]].shape[1]
        self.D = self.data["train"][POOLS[0]].shape[2]
        self.stats = {}
        for s in SPLITS:
            src = "train" if norm == "inductive" else s
            self.stats[s] = {p: split_stats(self.data[src], p) for p in POOLS}
        self.y = {s: self.data[s]["labels"].long() for s in SPLITS}

    def stream(self, split, pool, blk, idx=None):
        z = self.data[split][pool][:, blk - 1] if idx is None else \
            self.data[split][pool][idx, blk - 1]
        mu, sd = self.stats[split][pool]
        return (z.float() - mu[:, blk - 1]) / sd[:, blk - 1]

    def streams(self, split, picks, idx=None):
        return torch.stack([self.stream(split, p, b, idx) for p, b in picks], 1)

    def columns(self, split, cols, idx=None):
        """Flat column ids over all streams (pool-major, then block, then dim)."""
        per_pool = self.L * self.D
        out = []
        for pi, p in enumerate(POOLS):
            sel = cols[(cols >= pi * per_pool) & (cols < (pi + 1) * per_pool)] - pi * per_pool
            if len(sel) == 0:
                continue
            z = self.data[split][p] if idx is None else self.data[split][p][idx]
            z = z.reshape(z.shape[0], -1)[:, sel].float()
            mu, sd = (t.reshape(1, -1)[:, sel] for t in self.stats[split][p])
            out.append((z - mu) / sd)
        return torch.cat(out, 1)

    def all_columns(self, split, idx):
        return self.columns(split, torch.arange(len(POOLS) * self.L * self.D), idx)


BASE_ENCODER = {"panderm": "panderm_lp"}
RANDOM_SEED_OFFSET = 10_000


def ablation_streams(arm, L, sd):
    """The Table 4 controls: identical definitions to picks_ablation.py."""
    if arm == "last1":
        return [("cls", L)]
    if arm == "fixed8":
        return [("cls", b) for b in range(L - 7, L + 1)]
    names = [(p, b) for p in POOLS for b in range(1, L + 1)]
    rs = np.random.RandomState(RANDOM_SEED_OFFSET + sd)
    return [names[i] for i in rs.choice(len(names), TOPN, replace=False)]


def base_features(ds, norm, backbone="panderm"):
    enc = BASE_ENCODER.get(backbone, backbone)
    raw = {s: load_zoo(ds, enc, s)["embeddings"].float() for s in SPLITS}
    out = {}
    for s in SPLITS:
        ref = raw["train"] if norm == "inductive" else raw[s]
        out[s] = (raw[s] - ref.mean(0, keepdim=True)) / (ref.std(0, keepdim=True) + 1e-6)
    return out


# ----------------------------------------------------------------- selection
def select_streams(fe, K, C, mal, budget):
    """Rank all streams on a validation budget; probes fit on seeds 1000-1004."""
    names = [(p, b) for p in POOLS for b in range(1, fe.L + 1)]
    ytr, yva = fe.y["train"], fe.y["val"]
    sup = [build_support(ytr.numpy(), "kshot", K, seed=SEL_SEED_OFFSET + r) for r in range(5)]
    ysup = torch.stack([ytr[si] for si in sup]).to(DEVICE)
    va = (build_support(yva.numpy(), "kshot", K, seed=VAL_SUBSAMPLE_SEED)
          if budget == "kshot" else np.arange(len(yva)))
    yv = yva[va].numpy()
    scores = []
    for p, b in names:
        Zs = torch.stack([fe.stream("train", p, b, si) for si in sup]).to(DEVICE)
        W, bb, _ = batched_probe(Zs, ysup, C)
        Zv = fe.stream("val", p, b, va).to(DEVICE)
        pred = (torch.einsum("nd,sdc->snc", Zv, W) + bb[:, None]).argmax(-1).cpu().numpy()
        scores.append(np.mean([evaluate_all(np.eye(C)[pr], yv, mal, only="bacc")["bacc"]
                               for pr in pred]))
    order = np.argsort(-np.asarray(scores), kind="stable")
    return [names[i] for i in order[:TOPN]], len(va)


# --------------------------------------------------------------------- heads
class MAFHead(nn.Module):
    """LDC multi-level adapter fusion, fuse_type=2 (fixed depth weights)."""

    def __init__(self, D, C, hidden=2048):
        super().__init__()
        self.ad = nn.ModuleList([nn.Sequential(nn.Linear(D, D, bias=False),
                                               nn.LayerNorm(D), nn.ReLU())
                                 for _ in MAF_LEVELS])
        self.register_buffer("w", torch.tensor(MAF_WEIGHTS))
        self.mlp = nn.Sequential(nn.Linear(D, hidden, bias=False), nn.BatchNorm1d(hidden),
                                 nn.ReLU(), nn.Linear(hidden, C, bias=False))

    def forward(self, z):
        h = sum(self.w[i] * a(z[:, i]) for i, a in enumerate(self.ad))
        return self.mlp(h)


def fit_head(head, z, y, epochs=300, lr=1e-3, weight_decay=1e-4):
    head = head.to(DEVICE)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        head.train()
        opt.zero_grad()
        F.cross_entropy(head(z), y).backward()
        opt.step()
    return head.eval()


def adapter_probs(picks_or_levels, fe, si, ytr_s, C, head_kind):
    Ztr = fe.streams("train", picks_or_levels, si).to(DEVICE)
    Zte = fe.streams("test", picks_or_levels).to(DEVICE)
    if head_kind == "maf":
        head = fit_head(MAFHead(fe.D, C), Ztr, ytr_s)
    else:
        head = fit_adam(LDCAdapterHead(len(picks_or_levels), fe.D, C, learn_alpha=False),
                        Ztr, ytr_s, Ztr[:1], ytr_s[:1], C, None, val_select=False)
    with torch.no_grad():
        return F.softmax(head(Zte), 1).cpu().numpy()


def group_lasso_rank(X, y, C, steps=H2T_STEPS):
    """Proximal-gradient group lasso (one group per feature row of W)."""
    N, P = X.shape
    Y = F.one_hot(y, C).float()
    lip = torch.linalg.matrix_norm(X, ord=2) ** 2 / N * 0.5
    lr = 1.0 / float(lip)
    g0 = X.t() @ (Y - 1.0 / C) / N
    lam = H2T_LAMBDA_FRAC * float(g0.norm(dim=1).max())
    W = torch.zeros(P, C, device=X.device)
    b = torch.zeros(C, device=X.device)
    for _ in range(steps):
        G = (torch.softmax(X @ W + b, 1) - Y) / N
        W = W - lr * (X.t() @ G)
        b = b - lr * G.sum(0)
        nrm = W.norm(dim=1, keepdim=True).clamp_min(1e-12)
        W = W * (1 - lr * lam / nrm).clamp_min(0)
    return torch.argsort(W.norm(dim=1), descending=True)


def head2toe_probs(fe, si, vi, C, mal):
    ytr = fe.y["train"][si].to(DEVICE)
    X = fe.all_columns("train", si).to(DEVICE)
    rank = group_lasso_rank(X, ytr, C).cpu()
    del X
    yv = fe.y["val"][vi].numpy()
    best, best_f, best_wb = -1.0, None, None
    for f in H2T_FEATURES:
        cols = torch.sort(rank[:f]).values
        Xs = fe.columns("train", cols, si).to(DEVICE)
        W, b, _ = batched_probe(Xs[None], ytr[None], C)
        Xv = fe.columns("val", cols, vi).to(DEVICE)
        pv = torch.softmax(Xv @ W[0] + b[0], 1).cpu().numpy()
        s = evaluate_all(pv, yv, mal, only="bacc")["bacc"]
        if s > best:
            best, best_f, best_wb = s, cols, (W[0], b[0])
    Xt = fe.columns("test", best_f).to(DEVICE)
    with torch.no_grad():
        p = torch.softmax(Xt @ best_wb[0] + best_wb[1], 1).cpu().numpy()
    return p, len(best_f)


# ----------------------------------------------------------------------- run
def run(datasets, Ks, n_seeds, norm, tag, arms, backbone="panderm"):
    seeds = list(range(n_seeds))
    os.makedirs(os.path.join(RESULTS, f"revision_preds_{tag}"), exist_ok=True)
    out = os.path.join(RESULTS, f"revision_main_{tag}.json")
    rows, t0 = [], time.time()
    print(f"revision_main: backbone={backbone} norm={norm} datasets={datasets} "
          f"K={Ks} seeds={n_seeds} arms={arms} -> {out}", flush=True)
    for ds in datasets:
        print(f"[{time.time()-t0:7.1f}s] {ds}: loading features", flush=True)
        cfg = DATASETS[ds]
        C, mal = cfg["nb_classes"], cfg["malignant"]
        fe = Features(ds, backbone, norm)
        base = base_features(ds, norm, backbone)
        ytr, yva, yte = fe.y["train"], fe.y["val"], fe.y["test"]
        yte_np = yte.numpy()
        maf_streams = [("mean", b) for b in MAF_LEVELS]
        for K in Ks:
            sel = {}
            if "deepfuse" in arms:
                sel["deepfuse"] = select_streams(fe, K, C, mal, "kshot")
            if "deepfuse_full" in arms:
                sel["deepfuse_full"] = select_streams(fe, K, C, mal, "full")
            acc = {a: {m: [] for m in METRICS} for a in arms}
            preds = {a: np.zeros((n_seeds, len(yte)), np.int16) for a in arms}
            mscore = {a: np.zeros((n_seeds, len(yte)), np.float16) for a in arms}
            extra = {"head2toe_F": []}
            for j, sd in enumerate(seeds):
                print(f"[{time.time()-t0:7.1f}s] {ds:<14} K={K:<3} seed {j + 1}/{n_seeds}",
                      flush=True)
                si = build_support(ytr.numpy(), "kshot", K, seed=sd)
                vi = build_support(yva.numpy(), "kshot", K, seed=sd + 1000)
                ys = ytr[si].to(DEVICE)
                for a in arms:
                    set_seed(sd)
                    if a == "base":
                        h = fit_anchor(base["train"][si].to(DEVICE), ys, C, DEVICE,
                                       feat_dim=base["train"].shape[1])
                        with torch.no_grad():
                            p = F.softmax(h(base["test"].to(DEVICE)), 1).cpu().numpy()
                    elif a in ("deepfuse", "deepfuse_full"):
                        p = adapter_probs(sel[a][0], fe, si, ys, C, "ldc")
                    elif a in ("last1", "fixed8", "random8"):
                        p = adapter_probs(ablation_streams(a, fe.L, sd), fe, si, ys, C, "ldc")
                    elif a == "maf_ldc":
                        p = adapter_probs(maf_streams, fe, si, ys, C, "maf")
                    elif a == "head2toe":
                        p, nf = head2toe_probs(fe, si, vi, C, mal)
                        extra["head2toe_F"].append(nf)
                    m = evaluate_all(p, yte_np, mal)
                    for k in METRICS:
                        acc[a][k].append(m[k])
                    preds[a][j] = p.argmax(1)
                    mscore[a][j] = malignant_score(p, mal)
            np.savez_compressed(
                os.path.join(RESULTS, f"revision_preds_{tag}", f"{ds}_K{K}.npz"),
                y_test=yte_np, **{f"{a}__pred": preds[a] for a in arms},
                **{f"{a}__mal": mscore[a] for a in arms})
            for a in arms:
                rows.append({
                    "dataset": ds, "K": K, "arm": a, "norm": norm, "n_seeds": n_seeds,
                    **({"picks": [f"{p}{b}" for p, b in sel[a][0]],
                        "val_labels": int(sel[a][1])} if a in sel else {}),
                    **({"head2toe_F": extra["head2toe_F"]} if a == "head2toe" else {}),
                    **{k: float(np.nanmean(acc[a][k])) for k in METRICS},
                    **{f"{k}_seeds": [float(v) for v in acc[a][k]] for k in METRICS},
                })
            print(f"[{time.time()-t0:7.1f}s] {ds:<14} K={K:<3} bacc  " + "  ".join(
                f"{a} {np.mean(acc[a]['bacc']):.4f}" for a in arms), flush=True)
            with open(out, "w") as f:
                json.dump(rows, f, indent=1)
        del fe, base
    print(f"wrote {out}  ({len(rows)} rows, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ham"])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--Ks", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    ap.add_argument("--n_seeds", type=int, default=20)
    ap.add_argument("--norm", choices=["transductive", "inductive"], default="inductive")
    ap.add_argument("--arms", nargs="+",
                    default=["base", "deepfuse", "deepfuse_full", "maf_ldc", "head2toe"])
    ap.add_argument("--tag", default="smoke")
    ap.add_argument("--backbone", default="panderm",
                    help="deep cache name; dinov2_l needs deep_cache_timm first")
    a = ap.parse_args()
    run(ALL10 if a.all else a.datasets, a.Ks, a.n_seeds, a.norm, a.tag, a.arms,
        a.backbone)
