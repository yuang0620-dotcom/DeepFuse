"""Learnable layer weighting for MAF (LDC's fusion, ported properly).

`maf_eval.py` showed fixed concatenation of blocks {6,12,18,24} x {CLS, patch-mean}
beats the last layer significantly on HAM. This file asks the next question: should
the depth weights be LEARNED rather than concatenated or hard-coded to LDC's
0.1/0.2/0.3/0.4?

Note the expressiveness ordering: concat + linear head can already realise ANY
per-stream weighting (the head owns a separate block of weights per stream), so
`cat` >= `wsum` in capacity. The only thing a weighted sum buys is PARAMETER COUNT
(D*C instead of S*D*C). Hence the prediction this file tests:
    low K  -> wsum wins (fewer parameters to estimate)
    high K -> cat wins  (capacity pays off once labels are plentiful)

Streams: S = 4 levels x 2 poolings = 8, each [N,1024], z-scored independently.

Modes
  final       last-layer baseline (panderm_lp cache)
  cat         concat all streams  + linear head          (current champion)
  wsum_fixed  LDC's 0.1/0.2/0.3/0.4 depth weights        (no learning)
  wsum_learn  softmax-normalised learnable alpha[S]      + linear head
  proj_wsum   per-stream Linear(1024->d), learnable alpha + linear head
  ldc_adapter per-stream Linear->BN->ReLU->Linear, learnable alpha + head (LDC-faithful)
  ldc_noval   same module, FIXED epochs, never looks at the val split -- the honest
              control, since the LBFGS variants get no val-based selection either
  ldc_fixw    same as ldc_noval but depth weights FROZEN uniform -- isolates whether
              the learned weighting contributes anything beyond the adapters

All trainable variants are fit full-batch with LBFGS and the same L2 as
`fit_anchor` (C_reg = feat_dim*num_classes/100), except `ldc_adapter`, whose
BatchNorm forbids full-batch LBFGS and which therefore uses Adam + K-shot-val
early stopping. Learned alphas are dumped so the depth profile can be plotted.

Usage:
  python -m icd_pauc.maf_head --datasets ham pad --Ks 1 4 16 --n_seeds 20
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import ttest_rel

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from icd_pauc.model_zoo import load_zoo, DATASETS
from icd_pauc.maf_cache import load_maf, LEVELS
from icd_pauc.subsample import build_support
from icd_pauc.train_icd_pauc import fit_anchor, set_seed
from icd_pauc.metrics_pauc import evaluate_all

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS = os.path.join(_HERE, "..", "..", "results")
STREAMS = [f"l{lv}_{p}" for lv in LEVELS for p in ("cls", "mean")]
MODES = ["final", "cat", "wsum_fixed", "wsum_learn", "proj_wsum",
         "ldc_adapter", "ldc_noval", "ldc_fixw"]


def zscore(z):
    return (z - z.mean(0, keepdim=True)) / (z.std(0, keepdim=True) + 1e-6)


def load_streams(dataset, split):
    """-> (streams [N,S,D], baseline [N,D], labels [N])."""
    maf = load_maf(dataset, split)
    cls, avg = maf["cls"], maf["mean"]                        # [N,L,D]
    st = []
    for i in range(len(LEVELS)):
        st.append(zscore(cls[:, i])); st.append(zscore(avg[:, i]))
    base = zscore(load_zoo(dataset, "panderm_lp", split)["embeddings"])
    return torch.stack(st, 1), base, maf["labels"]


# --------------------------------------------------------------------- heads
class WSumHead(nn.Module):
    """softmax(alpha) weighted sum over streams -> linear classifier.
    proj_dim=None keeps the raw 1024-d streams; otherwise each stream gets its
    own Linear(D, proj_dim) first (LDC's adapter, minus the BN/ReLU)."""

    def __init__(self, S, D, C, proj_dim=None, fixed_w=None):
        super().__init__()
        self.proj = None
        if proj_dim is not None:
            self.proj = nn.ModuleList([nn.Linear(D, proj_dim, bias=False) for _ in range(S)])
            D = proj_dim
        if fixed_w is None:
            self.alpha = nn.Parameter(torch.zeros(S))          # uniform at init
            self.fixed = None
        else:
            self.alpha = None
            self.register_buffer("fixed", fixed_w)
        self.fc = nn.Linear(D, C)

    def weights(self):
        return self.fixed if self.alpha is None else torch.softmax(self.alpha, 0)

    def forward(self, z):                                       # z: [N,S,D]
        if self.proj is not None:
            z = torch.stack([p(z[:, i]) for i, p in enumerate(self.proj)], 1)
        w = self.weights().view(1, -1, 1)
        return self.fc((z * w).sum(1))


class LDCAdapterHead(nn.Module):
    """LDC-faithful per-stream adapter: Linear->BN->ReLU->Linear, learnable alpha.
    learn_alpha=False freezes the depth weights at uniform -- the ablation that
    separates 'the learned weighting helps' from 'the per-stream adapters help'."""

    def __init__(self, S, D, C, d=256, learn_alpha=True):
        super().__init__()
        self.ad = nn.ModuleList([nn.Sequential(
            nn.Linear(D, d, bias=False), nn.BatchNorm1d(d), nn.ReLU(inplace=True),
            nn.Linear(d, d, bias=False)) for _ in range(S)])
        if learn_alpha:
            self.alpha = nn.Parameter(torch.zeros(S))
        else:
            self.alpha = None
            self.register_buffer("fixed_w", torch.full((S,), 1.0 / S))
        self.fc = nn.Linear(d, C)

    def weights(self):
        return self.fixed_w if self.alpha is None else torch.softmax(self.alpha, 0)

    def forward(self, z):
        h = torch.stack([a(z[:, i]) for i, a in enumerate(self.ad)], 1)
        return self.fc((h * self.weights().view(1, -1, 1)).sum(1))


class BatchedLDCAdapterHead(nn.Module):
    """E independent LDCAdapterHeads fitted as one batched model.

    Profiling one cell (ham/K=16) put 78% of wall-clock inside `fit_adam`, and the
    reason is serialisation, not arithmetic: `adapter_arm_probs` loops over the 20
    episodes in Python, and `LDCAdapterHead.forward` loops again over the S streams,
    so a single cell issues ~500k `nn.Linear` calls on tensors of a few hundred rows.
    The GPU spends its time launching kernels rather than running them.

    Since the episodes share nothing -- separate parameters, separate data, separate
    losses -- they can be stacked into one set of bmm's. Two invariants make this a
    refactor rather than a new method:

      * parameters are COPIED from per-episode `LDCAdapterHead`s built exactly as the
        sequential path builds them, so the initialisation is bit-identical;
      * the batched loss is the SUM of the per-episode mean cross-entropies, not the
        mean. With disjoint parameters, d(sum_e L_e)/d(params_e) = d(L_e), so every
        episode sees exactly the gradient it saw before -- which matters because
        Adam's `weight_decay` is added to the raw gradient, so rescaling the CE term
        would silently change the regularisation balance.

    BatchNorm is reimplemented inline: torch's BatchNorm1d over a flattened
    [N, E*S*d] would be correct but allocates a module per call, and doing it by hand
    keeps the per-(episode, stream, feature) statistics explicit.
    """

    def __init__(self, heads, eps=1e-5, momentum=0.1):
        super().__init__()
        E, S = len(heads), len(heads[0].ad)
        self.E, self.S, self.eps, self.momentum = E, S, eps, momentum
        st = lambda f: torch.stack([torch.stack([f(h.ad[i]) for i in range(S)])
                                    for h in heads])
        # [E,S,D,d] and [E,S,d,d]: nn.Linear stores [out,in], bmm wants [in,out]
        self.W1 = nn.Parameter(st(lambda a: a[0].weight.detach().t().clone()))
        self.W2 = nn.Parameter(st(lambda a: a[3].weight.detach().t().clone()))
        self.gamma = nn.Parameter(st(lambda a: a[1].weight.detach().clone()))
        self.beta = nn.Parameter(st(lambda a: a[1].bias.detach().clone()))
        self.register_buffer("run_mean", st(lambda a: a[1].running_mean.detach().clone()))
        self.register_buffer("run_var", st(lambda a: a[1].running_var.detach().clone()))
        self.Wfc = nn.Parameter(torch.stack([h.fc.weight.detach().t().clone()
                                             for h in heads]))       # [E,d,C]
        self.bfc = nn.Parameter(torch.stack([h.fc.bias.detach().clone()
                                             for h in heads]))       # [E,C]
        w = torch.stack([h.weights().detach().clone() for h in heads])
        self.register_buffer("fixed_w", w)                            # [E,S]
        if heads[0].alpha is not None:
            raise ValueError("batched path supports learn_alpha=False only")

    def forward(self, z):
        """z: [E, N, S, D] -> logits [E, N, C]."""
        E, S = self.E, self.S
        N, D = z.shape[1], z.shape[3]
        d = self.W1.shape[-1]
        x = z.permute(0, 2, 1, 3).reshape(E * S, N, D)
        h = torch.bmm(x, self.W1.reshape(E * S, D, d))                # [E*S,N,d]
        if self.training:
            mean = h.mean(dim=1, keepdim=True)
            var = h.var(dim=1, unbiased=False, keepdim=True)
            with torch.no_grad():
                m = self.momentum
                self.run_mean.mul_(1 - m).add_(m * mean.detach().reshape(E, S, d))
                unb = var.detach() * (N / max(N - 1, 1))
                self.run_var.mul_(1 - m).add_(m * unb.reshape(E, S, d))
        else:
            mean = self.run_mean.reshape(E * S, 1, d)
            var = self.run_var.reshape(E * S, 1, d)
        h = (h - mean) / torch.sqrt(var + self.eps)
        h = h * self.gamma.reshape(E * S, 1, d) + self.beta.reshape(E * S, 1, d)
        h = torch.relu(h)
        h = torch.bmm(h, self.W2.reshape(E * S, d, d))                # [E*S,N,d]
        h = h.reshape(E, S, N, d).mul(self.fixed_w.view(E, S, 1, 1)).sum(1)
        return torch.baddbmm(self.bfc.unsqueeze(1), h, self.Wfc)      # [E,N,C]


def fit_adam_batched(head, z, y, epochs=300, lr=1e-3, weight_decay=1e-4):
    """z: [E,N,S,D], y: [E,N]. Sum of per-episode mean CE -- see the class docstring."""
    head = head.to(DEVICE)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    E = head.E
    for _ in range(epochs):
        head.train()
        opt.zero_grad()
        logits = head(z)
        loss = sum(F.cross_entropy(logits[e], y[e]) for e in range(E))
        loss.backward()
        opt.step()
    return head.eval()


# ------------------------------------------------------------------ fitting
def fit_lbfgs(head, z, y, C, feat_dim, max_iter=500):
    """Full-batch LBFGS + the same L2 strength fit_anchor uses."""
    head = head.to(DEVICE)
    opt = torch.optim.LBFGS(head.parameters(), lr=1.0, max_iter=max_iter,
                            line_search_fn="strong_wolfe")
    C_reg = feat_dim * C / 100.0

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(head(z), y)
        loss = loss + (1.0 / C_reg) * 0.5 * sum((p ** 2).sum() for p in head.parameters())
        loss.backward()
        return loss

    opt.step(closure)
    return head.eval()


def fit_adam(head, z, y, zv, yv, C, mal, epochs=300, patience=60, lr=1e-3,
             val_select=True):
    """For the BatchNorm variant: Adam. val_select=False never touches the val
    split (fixed epochs, final weights) so it matches what the LBFGS heads get."""
    head = head.to(DEVICE)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    best, best_state, bad = -1.0, None, 0
    for _ in range(epochs):
        head.train()
        opt.zero_grad()
        F.cross_entropy(head(z), y).backward()
        opt.step()
        if not val_select:
            continue
        head.eval()
        with torch.no_grad():
            vb = evaluate_all(F.softmax(head(zv), 1).cpu().numpy(), yv.cpu().numpy(), mal)["bacc"]
        if vb > best:
            best, bad = vb, 0
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        head.load_state_dict(best_state)
    return head.eval()


# --------------------------------------------------------------------- eval
def run_cell(mode, tr, te, va, ytr, yte, yva, si, vi, C, mal, D):
    S = tr.shape[1] if tr.dim() == 3 else 1
    if mode == "final":
        h = fit_anchor(tr[si].to(DEVICE), ytr[si].to(DEVICE), C, DEVICE, feat_dim=tr.shape[-1])
        with torch.no_grad():
            p = F.softmax(h(te.to(DEVICE)), 1).cpu().numpy()
        return evaluate_all(p, yte, mal), None
    if mode == "cat":
        Ztr = tr[si].reshape(len(si), -1).to(DEVICE)
        Zte = te.reshape(len(te), -1).to(DEVICE)
        h = fit_anchor(Ztr, ytr[si].to(DEVICE), C, DEVICE, feat_dim=Ztr.shape[1])
        with torch.no_grad():
            p = F.softmax(h(Zte), 1).cpu().numpy()
        return evaluate_all(p, yte, mal), None

    z, y = tr[si].to(DEVICE), ytr[si].to(DEVICE)
    if mode == "wsum_fixed":
        w = torch.tensor([0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4], device=DEVICE)
        head = fit_lbfgs(WSumHead(S, D, C, fixed_w=w / w.sum()), z, y, C, D)
    elif mode == "wsum_learn":
        head = fit_lbfgs(WSumHead(S, D, C), z, y, C, D)
    elif mode == "proj_wsum":
        head = fit_lbfgs(WSumHead(S, D, C, proj_dim=256), z, y, C, 256)
    elif mode in ("ldc_adapter", "ldc_noval", "ldc_fixw"):
        head = fit_adam(LDCAdapterHead(S, D, C, learn_alpha=(mode != "ldc_fixw")), z, y,
                        va[vi].to(DEVICE), yva[vi].to(DEVICE), C, mal,
                        val_select=(mode == "ldc_adapter"))
    else:
        raise ValueError(mode)
    with torch.no_grad():
        p = F.softmax(head(te.to(DEVICE)), 1).cpu().numpy()
        w = head.weights().detach().cpu().numpy().tolist()
    return evaluate_all(p, yte, mal), w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ham", "pad"])
    ap.add_argument("--Ks", nargs="+", type=int, default=[1, 4, 16])
    ap.add_argument("--n_seeds", type=int, default=20)
    ap.add_argument("--modes", nargs="+", default=MODES)
    args = ap.parse_args()
    seeds = list(range(args.n_seeds))
    rows, alphas = [], []

    for ds in args.datasets:
        cfg = DATASETS[ds]; C, mal = cfg["nb_classes"], cfg["malignant"]
        Str, base_tr, ytr = load_streams(ds, "train")
        Ste, base_te, yte = load_streams(ds, "test")
        Sva, base_va, yva = load_streams(ds, "val")
        yte_np = yte.numpy()
        D = Str.shape[-1]
        print(f"\n{'='*80}\n== {ds} (C={C})  streams={Str.shape[1]} x {D}   "
              f"n_seeds={len(seeds)}\n{'='*80}")
        print(f"{'mode':>12} " + "  ".join(f"{'K='+str(k):>13}" for k in args.Ks))

        acc = {m: {K: {"a": [], "b": []} for K in args.Ks} for m in args.modes}
        for K in args.Ks:
            for sd in seeds:
                si = build_support(ytr, "kshot", K, seed=sd)
                vi = build_support(yva, "kshot", K, seed=sd + 1000)
                for m in args.modes:
                    set_seed(sd)
                    tr_in = base_tr if m == "final" else Str
                    te_in = base_te if m == "final" else Ste
                    met, w = run_cell(m, tr_in, te_in, Sva, ytr, yte_np, yva,
                                      si, vi, C, mal, D)
                    acc[m][K]["a"].append(met["AUROC_bin"])
                    acc[m][K]["b"].append(met["bacc"])
                    if w is not None and sd == seeds[0]:
                        alphas.append({"dataset": ds, "K": K, "mode": m,
                                       "streams": STREAMS, "weights": w})
        for m in args.modes:
            cells = [f"{np.mean(acc[m][K]['a']):.3f}/{np.mean(acc[m][K]['b']):.3f}"
                     for K in args.Ks]
            print(f"{m:>12} " + "  ".join(f"{c:>13}" for c in cells))
        print("  (AUROC/bacc)")

        # paired tests vs both baselines
        print(f"\n  paired t-test (vs final / vs cat)")
        for m in args.modes:
            if m in ("final", "cat"):
                continue
            for K in args.Ks:
                for key, lab in (("a", "AUROC"), ("b", "bacc ")):
                    x = np.array(acc[m][K][key])
                    out = []
                    for basem in ("final", "cat"):
                        if basem not in acc:
                            continue
                        y = np.array(acc[basem][K][key])
                        _, p = ttest_rel(x, y)
                        s = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"
                        out.append(f"vs {basem}: {x.mean()-y.mean():+.4f} {s}")
                    print(f"    K={K:<3} {m:>11} {lab}  " + "   ".join(out))
                    rows.append({"dataset": ds, "mode": m, "K": K, "metric": lab.strip(),
                                 "mean": float(x.mean()),
                                 "vs_final": float(x.mean() - np.array(acc["final"][K][key]).mean()),
                                 "vs_cat": float(x.mean() - np.array(acc["cat"][K][key]).mean())})
        for m in args.modes:
            for K in args.Ks:
                rows.append({"dataset": ds, "mode": m, "K": K, "metric": "raw",
                             "AUROC_bin": float(np.mean(acc[m][K]["a"])),
                             "bacc": float(np.mean(acc[m][K]["b"])),
                             "bacc_std": float(np.std(acc[m][K]["b"]))})

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "maf_head.json"), "w") as f:
        json.dump(rows, f, indent=2)
    with open(os.path.join(RESULTS, "maf_learned_alpha.json"), "w") as f:
        json.dump(alphas, f, indent=2)
    print(f"\n-> {RESULTS}/maf_head.json, maf_learned_alpha.json")


if __name__ == "__main__":
    main()
