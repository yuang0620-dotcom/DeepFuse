"""Depth profile of PanDerm's 24 blocks, and fusion over the layers that matter.

Two stages:

  profile  probe every (block, pooling) pair on its own under the K-shot protocol.
           Answers: where along the depth axis does the useful signal live, and
           does that optimum move with imaging modality?

  fuse     take the top-N (block, pooling) streams by profile score and run them
           through the LDC per-stream adapter (the component §9 proved is the one
           that carries the gain -- learned depth weights were shown to contribute
           exactly nothing, so the weighting stays uniform here).

Selection honesty: `fuse` picks its streams using VAL scores only, never test.
A `--select_on test` switch exists purely to quantify the selection optimism.

Usage:
  python -m icd_pauc.layer_probe profile --datasets ham pad --Ks 1 4 16 --n_seeds 10
  python -m icd_pauc.layer_probe fuse    --datasets ham pad --Ks 1 4 16 --topn 8
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import ttest_rel

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from icd_pauc.model_zoo import load_zoo, DATASETS
from icd_pauc.deep_cache import load_deep, POOLS, N_BLOCKS
from icd_pauc.subsample import build_support
from icd_pauc.train_icd_pauc import fit_anchor, set_seed
from icd_pauc.metrics_pauc import evaluate_all
from icd_pauc.maf_head import LDCAdapterHead, fit_adam, zscore

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS = os.path.join(_HERE, "..", "..", "results")


def load_split(ds, split):
    d = load_deep(ds, split)
    return d, d["labels"]


def stream_stats(dtr, pool, blk):
    """Per-dimension mean/sd of one stream on the TRAIN split, for the inductive
    variant below."""
    z = dtr[pool][:, blk - 1].float()
    return z.mean(0, keepdim=True), z.std(0, keepdim=True)


def stream(d, pool, blk, stats=None):
    """(pool, block) -> [N,1024] fp32, z-scored.

    stats=None is the TRANSDUCTIVE default the paper's main results use: each
    split is standardised with its own statistics, so test features contribute
    to their own normalisation. It uses no labels, and every arm shares it, so
    it cannot move a comparison between arms -- but it does lift all arms
    together and makes absolute values incomparable with inductive results
    published elsewhere.

    Pass stats=stream_stats(train_split, pool, blk) for the INDUCTIVE variant,
    where val and test are standardised with train statistics only and nothing
    about the test split leaks into its own features.
    """
    z = d[pool][:, blk - 1].float()
    if stats is None:
        return zscore(z)
    mu, sd = stats
    return (z - mu) / (sd + 1e-6)


# ------------------------------------------------------------------ profile
def run_profile(datasets, Ks, seeds, rank_metric="AUROC_bin"):
    rows = []
    for ds in datasets:
        cfg = DATASETS[ds]; C, mal = cfg["nb_classes"], cfg["malignant"]
        dtr, ytr = load_split(ds, "train")
        dte, yte = load_split(ds, "test")
        yte_np = yte.numpy()
        base = zscore(load_zoo(ds, "panderm_lp", "train")["embeddings"])
        base_te = zscore(load_zoo(ds, "panderm_lp", "test")["embeddings"])

        print(f"\n{'='*94}\n== {ds} depth profile ({rank_metric}, {len(seeds)} seeds)\n{'='*94}")
        # baseline
        for K in Ks:
            a, b = [], []
            for sd in seeds:
                set_seed(sd)
                si = build_support(ytr, "kshot", K, seed=sd)
                h = fit_anchor(base[si].to(DEVICE), ytr[si].to(DEVICE), C, DEVICE,
                               feat_dim=base.shape[1])
                with torch.no_grad():
                    p = F.softmax(h(base_te.to(DEVICE)), 1).cpu().numpy()
                m = evaluate_all(p, yte_np, mal)
                a.append(m["AUROC_bin"]); b.append(m["bacc"])
            rows.append({"dataset": ds, "block": 0, "pool": "final_baseline", "K": K,
                         "AUROC_bin": float(np.nanmean(a)), "bacc": float(np.nanmean(b))})
            sel = np.nanmean(a) if rank_metric == "AUROC_bin" else np.nanmean(b)
            print(f"  [baseline: last-block attention pooling] K={K}: {rank_metric}={sel:.4f}")

        for pool in POOLS:
            print(f"\n  --- pool={pool} ---")
            print("  blk " + "".join(f"{'K='+str(k):>9}" for k in Ks))
            for blk in range(1, N_BLOCKS + 1):
                Ztr, Zte = stream(dtr, pool, blk), stream(dte, pool, blk)
                cells = []
                for K in Ks:
                    a, b = [], []
                    for sd in seeds:
                        set_seed(sd)
                        si = build_support(ytr, "kshot", K, seed=sd)
                        h = fit_anchor(Ztr[si].to(DEVICE), ytr[si].to(DEVICE), C, DEVICE,
                                       feat_dim=Ztr.shape[1])
                        with torch.no_grad():
                            p = F.softmax(h(Zte.to(DEVICE)), 1).cpu().numpy()
                        m = evaluate_all(p, yte_np, mal)
                        a.append(m["AUROC_bin"]); b.append(m["bacc"])
                    cells.append(np.nanmean(a) if rank_metric == "AUROC_bin" else np.nanmean(b))
                    rows.append({"dataset": ds, "block": blk, "pool": pool, "K": K,
                                 "AUROC_bin": float(np.nanmean(a)), "bacc": float(np.nanmean(b))})
                print(f"  {blk:>3} " + "".join(f"{c:>9.4f}" for c in cells), flush=True)

        for K in Ks:
            sub = [r for r in rows if r["dataset"] == ds and r["K"] == K and r["block"] > 0]
            top = sorted(sub, key=lambda r: -r[rank_metric])[:5]
            bl = next(r for r in rows if r["dataset"] == ds and r["K"] == K and r["block"] == 0)
            print(f"\n  K={K} best 5 streams ({rank_metric}, baseline {bl[rank_metric]:.4f}): " +
                  ", ".join(f"blk{t['block']}/{t['pool']}={t[rank_metric]:.4f}" for t in top))

    # Filename carries the dataset list: running one dataset alone must not clobber
    # another dataset's profile (the default ham+pad run keeps the original name).
    suffix = "" if datasets == ["ham", "pad"] else "_" + "_".join(datasets)
    out = os.path.join(RESULTS, f"layer_profile{suffix}.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n-> {out}")


# --------------------------------------------------------------------- fuse
def pick_streams(ds, K, topn, select_on, seeds, C, mal,
                 rank_metric="AUROC_bin", sel_budget="full", inductive=False):
    """Rank (block,pool) streams on the VAL split, return the top-n.

    rank_metric : AUROC_bin by default. Datasets whose malignant grouping is not
        meaningful (dermnet: only class 13 is cleanly malignant) must rank on bacc.
    sel_budget  : 'full' scores on the ENTIRE val split -- 575/344/3073 labels for
        ham/pad/dermnet, far above the K-shot budget the protocol claims. 'kshot'
        subsamples the scoring split to the same per-class budget as the support,
        so selection stops spending unbudgeted labels. Default stays 'full' so
        previously reported numbers reproduce bit-for-bit.
    inductive   : False keeps the transductive default (each split standardised
        with its own statistics). True standardises the scoring split with TRAIN
        statistics, so nothing about the split being scored enters its own
        features. See stream() above.
    """
    dtr, ytr = load_split(ds, "train")
    dsel, ysel = load_split(ds, select_on)
    sel_idx = None
    if sel_budget == "kshot":
        sel_idx = build_support(ysel, "kshot", K, seed=2000)
        ysel = ysel[sel_idx]
    ysel_np = ysel.numpy()
    scored = []
    for pool in POOLS:
        for blk in range(1, N_BLOCKS + 1):
            st = stream_stats(dtr, pool, blk) if inductive else None
            Ztr, Zs = stream(dtr, pool, blk, st), stream(dsel, pool, blk, st)
            if sel_idx is not None:
                Zs = Zs[sel_idx]
            a = []
            for sd in seeds:
                set_seed(sd)
                si = build_support(ytr, "kshot", K, seed=sd)
                h = fit_anchor(Ztr[si].to(DEVICE), ytr[si].to(DEVICE), C, DEVICE,
                               feat_dim=Ztr.shape[1])
                with torch.no_grad():
                    p = F.softmax(h(Zs.to(DEVICE)), 1).cpu().numpy()
                a.append(evaluate_all(p, ysel_np, mal)[rank_metric])
            scored.append(((pool, blk), float(np.nanmean(a))))
    scored.sort(key=lambda t: -t[1])
    return [s[0] for s in scored[:topn]], scored[:topn]


def run_fuse(datasets, Ks, topn, n_seeds, select_on, sel_seeds,
             rank_metric="AUROC_bin", sel_budget="full"):
    seeds = list(range(n_seeds))
    rows = []
    for ds in datasets:
        cfg = DATASETS[ds]; C, mal = cfg["nb_classes"], cfg["malignant"]
        dtr, ytr = load_split(ds, "train")
        dva, yva = load_split(ds, "val")
        dte, yte = load_split(ds, "test")
        yte_np = yte.numpy()
        base = zscore(load_zoo(ds, "panderm_lp", "train")["embeddings"])
        base_te = zscore(load_zoo(ds, "panderm_lp", "test")["embeddings"])

        print(f"\n{'='*88}\n== {ds} top-{topn} stream fusion (selected on {select_on}/{sel_budget}, "
              f"rank metric={rank_metric}, {len(seeds)} seeds)\n{'='*88}")
        for K in Ks:
            picks, scored = pick_streams(ds, K, topn, select_on,
                                         list(range(sel_seeds)), C, mal,
                                         rank_metric=rank_metric, sel_budget=sel_budget)
            print(f"\n  K={K} selected: " + ", ".join(f"blk{b}/{p}" for p, b in picks))
            Str = torch.stack([stream(dtr, p, b) for p, b in picks], 1)
            Sva = torch.stack([stream(dva, p, b) for p, b in picks], 1)
            Ste = torch.stack([stream(dte, p, b) for p, b in picks], 1)

            res = {"final": {"a": [], "b": []}, "deepfuse": {"a": [], "b": []}}
            for sd in seeds:
                si = build_support(ytr, "kshot", K, seed=sd)
                vi = build_support(yva, "kshot", K, seed=sd + 1000)
                set_seed(sd)
                h = fit_anchor(base[si].to(DEVICE), ytr[si].to(DEVICE), C, DEVICE,
                               feat_dim=base.shape[1])
                with torch.no_grad():
                    p = F.softmax(h(base_te.to(DEVICE)), 1).cpu().numpy()
                m = evaluate_all(p, yte_np, mal)
                res["final"]["a"].append(m["AUROC_bin"]); res["final"]["b"].append(m["bacc"])

                set_seed(sd)
                head = fit_adam(LDCAdapterHead(len(picks), Str.shape[-1], C,
                                               learn_alpha=False),
                                Str[si].to(DEVICE), ytr[si].to(DEVICE),
                                Sva[vi].to(DEVICE), yva[vi].to(DEVICE), C, mal,
                                val_select=False)
                with torch.no_grad():
                    p = F.softmax(head(Ste.to(DEVICE)), 1).cpu().numpy()
                m = evaluate_all(p, yte_np, mal)
                res["deepfuse"]["a"].append(m["AUROC_bin"]); res["deepfuse"]["b"].append(m["bacc"])

            for key, lab in (("a", "AUROC"), ("b", "bacc ")):
                x = np.array(res["deepfuse"][key]); y = np.array(res["final"][key])
                _, pv = ttest_rel(x, y)
                s = "***" if pv < .001 else "**" if pv < .01 else "*" if pv < .05 else "ns"
                print(f"    {lab}  final {y.mean():.4f} -> deepfuse {x.mean():.4f}  "
                      f"Δ={x.mean()-y.mean():+.4f}  p={pv:.2e} {s}")
                # Per-seed values are kept, not just their means: a paper table needs
                # dispersion, and the paired deltas are the only way to recompute an
                # effect size or a bootstrap CI later without re-running everything.
                rows.append({"dataset": ds, "K": K, "metric": lab.strip(), "topn": topn,
                             "select_on": select_on, "sel_budget": sel_budget,
                             "rank_metric": rank_metric,
                             "picks": [f"blk{b}/{p}" for p, b in picks],
                             "final": float(y.mean()), "deepfuse": float(x.mean()),
                             "final_std": float(y.std(ddof=1)),
                             "deepfuse_std": float(x.std(ddof=1)),
                             "final_seeds": [float(t) for t in y],
                             "deepfuse_seeds": [float(t) for t in x],
                             "n_seeds": int(len(x)),
                             "delta": float(x.mean() - y.mean()), "p_value": float(pv)})

    # Tag carries dataset list + any non-default selection setting, so a dermnet or
    # kshot-budget run cannot silently overwrite the reported ham/pad results.
    suffix = "" if datasets == ["ham", "pad"] else "_" + "_".join(datasets)
    if sel_budget != "full":
        suffix += f"_sel{sel_budget}"
    if rank_metric != "AUROC_bin":
        suffix += f"_{rank_metric}"
    tag = f"layer_fuse_top{topn}_{select_on}{suffix}"
    with open(os.path.join(RESULTS, f"{tag}.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n-> {RESULTS}/{tag}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["profile", "fuse"])
    ap.add_argument("--datasets", nargs="+", default=["ham", "pad"])
    ap.add_argument("--Ks", nargs="+", type=int, default=[1, 4, 16])
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--sel_seeds", type=int, default=5, help="seeds used to rank streams")
    ap.add_argument("--topn", type=int, default=8)
    ap.add_argument("--select_on", default="val", choices=["val", "test"],
                    help="'test' only to quantify selection optimism -- never report it")
    ap.add_argument("--rank_metric", default="AUROC_bin", choices=["AUROC_bin", "bacc"],
                    help="use bacc for dermnet, whose malignant grouping is a "
                         "single-class proxy; default keeps old runs reproducible")
    ap.add_argument("--sel_budget", default="full", choices=["full", "kshot"],
                    help="'kshot' caps stream selection at the same per-class label "
                         "budget as the support instead of using the whole val split")
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    if a.stage == "profile":
        run_profile(a.datasets, a.Ks, list(range(a.n_seeds)), rank_metric=a.rank_metric)
    else:
        run_fuse(a.datasets, a.Ks, a.topn, a.n_seeds, a.select_on, a.sel_seeds,
                 rank_metric=a.rank_metric, sel_budget=a.sel_budget)


if __name__ == "__main__":
    main()
