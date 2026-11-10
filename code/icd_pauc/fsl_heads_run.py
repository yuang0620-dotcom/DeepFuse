"""Driver for the LibFewShot-ported few-shot heads (icd_pauc.fsl_heads).

Isolation contract -- this script is additive only:
  * writes exclusively to results/fsl_heads_<tag>.json; never touches any
    existing result file
  * imports subsample / model_zoo / train_icd_pauc / metrics_pauc read-only and
    modifies none of them
  * reuses `build_support(y, "kshot", K, seed=sd)` for the support set and
    `seed=sd+1000` for the budget-capped validation split, byte-identical to
    fsl_baselines.py, so every new arm is *paired* with the existing ones and
    cross-table paired tests stay valid
  * never imports LibFewShot (open-ended `>=` pins could upgrade numpy/scipy in
    this environment and break reproducibility of already-published runs)

Each head is evaluated twice per cell:
  <head>@ref   the reference constants from LibFewShot's own config, verbatim
  <head>       the same scale-sensitive constant re-selected on the K-shot-capped
               validation split -- the same selection budget our method gets
Reporting both is the point: the reference constants were tuned for 84x84
meta-trained features and do not transfer to z-scored 1024-d ViT-L features, so
pinning them alone would understate the baselines.

Modes:
  --audit     print the provenance table: for each head, the LibFewShot source
              file and config it was ported from, plus the methods deliberately
              not ported and why. Runs nothing.
  --regress   recompute the cosine-prototype rule and diff it against the
              existing fsl_baselines_*.json cell by cell. Requires |delta| == 0,
              which proves this module shares no mutable state with the old
              pipeline. Writes nothing.
              NOTE: this checks `proto_cosine_equiv`, not `protonet`. The
              `protonet` head is now the reference's *euclidean* rule, so it is
              deliberately not equal to the old cosine arm.
  (default)   run all heads and write results/fsl_heads_<tag>.json

Usage:
  python -m icd_pauc.fsl_heads_run --audit
  python -m icd_pauc.fsl_heads_run --regress --datasets ham pad
  python -m icd_pauc.fsl_heads_run --datasets ham --heads protonet r2d2 --tag _ham
"""
import os
import sys
import json
import glob
import argparse
import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from icd_pauc.model_zoo import DATASETS, load_zoo
from icd_pauc.subsample import build_support
from icd_pauc.train_icd_pauc import set_seed
from icd_pauc.metrics_pauc import evaluate_all
from icd_pauc.maf_head import zscore
from icd_pauc.fsl_heads import (HEADS, NOT_PORTED, LIBFEWSHOT_REPORTED,
                                proto_cosine_equiv)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS = os.path.join(_HERE, "..", "..", "results")
LIBFEWSHOT = os.environ.get("LIBFEWSHOT", "/path/to/LibFewShot")


def audit():
    """Print the provenance table and verify each cited source file exists."""
    print(f"reference tree: {LIBFEWSHOT}")
    print(f"{'='*100}\nported heads (checked line by line against source and config)\n{'='*100}")
    missing = []
    for name, h in HEADS.items():
        src = h.libfewshot or "—  not in LibFewShot"
        ok = ""
        if h.libfewshot:
            path = os.path.join(LIBFEWSHOT, h.libfewshot.split(" ")[0])
            if os.path.exists(path):
                ok = "✓"
            else:
                ok = "✗ file not found"
                missing.append((name, path))
        print(f"\n  {name}")
        print(f"    source {src} {ok}")
        print(f"    config {h.config or '—'}")
        print(f"    reference {h.ref or '(no hyper-parameters)'}")
        print(f"    selection grid {h.grid[0] + '=' + str(h.grid[1]) if h.grid else '(none)'}")
        print(f"    note   {h.note}")

    print(f"\n{'='*100}\naccuracy reported by LibFewShot (miniImageNet + Conv64F, 5-way)\n"
          f"purpose: external consistency check; heads without a reported value must not be called 'validated by the library'\n{'='*100}")
    print(f"  {'method':<18}{'1-shot':>9}{'5-shot':>9}   source")
    for h, (s1, s5, where) in LIBFEWSHOT_REPORTED.items():
        a = f"{s1:.2f}" if s1 else "—"
        b = f"{s5:.2f}" if s5 else "—"
        print(f"  {h:<18}{a:>9}{b:>9}   {where}")
    unval = [h for h, (s1, _, _) in LIBFEWSHOT_REPORTED.items() if s1 is None]
    print(f"  → heads without a published value in the library: {', '.join(unval)}")

    print(f"\n{'='*100}\nmethods not ported (reasons from the source)\n{'='*100}")
    for name, why in NOT_PORTED.items():
        print(f"  {name:<22} {why}")

    n_lfs = sum(1 for h in HEADS.values() if h.libfewshot)
    print(f"\ntotal: {len(HEADS)} heads, {n_lfs} from LibFewShot, "
          f"{len(HEADS)-n_lfs} from other sources; exclusion reasons recorded for {len(NOT_PORTED)} method families.")
    if missing:
        print("\ncitation check failed:")
        for name, p in missing:
            print(f"  {name}: {p}")
        return False
    print("citation check passed: every cited source file exists in the reference tree.")
    return True


def load_split(ds, encoder, norm="transductive"):
    """norm='inductive' standardises val/test with TRAIN statistics; the default
    keeps the published split-wise standardisation bit-for-bit."""
    Z, y = {}, {}
    raw = {sp: load_zoo(ds, encoder, sp) for sp in ("train", "val", "test")}
    tr = raw["train"]["embeddings"]
    mu, sd = tr.mean(0, keepdim=True), tr.std(0, keepdim=True)
    for sp, d in raw.items():
        Z[sp] = (zscore(d["embeddings"]) if norm == "transductive"
                 else (d["embeddings"] - mu) / (sd + 1e-6)).to(DEVICE)
        y[sp] = d["labels"]
    return Z, y


def _predict(fn, Zs, ys, Zq, C, kw):
    with torch.no_grad():
        return F.softmax(fn(Zs, ys, Zq, C, **kw), 1).cpu().numpy()


def run_cell(head, Zs, ys, Zv, yv_np, Zte, C, mal, sel_metric):
    """One (dataset, K, seed) cell -> (ref prediction, selected prediction).

    Hyperparameters, when a head has one, are selected on the budget-capped
    validation split -- never on the full split.
    """
    p_ref = _predict(head.fn, Zs, ys, Zte, C, head.ref)
    if head.grid is None:
        return p_ref, p_ref, None
    key, values = head.grid
    best, best_v = -np.inf, values[0]
    for v in values:
        kw = dict(head.ref)
        kw[key] = v
        s = evaluate_all(_predict(head.fn, Zs, ys, Zv, C, kw), yv_np, mal)[sel_metric]
        if not np.isnan(s) and s > best:
            best, best_v = s, v
    if best_v == head.ref.get(key):
        return p_ref, p_ref, best_v          # selection landed on the reference
    kw = dict(head.ref)
    kw[key] = best_v
    return p_ref, _predict(head.fn, Zs, ys, Zte, C, kw), best_v


def run(datasets, Ks, n_seeds, encoder, metrics, heads, norm="transductive",
        pred_dir=None):
    rows = []
    for ds in datasets:
        cfg = DATASETS[ds]
        C, mal = cfg["nb_classes"], cfg["malignant"]
        Z, y = load_split(ds, encoder, norm)
        yte_np = y["test"].numpy()
        print(f"\n{'='*78}\n== {ds} ({encoder}, C={C}, {n_seeds} seeds)\n{'='*78}",
              flush=True)
        for K in Ks:
            arms = [a for h in heads for a in (h, h + "@ref")]
            acc = {a: {m: [] for m in metrics} for a in arms}
            picks = {h: [] for h in heads}
            pred = {h: np.zeros((n_seeds, len(yte_np)), np.int16) for h in heads}
            mscore = {h: np.zeros((n_seeds, len(yte_np)), np.float16) for h in heads}
            for sd in range(n_seeds):
                set_seed(sd)
                si = build_support(y["train"], "kshot", K, seed=sd)
                Zs, ys = Z["train"][si], y["train"][si].to(DEVICE)
                vi = build_support(y["val"], "kshot", K, seed=sd + 1000)
                Zv, yv_np = Z["val"][vi], y["val"][vi].numpy()
                for h in heads:
                    p_ref, p_sel, chosen = run_cell(
                        HEADS[h], Zs, ys, Zv, yv_np, Z["test"], C, mal, metrics[0])
                    if chosen is not None:
                        picks[h].append(chosen)
                    pred[h][sd] = p_sel.argmax(1)
                    mscore[h][sd] = p_sel[:, mal].sum(1)
                    ev_ref = evaluate_all(p_ref, yte_np, mal)
                    ev_sel = evaluate_all(p_sel, yte_np, mal)
                    for m in metrics:
                        acc[h][m].append(ev_sel[m])
                        acc[h + "@ref"][m].append(ev_ref[m])
            for m in metrics:
                line = f"  K={K:<3} {m:<10}"
                for h in heads:
                    for a in (h, h + "@ref"):
                        v = np.array(acc[a][m], float)
                        line += f"  {a}={np.nanmean(v):.4f}"
                        rows.append({"dataset": ds, "encoder": encoder, "K": K,
                                     "metric": m, "arm": a,
                                     "mean": float(np.nanmean(v)),
                                     "per_seed": v.tolist(),
                                     "selected": picks[h] if not a.endswith("@ref") else None,
                                     "libfewshot_src": HEADS[h].libfewshot})
                print(line, flush=True)
            if pred_dir:
                os.makedirs(pred_dir, exist_ok=True)
                np.savez_compressed(os.path.join(pred_dir, f"{ds}_K{K}.npz"),
                                    y_test=yte_np,
                                    **{f"{h}__pred": pred[h] for h in heads},
                                    **{f"{h}__mal": mscore[h] for h in heads})
    return rows


def regress(datasets, Ks, n_seeds, encoder, metrics):
    """Recompute the cosine-prototype rule here and diff against fsl_baselines_*."""
    ref = {}
    for f in glob.glob(os.path.join(RESULTS, "fsl_baselines*.json")):
        for r in json.load(open(f)):
            if r.get("arm") == "protonet":
                ref[(r["dataset"], r["K"], r["metric"])] = r["mean"]
    print(f"reference entries: {len(ref)} (protonet arm of fsl_baselines*.json)")
    print("compared with: proto_cosine_equiv (old cosine rule); the new protonet uses the "
          "reference Euclidean rule and by design differs from the old arm.")

    rows = []
    for ds in datasets:
        cfg = DATASETS[ds]
        C, mal = cfg["nb_classes"], cfg["malignant"]
        Z, y = load_split(ds, encoder)
        yte_np = y["test"].numpy()
        for K in Ks:
            acc = {m: [] for m in metrics}
            for sd in range(n_seeds):
                set_seed(sd)
                si = build_support(y["train"], "kshot", K, seed=sd)
                Zs, ys = Z["train"][si], y["train"][si].to(DEVICE)
                p = _predict(proto_cosine_equiv, Zs, ys, Z["test"], C, {})
                ev = evaluate_all(p, yte_np, mal)
                for m in metrics:
                    acc[m].append(ev[m])
            for m in metrics:
                rows.append({"dataset": ds, "K": K, "metric": m,
                             "mean": float(np.nanmean(np.array(acc[m], float)))})

    worst, n, miss = 0.0, 0, 0
    for r in rows:
        k = (r["dataset"], r["K"], r["metric"])
        if k not in ref:
            miss += 1
            continue
        d = abs(r["mean"] - ref[k])
        worst = max(worst, d)
        n += 1
        flag = "" if d == 0 else f"   <-- non-zero {d:.3e}"
        print(f"  {k[0]:<14} K={k[1]:<3} {k[2]:<10} new={r['mean']:.10f} "
              f"old={ref[k]:.10f} |Δ|={d:.3e}{flag}")
    print(f"\ncompared {n} cells ({miss} without reference), max |Δ| = {worst:.3e}")
    print("equivalence regression " + ("passed: new and old agree in every cell" if worst == 0.0
                        else "failed: non-zero differences, stop porting"))
    return worst == 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ham", "pad"])
    ap.add_argument("--Ks", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--n_seeds", type=int, default=20)
    ap.add_argument("--encoder", default="panderm_lp")
    ap.add_argument("--metrics", nargs="+", default=["bacc", "AUROC_bin"])
    ap.add_argument("--heads", nargs="+", default=list(HEADS))
    ap.add_argument("--tag", default="")
    ap.add_argument("--regress", action="store_true")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--norm", choices=["transductive", "inductive"],
                    default="transductive")
    ap.add_argument("--save_preds", action="store_true",
                    help="write per-image test predictions of the re-selected arm "
                         "to results/fsl_preds<tag>/")
    a = ap.parse_args()

    if a.audit:
        sys.exit(0 if audit() else 1)

    if a.regress:
        ok = regress(a.datasets, a.Ks, a.n_seeds, a.encoder, a.metrics)
        sys.exit(0 if ok else 1)

    pred_dir = (os.path.join(RESULTS, f"fsl_preds{a.tag or '_' + '_'.join(a.datasets)}")
                if a.save_preds else None)
    rows = run(a.datasets, a.Ks, a.n_seeds, a.encoder, a.metrics, a.heads,
               a.norm, pred_dir)
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, f"fsl_heads{a.tag or '_' + '_'.join(a.datasets)}.json")
    assert not os.path.basename(out).startswith("fsl_baselines"), "refusing to overwrite an existing result namespace"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
