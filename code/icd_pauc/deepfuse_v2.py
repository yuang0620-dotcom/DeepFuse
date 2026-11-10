"""DeepFuse-Adaptive v2: pick the fusion CONFIGURATION, not just its width.

v1 (`deepfuse_adaptive.py`) decided how many streams to fuse by counting how many
individual streams beat the last layer on the selection split. That rule can only
detect "no single stream is any good" -- which is why it rescued DermNet (0 streams
eligible -> fall back, +0.035 AUROC over fixed fusion) but not DDI, where 54 streams
individually beat the last layer at K=16 and the rule therefore fused, losing 0.023
AUROC against simply not fusing. The per-stream signal never asked whether the
FUSED model is any good. It is brittle too: `n_eligible` thresholds 96 scores against
base_val with a strict >, so one sample flipping on a K*C-row val split moves the
count (26 vs 19 on DDI) and with it the chosen width.

v2 asks the question directly. It evaluates a nested ladder of configurations on the
selection split and picks one:

    base        last layer + linear probe                     (D*C + C params)
    mean-n      uniform mean of the top-n streams + probe     (D*C + C params)
    adapter-n   LDC per-stream adapter, uniform weights       (~n*0.33M params)
                n in {1,2,4,8,12}

Two things fall out of putting `base` in the ladder as a first-class candidate:

  * Graceful degradation stops being a special case. Fusion is chosen only when it
    beats not-fusing on val, so "never harm the task" is a property of the rule
    rather than a fallback branch bolted onto it.
  * `mean-n` gives fusion a low-capacity form. It has EXACTLY the baseline's
    parameter count -- the streams are averaged before the probe sees them -- so at
    K=1, where the adapter's 0.33M params/stream cannot be estimated from C labels,
    fusion can still contribute signal instead of being priced out.

Selection uses the one-standard-error rule (Breiman/Hastie ESL 7.10): take the best
mean val score, drop one standard error below it, return the SIMPLEST candidate above
that line. This is what removes v1's hard-coded `if K == 1: cap the width at 4`. At
K=1 the standard error across selection episodes is large, the line falls far, and
parsimony wins; at K=16 it is small and the adapter has to earn its parameters. The
K-dependence is inferred, not asserted.

What selection CANNOT fix, and why the hedged arm exists: on DDI the val and test
splits disagree about the base arm by 0.119 bacc (val_full 0.561, test 0.680) while
agreeing on every fusion arm to within 0.04. That is split shift, not sampling noise
-- four independent val subsamples all score base at 0.55-0.57 -- so no rule reading
DDI's val split can discover that base is the best arm there. `v2h` therefore averages
the selected arm's posterior with the base arm's, which halves the exposure to that
failure mode at the cost of half the gain where fusion genuinely wins.

Selection touches train + a K-shot-budget val subsample only, never test.

Usage:
  python -m icd_pauc.deepfuse_v2 --datasets ham --Ks 1 4 16 --n_seeds 20
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
from icd_pauc.train_icd_pauc import set_seed
from icd_pauc.metrics_pauc import evaluate_all
from icd_pauc.maf_head import (LDCAdapterHead, BatchedLDCAdapterHead, fit_adam,
                               fit_adam_batched, zscore)
from icd_pauc.layer_probe import load_split
from icd_pauc.fastprobe import batched_probe, stream_stats, stream_rows, stream_scores

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS = os.path.join(_HERE, "..", "..", "results")

LADDER_N = (1, 2, 4, 8, 12)
SEL_SEED_OFFSET = 1000      # selection episodes disjoint from the evaluation ones
VAL_SUBSAMPLE_SEED = 2000   # one fixed K-shot val subsample = one annotation budget
INFER_CHUNK = 4096


def ladder_arms(adapter_ladder=LADDER_N, mean_ladder=LADDER_N, hedge=False,
                lambdas=(0.5,)):
    """Candidates in ascending order of complexity -- the parsimony ordering the
    one-standard-error rule walks.

    With hedge=True each fusion arm also appears shrunk halfway toward the base
    posterior, (1-l)*base + l*arm for l=0.5. A mixture with the baseline has lower
    effective capacity than the pure arm but higher than the baseline, so it sits
    between them: base < (a, 0.5) < (a, 1.0) within each arm, arms still ordered by
    width and head type. This lets the SAME rule choose how far to trust the
    selection instead of hedging by fiat -- which matters because on HAM the hedge
    is the only significant arm at K=1 (+0.0247 bacc) yet costs 0.032 at K=16.
    """
    fusion = ([("mean", n) for n in mean_ladder]
              + [("adapter", n) for n in adapter_ladder])
    lams = (sorted(l for l in lambdas if 0.0 < l < 1.0) if hedge else [])
    out = [("base", 0, 1.0)]
    for kind, n in fusion:
        out += [(kind, n, l) for l in lams] + [(kind, n, 1.0)]
    return out


def arm_label(arm):
    kind, n = arm[0], arm[1]
    lam = arm[2] if len(arm) > 2 else 1.0
    if kind == "base":
        return "base"
    return f"{kind}-{n}" if lam == 1.0 else f"{kind}-{n}@{lam:g}"


# ------------------------------------------------------------------ features
def pick_rows(d, stats, picks, idx):
    """z-scored rows for a set of streams -> [len(idx), n_picks, D]."""
    return torch.stack([stream_rows(d, p, b, idx, *stats[(p, b)]) for p, b in picks], 1)


def mean_rows(d, stats, picks, idx):
    """uniform mean over streams -> [len(idx), D] (baseline parameter count)."""
    return pick_rows(d, stats, picks, idx).mean(1)


def all_stats(d):
    return {(p, b): stream_stats(d, p, b) for p in POOLS for b in range(1, N_BLOCKS + 1)}


# ----------------------------------------------------- arms -> posteriors
def probe_arm_probs(Zsup_list, ysup, Zev_list, C, chunk=8):
    """Fit one linear probe per (arm, seed) in batches -> list of [n_seeds,N_ev,C].

    Same solver and objective as the baseline probe, so `base` here IS the FSL
    baseline arm and the mean-fusion arms are compared to it on equal footing.
    """
    n_arm, n_sd = len(Zsup_list), ysup.shape[0]
    out = [None] * n_arm
    for c0 in range(0, n_arm, chunk):
        c1 = min(c0 + chunk, n_arm)
        sc = c1 - c0
        Zb = torch.cat([Zsup_list[a] for a in range(c0, c1)]).to(DEVICE)
        yb = ysup.repeat(sc, 1).reshape(sc * n_sd, -1).long().to(DEVICE)
        W, b, _ = batched_probe(Zb, yb, C)
        del Zb
        D = W.shape[1]
        Wg, bg = W.view(sc, n_sd, D, C), b.view(sc, n_sd, C)
        for i, a in enumerate(range(c0, c1)):
            Zev = Zev_list[a]
            parts = []
            for r0 in range(0, len(Zev), INFER_CHUNK):
                zc = Zev[r0:r0 + INFER_CHUNK].to(DEVICE)
                lg = torch.einsum("nd,jdc->jnc", zc, Wg[i]) + bg[i][:, None, :]
                parts.append(torch.softmax(lg, -1).cpu())
                del zc, lg
            out[a] = torch.cat(parts, 1).numpy()
        del W, b, Wg, bg
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    return out


def adapter_arm_probs(Ssup, ysup, Sev, Sva_es, yva_es, C, mal, seeds):
    """LDC adapter head (uniform depth weights) per seed -> [n_seeds, N_ev, C].

    Ssup [n_seeds, N_sup, n, D]; Sev [N_ev, n, D]. Adam with fixed epochs and
    val_select=False, exactly as fixed DeepFuse fits it -- the arms differ only in
    which streams and how many.
    """
    outs = []
    for j, sd in enumerate(seeds):
        set_seed(sd)
        head = fit_adam(LDCAdapterHead(Ssup.shape[2], Ssup.shape[3], C,
                                       learn_alpha=False),
                        Ssup[j].to(DEVICE), ysup[j].long().to(DEVICE),
                        Sva_es.to(DEVICE), yva_es.long().to(DEVICE), C, mal,
                        val_select=False)
        parts = []
        with torch.no_grad():
            for r0 in range(0, len(Sev), INFER_CHUNK):
                sc = Sev[r0:r0 + INFER_CHUNK].to(DEVICE)
                parts.append(F.softmax(head(sc), 1).cpu())
                del sc
        outs.append(torch.cat(parts).numpy())
        del head
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    return np.stack(outs)


def adapter_arm_probs_batched(Ssup, ysup, Sev, C, seeds, infer_chunk=INFER_CHUNK):
    """Same contract as `adapter_arm_probs`, all seeds fitted in one batched model.

    The per-seed heads are constructed exactly as the sequential path constructs
    them -- `set_seed(sd)` then `LDCAdapterHead(...)`, in the same order -- and their
    weights are then stacked, so the two paths start from identical parameters. See
    `BatchedLDCAdapterHead` for why the batched loss must be a sum.
    """
    E, N_sup, n, D = Ssup.shape
    heads = []
    for sd in seeds:
        set_seed(sd)
        heads.append(LDCAdapterHead(n, D, C, learn_alpha=False))
    head = fit_adam_batched(BatchedLDCAdapterHead(heads),
                            Ssup.to(DEVICE), ysup.long().to(DEVICE))
    # Sev [N_ev, n, D] is one eval set shared by every episode; [E, N_ev, n, D] gives
    # each episode its own (see `--val_pair` in deepfuse_v4).
    paired = Sev.dim() == 4
    n_ev = Sev.shape[1] if paired else len(Sev)
    parts = []
    with torch.no_grad():
        for r0 in range(0, n_ev, infer_chunk):
            sc = (Sev[:, r0:r0 + infer_chunk] if paired
                  else Sev[r0:r0 + infer_chunk]).to(DEVICE)
            zb = sc if paired else sc.unsqueeze(0).expand(E, -1, -1, -1)
            parts.append(F.softmax(head(zb), dim=-1).cpu())
            del sc, zb
    out = torch.cat(parts, dim=1).numpy()                     # [E, N_ev, C]
    del head
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return out


def scores_from_probs(probs, y_np, mal, metric):
    return [evaluate_all(p, y_np, mal, only=metric)[metric] for p in probs]


# ----------------------------------------------------------- selection rule
def marginal_one_se_pick(val_scores):
    """Textbook one-standard-error rule: simplest arm whose mean val score is
    within one standard error OF THE BEST ARM's own mean.

    Kept only as a control. It fails here for a structural reason: with a K-shot
    val budget the marginal standard error is dominated by how small the eval split
    is (~0.07 on HAM at K=1, where the split is 7 labels), not by any arm's
    instability. Every arm inherits roughly the same figure, so the band carries no
    information about overfitting risk -- it just gets wide enough to swallow the
    whole ladder and parsimony then picks the cheapest arm regardless of merit.
    """
    V = np.asarray(val_scores, dtype=float)
    mu = np.nanmean(V, axis=1)
    n = np.sum(~np.isnan(V), axis=1)
    sd = np.array([np.nanstd(v, ddof=1) if k > 1 else 0.0 for v, k in zip(V, n)])
    se = sd / np.sqrt(np.maximum(n, 1))
    best = int(np.nanargmax(mu))
    thresh = mu[best] - se[best]
    eligible = [i for i in range(len(V)) if mu[i] >= thresh]
    return eligible[0], {"mu": mu.tolist(), "se": se.tolist(), "best": best,
                         "thresh": float(thresh), "eligible": eligible}


def paired_one_se_pick(val_scores):
    """One-standard-error rule on the PAIRED difference against the best arm.

    Every arm is scored on the same selection episodes and the same val subsample,
    so most of the variance in a single arm's score is shared noise that cancels
    when arms are differenced. The question the rule should ask is not "how noisy
    is the best arm" but "is this simpler arm meaningfully worse than the best
    one", and that is a paired comparison. Arm i survives when

        mean_ep(score_i - score_best) + se_ep(score_i - score_best) >= 0

    i.e. it is not worse than the best by more than one standard error of the
    difference. The simplest survivor wins. The best arm always survives (its
    difference is identically zero), so the rule is well defined.

    This is what makes the selected capacity K-dependent without any rule about K:
    at K=1 the paired differences are genuinely inconclusive and parsimony decides,
    while at K=16 they are resolved and a complex arm has to earn its parameters.
    """
    V = np.asarray(val_scores, dtype=float)
    mu = np.nanmean(V, axis=1)
    best = int(np.nanargmax(mu))
    d = V - V[best]
    dm = np.nanmean(d, axis=1)
    n = np.sum(~np.isnan(d), axis=1)
    dsd = np.array([np.nanstd(x, ddof=1) if k > 1 else 0.0 for x, k in zip(d, n)])
    dse = dsd / np.sqrt(np.maximum(n, 1))
    eligible = [i for i in range(len(V)) if dm[i] + dse[i] >= 0]
    return eligible[0], {"mu": mu.tolist(), "dm": dm.tolist(), "dse": dse.tolist(),
                         "best": best, "eligible": eligible}


def v1_topn(mean_scores, base_score, K, topn_max=12, k1_cap=4, margin=0.0):
    """v1's rule, reproduced so both rules can be scored inside one run."""
    n_elig = int((mean_scores > base_score + margin).sum())
    n = max(0, min(n_elig, topn_max))
    if K == 1:
        n = min(n, k1_cap)
    return n, n_elig


# ---------------------------------------------------------------------- main
def run_dataset(ds, Ks, n_seeds, sel_seeds, rank_metric, topn_fixed, chunk, rows,
                val_seed=VAL_SUBSAMPLE_SEED, hedge=False, lambdas=(0.5,)):
    cfg = DATASETS[ds]
    C, mal = cfg["nb_classes"], cfg["malignant"]
    dtr, ytr = load_split(ds, "train")
    dva, yva = load_split(ds, "val")
    dte, yte = load_split(ds, "test")
    yte_np = yte.numpy()
    names = [(p, b) for p in POOLS for b in range(1, N_BLOCKS + 1)]

    print(f"\n{'='*100}\n== {ds}  C={C}  train={len(ytr)} val={len(yva)} test={len(yte)}"
          f"  rank={rank_metric}\n{'='*100}", flush=True)
    print("computing z-scoring statistics for 96 streams x 3 splits ...", flush=True)
    st_tr, st_va, st_te = all_stats(dtr), all_stats(dva), all_stats(dte)
    Btr = zscore(load_zoo(ds, "panderm_lp", "train")["embeddings"])
    Bva = zscore(load_zoo(ds, "panderm_lp", "val")["embeddings"])
    Bte = zscore(load_zoo(ds, "panderm_lp", "test")["embeddings"])
    te_idx = np.arange(len(yte))
    eval_seeds = list(range(n_seeds))

    for K in Ks:
        # ---------------- selection: train supports + one K-shot val subsample
        sel_sup = [build_support(ytr, "kshot", K, seed=SEL_SEED_OFFSET + sd)
                   for sd in range(sel_seeds)]
        ysup_sel = torch.stack([ytr[si] for si in sel_sup])
        va_idx = build_support(yva, "kshot", K, seed=val_seed)
        yva_es = yva[va_idx]
        yva_np = yva_es.numpy()
        sel_ep_seeds = [SEL_SEED_OFFSET + s for s in range(sel_seeds)]

        # per-stream val scores -> the ranking every fusion arm draws its streams from
        Zsup = [torch.stack([stream_rows(dtr, p, b, si, *st_tr[(p, b)])
                             for si in sel_sup]) for p, b in names]
        Zev = [stream_rows(dva, p, b, va_idx, *st_va[(p, b)]) for p, b in names]
        mean_sc, _, _ = stream_scores(Zsup, Zev, ysup_sel, yva_es, C, mal,
                                      rank_metric, chunk_streams=chunk)
        ranked = [names[i] for i in np.argsort(-mean_sc)]
        del Zsup, Zev

        # ---------------- score every ladder candidate on the selection split
        # A shrunk arm (1-l)*base + l*arm needs no extra fit, so the ladder is built
        # from a smaller set of PURE configurations and the mixtures are derived.
        arms = ladder_arms(hedge=hedge, lambdas=lambdas)
        pure_cfgs = ([("base", 0)] + [("mean", n) for n in LADDER_N]
                     + [("adapter", n) for n in LADDER_N])
        lin_cfgs = [c for c in pure_cfgs if c[0] in ("base", "mean")]
        ad_cfgs = [c for c in pure_cfgs if c[0] == "adapter"]

        def pure_label(kind, n):
            return "base" if kind == "base" else f"{kind}-{n}"

        def linear_inputs(sup_sets, ev_d, ev_stats, ev_idx, ev_base):
            sup, evs = [], []
            for kind, n in lin_cfgs:
                if kind == "base":
                    sup.append(torch.stack([Btr[si] for si in sup_sets]))
                    evs.append(ev_base)
                else:
                    picks = ranked[:n]
                    sup.append(torch.stack([mean_rows(dtr, st_tr, picks, si)
                                            for si in sup_sets]))
                    evs.append(mean_rows(ev_d, ev_stats, picks, ev_idx))
            return sup, evs

        def derive(pure_probs):
            """pure posteriors -> one posterior per ladder arm, mixtures included."""
            out = {}
            for a in arms:
                lam = a[2] if len(a) > 2 else 1.0
                p = pure_probs[pure_label(a[0], a[1])]
                out[arm_label(a)] = (p if lam == 1.0 else
                                     (1.0 - lam) * pure_probs["base"] + lam * p)
            return out

        vpure = {}
        sup, evs = linear_inputs(sel_sup, dva, st_va, va_idx, Bva[va_idx])
        lp = probe_arm_probs(sup, ysup_sel, evs, C)
        for (kind, n), p in zip(lin_cfgs, lp):
            vpure[pure_label(kind, n)] = p
        del sup, evs, lp

        for kind, n in ad_cfgs:
            picks = ranked[:n]
            Ssup = torch.stack([pick_rows(dtr, st_tr, picks, si) for si in sel_sup])
            Sva = pick_rows(dva, st_va, picks, va_idx)
            vpure[f"adapter-{n}"] = adapter_arm_probs(
                Ssup, ysup_sel, Sva, Sva, yva_es, C, mal, sel_ep_seeds)
            del Ssup, Sva

        varm = derive(vpure)
        val_scores = [scores_from_probs(varm[arm_label(a)], yva_np, mal, rank_metric)
                      for a in arms]
        del vpure, varm

        pick, diag = paired_one_se_pick(val_scores)
        pick_m, diag_m = marginal_one_se_pick(val_scores)
        pick_a = diag["best"]
        n_v1, n_elig = v1_topn(mean_sc, float(np.nanmean(val_scores[0])), K)

        print(f"\n  K={K}  selection ({sel_seeds} episodes, val budget {len(va_idx)} "
              f"labels)  base_val={np.nanmean(val_scores[0]):.4f}  #streams>base={n_elig}")
        print(f"    {'candidate':>12} {'val mu':>8} {'marg se':>8} "
              f"{'d vs best':>10} {'paired se':>10}    (best = "
              f"{arm_label(arms[diag['best']])})")
        for i, a in enumerate(arms):
            tag = ""
            if i == pick:
                tag = " <- v2"
            elif i in diag["eligible"]:
                tag = " *"
            print(f"    {arm_label(a):>12} {diag['mu'][i]:8.4f} {diag_m['se'][i]:8.4f} "
                  f"{diag['dm'][i]:10.4f} {diag['dse'][i]:10.4f}{tag}")
        print(f"    v2 (paired 1-SE) picks {arm_label(arms[pick])};  "
              f"v2a (argmax) {arm_label(arms[pick_a])};  "
              f"v2m (marginal 1-SE) {arm_label(arms[pick_m])};  "
              f"v1 {'base' if n_v1 == 0 else f'adapter-{n_v1}'}", flush=True)

        # ---------------- test evaluation of every candidate + v1's pick
        eval_sup = [build_support(ytr, "kshot", K, seed=sd) for sd in eval_seeds]
        ysup_ev = torch.stack([ytr[si] for si in eval_sup])
        vi0 = build_support(yva, "kshot", K, seed=eval_seeds[0] + 1000)

        tpure = {}
        sup, evs = linear_inputs(eval_sup, dte, st_te, te_idx, Bte)
        lp = probe_arm_probs(sup, ysup_ev, evs, C)
        for (kind, n), p in zip(lin_cfgs, lp):
            tpure[pure_label(kind, n)] = p
        del sup, evs, lp

        adapter_ns = sorted({n for _, n in ad_cfgs}
                            | ({topn_fixed} if topn_fixed > 0 else set())
                            | ({n_v1} if n_v1 > 0 else set()))
        for n in adapter_ns:
            picks = ranked[:n]
            Ssup = torch.stack([pick_rows(dtr, st_tr, picks, si) for si in eval_sup])
            Ste = pick_rows(dte, st_te, picks, te_idx)
            Sva0 = pick_rows(dva, st_va, picks, vi0)
            tpure[f"adapter-{n}"] = adapter_arm_probs(
                Ssup, ysup_ev, Ste, Sva0, yva[vi0], C, mal, eval_seeds)
            del Ssup, Ste, Sva0
            print(f"    tested adapter-{n}", flush=True)

        probs = derive(tpure)
        for k, v in tpure.items():           # keep fixed8 / v1's pick even if the
            probs.setdefault(k, v)           # ladder does not contain that width

        # ---------------- derived arms (free: the posteriors are already here)
        cand_keys = [arm_label(a) for a in arms]
        v2_key = cand_keys[pick]
        v2a_key, v2m_key = cand_keys[pick_a], cand_keys[pick_m]
        v1_key = "base" if n_v1 == 0 else f"adapter-{n_v1}"
        fixed_key = f"adapter-{topn_fixed}"
        probs["v2h"] = 0.5 * (probs["base"] + probs[v2_key])
        el_keys = [cand_keys[i] for i in diag["eligible"]]
        probs["ens1se"] = np.mean([probs[k] for k in el_keys], axis=0)

        # ---------------- report
        met = {k: {} for k in list(probs)}
        for k, pr in probs.items():
            for metric in ("AUROC_bin", "bacc"):
                met[k][metric] = np.array(scores_from_probs(pr, yte_np, mal, metric))

        print()
        for metric, lab in (("AUROC_bin", "AUROC"), ("bacc", "bacc")):
            base_a = met["base"][metric]
            oracle_key = max(cand_keys, key=lambda k: met[k][metric].mean())
            named = ((fixed_key, "fixed8"), (v1_key, "v1"), (v2m_key, "v2m"),
                     (v2_key, "v2"), (v2a_key, "v2a"), ("v2h", "v2h"),
                     ("ens1se", "ens"), (oracle_key, "oracle"))
            line = []
            for key, tag in named:
                x = met[key][metric]
                _, p = ttest_rel(x, base_a)
                line.append(f"{tag} {x.mean():.4f} ({x.mean()-base_a.mean():+.4f} "
                            f"p={p:.0e})")
            print(f"    {lab}  base {base_a.mean():.4f}")
            print(f"      " + " | ".join(line[:4]))
            print(f"      " + " | ".join(line[4:]))
            print(f"      candidates: " + "  ".join(
                f"{k}={met[k][metric].mean():.4f}" for k in cand_keys))

            rows.append({
                "dataset": ds, "K": K, "metric": lab, "rank_metric": rank_metric,
                "n_seeds": n_seeds, "sel_seeds": sel_seeds, "val_seed": val_seed,
                "v2_pick": v2_key, "v2a_pick": v2a_key, "v2m_pick": v2m_key,
                "v1_pick": v1_key, "fixed": fixed_key,
                "oracle_pick": oracle_key, "eligible": el_keys,
                "n_eligible_streams": n_elig,
                "ranked_top12": [list(t) for t in ranked[:12]],
                # per-episode val scores, so any other selection rule can be scored
                # post hoc from this file without re-running the expensive test arms
                "val_scores": [[float(v) for v in vs] for vs in val_scores],
                "val_mu": diag["mu"], "val_se": diag_m["se"],
                "val_dm": diag["dm"], "val_dse": diag["dse"],
                "val_best": arm_label(arms[diag["best"]]),
                "test_mean": {k: float(met[k][metric].mean()) for k in met},
                "test_seeds": {k: [float(v) for v in met[k][metric]] for k in met},
                "base": float(base_a.mean()),
                "fixed8": float(met[fixed_key][metric].mean()),
                "v1": float(met[v1_key][metric].mean()),
                "v2m": float(met[v2m_key][metric].mean()),
                "v2": float(met[v2_key][metric].mean()),
                "v2a": float(met[v2a_key][metric].mean()),
                "v2h": float(met["v2h"][metric].mean()),
                "ens1se": float(met["ens1se"][metric].mean()),
                "oracle": float(met[oracle_key][metric].mean()),
                "p_v2_vs_base": float(ttest_rel(met[v2_key][metric], base_a)[1]),
                "p_v2a_vs_base": float(ttest_rel(met[v2a_key][metric], base_a)[1]),
                "p_v2h_vs_base": float(ttest_rel(met["v2h"][metric], base_a)[1]),
                "p_ens_vs_base": float(ttest_rel(met["ens1se"][metric], base_a)[1]),
            })
        del probs, met


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ham"])
    ap.add_argument("--Ks", nargs="+", type=int, default=[1, 4, 16])
    ap.add_argument("--n_seeds", type=int, default=20)
    ap.add_argument("--sel_seeds", type=int, default=8)
    ap.add_argument("--rank_metric", default="bacc", choices=["AUROC_bin", "bacc"])
    ap.add_argument("--topn_fixed", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--tag", default="")
    ap.add_argument("--val_seed", type=int, default=VAL_SUBSAMPLE_SEED,
                    help="which K-shot val subsample is the selection budget")
    ap.add_argument("--hedge", action="store_true",
                    help="also put (1-l)*base + l*arm mixtures in the ladder, so the "
                         "same rule chooses how far to trust the selection")
    ap.add_argument("--lambdas", nargs="+", type=float, default=[0.5],
                    help="shrinkage weights added to the ladder under --hedge. A finer "
                         "grid buys resolution but enlarges the ladder, and selection "
                         "noise grows with it -- patch16 K=1 already lost 0.037 bacc "
                         "going from 11 to 21 candidates.")
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    rows = []
    for ds in a.datasets:
        run_dataset(ds, a.Ks, a.n_seeds, a.sel_seeds, a.rank_metric,
                    a.topn_fixed, a.chunk, rows, val_seed=a.val_seed,
                    hedge=a.hedge, lambdas=tuple(a.lambdas))
        tag = a.tag or "_".join(a.datasets)
        out = os.path.join(RESULTS, f"deepfuse_v2_{tag}_{a.rank_metric}.json")
        with open(out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\n-> {out}", flush=True)


if __name__ == "__main__":
    main()
