"""Few-shot classifier heads, ported line-by-line from LibFewShot.

Provenance
----------
Reference tree: /path/to/LibFewShot (commit as vendored).
Every head below was written by reading the reference *source file* and its
*config yaml*, not from memory. `PROVENANCE` at the bottom records, per head,
the exact file the math came from, the config the constants came from, and any
deviation forced by our setting. `fsl_heads_run.py --audit` prints that table so
each claim can be checked against the reference tree directly.

Why the heads and not the framework
-----------------------------------
LibFewShot's native protocol is episodic N-way K-shot over a backbone trained
from scratch at 84x84. Ours is a frozen 224x224 ViT-L scored over the full label
space. Running LibFewShot end-to-end would compare *backbones*, not methods. So
the inference-time head is ported and run on the identical cached features,
identical support sets and identical seeds as every other arm in this project.
LibFewShot is never imported: its requirements.txt uses open-ended `>=` pins that
would silently upgrade numpy/scipy/sklearn in the `panderm` env and break
reproducibility of already-published runs.

What changed in this revision (the previous version was wrong)
--------------------------------------------------------------
1. protonet: was cosine. LibFewShot's `ProtoLayer` defaults to `mode="euclidean"`
   and `ProtoNet.set_forward` never passes a mode, so the reference rule is
   **negative squared Euclidean on un-normalized features**. Cosine-to-prototype
   is a different reference method (`MetaBaseline`), now added separately.
2. negative_margin: was margin=-0.3, scale=30 with a weight-normalised layer.
   All three are wrong. In `config/classifiers/Negative_Margin.yaml`, -0.02/30
   are the *base pre-training* margin/scale; the episodic head uses
   `inner_margin: 0.0` and `inner_scale_factor: 5.0`, and `NegLayer` hard-
   normalises its weight (`F.normalize`), so unlike DistLinear it has **no**
   per-class learnable magnitude.
3. matchingnet: the previous docstring cited "LibFewShot core/model/metric/..
   matchingnet". **That file does not exist.** The only MatchingNet in the tree
   is `core/model/meta/matchingnet_ifsl.py` (`DMatchingNet`, the IFSL causal
   variant), which needs 16 meta-trained BiLSTM/FCE blocks plus two external
   pre-trained artefacts and consumes raw images. It is not portable. The
   classic Vinyals rule is kept here but is flagged `libfewshot=None` and must
   not be described as a LibFewShot-validated baseline.
4. dsn: added the reference's `logits /= d` (`normalize=True` default) and the
   `scale` parameter (init 1.0).
5. r2d2: the reference `binv` uses `torch.solve`, removed in torch>=1.13. Uses
   `torch.linalg.solve`, as the reference's own `r2d2_mcl.py` does.
6. added baseline, metabaseline, rfs -- three further heads that the source
   survey confirmed are faithfully portable to pooled features.

Hyperparameters
---------------
Reference constants (gamma=50, temp=10, scale_factor=2/5) were tuned against
meta-trained Conv64F/ResNet-12 features at 84x84. On z-scored 1024-d ViT-L
features the feature scale is entirely different, so those constants are not
transferable -- pinning them would handicap the baselines. Each head therefore
carries both:
  * `ref`  -- the reference constants, reported verbatim as `<head>@ref`
  * `grid` -- the same scale-sensitive constant re-selected on the K-shot-capped
              validation split, i.e. the *same* budget our own method gets
The headline number is the val-selected one, which is the setting generous to
the baseline. Both are written to the results JSON.

Every head has the same signature and returns raw logits:

    head(Zs, ys, Zq, C, **hp) -> [n_query, C]

  Zs : [n_support, d]  support features (z-scored, as in every other arm)
  ys : [n_support]     support labels, int64, same device
  Zq : [n_query, d]    query features
  C  : int             number of classes in the full label space

Classes absent from the support set are masked to -inf, so they receive exactly
zero probability and can never be predicted. (The older arms used -1e4; that is
a soft version of the same thing, but -1e4 is not safe here because the
Euclidean heads produce logits of order -1e3..-1e4 themselves.)
"""
import math
from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F

# DistLinear's weight_norm(dim=0) is implemented explicitly as g * v/||v|| in
# `_train_support` rather than by calling torch.nn.utils.weight_norm, because
# the module form cannot carry a leading problem axis. The parameterisation and
# its initialisation (g <- ||W_init||) are identical.


# --------------------------------------------------------------- shared utils
def _seen_mask(ys, C, device):
    seen = torch.zeros(C, dtype=torch.bool, device=device)
    seen[torch.unique(ys)] = True
    return seen


def _mask_unseen(logits, seen):
    out = logits.clone()
    out[:, ~seen] = -torch.inf
    return out


def _class_means(Zs, ys, C):
    """Prototypes. Reference: `torch.mean(support_feat, dim=2)` over the shot
    axis, on raw features -- no normalisation before the mean."""
    P = torch.zeros(C, Zs.shape[1], device=Zs.device, dtype=Zs.dtype)
    for c in range(C):
        m = ys == c
        if m.any():
            P[c] = Zs[m].mean(0)
    return P


# --------------------------------------------------------------- ProtoNet
def protonet(Zs, ys, Zq, C, **_):
    """Negative squared Euclidean distance to the class mean.

    core/model/metric/proto_net.py, `ProtoLayer.forward`:
        "euclidean": -torch.sum(torch.pow(x.unsqueeze(2) - y.unsqueeze(1), 2), dim=3)
    `ProtoNet.set_forward` calls the layer without a `mode`, so `mode="euclidean"`
    is what ProtoNet actually runs. There is no temperature, no scale, and the
    features are **not** L2-normalised anywhere on this path.
    config/classifiers/Proto.yaml: `kwargs: ~` -- zero head hyperparameters.
    Snell et al., NeurIPS 2017.
    """
    P = _class_means(Zs, ys, C)
    out = torch.stack([-((Zq - P[c]) ** 2).sum(1) for c in range(C)], dim=1)
    return _mask_unseen(out, _seen_mask(ys, C, Zs.device))


def proto_cosine_equiv(Zs, ys, Zq, C, **_):
    """The *old* cosine prototype rule, kept only so the equivalence regression
    can still prove this module shares no mutable state with the original
    pipeline. Byte-identical to `fsl_baselines.proto_logits`, including its
    -1e4 mask. Not a LibFewShot method and not reported in any table."""
    P = torch.zeros(C, Zs.shape[1], device=Zs.device)
    seen = torch.zeros(C, dtype=torch.bool, device=Zs.device)
    for c in range(C):
        m = ys == c
        if m.any():
            P[c] = Zs[m].mean(0)
            seen[c] = True
    P = F.normalize(P, dim=1)
    out = F.normalize(Zq, dim=1) @ P.T
    out[:, ~seen] = -1e4
    return out


# ------------------------------------------------------------- Meta-Baseline
def metabaseline(Zs, ys, Zq, C, temp=10.0, **_):
    """Cosine to prototypes, scaled by a learnable temperature.

    core/model/metric/meta_baseline.py: `ProtoLayer_temperature` defaults to
    `mode="cos_sim"`, which normalises **both** sides
    (`F.normalize(x, p=2, dim=-1)` and the transposed normalised prototypes),
    and `MetaBaseline.set_forward` multiplies the result by
    `self.temp = nn.Parameter(torch.tensor(10.))`.
    The temperature is meta-learned; 10.0 is its initialisation and the only
    defined value without meta-training.
    Chen et al., ICCV 2021.
    """
    P = _class_means(Zs, ys, C)
    out = temp * (F.normalize(Zq, p=2, dim=-1) @ F.normalize(P, p=2, dim=-1).T)
    return _mask_unseen(out, _seen_mask(ys, C, Zs.device))


# ------------------------------------------------------------------- R2D2
def r2d2(Zs, ys, Zq, C, gamma=50.0, alpha=1.0, beta=0.0, **_):
    """Closed-form ridge regression in the dual.

    core/model/meta/r2d2.py, `R2D2Layer.forward`:
        W     = support^T @ inv(support @ support^T + gamma*I) @ one_hot(y)
        logit = alpha * (query @ W) + beta
    Features are **not** normalised, which is why gamma is scale-sensitive.
    config/classifiers/R2D2.yaml has `kwargs: ~`; alpha/beta/gamma are
    nn.Parameters hardcoded in `R2D2Layer.__init__` to 1.0 / 0.0 / 50.0.
    alpha>0 and beta are a per-row affine map and cannot change the ranking;
    gamma can, so it is the one exposed to selection.
    The reference `binv` calls `torch.solve`, removed in torch>=1.13; the
    reference's own r2d2_mcl.py uses `torch.linalg.solve`, followed here.
    Bertinetto et al., ICLR 2019.
    """
    Y = F.one_hot(ys.long(), C).to(Zs.dtype)
    G = Zs @ Zs.T
    G = G + gamma * torch.eye(G.shape[0], device=G.device, dtype=G.dtype)
    W = Zs.T @ torch.linalg.solve(G, Y)
    return _mask_unseen(alpha * (Zq @ W) + beta, _seen_mask(ys, C, Zs.device))


# --------------------------------------------------------------------- DSN
def dsn(Zs, ys, Zq, C, scale=1.0, normalize=True, **_):
    """Adaptive subspace: negative residual after projecting onto a class subspace.

    core/model/metric/dsn.py, `DSNLayer.forward`:
        UU       = svd(support.permute(0,1,3,2))[0]     # SVD of the (d x shot) block
        subspace = UU[:, :, :, :shot_num-1]             # rank shot-1, un-centred
        dist     = sum((query - subspace^T subspace query)^2, -1)
        logits   = -dist / d      (normalize=True is the default)
        logits   = logits * self.scale                  (enable_scale=True, init 1.0)
    `self.scale` is a meta-learned nn.Parameter; 1.0 is its initialisation.
    config/dsn.yaml passes only `discriminative`, which affects the *loss*, not
    inference.

    K=1 deviation: the reference has no valid subspace at 1 shot (rank shot-1=0)
    and works around it by stacking `torch.flip(support_image, dims=[3])` -- an
    image-space op with no equivalent on cached features. A class with a single
    shot therefore falls back to the squared distance to that shot. Documented
    rather than silently approximated. (Incidentally, `dims=[3]` on the 6-D
    (E,way,shot,c,h,w) tensor is the *channel* axis, so the reference's
    "horizontal flip" is actually an RGB reversal -- likely a bug upstream.)
    Simon et al., CVPR 2020.
    """
    d = Zs.shape[1]
    scores = torch.full((Zq.shape[0], C), -torch.inf,
                        device=Zs.device, dtype=Zs.dtype)
    for c in range(C):
        m = ys == c
        if not m.any():
            continue
        Xc = Zs[m].double()                                     # [shot, d]
        k = Xc.shape[0] - 1
        if k < 1:
            dist = ((Zq.double() - Xc) ** 2).sum(1)
        else:
            U = torch.linalg.svd(Xc.T, full_matrices=False)[0][:, :k]   # [d, k]
            Qd = Zq.double()
            dist = ((Qd - Qd @ U @ U.T) ** 2).sum(1)
        out = -dist
        if normalize:
            out = out / d
        scores[:, c] = (out * scale).to(Zs.dtype)
    return scores


# ------------------------------------------------- support-set trained heads
#
# Baseline, Baseline++ and Negative-Margin all fit a fresh head on the support
# set with the reference's inner loop:
#
#   core/model/finetuning/baseline_plus.py `set_forward_adaptation`:
#       for epoch in range(inner_train_iter):                  # 100
#           rand_id = torch.randperm(way*shot)
#           for i in range(0, way*shot, inner_batch_size):     # 4
#               sel  = rand_id[i : min(i+bs, n)]
#               loss = CrossEntropy(head(support[sel]), target[sel]); step
#
# That loop is 100 x ceil(n/4) sequential optimizer steps on a head with a few
# thousand parameters -- entirely Python and kernel-launch bound, not compute
# bound. Fitting them one (seed, hyperparameter) at a time wasted ~95% of the
# wall clock.
#
# So the loop is written **once**, over a leading problem axis B: `_train_support`
# takes Zs of shape [B, n, d] and trains B independent heads simultaneously. The
# per-problem loss is a mean over its own minibatch and the B losses are SUMMED,
# so dL/dW_b is exactly what problem b would have seen alone; Adam and SGD are
# elementwise, so their updates are identical to training b by itself. Each
# problem draws its own permutation each epoch, as in the reference.
#
# The public heads accept either [n, d] (one problem) or [B, n, d] (many) and
# return [n_query, C] or [B, n_query, C] to match. There is only one code path,
# so the batched and unbatched cases cannot drift apart; `fsl_heads_run.py
# --vtest` additionally checks B-at-once against B-one-at-a-time numerically.


def _uniform_(shape, fan_in, device):
    """nn.Linear's default init: U(-1/sqrt(fan_in), +1/sqrt(fan_in)).

    `kaiming_uniform_(w, a=sqrt(5))` -- which both nn.Linear and the reference's
    NegLayer use -- reduces to exactly this bound. It is spelled out rather than
    called because torch computes fan_in from dim 1 of a 3-D tensor, which would
    be wrong once a problem axis is prepended.
    """
    b = 1.0 / math.sqrt(fan_in)
    return torch.empty(shape, device=device).uniform_(-b, b)


def _train_support(kind, Zs, ys, C, *, scale, margin, iters, batch, opt_name,
                   opt_kw, w0=None, perms=None):
    """Train B independent support-set heads at once. Returns a predict closure.

    kind:
      "linear"  nn.Linear(d, C) with bias        -- Baseline
      "dist"    DistLinear, weight_norm(dim=0)   -- Baseline++
      "neg"     NegLayer, hard-normalised weight -- Negative-Margin
    w0 / perms are test hooks (see --vtest); leave them None in normal use.
    """
    B, n, d = Zs.shape
    dev = Zs.device
    y = ys.long()
    W0 = _uniform_((B, C, d), d, dev) if w0 is None else w0.clone()

    if kind == "linear":
        W = W0.requires_grad_()
        bias = _uniform_((B, C), d, dev).requires_grad_()
        params = [W, bias]
    elif kind == "dist":
        # weight_norm(fc, "weight", dim=0) splits W into a unit-norm direction v
        # and a learnable per-class magnitude g, initialised to ||W_init||.
        v = W0.requires_grad_()
        g = W0.detach().norm(p=2, dim=-1, keepdim=True).clone().requires_grad_()
        params = [v, g]
    elif kind == "neg":
        W = W0.requires_grad_()
        params = [W]
    else:
        raise ValueError(kind)

    def forward(x, label=None, idx=None):
        """x: [B, m, d] or [m, d] (shared across problems). -> [B, m, C]"""
        sl = slice(None) if idx is None else idx
        if kind == "linear":
            out = torch.matmul(x, W[sl].transpose(-1, -2)) + bias[sl].unsqueeze(-2)
            return out
        if kind == "dist":
            # DistLinear.forward: x / (||x||_2 + 1e-5), then the weight-normed fc
            xn = x / (x.norm(p=2, dim=-1, keepdim=True) + 0.00001)
            Wn = g[sl] * v[sl] / v[sl].norm(p=2, dim=-1, keepdim=True)
            return scale * torch.matmul(xn, Wn.transpose(-1, -2))
        # "neg": both sides hard-normalised, margin only on the true class
        cos = torch.matmul(F.normalize(x, dim=-1),
                           F.normalize(W[sl], dim=-1).transpose(-1, -2))
        if label is None:
            return cos * scale
        oh = F.one_hot(label, C).bool()
        return torch.where(oh, cos - margin, cos) * scale

    opt = (torch.optim.Adam(params, **opt_kw) if opt_name == "Adam"
           else torch.optim.SGD(params, **opt_kw))
    bidx = torch.arange(B, device=dev).unsqueeze(1)
    with torch.enable_grad():
        for ep in range(iters):
            # one independent permutation per problem, as `torch.randperm` gives
            # per episode in the reference
            perm = (torch.argsort(torch.rand(B, n, device=dev), dim=1)
                    if perms is None else perms[ep])
            for i in range(0, n, batch):
                sel = perm[:, i:min(i + batch, n)]                  # [B, m]
                xb, yb = Zs[bidx, sel], y[bidx, sel]                # [B,m,d] [B,m]
                lg = forward(xb, yb)
                # per-problem mean, then SUM over problems: each problem's
                # gradient is exactly what it would be if trained alone
                loss = F.cross_entropy(lg.reshape(-1, C), yb.reshape(-1),
                                       reduction="none").view(B, -1).mean(1).sum()
                opt.zero_grad()
                loss.backward()
                opt.step()

    def predict(Zq, idx=None):
        with torch.no_grad():
            return forward(Zq, None, idx)
    return predict


def _as_batched(Zs, ys):
    """Accept [n,d]/[n] or [B,n,d]/[B,n]; report whether a dim was added."""
    if Zs.dim() == 2:
        return Zs.detach().unsqueeze(0), ys.unsqueeze(0), True
    return Zs.detach(), ys, False


def _finish(out, ys, C, squeezed):
    """Mask classes absent from each problem's own support set."""
    seen = torch.zeros(ys.shape[0], C, dtype=torch.bool, device=out.device)
    seen.scatter_(1, ys.long(), True)
    out = out.masked_fill(~seen.unsqueeze(1), -torch.inf)
    return out.squeeze(0) if squeezed else out


# Per-head arguments to `_train_support`, straight from each reference config.
# scale_factor for Baseline++ is derived from the class count inside DistLinear
# (2 for <=200 classes, i.e. every dataset here), not configured.
_TRAINED = {
    "baseline": lambda C, hp: dict(
        kind="linear", scale=1.0, margin=0.0, opt_name="Adam",
        opt_kw={"lr": hp.get("lr", 1e-3)}),
    "baseline_pp": lambda C, hp: dict(
        kind="dist", margin=0.0, opt_name="Adam",
        scale=(2.0 if C <= 200 else 10.0) if hp.get("scale_factor") is None
        else hp["scale_factor"],
        opt_kw={"lr": hp.get("lr", 1e-2)}),
    "negative_margin": lambda C, hp: dict(
        kind="neg", scale=hp.get("scale_factor", 5.0),
        margin=hp.get("margin", 0.0), opt_name="SGD",
        opt_kw={"lr": hp.get("lr", 0.01), "momentum": hp.get("momentum", 0.9),
                "dampening": hp.get("dampening", 0.9),
                "weight_decay": hp.get("weight_decay", 0.001)}),
}


def fit_trained(name, Zs, ys, C, **hp):
    """Fit the B support-set heads once and hand back a reusable predictor.

    The runner needs two predictions from one fit -- the K-shot validation split
    to select a hyperparameter, and the test split to score -- so the fit is
    exposed separately rather than hidden inside the head function. Returns
    (predict, ys_batched, squeezed); feed `predict` output through `_finish`.
    """
    Z, y, sq = _as_batched(Zs, ys)
    cfg = _TRAINED[name](C, hp)
    pred = _train_support(Zs=Z, ys=y, C=C, iters=hp.get("iters", 100),
                          batch=hp.get("batch", 4), w0=hp.get("w0"),
                          perms=hp.get("perms"), **cfg)
    return pred, y, sq


def baseline(Zs, ys, Zq, C, **hp):
    """Baseline: a plain linear head (with bias) fit on the support set.

    core/model/finetuning/baseline.py `set_forward_adaptation`:
        clf = nn.Linear(feat_dim, way_num)   # default init, bias=True
    config/classifiers/Baseline.yaml: inner_optim Adam(lr=1e-3),
    inner_train_iter 100, inner_batch_size 4. No feature normalisation anywhere.
    Chen et al., ICLR 2019 ("A Closer Look at Few-shot Classification").
    """
    pred, y, sq = fit_trained("baseline", Zs, ys, C, **hp)
    return _finish(pred(Zq), y, C, sq)


def baseline_pp(Zs, ys, Zq, C, **hp):
    """Baseline++: the DistLinear cosine head fit on the support set.

    core/model/finetuning/baseline_plus.py, `DistLinear`:
        fc = nn.Linear(in, out, bias=False); weight_norm(fc, "weight", dim=0)
        scale_factor = 2 if out_channel <= 200 else 10
        x_normalized = x / (||x||_2 + 0.00001)
        score        = scale_factor * fc(x_normalized)
    weight_norm(dim=0) keeps a learnable per-class magnitude g_c, so the logit is
    s * g_c * cos(x, v_c) -- NOT a hard cosine. The reference comments that this
    is deliberate (CloserLookFewShot issues #4 and #8).
    config/classifiers/Baseline++.yaml: inner_optim Adam(lr=1e-2),
    inner_train_iter 100, inner_batch_size 4.
    Chen et al., ICLR 2019.
    """
    pred, y, sq = fit_trained("baseline_pp", Zs, ys, C, **hp)
    return _finish(pred(Zq), y, C, sq)


def negative_margin(Zs, ys, Zq, C, **hp):
    """Negative-Margin: hard-cosine head with an additive margin on the true class.

    core/model/finetuning/negative_margin.py, `NegLayer`:
        weight = Parameter(out, in); kaiming_uniform_(weight, a=sqrt(5))
        cosine = F.linear(F.normalize(feature), F.normalize(self.weight))
        if label is None:  return cosine * scale_factor          # inference
        output = where(one_hot(label), cosine - margin, cosine) * scale_factor
    Both sides are hard-normalised every forward, so there is no per-class
    magnitude -- that is the concrete difference from DistLinear.

    config/classifiers/Negative_Margin.yaml, `inner_param`:
        inner_optim: SGD(lr=0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)
        inner_train_iter: 100
        inner_scale_factor: 5.0
        inner_margin: 0.0
    The -0.02 margin and scale 30 in the same file are `margin`/`scale_factor`
    for the **base-class pre-training** head, which is discarded before the
    episode. So at episode time the reference NegNet is a scale-5 hard-cosine
    head with no margin, and its only difference from Baseline++ is the absence
    of DistLinear's per-class learnable magnitude. That is reported as-is; the
    margin is still exposed to selection so the method is not judged solely at
    the setting where it degenerates.
    `inner_batch_size` in the yaml is dead -- negative_margin.py:95 hardcodes 4.
    Liu et al., ECCV 2020.
    """
    pred, y, sq = fit_trained("negative_margin", Zs, ys, C, **hp)
    return _finish(pred(Zq), y, C, sq)


def rfs(Zs, ys, Zq, C, penalty_C=1.0, **_):
    """RFS / SKD: multinomial logistic regression on L2-normalised support features.

    core/model/finetuning/rfs_model.py `set_forward_adaptation`:
        LogisticRegression(penalty="l2", random_state=0, C=1.0,
                           solver="lbfgs", max_iter=1000,
                           multi_class="multinomial")
        .fit(F.normalize(support_feat, p=2, dim=1), support_target)
    skd_model.py is identical except it omits C (sklearn default C=1.0), so one
    head covers both methods; they differ only in pre-training.
    `multi_class` is deprecated in sklearn>=1.5 and multinomial is already the
    lbfgs default for >2 classes, so it is omitted rather than pinned.
    The reference returns `classifier.predict(...)` (hard labels); that would
    make AUROC undefined, so `predict_log_proba` is used -- the same fitted
    model, read out as scores instead of argmax.
    Tian et al., ECCV 2020 / Rajasegaran et al., 2020.
    """
    from sklearn.linear_model import LogisticRegression

    Xs = F.normalize(Zs, p=2, dim=1).detach().cpu().numpy()
    Xq = F.normalize(Zq, p=2, dim=1).detach().cpu().numpy()
    yn = ys.detach().cpu().numpy()
    seen = np.unique(yn)
    out = np.full((Xq.shape[0], C), -np.inf, dtype=np.float64)
    if seen.size == 1:
        out[:, seen[0]] = 0.0
    else:
        clf = LogisticRegression(penalty="l2", random_state=0, C=penalty_C,
                                 solver="lbfgs", max_iter=1000)
        clf.fit(Xs, yn)
        out[:, clf.classes_.astype(int)] = clf.predict_log_proba(Xq)
    return torch.as_tensor(out, device=Zq.device, dtype=Zq.dtype)


# ------------------------------------------------------- NOT from LibFewShot
def matchingnet(Zs, ys, Zq, C, temp=10.0, **_):
    """Classic MatchingNet inference rule -- attention over support *examples*.

    NOT PORTED FROM LibFewShot. The reference tree contains no vanilla
    MatchingNet: `grep -ril matching` hits only core/model/meta/matchingnet_ifsl.py
    (`DMatchingNet`, the IFSL causal-intervention variant), which needs 16
    meta-trained BiLSTM + FullyContextualEmbedding blocks, an external
    pre-trained nn.Linear checkpoint and an external .npy class-mean bank, and
    consumes raw images rather than pooled features. None of that survives into
    a frozen-feature setting, so LibFewShot's MatchingNet is *not reproducible*
    here and this head must not be presented as one of its validated baselines.

    What is implemented is the original rule from Vinyals et al., NeurIPS 2016:
        a = softmax(temp * cos(q, s_i)) over support examples ; p = a @ Y_onehot
    Reported separately from the LibFewShot-sourced heads.
    """
    A = torch.softmax(temp * (F.normalize(Zq, dim=1) @ F.normalize(Zs, dim=1).T), dim=1)
    Y = F.one_hot(ys.long(), C).to(Zq.dtype)
    return _mask_unseen(torch.log(A @ Y + 1e-8), _seen_mask(ys, C, Zs.device))


def tip_adapter(Zs, ys, Zq, C, beta=5.5, alpha=1.0, **_):
    """Tip-Adapter training-free cache model. NOT in LibFewShot.

        A = exp(-beta * (1 - cos(q, s))) ; logits = alpha * A @ Y_onehot
    Only beta is selected: alpha>0 is a positive scale on the logits.
    Zhang et al., ECCV 2022.
    """
    A = torch.exp(-beta * (1.0 - F.normalize(Zq, dim=1) @ F.normalize(Zs, dim=1).T))
    Y = F.one_hot(ys.long(), C).to(Zq.dtype)
    return _mask_unseen(alpha * (A @ Y), _seen_mask(ys, C, Zs.device))


# ------------------------------------------------------------------ registry
Head = namedtuple("Head", "fn ref grid libfewshot config note")

HEADS = {
    "protonet": Head(
        protonet, {}, None,
        "core/model/metric/proto_net.py", "config/classifiers/Proto.yaml",
        "exact; euclidean mode, no hyperparameters"),
    "metabaseline": Head(
        metabaseline, {"temp": 10.0}, ("temp", (1.0, 5.0, 10.0, 20.0)),
        "core/model/metric/meta_baseline.py", "(constants are in __init__)",
        "temp is a meta-learned nn.Parameter; 10.0 is its init"),
    "r2d2": Head(
        r2d2, {"gamma": 50.0}, ("gamma", (1.0, 10.0, 50.0, 250.0, 1000.0)),
        "core/model/meta/r2d2.py", "config/classifiers/R2D2.yaml",
        "gamma is a meta-learned nn.Parameter tuned on un-normalised 640-d "
        "features; init 50.0"),
    "dsn": Head(
        dsn, {}, None,
        "core/model/metric/dsn.py", "config/dsn.yaml",
        "K=1 has no valid subspace; reference's image-flip workaround is "
        "unavailable on cached features, so K=1 falls back to distance-to-shot"),
    "baseline": Head(
        baseline, {}, None,
        "core/model/finetuning/baseline.py", "config/classifiers/Baseline.yaml",
        "exact; Adam(1e-3), 100 iters, batch 4"),
    "baseline_pp": Head(
        baseline_pp, {}, None,
        "core/model/finetuning/baseline_plus.py",
        "config/classifiers/Baseline++.yaml",
        "exact; DistLinear with weight_norm(dim=0), scale_factor=2 for C<=200"),
    "negative_margin": Head(
        negative_margin, {"margin": 0.0, "scale_factor": 5.0},
        ("margin", (0.0, -0.1, -0.3, -0.5)),
        "core/model/finetuning/negative_margin.py",
        "config/classifiers/Negative_Margin.yaml",
        "reference episodic margin is 0.0 / scale 5.0; -0.02 / 30 belong to the "
        "discarded pre-training head"),
    "rfs": Head(
        rfs, {"penalty_C": 1.0}, ("penalty_C", (0.1, 1.0, 10.0)),
        "core/model/finetuning/rfs_model.py (== skd_model.py at inference)",
        "config/classifiers/RFS.yaml",
        "predict_log_proba instead of predict, so AUROC is defined"),
    "matchingnet": Head(
        matchingnet, {"temp": 10.0}, ("temp", (1.0, 5.0, 10.0, 20.0)),
        None, None,
        "LibFewShot has no vanilla MatchingNet; DMatchingNet (IFSL) is not "
        "portable. Vinyals et al. 2016 rule, reported outside the ported set"),
    "tip": Head(
        tip_adapter, {"beta": 5.5}, ("beta", (1.0, 5.0, 5.5, 10.0)),
        None, None,
        "not a LibFewShot method; Zhang et al., ECCV 2022"),
}

# Heads that fit a head on the support set. These accept a leading problem axis,
# so the driver trains all seeds (and, for negative_margin, all grid values) in
# one call instead of one Python inner loop per (seed, hyperparameter).
TRAINED_HEADS = tuple(_TRAINED)

# LibFewShot's OWN reproduced accuracies, miniImageNet + Conv64F, 5-way, from
# reproduce/<Method>/README.md (the "reproduced" column, not the column quoting
# the original paper). This is the external sanity check on the port: a head
# whose relative standing here is wildly different from ours deserves scrutiny.
# Note the two entries with no number: LibFewShot ships configs for DSN and
# Meta-Baseline but publishes **no reproduction README** for either, so neither
# is validated by the library's own table and neither may be described as such.
LIBFEWSHOT_REPORTED = {                 # head -> (1-shot, 5-shot, config used)
    "r2d2":         (51.19, 67.29, "R2D2-miniImageNet--ravi-Conv64F-5-{1,5}-Table2"),
    "baseline_pp":  (48.86, 63.29, "BaselinePlus-miniImageNet--ravi-Conv64F-Table2"),
    "rfs":          (47.97, 65.88, "RFS-simple-miniImageNet--ravi-Conv64F-Table2 "
                                   "(SKD-Gen0 is 48.14 / 66.36)"),
    "protonet":     (47.05, 68.56, "ProtoNet-miniImageNet-Conv64F-5-{1,5}-Table2"),
    "baseline":     (44.90, 63.96, "Baseline-miniImageNet--ravi-Conv64F-Table2"),
    "dsn":          (None,  None,  "reproduce/DSN/ has configs but NO README"),
    "metabaseline": (None,  None,  "reproduce/MetaBaseline/ has configs but NO README"),
    "negative_margin": (None, None, "no reproduce/ directory at all"),
}

# Reference methods deliberately NOT ported, with the reason from the source.
# Printed by `fsl_heads_run.py --audit` so the exclusions are auditable too.
NOT_PORTED = {
    "DN4":          "top-k cosine over h*w local descriptors (query_feat.view(t,wq,c,h*w))",
    "ADM / ADM_KL": "per-image covariance from h*w descriptors; a pooled vector gives a rank-1 covariance",
    "ATL-Net":      "nn.Conv2d over (c,h,w) plus a meta-trained AEAModule threshold MLP",
    "CAN":          "cross-attention ConvBlock on the HW x HW correlation map",
    "FRN":          "self.resolution=25 spatial positions; degenerate at resolution 1",
    "MCL":          "Katz centrality over a query-patch x support-patch graph",
    "ConvMNet":     "local-descriptor covariance + meta-trained nn.Conv1d aggregation",
    "RelationNet":  "meta-trained Conv2d comparator over concatenated feature maps",
    "RENet":        "nn.Unfold self-correlation + SepConv4d cross-correlation on feature maps",
    "CPEA":         "needs the 14x14 patch-token grid; Mlp(in_features=196**2) is meta-trained",
    "FEAT":         "prototypes adapted by a meta-trained MultiHeadAttention; at random init it is noise",
    "LEO / VERSA":  "classifier weights are generated by meta-trained encoders/decoders",
    "METAL":        "the loss function itself is a meta-trained network",
    "MAML / BOIL":  "inner loop takes gradients through the backbone on raw images",
    "MTL":          "convert_mtl_module injects meta-learned scale/shift into backbone convs",
    "DMatchingNet": "16 meta-trained BiLSTM/FCE blocks + two external pre-trained artefacts; consumes images",
    "FGFL":         "DCT frequency branch + backward hooks on the backbone; needs raw images",
    "DeepBDC":      "the head is a ProtoLayer, but its input must be the r(r+1)/2 BDC matrix from the "
                    "backbone pooling layer -- porting the head without the representation would not be DeepBDC",
    "MetaBaselineKendall": "portable in principle, but Kendall's tau over C(1024,2)=523,776 coordinate "
                    "pairs per (query, prototype) is cost-prohibitive at our query counts; the reference "
                    "only ever runs it at d=640 with 5 ways",
    "S2M2":         "inference head is distLinear, identical to Baseline++ -- covered, not duplicated",
    "SKD":          "inference is identical to RFS -- covered by the `rfs` head",
    "ANIL":         "head init is meta-learned; with random init it degenerates to `baseline`",
}
