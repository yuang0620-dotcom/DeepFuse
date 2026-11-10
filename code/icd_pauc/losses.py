"""Losses for the ICD+pAUC method.

Self-contained pAUC surrogate (no libauc dependency). SOPA-style squared-hinge on
positive/negative score pairs, with positive-difficulty weighting so gradient
concentrates on hard (low-scoring) positives -- exactly the ones that govern the
high-TPR (TPR>=80%) region of the ROC that the clinical operating point cares about.

  L_pAUC = mean_{i in pos, j in neg}  w_i * [max(0, m - (s_i - s_j))]^2
  w_i    = softmax-free normalized (1 - s_i)   (harder positive -> larger weight)

Plus the two auxiliary terms from the design:
  CE(final_logits, y_multi)                 diagnostic supervision
  L1(final_logits, base_logits.detach())    prior anchoring (small-sample stabilizer)
"""
import torch
import torch.nn.functional as F


def pauc_surrogate_loss(s, y_bin, margin=1.0, tpr_focus=True, eps=1e-8):
    """s: [B] malignant aggregate score in [0,1]. y_bin: [B] {0,1}.
    Returns (loss, valid) where valid is False if the batch lacks pos or neg
    (caller should then skip / fall back to the frozen anchor)."""
    pos = s[y_bin == 1]
    neg = s[y_bin == 0]
    if pos.numel() == 0 or neg.numel() == 0:
        return s.new_tensor(0.0), False

    # pairwise margins: [P, N]
    diff = pos.unsqueeze(1) - neg.unsqueeze(0)          # s_i - s_j
    hinge = F.relu(margin - diff) ** 2                   # squared hinge

    if tpr_focus:
        w = (1.0 - pos).detach()                         # focus hard positives
        w = w / (w.sum() + eps)                          # normalize over positives
        loss = (w.unsqueeze(1) * hinge).sum(dim=1).sum() / neg.numel()
    else:
        loss = hinge.mean()
    return loss, True


def composite_loss(final_logits, base_logits, y_multi, malignant_idx,
                   lambda1=0.1, lambda2=1.0, margin=1.0, pauc_weight=1.0):
    """Full training loss. Returns (total, dict_of_parts, pauc_valid).
    pauc_weight=0 gives the +ICD(CE-only) ablation (arm ③) for pAUC attribution."""
    p = F.softmax(final_logits, dim=1)
    s = p[:, malignant_idx].sum(dim=1)                   # malignant aggregate [B]
    y_bin = torch.zeros_like(y_multi)
    for c in malignant_idx:
        y_bin = y_bin | (y_multi == c).long()

    l_pauc, valid = pauc_surrogate_loss(s, y_bin, margin=margin)
    l_ce = F.cross_entropy(final_logits, y_multi)
    l_l1 = F.l1_loss(final_logits, base_logits.detach())

    total = pauc_weight * l_pauc + lambda2 * l_ce + lambda1 * l_l1
    parts = {"pauc": float(l_pauc.detach()), "ce": float(l_ce.detach()),
             "l1": float(l_l1.detach()), "total": float(total.detach())}
    return total, parts, valid
