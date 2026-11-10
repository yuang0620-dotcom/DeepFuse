"""Evaluation metrics for the clinical-operating-point screening task.

Binary (malignant-aggregate score s):
  - pAUC above {80,90,95}% TPR  (official ISIC-2024 definition via McClish un-scaling)
  - full AUROC
  - NNB@TPR (number needed to biopsy at a sensitivity operating point)
Multiclass (final_logits argmax):
  - acc, balanced acc, weighted F1
"""
import numpy as np
from sklearn.metrics import (roc_auc_score, roc_curve, balanced_accuracy_score,
                             accuracy_score, f1_score)


def pauc_above_tpr(y_true, y_score, min_tpr=0.80):
    """Partial AUC above `min_tpr` TPR, ISIC-2024 official formulation.
    Returns raw partial AUC in [0.5*max_fpr^2, max_fpr]."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    v_gt = np.abs(y_true - 1)          # flip labels
    v_pred = -1.0 * y_score           # flip scores -> maps TPR>=t region to FPR<=1-t
    max_fpr = 1.0 - min_tpr
    partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    # invert sklearn's McClish scaling to recover raw partial area
    partial_auc = 0.5 * max_fpr ** 2 + (max_fpr - 0.5 * max_fpr ** 2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)
    return float(partial_auc)


def nnb_at_tpr(y_true, y_score, min_tpr=0.90):
    """Number needed to biopsy at the operating point achieving >= min_tpr sensitivity.
    NNB = (TP + FP) / TP = 1 / precision at that threshold."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, thr = roc_curve(y_true, y_score)
    P = int((y_true == 1).sum())
    N = int((y_true == 0).sum())
    ok = np.where(tpr >= min_tpr)[0]
    if len(ok) == 0:
        return float("nan")
    i = ok[0]                          # lowest-threshold point reaching the TPR target
    tp = tpr[i] * P
    fp = fpr[i] * N
    if tp <= 0:
        return float("nan")
    return float((tp + fp) / tp)


def malignant_score(probs, malignant_idx):
    """probs: [N,C] softmax. Returns s = sum of malignant-class probs, [N]."""
    return np.asarray(probs)[:, malignant_idx].sum(axis=1)


def to_binary(y_multi, malignant_idx):
    y = np.asarray(y_multi)
    return np.isin(y, list(malignant_idx)).astype(int)


def evaluate_all(probs, y_multi, malignant_idx, only=None):
    """probs [N,C], y_multi [N]. Returns metric dict.

    `only` restricts the computation to the named metrics. The selection pipeline
    calls this once per (arm, episode) and reads a SINGLE key, so the default was
    spending most of its time on three pAUC curves, two NNB points and a weighted F1
    that the caller discards -- 8.9s of a 116s cell in one profile. Values are
    identical either way; only the unrequested keys are absent.
    """
    probs = np.asarray(probs)
    y_multi = np.asarray(y_multi)
    y_bin = to_binary(y_multi, malignant_idx)
    s = malignant_score(probs, malignant_idx)
    y_pred = probs.argmax(axis=1)

    if only is not None:
        want = {only} if isinstance(only, str) else set(only)
        lazy = {
            "pAUC@80": lambda: pauc_above_tpr(y_bin, s, 0.80),
            "pAUC@90": lambda: pauc_above_tpr(y_bin, s, 0.90),
            "pAUC@95": lambda: pauc_above_tpr(y_bin, s, 0.95),
            "AUROC_bin": lambda: (float(roc_auc_score(y_bin, s))
                                  if len(np.unique(y_bin)) > 1 else float("nan")),
            "NNB@80": lambda: nnb_at_tpr(y_bin, s, 0.80),
            "NNB@90": lambda: nnb_at_tpr(y_bin, s, 0.90),
            "acc": lambda: float(accuracy_score(y_multi, y_pred)),
            "bacc": lambda: float(balanced_accuracy_score(y_multi, y_pred)),
            "wF1": lambda: float(f1_score(y_multi, y_pred, average="weighted")),
            "n_pos": lambda: int(y_bin.sum()),
            "n": lambda: int(len(y_bin)),
        }
        unknown = want - set(lazy)
        if unknown:
            raise KeyError(f"unknown metric(s): {sorted(unknown)}")
        return {k: lazy[k]() for k in want}

    out = {
        # binary clinical-operating-point metrics on malignant aggregate score
        "pAUC@80": pauc_above_tpr(y_bin, s, 0.80),
        "pAUC@90": pauc_above_tpr(y_bin, s, 0.90),
        "pAUC@95": pauc_above_tpr(y_bin, s, 0.95),
        "AUROC_bin": float(roc_auc_score(y_bin, s)) if len(np.unique(y_bin)) > 1 else float("nan"),
        "NNB@80": nnb_at_tpr(y_bin, s, 0.80),
        "NNB@90": nnb_at_tpr(y_bin, s, 0.90),
        # multiclass diagnostic metrics
        "acc": float(accuracy_score(y_multi, y_pred)),
        "bacc": float(balanced_accuracy_score(y_multi, y_pred)),
        "wF1": float(f1_score(y_multi, y_pred, average="weighted")),
        "n_pos": int(y_bin.sum()),
        "n": int(len(y_bin)),
    }
    return out
