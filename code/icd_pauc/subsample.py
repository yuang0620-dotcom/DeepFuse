"""Few-shot support-set construction on cached features.

Two protocols share one entry point:
  - mode='kshot'   : per-class K examples  (formal N-way K-shot FSL; K=1/2/4/8/16)
  - mode='percent' : stratified fraction per class (fixes the non-stratified
                     head(len*percent) bug in datasets/derm_data.py)

Only the training split is subsampled; val/test (query) stay full-size.
"""
import numpy as np


def build_support(labels, mode, value, seed=42):
    """labels: 1D int array/tensor of multiclass labels for the train split.
    Returns a sorted np.array of selected indices into the train split.
    """
    y = np.asarray(labels).reshape(-1)
    rng = np.random.RandomState(seed)
    classes = np.unique(y)
    sel = []

    if mode == "kshot":
        k = int(value)
        for c in classes:
            idx = np.where(y == c)[0]
            rng.shuffle(idx)
            sel.append(idx[:min(k, len(idx))])  # if a class has < K, take all it has
    elif mode == "percent":
        frac = float(value)
        for c in classes:
            idx = np.where(y == c)[0]
            rng.shuffle(idx)
            n = max(1, int(round(len(idx) * frac)))  # at least 1 per class
            sel.append(idx[:n])
    else:
        raise ValueError(f"unknown mode {mode!r}")

    out = np.sort(np.concatenate(sel))
    return out
