"""
evaluate.py — metrics + comparison helpers, implemented FROM SCRATCH in
NumPy. scikit-learn is used only in tests to sanity-check these numbers.
"""

from __future__ import annotations

import time

import numpy as np


# --- Regression (Task A: price) ----------------------------------------------
def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot)


# --- Classification (Task B: price tier) --------------------------------------
def confusion_matrix(y_true, y_pred):
    """2x2 matrix [[TN, FP], [FN, TP]] for labels in {0, 1}."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return np.array([[tn, fp], [fn, tp]])


def accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def precision_recall_f1(y_true, y_pred):
    """(precision, recall, f1) for the positive (premium = 1) class."""
    (_, fp), (fn, tp) = confusion_matrix(y_true, y_pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    return float(precision), float(recall), float(f1)


def roc_auc(y_true, scores) -> float:
    """
    AUC from decision scores via the rank (Mann-Whitney U) formulation,
    with midranks for tied scores.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, float)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), float)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0  # midrank (1-based)
        i = j + 1
    sum_pos = ranks[y_true == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def roc_curve(y_true, scores, n_points: int = 200):
    """(fpr, tpr) arrays for plotting, swept over score thresholds."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, float)
    thresholds = np.quantile(scores, np.linspace(0, 1, n_points))
    fpr, tpr = [], []
    n_pos, n_neg = y_true.sum(), len(y_true) - y_true.sum()
    for thr in thresholds[::-1]:
        pred = scores >= thr
        tpr.append(np.sum(pred & (y_true == 1)) / n_pos)
        fpr.append(np.sum(pred & (y_true == 0)) / n_neg)
    return np.array(fpr), np.array(tpr)


# --- helpers -------------------------------------------------------------------
def timed(fn, *args, **kwargs):
    """Run fn(*args, **kwargs); return (result, seconds)."""
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


def kfold_indices(n: int, k: int = 5, seed: int = 42):
    """Deterministic k-fold index pairs [(train_idx, val_idx), ...]."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, k)
    return [(np.concatenate([folds[j] for j in range(k) if j != i]), folds[i])
            for i in range(k)]


def format_pm(values) -> str:
    """mean ± std string for a list of fold scores."""
    v = np.asarray(values, float)
    return f"{v.mean():.4f} ± {v.std():.4f}"


def compare(name_to_metrics: dict, title: str = "") -> str:
    """
    Pretty-print (and return) a fixed-width comparison table.
    `name_to_metrics` maps model name -> {metric: value_or_string}.
    """
    rows = list(name_to_metrics.items())
    metrics = []
    for _, m in rows:
        for key in m:
            if key not in metrics:
                metrics.append(key)
    w0 = max(len(n) for n, _ in rows) + 2
    widths = [max(len(k) + 2, 14) for k in metrics]
    lines = []
    if title:
        lines.append(title)
    header = "Model".ljust(w0) + "".join(k.rjust(w) for k, w in zip(metrics, widths))
    lines.append(header)
    lines.append("-" * len(header))
    for name, m in rows:
        cells = []
        for k, w in zip(metrics, widths):
            v = m.get(k, "")
            cells.append((f"{v:.4f}" if isinstance(v, float) else str(v)).rjust(w))
        lines.append(name.ljust(w0) + "".join(cells))
    table = "\n".join(lines)
    print(table)
    return table
