"""Evaluation metrics shared by every FaunaVision model.

All functions take class probabilities ``probs`` (N x C) and integer labels
``y`` (N,), so baselines and fine-tuned models are scored identically.

Beyond standard accuracy and F1, two metrics are specific to this project:
- **Lookalike confusion**: of the errors on a cluster's species, how many
  went to a sibling in the same cluster (e.g. white rhino -> black rhino).
- **Mistake severity**: share of all errors that stay inside the true
  cluster (a "near miss") versus leaving it (e.g. antelope -> lion).
"""

from __future__ import annotations

import numpy as np


def topk_accuracy(probs: np.ndarray, y: np.ndarray, k: int = 1) -> float:
    k = min(k, probs.shape[1])
    topk = np.argpartition(-probs, kth=k - 1, axis=1)[:, :k]
    return float((topk == y[:, None]).any(axis=1).mean())


def per_class_prf(pred: np.ndarray, y: np.ndarray, n_classes: int):
    """Precision, recall, F1 and support per class (0 where undefined)."""
    cm = confusion_matrix(pred, y, n_classes)
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1).astype(float)  # rows = true class
    predicted = cm.sum(axis=0).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(support > 0, tp / support, 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    return precision, recall, f1, support


def confusion_matrix(pred: np.ndarray, y: np.ndarray, n_classes: int) -> np.ndarray:
    """Rows are true classes, columns are predictions."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    return cm


def macro_f1(pred: np.ndarray, y: np.ndarray, n_classes: int) -> float:
    """Mean F1 over classes present in ``y`` (absent classes are skipped)."""
    _, _, f1, support = per_class_prf(pred, y, n_classes)
    present = support > 0
    return float(f1[present].mean()) if present.any() else 0.0


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature that minimises negative log-likelihood (grid + refine).

    Fitted on validation data, then applied to test logits.
    """
    def nll(t: float) -> float:
        z = logits / t
        z = z - z.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return float(-logp[np.arange(len(y)), y].mean())

    grid = np.exp(np.linspace(np.log(0.05), np.log(20), 60))
    best = min(grid, key=nll)
    fine = np.linspace(best * 0.8, best * 1.25, 40)
    return float(min(fine, key=nll))


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cluster_report(pred: np.ndarray, y: np.ndarray, class_names: list[str], clusters: dict[str, str]) -> dict:
    """Per-cluster accuracy and lookalike confusion.

    ``clusters`` maps class name -> cluster name.
    """
    cluster_of = np.array([clusters[c] for c in class_names])
    true_cluster, pred_cluster = cluster_of[y], cluster_of[pred]
    wrong = pred != y
    out = {}
    for cl in sorted(set(cluster_of)):
        m = true_cluster == cl
        if not m.any():
            continue
        errs = wrong & m
        out[cl] = {
            "n": int(m.sum()),
            "accuracy": float((~wrong[m]).mean()),
            # errors that went to a sibling species in the same cluster
            "lookalike_share_of_errors": float((pred_cluster[errs] == cl).mean()) if errs.any() else 0.0,
        }
    return out


def pair_accuracy(pred: np.ndarray, y: np.ndarray, a: int, b: int) -> tuple[float, int]:
    """How well two lookalikes are told apart.

    Among images of A or B that the model assigned to A or B, the share given
    the right one. Errors to other classes are excluded, so this isolates
    the "which of the two" decision. Returns (accuracy, n).
    """
    m = np.isin(y, [a, b]) & np.isin(pred, [a, b])
    return (float((pred[m] == y[m]).mean()) if m.any() else float("nan")), int(m.sum())


def summarize(logits: np.ndarray, y: np.ndarray, class_names: list[str], clusters: dict[str, str],
              temperature: float = 1.0) -> dict:
    probs = softmax(logits, temperature)
    pred = probs.argmax(axis=1)
    n = len(class_names)
    precision, recall, f1, support = per_class_prf(pred, y, n)
    cluster_of = np.array([clusters[c] for c in class_names])
    wrong = pred != y
    present = support > 0
    return {
        "n": int(len(y)),
        "top1": topk_accuracy(probs, y, 1),
        "top5": topk_accuracy(probs, y, 5),
        "macro_f1": macro_f1(pred, y, n),
        "balanced_accuracy": float(recall[present].mean()) if present.any() else 0.0,
        "ece": expected_calibration_error(probs, y),
        "temperature": temperature,
        "near_miss_share_of_errors": float((cluster_of[pred[wrong]] == cluster_of[y[wrong]]).mean()) if wrong.any() else 0.0,
        "per_class": {
            c: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])}
            for i, c in enumerate(class_names) if support[i] > 0
        },
        "clusters": cluster_report(pred, y, class_names, clusters),
    }
