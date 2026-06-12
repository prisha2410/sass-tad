"""
Evaluation metrics for style change detection.

Includes standard classification metrics (precision/recall/F1) plus
WindowDiff, the official PAN metric for boundary detection tasks.
"""

from typing import List, Sequence
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def binary_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    """Standard binary classification metrics for boundary labels."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def window_diff(y_true: Sequence[int], y_pred: Sequence[int], k: int = None) -> float:
    """
    WindowDiff metric (Pevzner & Hearst, 2002) — the official PAN metric.

    Slides a window of size k across the boundary sequence and compares
    the number of boundaries inside the window for true vs predicted.
    Lower is better (0 = perfect).

    y_true, y_pred: sequences of 0/1 boundary labels, length = n_sentences - 1
    k: window size. If None, defaults to half the average segment length,
       per the standard PAN convention (k = round(n / (2 * num_boundaries))).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    if n == 0:
        return 0.0

    if k is None:
        num_true_boundaries = max(y_true.sum(), 1)
        k = max(1, round(n / (2 * num_true_boundaries)))

    k = min(k, n)  # window can't exceed sequence length

    errors = 0
    num_windows = n - k + 1
    if num_windows <= 0:
        return float(y_true.sum() != y_pred.sum())

    for i in range(num_windows):
        true_count = y_true[i:i + k].sum()
        pred_count = y_pred[i:i + k].sum()
        if true_count != pred_count:
            errors += 1

    return errors / num_windows


def evaluate_predictions(
    all_true: List[Sequence[int]],
    all_pred: List[Sequence[int]],
) -> dict:
    """
    Evaluate across a list of documents (each a sequence of boundary labels).

    Returns micro-averaged binary metrics + mean WindowDiff across docs.
    """
    flat_true, flat_pred = [], []
    wd_scores = []

    for t, p in zip(all_true, all_pred):
        flat_true.extend(t)
        flat_pred.extend(p)
        wd_scores.append(window_diff(t, p))

    metrics = binary_metrics(flat_true, flat_pred)
    metrics["window_diff"] = float(np.mean(wd_scores))
    return metrics