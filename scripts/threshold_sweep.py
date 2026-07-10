#!/usr/bin/env python3
"""Find the optimal decision threshold per tier from saved probabilities."""
import argparse
import numpy as np


def compute_f1(preds, labels):
    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def main(args):
    data = np.load(args.predictions)
    for tier in ["easy", "medium", "hard"]:
        key = f"2025_{tier}"
        if f"{key}__probs" not in data:
            print(f"[{tier}] no saved probabilities, skipping")
            continue
        probs  = data[f"{key}__probs"]
        labels = data[f"{key}__labels"]
        best_f1, best_t = 0.0, 0.40
        for t in np.arange(0.20, 0.65, 0.01):
            f1 = compute_f1((probs >= t).astype(int), labels)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        default_f1 = compute_f1((probs >= 0.40).astype(int), labels)
        print(f"[{tier}] default(0.40) F1={default_f1:.4f} | best(t={best_t:.2f}) F1={best_f1:.4f} | gain={best_f1-default_f1:+.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", default="checkpoints/fulldata/test_predictions.npz")
    args = p.parse_args()
    main(args)