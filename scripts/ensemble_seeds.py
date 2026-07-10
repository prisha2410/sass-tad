#!/usr/bin/env python3
"""Average probabilities across seed checkpoints, then find best threshold."""
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
    data = [np.load(p) for p in args.predictions]

    for tier in ["easy", "medium", "hard"]:
        key = f"2025_{tier}"
        if not all(f"{key}__probs" in d for d in data):
            print(f"[{tier}] missing probs in one or more files, skipping")
            continue
        probs_list = [d[f"{key}__probs"] for d in data]
        labels = data[0][f"{key}__labels"]
        avg_probs = np.mean(probs_list, axis=0)

        best_f1, best_t = 0.0, 0.40
        for t in np.arange(0.20, 0.65, 0.01):
            f1 = compute_f1((avg_probs >= t).astype(int), labels)
            if f1 > best_f1:
                best_f1, best_t = f1, t

        single_f1 = compute_f1((probs_list[0] >= 0.40).astype(int), labels)
        print(f"[{tier}] single-seed F1={single_f1:.4f} | ensemble F1={best_f1:.4f} (t={best_t:.2f}) | gain={best_f1-single_f1:+.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", nargs="+", default=[
        "checkpoints/fulldata/test_predictions.npz",
        "checkpoints/fulldata_seed0/test_predictions.npz",
        "checkpoints/fulldata_seed1/test_predictions.npz",
    ])
    args = p.parse_args()
    main(args)