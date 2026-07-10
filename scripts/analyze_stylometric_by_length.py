#!/usr/bin/env python3
"""
Tests whether stylometric fusion's benefit is concentrated in
short-paragraph pairs (where DeBERTa has less context).
Compares full model vs no_stylometric ablation, segmented by paragraph length.
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = ROOT / "data" / "raw" / "pan2025_full"
TIERS = ["easy", "medium", "hard"]


def parse_problem(txt_path):
    text = txt_path.read_text(encoding="utf-8", errors="replace")
    return [p.strip() for p in text.strip().split("\n") if p.strip()]


def get_pair_lengths(tier):
    """For each boundary, length = word count of the shorter of the two paragraphs."""
    test_dir = TEST_ROOT / tier / "test"
    truth_files = {p.stem.replace("truth-problem-", ""): p
                   for p in test_dir.glob("truth-problem-*.json")}
    prob_files  = {p.stem.replace("problem-", ""): p
                   for p in test_dir.glob("problem-*.txt")}
    common_ids = sorted(set(truth_files) & set(prob_files))

    lengths = []
    for pid in common_ids:
        paragraphs = parse_problem(prob_files[pid])
        labels = json.loads(truth_files[pid].read_text())["changes"]
        if len(paragraphs) < 2 or len(labels) != len(paragraphs) - 1:
            continue
        for j in range(len(paragraphs) - 1):
            wc = min(len(paragraphs[j].split()), len(paragraphs[j+1].split()))
            lengths.append(wc)
    return np.array(lengths)


def compute_f1(preds, labels):
    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def main():
    full = np.load("checkpoints/fulldata/test_predictions.npz")
    nosty = np.load("checkpoints/ablation_no_stylometric/test_predictions.npz")

    for tier in TIERS:
        key = f"2025_{tier}"
        if f"{key}__preds" not in full or f"{key}__preds" not in nosty:
            print(f"[{tier}] missing predictions, skipping")
            continue

        lengths = get_pair_lengths(tier)
        full_preds  = full[f"{key}__preds"]
        nosty_preds = nosty[f"{key}__preds"]
        labels      = full[f"{key}__labels"]

        if not (len(lengths) == len(full_preds) == len(nosty_preds)):
            print(f"[{tier}] length mismatch ({len(lengths)} vs {len(full_preds)}) — alignment issue, skipping")
            continue

        short_mask = lengths < 30
        long_mask  = ~short_mask

        for label, mask in [("SHORT (<30 words)", short_mask), ("LONG (>=30 words)", long_mask)]:
            if mask.sum() == 0:
                continue
            f1_full  = compute_f1(full_preds[mask],  labels[mask])
            f1_nosty = compute_f1(nosty_preds[mask], labels[mask])
            print(f"[{tier}] {label} (n={mask.sum()}): full={f1_full:.4f} | no_stylometric={f1_nosty:.4f} | gain={f1_full-f1_nosty:+.4f}")


if __name__ == "__main__":
    main()