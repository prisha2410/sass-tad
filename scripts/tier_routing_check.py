#!/usr/bin/env python3
"""Tier-conditional model selection: full model for Easy/Medium, no_stylometric for Hard."""
import numpy as np

def compute_f1(preds, labels):
    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

full  = np.load("checkpoints/fulldata/test_predictions.npz")
nosty = np.load("checkpoints/ablation_no_stylometric/test_predictions.npz")

# Routed: full model for easy/medium, no_stylometric for hard
routed_f1 = {}
routed_f1["easy"]   = compute_f1(full["2025_easy__preds"],   full["2025_easy__labels"])
routed_f1["medium"] = compute_f1(full["2025_medium__preds"], full["2025_medium__labels"])
routed_f1["hard"]   = compute_f1(nosty["2025_hard__preds"],  nosty["2025_hard__labels"])

print("Tier-routed model results:")
for tier, f1 in routed_f1.items():
    print(f"  {tier}: {f1:.4f}")
print(f"  Macro: {np.mean(list(routed_f1.values())):.4f}")

print(f"\nCompare to plain full model Hard: {compute_f1(full['2025_hard__preds'], full['2025_hard__labels']):.4f}")