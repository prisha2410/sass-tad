"""
Run all three baselines on PAN 2024 hard and print a comparison table.

Usage:
    python scripts/run_baselines.py
    python scripts/run_baselines.py --year 2024 --tier hard
"""

import argparse
from pathlib import Path

from src.utils.data_loader import load_pan_labeled
from src.models.baselines import StylometricBaseline, SBERTBaseline, FusedBaseline
from src.evaluation.metrics import evaluate_predictions

CACHE_DIR = Path("data/processed/embeddings")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", default="2024")
    parser.add_argument("--tier", default="hard")
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"Baselines — PAN {args.year} {args.tier}")
    print(f"{'=' * 60}\n")

    train, val, test = load_pan_labeled("data/raw", year=args.year, tier=args.tier)

    train_cache = CACHE_DIR / f"pan{args.year}_{args.tier}_train.pt"
    val_cache = CACHE_DIR / f"pan{args.year}_{args.tier}_val.pt"

    results = {}

    # --- 1. Stylometric baseline ---
    print("[1/3] Stylometric baseline...")
    sty = StylometricBaseline().fit(train)
    res = sty.evaluate(val)
    results["Stylometric"] = evaluate_predictions(res.doc_true, res.doc_pred)

    # --- 2. SBERT baseline ---
    print("[2/3] SBERT baseline...")
    sbert = SBERTBaseline().fit(str(train_cache))
    res = sbert.evaluate(str(val_cache))
    results["SBERT"] = evaluate_predictions(res.doc_true, res.doc_pred)

    # --- 3. Fused baseline ---
    print("[3/3] Fused baseline...")
    fused = FusedBaseline().fit(train, str(train_cache))
    res = fused.evaluate(val, str(val_cache))
    results["Fused"] = evaluate_predictions(res.doc_true, res.doc_pred)

    # --- Print table ---
    print(f"\n{'Model':<15} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'WinDiff':>8}")
    print("-" * 60)
    for name, m in results.items():
        print(f"{name:<15} {m['accuracy']:>8.4f} {m['precision']:>8.4f} "
              f"{m['recall']:>8.4f} {m['f1']:>8.4f} {m['window_diff']:>8.4f}")

    print("\nNote: WindowDiff lower = better. All others higher = better.")
    print("These are your 'before SASS-TAD' reference numbers.")


if __name__ == "__main__":
    main()