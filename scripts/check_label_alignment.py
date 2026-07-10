"""
Diagnostic: quantify paragraph/boundary-label mismatch across PAN datasets
=============================================================================

Purpose
-------
src/utils/data_loader.py's _read_pan_document() silently pads or truncates
boundary_labels to match len(sentences) - 1, with this line:

    expected = max(0, len(sentences) - 1)
    boundary_labels = (boundary_labels + [0] * expected)[:expected]

This script scans every problem-X.txt / truth-problem-X.json pair across
PAN2023/2024/2025, all tiers, both splits, and reports — BEFORE any
padding/truncation happens — how often len(lines) - 1 != len(changes).

This tells us whether the silent pad/truncate behavior is corrupting a
meaningful fraction of the dataset, or just smoothing over rare edge cases.

Usage
-----
Run from your sass-tad project root (where data/raw/ lives):

    python check_label_alignment.py

Or point it at a different root:

    python check_label_alignment.py --root data/raw
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict


_PAN_TIER_DIRS = {
    "pan2023": {"easy": "pan23-multi-author-analysis-dataset1",
                "medium": "pan23-multi-author-analysis-dataset2",
                "hard": "pan23-multi-author-analysis-dataset3"},
    "pan2024": {"easy": "easy", "medium": "medium", "hard": "hard"},
    "pan2025": {"easy": "easy", "medium": "medium", "hard": "hard"},
}

_PAN_SPLIT_DIRNAME = {
    "pan2023": lambda tier_dir, split: f"{tier_dir}-{split}",
    "pan2024": lambda tier_dir, split: split,
    "pan2025": lambda tier_dir, split: split,
}


def check_one_doc(txt_path: Path, truth_path: Path):
    """
    Returns (n_lines, n_changes, mismatch_delta) for one document,
    or None if truth file missing/unreadable.

    mismatch_delta = (n_lines - 1) - n_changes
        0   -> perfectly aligned (no padding/truncation needed)
        >0  -> lines exceed changes -> labels get PADDED with zeros
        <0  -> changes exceed lines -> labels get TRUNCATED (real labels dropped!)
    """
    if not truth_path.exists():
        return None

    raw = txt_path.read_text(encoding="utf-8").strip()
    lines = [s.strip() for s in raw.split("\n") if s.strip()]

    try:
        truth = json.loads(truth_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None

    changes = truth.get("changes", None)
    if changes is None:
        return None

    expected_boundaries = max(0, len(lines) - 1)
    actual_changes = len(changes)
    delta = expected_boundaries - actual_changes

    return len(lines), actual_changes, delta


def scan_split(split_dir: Path):
    """Scan one split directory (e.g. .../hard/train), return list of deltas."""
    results = []
    if not split_dir.exists():
        return results

    for txt_path in sorted(split_dir.glob("problem-*.txt")):
        problem_id = txt_path.stem
        truth_path = split_dir / f"truth-{problem_id}.json"
        result = check_one_doc(txt_path, truth_path)
        if result is not None:
            results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data/raw")
    args = parser.parse_args()

    root = Path(args.root)

    overall_summary = []

    for source_key, tiers in _PAN_TIER_DIRS.items():
        for tier, tier_dirname in tiers.items():
            tier_path = root / source_key / tier_dirname
            if not tier_path.exists():
                print(f"[SKIP] {source_key}/{tier}: directory not found at {tier_path}")
                continue

            split_fn = _PAN_SPLIT_DIRNAME[source_key]
            all_results = []
            for pan_split in ("train", "validation"):
                split_dirname = split_fn(tier_path.name, pan_split)
                split_dir = tier_path / split_dirname
                results = scan_split(split_dir)
                all_results.extend(results)

            if not all_results:
                print(f"[EMPTY] {source_key}/{tier}: no documents found")
                continue

            n_docs = len(all_results)
            n_aligned = sum(1 for (_, _, d) in all_results if d == 0)
            n_padded = sum(1 for (_, _, d) in all_results if d > 0)
            n_truncated = sum(1 for (_, _, d) in all_results if d < 0)

            pct_aligned = 100.0 * n_aligned / n_docs
            pct_padded = 100.0 * n_padded / n_docs
            pct_truncated = 100.0 * n_truncated / n_docs

            max_pad = max((d for (_, _, d) in all_results if d > 0), default=0)
            max_trunc = min((d for (_, _, d) in all_results if d < 0), default=0)

            print(f"\n=== {source_key} / {tier} ===")
            print(f"  Total docs checked:  {n_docs}")
            print(f"  Perfectly aligned:   {n_aligned} ({pct_aligned:.1f}%)")
            print(f"  Padded (lines>changes): {n_padded} ({pct_padded:.1f}%)  max pad = {max_pad}")
            print(f"  Truncated (changes>lines): {n_truncated} ({pct_truncated:.1f}%)  max truncation = {max_trunc}")

            overall_summary.append({
                "source": source_key,
                "tier": tier,
                "n_docs": n_docs,
                "pct_aligned": pct_aligned,
                "pct_padded": pct_padded,
                "pct_truncated": pct_truncated,
            })

    print("\n" + "=" * 70)
    print("SUMMARY (sorted by worst alignment)")
    print("=" * 70)
    overall_summary.sort(key=lambda x: x["pct_aligned"])
    for row in overall_summary:
        flag = "  <-- INVESTIGATE" if row["pct_aligned"] < 95.0 else ""
        print(
            f"{row['source']:10s} {row['tier']:7s} "
            f"n={row['n_docs']:5d}  "
            f"aligned={row['pct_aligned']:5.1f}%  "
            f"padded={row['pct_padded']:5.1f}%  "
            f"truncated={row['pct_truncated']:5.1f}%"
            f"{flag}"
        )


if __name__ == "__main__":
    main()