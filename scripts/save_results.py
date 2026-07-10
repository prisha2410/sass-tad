#!/usr/bin/env python3
"""
Save all evaluation results to results/results.json and results/results.csv
Run after any threshold sweep or evaluation.
"""
import json, csv, sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---- Fill this in after each experiment ----
entry = {
    "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M"),
    "experiment": "hard_oversample_epoch5",          # change each time
    "checkpoint": "checkpoints/hard_oversample/hard_oversample_best.pt",
    "threshold":  0.40,
    "notes":      "finetuned DeBERTa features, hard weight=3x, epoch 5/10 best so far",
    "val": {
        "2024_easy":   0.9909,
        "2024_medium": 0.8947,
        "2024_hard":   0.8668,
        "2025_easy":   0.9549,
        "2025_medium": 0.7886,
        "2025_hard":   0.7924,
        "avg_hard":    0.8296,
    },
    "val_thresh040": {
        "2024_hard":   0.8775,
        "2025_hard":   0.7999,
        "2024_medium": 0.8849,
        "2025_medium": 0.7901,
        "avg_hard":    0.8387,
    },
    "test": {}   # fill after test eval
}
# --------------------------------------------

json_path = RESULTS_DIR / "results.json"
if json_path.exists():
    all_results = json.loads(json_path.read_text())
else:
    all_results = []

all_results.append(entry)
json_path.write_text(json.dumps(all_results, indent=2))
print(f"Saved to {json_path}")

# Also write/append CSV for easy viewing in Excel
csv_path = RESULTS_DIR / "results.csv"
flat = {"timestamp": entry["timestamp"], "experiment": entry["experiment"],
        "checkpoint": entry["checkpoint"], "threshold": entry["threshold"],
        "notes": entry["notes"]}
for split in ("val", "val_thresh040", "test"):
    for k, v in entry.get(split, {}).items():
        flat[f"{split}_{k}"] = v

write_header = not csv_path.exists()
with open(csv_path, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=flat.keys())
    if write_header:
        writer.writeheader()
    writer.writerow(flat)
print(f"Saved to {csv_path}")