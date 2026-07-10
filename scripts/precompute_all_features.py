#!/usr/bin/env python3
"""
Precomputes DeBERTa CLS embeddings + stylometric diffs for all docs.
Run once — takes ~30-60 min. Saves to data/processed/features/{year}_{tier}.npz

Usage:
    python scripts/precompute_all_features.py
"""
import sys, gc
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.deberta_fusion import DeBERTaEncoder
from src.features.stylometric import StylometricExtractor
from src.utils.data_loader import PANDataset

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
YEARS   = ["2024", "2025"]
TIERS   = ["easy", "medium", "hard"]
OUT_DIR = ROOT / "data" / "processed" / "features"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Device: {DEVICE}")
encoder   = DeBERTaEncoder(device=DEVICE, batch_size=8)
extractor = StylometricExtractor()

for year in YEARS:
    for tier in TIERS:
        out_path = OUT_DIR / f"{year}_{tier}.npz"
        if out_path.exists():
            print(f"[{year}/{tier}] Already exists, skipping.")
            continue

        try:
            docs = PANDataset(root="data/raw", year=year, tier=tier).load()
        except FileNotFoundError as e:
            print(f"[{year}/{tier}] Not found, skipping.")
            continue

        print(f"\n[{year}/{tier}] {len(docs)} docs")

        all_cls    = []
        all_style  = []
        all_labels = []
        doc_lengths = []

        for doc in tqdm(docs, desc=f"{year}/{tier}"):
            sents  = doc.sentences
            labels = doc.boundary_labels
            n = min(len(sents) - 1, len(labels))
            if n == 0:
                doc_lengths.append(0)
                continue

            # DeBERTa CLS — uses DeBERTaEncoder already in deberta_fusion.py
            cls_block = encoder.encode_pairs(sents).numpy()  # (n, 768) float32

            # Stylometric diffs — uses extract_document already in stylometric.py
            _, diffs = extractor.extract_document(sents)     # (n, D) float32
            style_block = diffs[:n, :193].astype(np.float32) # (n, 193)

            all_cls.append(cls_block)
            all_style.append(style_block)
            all_labels.append(np.array(labels[:n], dtype=np.int8))
            doc_lengths.append(n)

        if not all_cls:
            continue

        np.savez_compressed(
            out_path,
            cls_reprs   = np.concatenate(all_cls,    axis=0),  # (N_total, 768)
            style_diffs = np.concatenate(all_style,  axis=0),  # (N_total, 193)
            labels      = np.concatenate(all_labels, axis=0),  # (N_total,)
            doc_lengths = np.array(doc_lengths, dtype=np.int32),
        )
        size_mb = out_path.stat().st_size / 1e6
        print(f"  Saved {sum(doc_lengths)} pairs → {out_path.name} ({size_mb:.1f} MB)")
        torch.cuda.empty_cache()
        gc.collect()

print("\nDone.")
for f in sorted(OUT_DIR.glob("*.npz")):
    print(f"  {f.name}  {f.stat().st_size/1e6:.1f} MB")