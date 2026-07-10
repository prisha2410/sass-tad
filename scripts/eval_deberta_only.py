#!/usr/bin/env python3
"""
Evaluates the DeBERTa-only baseline on precomputed test features.
Saves predictions + probabilities for significance testing.

Usage:
    python scripts/eval_deberta_only.py --checkpoint checkpoints/deberta_only/deberta_only_final.pt
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_deberta_only import DeBERTaOnlyHead, load_cached, compute_f1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TIERS  = ["easy", "medium", "hard"]


def main(args):
    feat_dir = ROOT / args.feat_dir
    model = DeBERTaOnlyHead(cls_dim=768, hidden=args.hidden)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model = model.to(DEVICE)
    model.eval()

    results = {}
    pred_dump = {}

    for year in args.years:
        print(f"--- {year} ---")
        for tier in TIERS:
            items = load_cached(feat_dir, year, tier, "test")
            if not items:
                print(f"  [{year}_{tier}] MISSING test npz")
                continue
            all_preds, all_labels, all_probs, doc_lengths = [], [], [], []
            with torch.no_grad():
                for cls_np, lbl_np in tqdm(items, desc=f"  {year}_{tier}", ncols=70):
                    cls_t  = torch.tensor(cls_np).to(DEVICE)
                    logits = model(cls_t)
                    probs  = torch.sigmoid(logits).cpu().numpy()
                    preds  = (probs >= args.threshold).astype(int).flatten().tolist()
                    lbls   = lbl_np.astype(int).flatten().tolist()
                    all_preds.append(preds)
                    all_labels.append(lbls)
                    all_probs.append(probs.flatten().tolist())
                    doc_lengths.append(len(preds))
            f1 = compute_f1(all_preds, all_labels)
            key = f"{year}_{tier}"
            results[key] = f1
            print(f"  [{key}] F1 = {f1:.4f}")

            pred_dump[f"{key}__preds"]   = np.array([p for doc in all_preds for p in doc], dtype=np.int8)
            pred_dump[f"{key}__labels"]  = np.array([l for doc in all_labels for l in doc], dtype=np.int8)
            pred_dump[f"{key}__probs"]   = np.array([p for doc in all_probs for p in doc], dtype=np.float32)
            pred_dump[f"{key}__lengths"] = np.array(doc_lengths, dtype=np.int32)

    print("\n" + "="*50)
    print("DEBERTA-ONLY BASELINE — TEST SET RESULTS")
    print("="*50)
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
    if results:
        print(f"  Macro avg: {np.mean(list(results.values())):.4f}")

    out_dir = Path(args.checkpoint).parent
    with open(out_dir / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    np.savez_compressed(out_dir / "test_predictions.npz", **pred_dump)
    print(f"\nSaved -> {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/deberta_only/deberta_only_final.pt")
    p.add_argument("--feat-dir",   default="data/processed/features_finetuned_official_test")
    p.add_argument("--threshold",  type=float, default=0.40)
    p.add_argument("--hidden",     type=int,   default=256)
    p.add_argument("--years",      nargs="+",  default=["2025"])
    args = p.parse_args()
    main(args)