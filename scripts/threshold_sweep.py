#!/usr/bin/env python3
"""
Sweep decision threshold on val set using precomputed finetuned features.
Run while training is still going — uses hard_oversample_best.pt as it improves.

Usage:
    python scripts/threshold_sweep.py
    python scripts/threshold_sweep.py --checkpoint checkpoints/hard_oversample/hard_oversample_best.pt
"""
import sys, numpy as np, torch, torch.nn as nn
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.deberta_fusion import DeBERTaFusionE2E

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEAT_DIR = ROOT / "data" / "processed" / "features_finetuned"
TIERS    = ["easy", "medium", "hard"]
YEARS    = ["2024", "2025"]


def load_cached(year, tier, split):
    path = FEAT_DIR / f"{year}_{tier}_{split}.npz"
    if not path.exists():
        return []
    data    = np.load(path)
    cls     = data["cls_reprs"].astype(np.float32)
    style   = data["style_diffs"].astype(np.float32)
    labels  = data["labels"].astype(np.int8)
    lengths = data["doc_lengths"].astype(np.int32)
    items, idx = [], 0
    for n in lengths.tolist():
        if n > 0:
            items.append((cls[idx:idx+n], style[idx:idx+n], labels[idx:idx+n]))
        idx += n
    return items


def load_model(ckpt_path):
    model = DeBERTaFusionE2E(
        style_dim=193, unfreeze_layers=2,
        use_style=True, use_attention_fusion=True,
        use_bilstm=True, use_adversarial=False,
    )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    model.deberta = model.deberta.cpu()
    model.eval()
    return model.to(DEVICE)


def forward_doc(model, cls_np, style_np):
    pair_reprs = []
    for j in range(cls_np.shape[0]):
        cls   = torch.tensor(cls_np[j]).unsqueeze(0).to(DEVICE)
        style = torch.tensor(style_np[j]).unsqueeze(0).to(DEVICE)
        fused = model.style_fusion(cls, style) if model.use_attention_fusion else torch.cat([cls, style], dim=-1)
        pair_reprs.append(fused)
    pair_reprs_t = torch.cat(pair_reprs, dim=0).unsqueeze(0)
    return model.forward_document(pair_reprs_t).squeeze(0).reshape(-1)


def compute_f1(all_preds, all_labels):
    tp = fp = fn = 0
    for preds, lbls in zip(all_preds, all_labels):
        for p, l in zip(preds, lbls):
            if p == 1 and l == 1: tp += 1
            elif p == 1 and l == 0: fp += 1
            elif p == 0 and l == 1: fn += 1
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/hard_oversample/hard_oversample_best.pt")
    args = p.parse_args()

    print(f"Loading {args.checkpoint}...")
    model = load_model(args.checkpoint)

    print("Loading val sets...")
    val_data = {}
    for year in YEARS:
        for tier in TIERS:
            items = load_cached(year, tier, "val")
            if items:
                val_data[f"{year}_{tier}"] = items

    # Collect all probs and labels
    print("Running inference...")
    all_probs_by_key, all_labels_by_key = {}, {}
    with torch.no_grad():
        for key, items in val_data.items():
            probs_list, labels_list = [], []
            for cls_np, style_np, lbl_np in tqdm(items, desc=key, ncols=70):
                logits = forward_doc(model, cls_np, style_np)
                probs_list.append(torch.sigmoid(logits).cpu().numpy())
                labels_list.append(lbl_np.astype(int))
            all_probs_by_key[key]  = probs_list
            all_labels_by_key[key] = labels_list

    # Sweep thresholds
    thresholds = np.arange(0.30, 0.71, 0.02)
    print(f"\n{'Thresh':>7}  {'2024_hard':>10}  {'2025_hard':>10}  {'avg_hard':>10}  {'2024_med':>10}  {'2025_med':>10}")
    print("-" * 65)

    best_thresh, best_avg_hard = 0.5, 0.0
    for thresh in thresholds:
        row = {}
        for key in val_data:
            preds = [(p >= thresh).astype(int).flatten().tolist() for p in all_probs_by_key[key]]
            lbls  = [l.flatten().tolist() for l in all_labels_by_key[key]]
            row[key] = compute_f1(preds, lbls)

        hard_f1s = [row[k] for k in row if "hard" in k]
        avg_hard = np.mean(hard_f1s) if hard_f1s else 0.0
        if avg_hard > best_avg_hard:
            best_avg_hard, best_thresh = avg_hard, thresh

        print(f"  {thresh:.2f}   {row.get('2024_hard',0):.4f}      {row.get('2025_hard',0):.4f}      {avg_hard:.4f}      {row.get('2024_medium',0):.4f}      {row.get('2025_medium',0):.4f}")

    print(f"\nBest threshold: {best_thresh:.2f}  →  Avg Hard F1 = {best_avg_hard:.4f}")


if __name__ == "__main__":
    main()