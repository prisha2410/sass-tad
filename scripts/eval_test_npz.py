#!/usr/bin/env python3
"""
Evaluates DeSAF checkpoint on precomputed test features.
Saves per-document predictions, raw probabilities, and labels
for threshold tuning and ensembling.

Usage:
    python scripts/eval_test_npz.py --checkpoint checkpoints/fulldata/fulldata_final.pt --feat-dir data/processed/features_finetuned_official_test --years 2025
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.deberta_fusion import DeBERTaFusionE2E

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TIERS  = ["easy", "medium", "hard"]
YEARS  = ["2024", "2025"]


def load_cached(feat_dir, year, tier, split):
    path = feat_dir / f"{year}_{tier}_{split}.npz"
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


def forward_doc(model, cls_block, style_block):
    pair_reprs = []
    for j in range(cls_block.shape[0]):
        cls   = cls_block[j].unsqueeze(0).to(DEVICE)
        style = style_block[j].unsqueeze(0).to(DEVICE)
        if model.use_attention_fusion:
            fused = model.style_fusion(cls, style)
        else:
            fused = torch.cat([cls, style], dim=-1)
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


def evaluate(model, items, threshold, key):
    model.eval()
    all_preds, all_labels, all_probs, doc_lengths = [], [], [], []
    with torch.no_grad():
        for cls_np, style_np, lbl_np in tqdm(items, desc=f"  {key}", ncols=70):
            cls_t   = torch.tensor(cls_np)
            style_t = torch.tensor(style_np)
            logits  = forward_doc(model, cls_t, style_t)
            probs   = torch.sigmoid(logits).cpu().numpy()
            preds   = (probs >= threshold).astype(int).flatten().tolist()
            lbls    = lbl_np.astype(int).flatten().tolist()
            prob_l  = probs.flatten().tolist()
            all_preds.append(preds)
            all_labels.append(lbls)
            all_probs.append(prob_l)
            doc_lengths.append(len(preds))
    f1 = compute_f1(all_preds, all_labels)
    print(f"  [{key}] F1 = {f1:.4f}")
    return f1, all_preds, all_labels, all_probs, doc_lengths


def main(args):
    feat_dir = ROOT / args.feat_dir
    print(f"Loading checkpoint: {args.checkpoint}")
    print(f"Feature dir: {feat_dir}")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    model = DeBERTaFusionE2E(
        style_dim=193, unfreeze_layers=2,
        use_style=True, use_attention_fusion=True,
        use_bilstm=True, use_adversarial=False,
    )
    model.load_state_dict(state_dict, strict=False)
    model.deberta = model.deberta.cpu()
    model = model.to(DEVICE)
    model.eval()
    print("Model loaded.\n")

    results = {}
    pred_dump = {}  # key -> {preds_flat, labels_flat, probs_flat, doc_lengths}

    for year in args.years:
        print(f"--- {year} ---")
        for tier in TIERS:
            items = load_cached(feat_dir, year, tier, "test")
            if not items:
                print(f"  [{year}_{tier}] MISSING test npz")
                continue
            key = f"{year}_{tier}"
            f1, all_preds, all_labels, all_probs, doc_lengths = evaluate(model, items, args.threshold, key)
            results[key] = f1
            pred_dump[f"{key}__preds"]   = np.array([p for doc in all_preds for p in doc], dtype=np.int8)
            pred_dump[f"{key}__labels"]  = np.array([l for doc in all_labels for l in doc], dtype=np.int8)
            pred_dump[f"{key}__probs"]   = np.array([p for doc in all_probs for p in doc], dtype=np.float32)
            pred_dump[f"{key}__lengths"] = np.array(doc_lengths, dtype=np.int32)

    print("\n" + "="*50)
    print("TEST SET RESULTS")
    print("="*50)
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    hard_f1s = [v for k, v in results.items() if "hard" in k]
    all_f1s  = list(results.values())
    if hard_f1s:
        print(f"  Avg Hard F1:  {np.mean(hard_f1s):.4f}")
    if all_f1s:
        print(f"  Macro avg:    {np.mean(all_f1s):.4f}")

    out_dir = Path(args.checkpoint).parent
    out_path = out_dir / "test_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_path}")

    preds_path = out_dir / "test_predictions.npz"
    np.savez_compressed(preds_path, **pred_dump)
    print(f"Saved predictions -> {preds_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/fulldata/fulldata_final.pt")
    p.add_argument("--feat-dir",   default="data/processed/features_finetuned")
    p.add_argument("--threshold",  type=float, default=0.40)
    p.add_argument("--years",      nargs="+", default=["2024", "2025"])
    args = p.parse_args()
    main(args)