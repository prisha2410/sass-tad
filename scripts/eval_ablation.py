#!/usr/bin/env python3
"""
Evaluates ablation checkpoints on the official PAN 2025 test set,
matching each ablation's exact training-time forward pass.
Now also saves raw probabilities for threshold tuning.

Usage:
    python scripts/eval_ablation.py --checkpoint checkpoints/ablation_no_bilstm/best.pt --mode no_bilstm
    python scripts/eval_ablation.py --checkpoint checkpoints/ablation_no_stylometric/best.pt --mode no_stylometric
    python scripts/eval_ablation.py --checkpoint checkpoints/ablation_no_oversampling/best.pt --mode full
    python scripts/eval_ablation.py --checkpoint checkpoints/ablation_no_curriculum/best.pt --mode full
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.deberta_fusion import DeBERTaFusionE2E

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TIERS  = ["easy", "medium", "hard"]
FUSION_DIM = 961
HEAD_INPUT_DIM = 256


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


def forward_full(model, proj, cls_block, style_block):
    pair_reprs = []
    for j in range(cls_block.shape[0]):
        cls   = cls_block[j].unsqueeze(0).to(DEVICE)
        style = style_block[j].unsqueeze(0).to(DEVICE)
        fused = model.style_fusion(cls, style)
        pair_reprs.append(fused)
    pair_reprs_t = torch.cat(pair_reprs, dim=0).unsqueeze(0)
    return model.forward_document(pair_reprs_t).squeeze(0).reshape(-1)


def forward_no_bilstm(model, proj, cls_block, style_block):
    pair_reprs = []
    for j in range(cls_block.shape[0]):
        cls   = cls_block[j].unsqueeze(0).to(DEVICE)
        style = style_block[j].unsqueeze(0).to(DEVICE)
        fused = model.style_fusion(cls, style)
        pair_reprs.append(fused)
    stacked   = torch.cat(pair_reprs, dim=0)
    projected = proj(stacked)
    return model.head(projected).reshape(-1)


def forward_no_stylometric(model, proj, cls_block, style_block):
    zeroed_style = torch.zeros_like(style_block)
    return forward_full(model, proj, cls_block, zeroed_style)


FORWARD_FNS = {
    "full":            forward_full,
    "no_bilstm":       forward_no_bilstm,
    "no_stylometric":  forward_no_stylometric,
}


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


def evaluate(model, proj, items, threshold, key, forward_fn):
    model.eval()
    all_preds, all_labels, all_probs, doc_lengths = [], [], [], []
    with torch.no_grad():
        for cls_np, style_np, lbl_np in tqdm(items, desc=f"  {key}", ncols=70):
            cls_t   = torch.tensor(cls_np)
            style_t = torch.tensor(style_np)
            logits  = forward_fn(model, proj, cls_t, style_t)
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
    forward_fn = FORWARD_FNS[args.mode]
    print(f"Mode: {args.mode} | Checkpoint: {args.checkpoint}")
    print(f"Feature dir: {feat_dir}")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    proj = None
    if args.mode == "no_bilstm":
        if isinstance(ckpt, dict) and "model" in ckpt and "proj" in ckpt:
            model_state = ckpt["model"]
            proj_state  = ckpt["proj"]
        else:
            raise ValueError("Expected {'model':..., 'proj':...} checkpoint for no_bilstm mode")
        proj = nn.Linear(FUSION_DIM, HEAD_INPUT_DIM).to(DEVICE)
        proj.load_state_dict(proj_state)
        proj.eval()
    else:
        model_state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    model = DeBERTaFusionE2E(
        style_dim=193, unfreeze_layers=2,
        use_style=True, use_attention_fusion=True,
        use_bilstm=True, use_adversarial=False,
    )
    model.load_state_dict(model_state, strict=False)
    model.deberta = model.deberta.cpu()
    model = model.to(DEVICE)
    model.eval()
    print("Model loaded.\n")

    results = {}
    pred_dump = {}

    for tier in TIERS:
        items = load_cached(feat_dir, "2025", tier, "test")
        if not items:
            print(f"  [2025_{tier}] MISSING test npz")
            continue
        key = f"2025_{tier}"
        f1, all_preds, all_labels, all_probs, doc_lengths = evaluate(model, proj, items, args.threshold, key, forward_fn)
        results[key] = f1
        pred_dump[f"{key}__preds"]   = np.array([p for doc in all_preds for p in doc], dtype=np.int8)
        pred_dump[f"{key}__labels"]  = np.array([l for doc in all_labels for l in doc], dtype=np.int8)
        pred_dump[f"{key}__probs"]   = np.array([p for doc in all_probs for p in doc], dtype=np.float32)
        pred_dump[f"{key}__lengths"] = np.array(doc_lengths, dtype=np.int32)

    print("\n" + "="*50)
    print(f"ABLATION [{args.mode}] — TEST SET RESULTS")
    print("="*50)
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
    if results:
        print(f"  Macro avg: {np.mean(list(results.values())):.4f}")

    out_dir = Path(args.checkpoint).parent
    with open(out_dir / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    np.savez_compressed(out_dir / "test_predictions.npz", **pred_dump)
    print(f"\nSaved -> {out_dir}/test_results.json and test_predictions.npz")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--mode", required=True, choices=["full", "no_bilstm", "no_stylometric"])
    p.add_argument("--feat-dir", default="data/processed/features_finetuned_official_test")
    p.add_argument("--threshold", type=float, default=0.40)
    args = p.parse_args()
    main(args)