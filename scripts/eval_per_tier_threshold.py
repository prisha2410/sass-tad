"""
Per-tier threshold optimization on a saved checkpoint.

Usage:
    python scripts/eval_per_tier_threshold.py --checkpoint checkpoints/bilstm_best.pt
    python scripts/eval_per_tier_threshold.py --checkpoint checkpoints/curriculum_best.pt
"""

import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.amp import autocast

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.data_loader import PANDataset, split_documents
from src.features.stylometric import StylometricExtractor
from src.evaluation.metrics import binary_metrics, window_diff
from src.models.deberta_fusion import DeBERTaFusionE2E, DocLevelDataset
from transformers import AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to saved checkpoint .pt file")
    p.add_argument("--year", default="2024,2025")
    p.add_argument("--bilstm-hidden", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--sub-batch", type=int, default=8)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_style_diffs(docs, extractor):
    style_diffs = []
    for doc in docs:
        vecs = [extractor.extract_vector(s) for s in doc.sentences]
        diffs = []
        for i in range(len(doc.sentences) - 1):
            diff = np.abs(np.asarray(vecs[i + 1]) - np.asarray(vecs[i]))
            diffs.append(diff)
        style_diffs.append(np.stack(diffs) if diffs else np.zeros((0, extractor.feature_dim)))
    return style_diffs


def build_ds(docs, extractor):
    sents_list = [d.sentences for d in docs]
    labels_list = [d.boundary_labels for d in docs]
    doc_ids = [d.doc_id for d in docs]
    style_diffs = build_style_diffs(docs, extractor)
    return DocLevelDataset(sents_list, labels_list, style_diffs, doc_ids=doc_ids)


def encode_document(model, doc, tokenizer, device, max_length, sub_batch_size):
    pairs = doc["pairs"]
    style_diffs = doc["style_diffs"]
    n = len(pairs)
    all_fused = []
    for start in range(0, n, sub_batch_size):
        end = min(start + sub_batch_size, n)
        batch_a = [pairs[i][0] for i in range(start, end)]
        batch_b = [pairs[i][1] for i in range(start, end)]
        encoded = tokenizer(
            batch_a, batch_b,
            padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        ).to(device)
        style = None
        if style_diffs is not None:
            style = torch.tensor(style_diffs[start:end], dtype=torch.float32).to(device)
        with autocast("cuda", dtype=torch.float16):
            fused = model.encode_pairs(
                encoded["input_ids"], encoded["attention_mask"], style_diff=style,
            )
        all_fused.append(fused)
    return torch.cat(all_fused, dim=0)


def get_probs(model, dataset, tokenizer, device, args):
    model.eval()
    all_probs, all_labels, doc_sizes = [], [], []
    with torch.no_grad():
        for idx in range(len(dataset)):
            doc = dataset[idx]
            fused = encode_document(model, doc, tokenizer, device, args.max_length, args.sub_batch)
            with autocast("cuda", dtype=torch.float16):
                logits = model.forward_document(fused.unsqueeze(0))
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(doc["labels"])
            doc_sizes.append(len(doc["labels"]))
    return np.array(all_probs), np.array(all_labels, dtype=int), doc_sizes


def best_threshold(probs, labels, doc_sizes):
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.20, 0.81, 0.02):
        preds = (probs >= t).astype(int)
        f1 = binary_metrics(labels, preds)["f1"]
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def evaluate(probs, labels, doc_sizes, threshold):
    preds = (probs >= threshold).astype(int)
    metrics = binary_metrics(labels, preds)
    wd_scores = []
    idx = 0
    for n in doc_sizes:
        if n > 0:
            wd_scores.append(window_diff(labels[idx:idx+n], preds[idx:idx+n]))
            idx += n
    metrics["window_diff"] = float(np.mean(wd_scores)) if wd_scores else 0.0
    return metrics


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = StylometricExtractor()

    model = DeBERTaFusionE2E(
        style_dim=extractor.feature_dim,
        unfreeze_layers=2,
        use_style=True,
        use_attention_fusion=True,
        use_bilstm=True,
        bilstm_hidden=args.bilstm_hidden,
        dropout=args.dropout,
        use_adversarial=False,
    ).to(device)

    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    tokenizer = AutoTokenizer.from_pretrained(DeBERTaFusionE2E.DEBERTA_MODEL)

    baselines = {
        "easy":   {"wqd": 0.926, "SCL-DeBERTa": 0.932},
        "medium": {"wqd": 0.753, "SCL-DeBERTa": 0.776},
        "hard":   {"wqd": 0.830, "SCL-DeBERTa": 0.829},
    }

    eval_year = args.year.split(",")[-1]
    print(f"\n{'='*60}")
    print(f"PER-TIER THRESHOLD OPTIMIZATION (eval year: {eval_year})")
    print(f"{'='*60}")

    tier_results = {}
    for tier in ["easy", "medium", "hard"]:
        docs = PANDataset(root="data/raw", year=eval_year, tier=tier).load()
        _, val_docs, test_docs = split_documents(docs, seed=args.seed)

        # Optimize threshold on val split
        val_ds = build_ds(val_docs, extractor)
        val_probs, val_labels, val_sizes = get_probs(model, val_ds, tokenizer, device, args)
        opt_thresh, val_f1 = best_threshold(val_probs, val_labels, val_sizes)

        # Evaluate on test split with tier-specific threshold
        test_ds = build_ds(test_docs, extractor)
        test_probs, test_labels, test_sizes = get_probs(model, test_ds, tokenizer, device, args)
        metrics = evaluate(test_probs, test_labels, test_sizes, opt_thresh)

        tier_results[tier] = {"threshold": opt_thresh, "metrics": metrics}

        print(f"\n{tier.upper()} tier:")
        print(f"  Optimized threshold: {opt_thresh:.2f}  (val F1={val_f1:.4f})")
        print(f"  Test F1:    {metrics['f1']:.4f}")
        print(f"  Test WD:    {metrics['window_diff']:.4f}")
        print(f"  Precision:  {metrics['precision']:.4f}")
        print(f"  Recall:     {metrics['recall']:.4f}")
        for name, f1 in baselines[tier].items():
            delta = metrics["f1"] - f1
            marker = "+" if delta >= 0 else ""
            print(f"  vs {name}: F1={f1:.3f}  (ours {marker}{delta:.3f})")


if __name__ == "__main__":
    main()