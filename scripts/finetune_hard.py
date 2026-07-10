"""
Hard-tier fine-tuning on top of an existing checkpoint.
Loads a pre-trained checkpoint and fine-tunes for N additional epochs
on Hard-tier documents only, with a lower learning rate.

Usage:
    python scripts/finetune_hard.py --checkpoint checkpoints/bilstm_best.pt --epochs 5
"""

import sys
import argparse
import time
import json
import os
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.amp import GradScaler, autocast

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.data_loader import PANDataset, split_documents
from src.features.stylometric import StylometricExtractor
from src.evaluation.metrics import binary_metrics, window_diff
from src.models.deberta_fusion import DeBERTaFusionE2E, DocLevelDataset
from transformers import AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--year", default="2024,2025")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr-deberta", type=float, default=5e-6, help="Lower LR for DeBERTa layers")
    p.add_argument("--lr-head", type=float, default=1e-4, help="Lower LR for head")
    p.add_argument("--bilstm-hidden", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--sub-batch", type=int, default=8)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--doc-accum", type=int, default=4)
    p.add_argument("--pos-weight", type=float, default=2.0)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-checkpoint", type=str, default="checkpoints/hard_ft_best.pt")
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


def best_threshold(probs, labels):
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
    metrics["threshold"] = threshold
    return metrics


def train_epoch(model, dataset, tokenizer, optimizer, scaler, bce, device, args):
    model.train()
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    epoch_loss = 0.0
    optimizer.zero_grad()

    for doc_idx, idx in enumerate(indices):
        doc = dataset[idx]
        labels = torch.tensor(doc["labels"], dtype=torch.float32).to(device)
        fused = encode_document(model, doc, tokenizer, device, args.max_length, args.sub_batch)

        with autocast("cuda", dtype=torch.float16):
            logits = model.forward_document(fused.unsqueeze(0))
            loss = bce(logits, labels) / args.doc_accum

        scaler.scale(loss).backward()

        if (doc_idx + 1) % args.doc_accum == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        epoch_loss += loss.item() * args.doc_accum

    return epoch_loss / max(len(dataset), 1)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = StylometricExtractor()

    print("=" * 60)
    print("Hard-tier Fine-tuning on Pre-trained Checkpoint")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"LR: DeBERTa={args.lr_deberta} | Head={args.lr_head}")
    print(f"Epochs: {args.epochs} | Patience: {args.patience}")
    print()

    # --- Load Hard-tier docs only ---
    print("Loading Hard-tier data...")
    years = [y.strip() for y in args.year.split(",")]
    all_hard_docs = []
    for year in years:
        docs = PANDataset(root="data/raw", year=year, tier="hard").load()
        all_hard_docs.extend(docs)
        print(f"  PAN {year} hard: {len(docs)} docs")

    train_docs, val_docs, test_docs = split_documents(all_hard_docs, seed=args.seed)
    print(f"  Train: {len(train_docs)} | Val: {len(val_docs)} | Test: {len(test_docs)}")

    # Also load all-tier test docs for full per-tier eval at the end
    all_test = {}
    eval_year = args.year.split(",")[-1]
    for tier in ["easy", "medium", "hard"]:
        docs = PANDataset(root="data/raw", year=eval_year, tier=tier).load()
        _, _, test = split_documents(docs, seed=args.seed)
        all_test[tier] = test

    print("\nBuilding datasets...")
    t0 = time.time()
    train_ds = build_ds(train_docs, extractor)
    val_ds = build_ds(val_docs, extractor)
    test_ds = build_ds(test_docs, extractor)
    print(f"  Done in {time.time()-t0:.1f}s")

    tokenizer = AutoTokenizer.from_pretrained(DeBERTaFusionE2E.DEBERTA_MODEL)

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
    print(f"Loaded checkpoint: {args.checkpoint}\n")

    optimizer = torch.optim.AdamW([
        {"params": model.get_deberta_params(), "lr": args.lr_deberta},
        {"params": model.get_head_params(), "lr": args.lr_head},
    ], weight_decay=0.01)

    total_steps = (len(train_ds) * args.epochs) // args.doc_accum
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))
    scaler = GradScaler("cuda")
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.pos_weight]).to(device))

    best_val_f1, best_state, best_threshold_val = 0, None, 0.5
    patience_counter = 0

    # --- Training ---
    print("Fine-tuning on Hard tier...")
    for epoch in range(1, args.epochs + 1):
        avg_loss = train_epoch(model, train_ds, tokenizer, optimizer, scaler, bce, device, args)
        scheduler.step()

        val_probs, val_labels, val_sizes = get_probs(model, val_ds, tokenizer, device, args)
        opt_thresh, val_f1 = best_threshold(val_probs, val_labels)

        print(f"Epoch {epoch}/{args.epochs} | loss={avg_loss:.4f} | val_F1={val_f1:.4f} thr={opt_thresh:.2f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_threshold_val = opt_thresh
            print(f"  -> New best Hard val F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping (patience={args.patience})")
                break

    # --- Test evaluation ---
    print("\n" + "=" * 60)
    print("TEST EVALUATION")
    print("=" * 60)

    model.load_state_dict(best_state)
    model.to(device)

    if args.save_checkpoint:
        os.makedirs(os.path.dirname(args.save_checkpoint) or ".", exist_ok=True)
        torch.save(best_state, args.save_checkpoint)
        print(f"Checkpoint saved to {args.save_checkpoint}")

    # Hard-tier test
    test_probs, test_labels, test_sizes = get_probs(model, test_ds, tokenizer, device, args)
    hard_metrics = evaluate(test_probs, test_labels, test_sizes, best_threshold_val)
    print(f"\nHard tier (fine-tuned threshold={best_threshold_val:.2f}):")
    print(f"  F1:  {hard_metrics['f1']:.4f}")
    print(f"  WD:  {hard_metrics['window_diff']:.4f}")

    # Full per-tier eval
    baselines = {
        "easy":   {"wqd": 0.926, "SCL-DeBERTa": 0.932},
        "medium": {"wqd": 0.753, "SCL-DeBERTa": 0.776},
        "hard":   {"wqd": 0.830, "SCL-DeBERTa": 0.829},
    }

    print(f"\n--- Full per-tier breakdown (tier-specific thresholds) ---")
    per_tier = {}
    for tier in ["easy", "medium", "hard"]:
        docs = all_test[tier]
        ds = build_ds(docs, extractor)
        probs, labels, sizes = get_probs(model, ds, tokenizer, device, args)

        # Use val-optimized threshold for hard, search for others
        val_tier_docs = PANDataset(root="data/raw", year=eval_year, tier=tier).load()
        _, val_t, _ = split_documents(val_tier_docs, seed=args.seed)
        val_t_ds = build_ds(val_t, extractor)
        val_p, val_l, _ = get_probs(model, val_t_ds, tokenizer, device, args)
        t_opt, _ = best_threshold(val_p, val_l)

        m = evaluate(probs, labels, sizes, t_opt)
        per_tier[tier] = m
        print(f"  {tier:6s} | F1={m['f1']:.4f}  WD={m['window_diff']:.4f}  thr={t_opt:.2f}")
        for name, f1 in baselines[tier].items():
            delta = m["f1"] - f1
            marker = "+" if delta >= 0 else ""
            print(f"           vs {name}: F1={f1:.3f}  (ours {marker}{delta:.3f})")

    # Save results
    os.makedirs("results", exist_ok=True)
    result = {
        "variant": "deberta+style+attn+bilstm+hard_ft",
        "base_checkpoint": args.checkpoint,
        "year": args.year,
        "hard_ft_epochs": args.epochs,
        "lr_deberta": args.lr_deberta,
        "lr_head": args.lr_head,
        "best_hard_val_f1": best_val_f1,
        "per_tier_results": {k: v for k, v in per_tier.items()},
    }
    results_path = "results/deberta_fusion.json"
    existing = []
    if os.path.exists(results_path):
        with open(results_path) as f:
            existing = json.load(f)
    existing.append(result)
    with open(results_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()