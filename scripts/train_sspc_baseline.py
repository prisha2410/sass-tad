"""
Train SSPC Baseline — BiLSTM over frozen SBERT embeddings.

Usage:
    python scripts/train_sspc_baseline.py
    python scripts/train_sspc_baseline.py --year 2024 --tier hard
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.utils.data_loader import load_pan_labeled, Document
from src.utils.sbert_cache import EmbeddingCache
from src.evaluation.metrics import evaluate_predictions


class SSPCDataset(Dataset):
    def __init__(self, docs, sbert_embeddings):
        self.docs = docs
        self.sbert_embeddings = sbert_embeddings

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        doc = self.docs[idx]
        return {
            "sbert": self.sbert_embeddings[idx],
            "boundary_labels": torch.tensor(doc.boundary_labels or [], dtype=torch.float32),
            "length": len(doc.sentences),
            "doc_id": doc.doc_id,
        }


def collate_fn(batch):
    max_sents = max(item["length"] for item in batch)
    B = len(batch)
    sbert_dim = batch[0]["sbert"].shape[-1]

    sbert_padded = torch.zeros(B, max_sents, sbert_dim)
    boundary_padded = torch.zeros(B, max_sents - 1)
    boundary_mask = torch.zeros(B, max_sents - 1, dtype=torch.bool)
    lengths = torch.zeros(B, dtype=torch.long)
    doc_ids = []

    for i, item in enumerate(batch):
        n = item["length"]
        lengths[i] = n
        sbert_padded[i, :n] = item["sbert"][:n]
        n_b = min(len(item["boundary_labels"]), max_sents - 1)
        if n_b > 0:
            boundary_padded[i, :n_b] = item["boundary_labels"][:n_b]
            boundary_mask[i, :n_b] = True
        doc_ids.append(item["doc_id"])

    return {
        "sbert": sbert_padded,
        "boundary_labels": boundary_padded,
        "boundary_mask": boundary_mask,
        "lengths": lengths,
        "doc_ids": doc_ids,
    }


def main():
    parser = argparse.ArgumentParser(description="Train SSPC Baseline")
    parser.add_argument("--year", default="2024")
    parser.add_argument("--tier", default="hard")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--cache-dir", default="data/processed/embeddings")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--results-file", default="results/ablations.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading PAN {args.year} {args.tier}...")
    train_docs, val_docs, test_docs = load_pan_labeled(args.data_root, args.year, args.tier)

    cache = EmbeddingCache(args.cache_dir)
    source = f"pan{args.year}"
    train_emb, _, _ = cache.load(source, args.tier, "train")
    val_emb, _, _ = cache.load(source, args.tier, "val")

    train_dataset = SSPCDataset(train_docs, train_emb)
    val_dataset = SSPCDataset(val_docs, val_emb)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0, pin_memory=True)

    all_labels = [l for doc in train_docs for l in (doc.boundary_labels or [])]
    n_pos = sum(all_labels)
    n_neg = len(all_labels) - n_pos
    pos_weight = torch.tensor(n_neg / max(n_pos, 1), device=device)
    print(f"Class balance: {n_pos} pos / {n_neg} neg -> pos_weight={pos_weight:.2f}")

    from src.models.sspc_baseline import SSPCBaseline
    model = SSPCBaseline(sbert_dim=384, lstm_hidden=128, lstm_layers=2, mlp_hidden=128, dropout=0.3)
    model = model.to(device)
    print(f"SSPC Baseline: {model.count_parameters():,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    best_wd = float("inf")
    best_threshold = 0.5
    patience_counter = 0
    patience = 12
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        total_loss = 0
        n_batches = 0

        for batch in train_loader:
            sbert = batch["sbert"].to(device)
            bl = batch["boundary_labels"].to(device)
            bm = batch["boundary_mask"].to(device)
            lengths = batch["lengths"].to(device)

            if scaler:
                with torch.amp.autocast("cuda"):
                    out = model(sbert, lengths)
                    loss = F.binary_cross_entropy_with_logits(
                        out["boundary_logits"], bl, pos_weight=pos_weight, reduction="none"
                    )
                    loss = (loss * bm.float()).sum() / bm.float().sum().clamp(min=1)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(sbert, lengths)
                loss = F.binary_cross_entropy_with_logits(
                    out["boundary_logits"], bl, pos_weight=pos_weight, reduction="none"
                )
                loss = (loss * bm.float()).sum() / bm.float().sum().clamp(min=1)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            n_batches += 1

        model.eval()
        all_true, all_probs = [], []
        with torch.no_grad():
            for batch in val_loader:
                sbert = batch["sbert"].to(device)
                bl = batch["boundary_labels"].to(device)
                lengths = batch["lengths"].to(device)
                out = model(sbert, lengths)
                probs = torch.sigmoid(out["boundary_logits"])
                for i in range(len(batch["doc_ids"])):
                    n = lengths[i].item() - 1
                    if n > 0:
                        all_true.append(bl[i, :n].cpu().long().tolist())
                        all_probs.append(probs[i, :n].cpu().tolist())

        best_t_epoch, best_wd_epoch = 0.5, float("inf")
        for t in np.arange(0.30, 0.71, 0.02):
            preds = [[1 if p > t else 0 for p in doc] for doc in all_probs]
            metrics = evaluate_predictions(all_true, preds)
            if metrics["window_diff"] < best_wd_epoch:
                best_wd_epoch = metrics["window_diff"]
                best_t_epoch = t

        preds_best = [[1 if p > best_t_epoch else 0 for p in doc] for doc in all_probs]
        metrics = evaluate_predictions(all_true, preds_best)
        scheduler.step(best_wd_epoch)
        elapsed = time.time() - t0

        print(f"Epoch {epoch+1:3d}/{args.epochs} | loss={total_loss/n_batches:.4f} | "
              f"F1={metrics['f1']:.4f} | WD={metrics['window_diff']:.4f} | t={best_t_epoch:.2f} | {elapsed:.1f}s")

        if best_wd_epoch < best_wd:
            best_wd = best_wd_epoch
            best_threshold = best_t_epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "best_wd": best_wd,
                "best_threshold": best_threshold,
                "val_metrics": metrics,
            }, ckpt_dir / "best_model_sspc.pt")
            print(f"  -> New best! WD={best_wd:.4f} t={best_threshold:.2f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  -> Early stopping at epoch {epoch+1}")
                break

    print("\n" + "=" * 70)
    print("Final evaluation on TEST set:")
    test_emb, _, _ = cache.load(source, args.tier, "test")
    test_dataset = SSPCDataset(test_docs, test_emb)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    best_ckpt = ckpt_dir / "best_model_sspc.pt"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        best_threshold = ckpt.get("best_threshold", 0.5)

    model.eval()
    all_true, all_probs = [], []
    with torch.no_grad():
        for batch in test_loader:
            sbert = batch["sbert"].to(device)
            bl = batch["boundary_labels"].to(device)
            lengths = batch["lengths"].to(device)
            out = model(sbert, lengths)
            probs = torch.sigmoid(out["boundary_logits"])
            for i in range(len(batch["doc_ids"])):
                n = lengths[i].item() - 1
                if n > 0:
                    all_true.append(bl[i, :n].cpu().long().tolist())
                    all_probs.append(probs[i, :n].cpu().tolist())

    preds_final = [[1 if p > best_threshold else 0 for p in doc] for doc in all_probs]
    test_metrics = evaluate_predictions(all_true, preds_final)

    print(f"\n  F1={test_metrics['f1']:.4f}  WD={test_metrics['window_diff']:.4f}  "
          f"P={test_metrics['precision']:.4f}  R={test_metrics['recall']:.4f}")

    result_entry = {
        "config": {"model": "sspc_baseline", "year": args.year, "tier": args.tier},
        "metrics": {
            "f1": round(test_metrics["f1"], 4),
            "window_diff": round(test_metrics["window_diff"], 4),
            "precision": round(test_metrics["precision"], 4),
            "recall": round(test_metrics["recall"], 4),
        },
    }
    results_path = Path(args.results_file)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
    else:
        all_results = []
    all_results.append(result_entry)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()