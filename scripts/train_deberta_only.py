#!/usr/bin/env python3
"""
Baseline: DeBERTa CLS representation + linear head only.
No stylometric features, no attention fusion, no BiLSTM.
Same fine-tuned DeBERTa backbone (via precomputed cls_reprs) as the full model —
isolates the contribution of stylometric fusion + BiLSTM.

Usage:
    python scripts/train_deberta_only.py --feat-dir data/processed/features_finetuned --out-dir checkpoints/deberta_only
"""
import argparse, json, sys, gc
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TIERS  = ["easy", "medium", "hard"]


class DeBERTaOnlyHead(nn.Module):
    def __init__(self, cls_dim=768, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cls_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, 1),
        )
    def forward(self, cls_block):
        return self.net(cls_block).reshape(-1)


def load_cached(feat_dir, year, tier, split):
    path = feat_dir / f"{year}_{tier}_{split}.npz"
    if not path.exists():
        return []
    data    = np.load(path)
    cls     = data["cls_reprs"].astype(np.float32)
    labels  = data["labels"].astype(np.int8)
    lengths = data["doc_lengths"].astype(np.int32)
    items, idx = [], 0
    for n in lengths.tolist():
        if n > 0:
            items.append((cls[idx:idx+n], labels[idx:idx+n]))
        idx += n
    return items


class FlatDocDataset(Dataset):
    def __init__(self, items):
        self.items = items
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        return self.items[idx]


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


def train(args):
    torch.cuda.empty_cache(); gc.collect()
    print(f"Device: {DEVICE}")

    feat_dir = ROOT / args.feat_dir
    years    = args.years
    print(f"Feature dir: {feat_dir} | years: {years}")

    model = DeBERTaOnlyHead(cls_dim=768, hidden=args.hidden).to(DEVICE)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable: {trainable:,} params (linear head only)")

    print("\nLoading train + val (100% of training data)...")
    items_by_tier = {t: [] for t in TIERS}
    for year in years:
        for tier in TIERS:
            tr = load_cached(feat_dir, year, tier, "train")
            vl = load_cached(feat_dir, year, tier, "val")
            combined = tr + vl
            if not combined:
                print(f"  [{year}/{tier}] MISSING — skipping")
                continue
            items_by_tier[tier].extend(combined)
            print(f"  [{year}/{tier}] {len(tr)} train + {len(vl)} val = {len(combined)} total")

    weights_map = {"easy": 1.0, "medium": 1.0, "hard": args.hard_weight}
    flat_items, flat_weights = [], []
    for tier in TIERS:
        for item in items_by_tier[tier]:
            flat_items.append(item)
            flat_weights.append(weights_map[tier])

    if len(flat_items) == 0:
        print("\nERROR: No training data found. Check --feat-dir / --years.")
        return

    print(f"\nTotal train docs: {len(flat_items)} | hard weight={args.hard_weight}x")

    ds      = FlatDocDataset(flat_items)
    sampler = WeightedRandomSampler(flat_weights, len(flat_weights), replacement=True)
    loader  = DataLoader(ds, batch_size=1, sampler=sampler,
                         collate_fn=lambda b: b, num_workers=0)

    optimizer    = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps  = len(loader) * args.epochs
    warmup_steps = int(0.06 * total_steps)
    scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler       = GradScaler()
    criterion    = nn.BCEWithLogitsLoss()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for batch in tqdm(loader, desc=f"Epoch {epoch}", ncols=80):
            cls_np, lbl_np = batch[0]
            cls_t = torch.tensor(cls_np).to(DEVICE)
            lbl_t = torch.tensor(lbl_np.astype(np.float32)).to(DEVICE)
            with autocast(device_type="cuda"):
                logits = model(cls_t)
                loss   = criterion(logits, lbl_t.reshape(-1))
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            epoch_loss += loss.item(); n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"\nEpoch {epoch}/{args.epochs} | Loss={avg_loss:.4f}")
        torch.cuda.empty_cache()

        with open(out_dir / f"epoch_{epoch}.json", "w") as f:
            json.dump({"epoch": epoch, "loss": avg_loss}, f, indent=2)

    save_path = out_dir / "deberta_only_final.pt"
    torch.save(model.state_dict(), save_path)
    print(f"\nDone. Final model saved -> {save_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--feat-dir",    default="data/processed/features_finetuned")
    p.add_argument("--out-dir",     default="checkpoints/deberta_only")
    p.add_argument("--epochs",      type=int,   default=10)
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--hard-weight", type=float, default=3.0)
    p.add_argument("--hidden",      type=int,   default=256)
    p.add_argument("--years",       nargs="+",  default=["2024", "2025"])
    args = p.parse_args()
    train(args)