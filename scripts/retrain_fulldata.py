#!/usr/bin/env python3
"""
Retrains on 100% of training data (train + val combined).
No validation split — fixed 10 epochs, saves final checkpoint.

Usage:
    python scripts/retrain_fulldata.py --checkpoint checkpoints/curriculum_228_best.pt --feat-dir data/processed/features_finetuned
    python scripts/retrain_fulldata.py --checkpoint checkpoints/curriculum_228_best.pt --feat-dir data/processed/features_finetuned --years 2025 --out-dir checkpoints/fulldata_2025only
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

from src.models.deberta_fusion import DeBERTaFusionE2E

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TIERS  = ["easy", "medium", "hard"]


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


class FlatDocDataset(Dataset):
    def __init__(self, items):
        self.items = items
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        return self.items[idx]


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


def train(args):
    torch.cuda.empty_cache(); gc.collect()
    print(f"Device: {DEVICE}")

    if args.feat_dir:
        feat_dir = ROOT / args.feat_dir
    else:
        feat_dir = ROOT / "data" / "processed" / f"features_finetuned_seed{args.seed}"

    print(f"Feature dir: {feat_dir}")
    print(f"Years: {args.years}")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    model = DeBERTaFusionE2E(
        style_dim=193, unfreeze_layers=2,
        use_style=True, use_attention_fusion=True,
        use_bilstm=True, use_adversarial=False,
    )
    model.load_state_dict(state_dict, strict=False)
    for name, param in model.named_parameters():
        if 'deberta' in name:
            param.requires_grad = False
    model.deberta = model.deberta.cpu()
    model = model.to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable: {trainable:,} params")

    # Load train + val combined
    print("\nLoading train + val (100% of training data)...")
    items_by_tier = {t: [] for t in TIERS}
    for year in args.years:
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

    optimizer    = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01
    )
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
            cls_np, style_np, lbl_np = batch[0]
            cls_t   = torch.tensor(cls_np)
            style_t = torch.tensor(style_np)
            lbl_t   = torch.tensor(lbl_np.astype(np.float32)).to(DEVICE)
            with autocast(device_type="cuda"):
                logits = forward_doc(model, cls_t, style_t)
                loss   = criterion(logits.reshape(-1), lbl_t.reshape(-1))
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            epoch_loss += loss.item(); n_batches += 1
            del cls_t, style_t, lbl_t, logits, loss

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"\nEpoch {epoch}/{args.epochs} | Loss={avg_loss:.4f}")
        torch.cuda.empty_cache()

        with open(out_dir / f"epoch_{epoch}.json", "w") as f:
            json.dump({"epoch": epoch, "loss": avg_loss}, f, indent=2)

    save_path = out_dir / "fulldata_final.pt"
    torch.save(model.state_dict(), save_path)
    print(f"\nDone. Final model saved -> {save_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  default="checkpoints/curriculum_228_best.pt")
    p.add_argument("--out-dir",     default="checkpoints/fulldata")
    p.add_argument("--epochs",      type=int,   default=10)
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--hard-weight", type=float, default=3.0)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--feat-dir",    default=None)
    p.add_argument("--years",       nargs="+", default=["2024", "2025"])
    args = p.parse_args()
    train(args)