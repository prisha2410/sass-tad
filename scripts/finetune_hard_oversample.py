#!/usr/bin/env python3
"""
Fine-tunes fusion + BiLSTM + head using precomputed FINE-TUNED features.
DeBERTa is NOT on GPU during training — fits easily in 4GB VRAM.

Prerequisite: python scripts/precompute_finetuned_features.py

Usage:
    python scripts/finetune_hard_oversample.py --checkpoint checkpoints/curriculum_228_best.pt
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

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TIERS    = ["easy", "medium", "hard"]
YEARS    = ["2024", "2025"]
FEAT_DIR = ROOT / "data" / "processed" / "features_finetuned"


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
            items.append((
                cls[idx:idx+n],
                style[idx:idx+n],
                labels[idx:idx+n],
            ))
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
    n = cls_block.shape[0]
    pair_reprs = []
    for j in range(n):
        cls   = cls_block[j].unsqueeze(0).to(DEVICE)
        style = style_block[j].unsqueeze(0).to(DEVICE)
        if model.use_attention_fusion:
            fused = model.style_fusion(cls, style)
        else:
            fused = torch.cat([cls, style], dim=-1)
        pair_reprs.append(fused)

    pair_reprs_t = torch.cat(pair_reprs, dim=0).unsqueeze(0)
    logits = model.forward_document(pair_reprs_t).squeeze(0)
    return logits.reshape(-1)


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


def evaluate(model, all_items_by_key, threshold=0.5):
    model.eval()
    results = {}
    with torch.no_grad():
        for key, items in all_items_by_key.items():
            all_preds, all_lbls = [], []
            for cls_np, style_np, lbl_np in items:
                cls_t   = torch.tensor(cls_np)
                style_t = torch.tensor(style_np)
                logits  = forward_doc(model, cls_t, style_t)
                probs   = torch.sigmoid(logits).cpu().numpy()
                all_preds.append((probs >= threshold).astype(int).flatten().tolist())
                all_lbls.append(lbl_np.astype(int).flatten().tolist())
            f1 = compute_f1(all_preds, all_lbls)
            results[key] = f1
            print(f"  [{key}] F1={f1:.4f}")
    return results


def train(args):
    torch.cuda.empty_cache()
    gc.collect()
    print(f"Device: {DEVICE}")
    print(f"Feature dir: {FEAT_DIR}")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

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
    print(f"Trainable: {trainable:,} params (fusion + BiLSTM + head only)")

    print("\nLoading precomputed features...")
    train_items_by_tier = {t: [] for t in TIERS}
    val_items_by_key    = {}

    for year in YEARS:
        for tier in TIERS:
            tr = load_cached(year, tier, "train")
            vl = load_cached(year, tier, "val")
            if not tr:
                print(f"  [{year}/{tier}] MISSING train — run precompute_finetuned_features.py first!")
                continue
            train_items_by_tier[tier].extend(tr)
            val_items_by_key[f"{year}_{tier}"] = vl
            print(f"  [{year}/{tier}] train={len(tr)}, val={len(vl)}")

    weights_map = {"easy": 1.0, "medium": 1.0, "hard": args.hard_weight}
    flat_items, flat_weights = [], []
    for tier in TIERS:
        for item in train_items_by_tier[tier]:
            flat_items.append(item)
            flat_weights.append(weights_map[tier])

    print(f"\nTotal train docs: {len(flat_items)} | hard weight={args.hard_weight}x")

    ds      = FlatDocDataset(flat_items)
    sampler = WeightedRandomSampler(flat_weights, len(flat_weights), replacement=True)
    loader  = DataLoader(ds, batch_size=1, sampler=sampler,
                         collate_fn=lambda b: b, num_workers=0)

    optimizer = torch.optim.AdamW(
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
    best_hard_f1 = 0.0

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

            epoch_loss += loss.item()
            n_batches  += 1
            del cls_t, style_t, lbl_t, logits, loss

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"\nEpoch {epoch}/{args.epochs} | Loss={avg_loss:.4f}")
        print("Evaluating on val set...")

        results     = evaluate(model, val_items_by_key, threshold=args.threshold)
        hard_f1s    = [v for k, v in results.items() if "hard" in k]
        avg_hard_f1 = np.mean(hard_f1s) if hard_f1s else 0.0
        print(f"Avg Hard Val F1={avg_hard_f1:.4f} | Best={best_hard_f1:.4f}")

        if avg_hard_f1 > best_hard_f1:
            best_hard_f1 = avg_hard_f1
            save_path = out_dir / "hard_oversample_best.pt"
            torch.save(model.state_dict(), save_path)
            print(f"  ** New best! -> {save_path}")

        with open(out_dir / f"epoch_{epoch}.json", "w") as f:
            json.dump({"epoch": epoch, "loss": avg_loss,
                       "avg_hard_f1": avg_hard_f1, "results": results}, f, indent=2)

        torch.cuda.empty_cache()

    print(f"\nDone. Best Hard Val F1: {best_hard_f1:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  default="checkpoints/curriculum_228_best.pt")
    p.add_argument("--out-dir",     default="checkpoints/hard_oversample")
    p.add_argument("--epochs",      type=int,   default=10)
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--hard-weight", type=float, default=3.0)
    p.add_argument("--threshold",   type=float, default=0.40)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--feat-dir",    default=None,
                   help="override feature dir (default: features_finetuned_seed{seed})")
    args = p.parse_args()

    # Set feat dir based on seed
    if args.feat_dir:
        FEAT_DIR = ROOT / args.feat_dir
    else:
        FEAT_DIR = ROOT / "data" / "processed" / f"features_finetuned_seed{args.seed}"

    train(args)