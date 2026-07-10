#!/usr/bin/env python3
"""Ablation: no Hard oversampling (uniform weights)."""
import argparse, json, sys, gc
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, RandomSampler, Dataset
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.deberta_fusion import DeBERTaFusionE2E

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TIERS  = ["easy", "medium", "hard"]
YEARS  = ["2024", "2025"]


def load_cached(feat_dir, year, tier, split):
    path = feat_dir / f"{year}_{tier}_{split}.npz"
    if not path.exists(): return []
    data = np.load(path)
    cls, style, labels, lengths = (data["cls_reprs"].astype(np.float32),
        data["style_diffs"].astype(np.float32),
        data["labels"].astype(np.int8),
        data["doc_lengths"].astype(np.int32))
    items, idx = [], 0
    for n in lengths.tolist():
        if n > 0:
            items.append((cls[idx:idx+n], style[idx:idx+n], labels[idx:idx+n]))
        idx += n
    return items


class FlatDocDataset(Dataset):
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, idx): return self.items[idx]


def forward_doc(model, cls_block, style_block):
    pair_reprs = []
    for j in range(cls_block.shape[0]):
        cls   = cls_block[j].unsqueeze(0).to(DEVICE)
        style = style_block[j].unsqueeze(0).to(DEVICE)
        fused = model.style_fusion(cls, style)
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


def evaluate(model, val_items_by_key, threshold):
    model.eval()
    results = {}
    with torch.no_grad():
        for key, items in val_items_by_key.items():
            all_preds, all_lbls = [], []
            for cls_np, style_np, lbl_np in items:
                logits = forward_doc(model, torch.tensor(cls_np), torch.tensor(style_np))
                probs  = torch.sigmoid(logits).cpu().numpy()
                all_preds.append((probs >= threshold).astype(int).flatten().tolist())
                all_lbls.append(lbl_np.astype(int).flatten().tolist())
            results[key] = compute_f1(all_preds, all_lbls)
            print(f"  [{key}] F1={results[key]:.4f}")
    return results


def train(args):
    if args.feat_dir:
        feat_dir = ROOT / args.feat_dir
    else:
        feat_dir = ROOT / "data" / "processed" / f"features_finetuned_seed{args.seed}"
    print(f"Ablation: NO OVERSAMPLING | Device: {DEVICE}")
    print(f"Feature dir: {feat_dir}")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    model = DeBERTaFusionE2E(style_dim=193, unfreeze_layers=2,
        use_style=True, use_attention_fusion=True,
        use_bilstm=True, use_adversarial=False)
    model.load_state_dict(state_dict, strict=False)
    for name, param in model.named_parameters():
        if 'deberta' in name: param.requires_grad = False
    model.deberta = model.deberta.cpu()
    model = model.to(DEVICE)

    train_items_by_tier = {t: [] for t in TIERS}
    val_items_by_key = {}
    for year in YEARS:
        for tier in TIERS:
            tr = load_cached(feat_dir, year, tier, "train")
            vl = load_cached(feat_dir, year, tier, "val")
            if not tr: continue
            train_items_by_tier[tier].extend(tr)
            val_items_by_key[f"{year}_{tier}"] = vl

    flat_items = []
    for tier in TIERS:
        flat_items.extend(train_items_by_tier[tier])

    if len(flat_items) == 0:
        print("\nERROR: No training data found. Check --feat-dir.")
        return

    print(f"Total train docs: {len(flat_items)} | uniform weights (no oversampling)")

    ds = FlatDocDataset(flat_items)
    loader = DataLoader(ds, batch_size=1, sampler=RandomSampler(ds),
                        collate_fn=lambda b: b, num_workers=0)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4, weight_decay=0.01)
    total_steps = len(loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(0.06 * total_steps), total_steps)
    scaler = GradScaler()
    criterion = nn.BCEWithLogitsLoss()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_hard_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for batch in tqdm(loader, desc=f"Epoch {epoch}", ncols=80):
            cls_np, style_np, lbl_np = batch[0]
            cls_t = torch.tensor(cls_np)
            style_t = torch.tensor(style_np)
            lbl_t = torch.tensor(lbl_np.astype(np.float32)).to(DEVICE)
            with autocast(device_type="cuda"):
                logits = forward_doc(model, cls_t, style_t)
                loss = criterion(logits.reshape(-1), lbl_t.reshape(-1))
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            epoch_loss += loss.item()
            n_batches += 1

        print(f"\nEpoch {epoch} | Loss={epoch_loss/max(n_batches,1):.4f}")
        results = evaluate(model, val_items_by_key, args.threshold)
        hard_f1s = [v for k, v in results.items() if "hard" in k]
        avg_hard_f1 = np.mean(hard_f1s) if hard_f1s else 0.0
        print(f"Avg Hard F1={avg_hard_f1:.4f} | Best={best_hard_f1:.4f}")
        if avg_hard_f1 > best_hard_f1:
            best_hard_f1 = avg_hard_f1
            torch.save(model.state_dict(), out_dir / "best.pt")
            print(f"  ** New best!")
        with open(out_dir / f"epoch_{epoch}.json", "w") as f:
            json.dump({"epoch": epoch, "loss": epoch_loss/max(n_batches,1),
                       "avg_hard_f1": avg_hard_f1, "results": results}, f, indent=2)
        torch.cuda.empty_cache()

    print(f"\nDone. Best Hard Val F1: {best_hard_f1:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/curriculum_228_best.pt")
    p.add_argument("--out-dir", default="checkpoints/ablation_no_oversampling")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--threshold", type=float, default=0.40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--feat-dir", default=None)
    args = p.parse_args()
    train(args)