#!/usr/bin/env python3
"""
Joint fine-tuning: unfreezes DeBERTa's last 2 layers and trains them
TOGETHER with the stylometric fusion + BiLSTM + head, end-to-end.
Unlike retrain_fulldata.py, this does NOT use precomputed CLS features —
it loads raw text and runs DeBERTa forward+backward every step, since
frozen precomputed features can't backprop into the encoder.

This is slower than the frozen-backbone pipeline. Expect a long overnight run.

Usage:
    python scripts/finetune_joint.py --checkpoint checkpoints/curriculum_228_best.pt --years 2025 --epochs 5
"""
import argparse, json, sys, gc
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup, AutoTokenizer
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.deberta_fusion import DeBERTaFusionE2E
from src.features.stylometric import StylometricExtractor
from src.utils.data_loader import PANDataset, split_documents

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TIERS  = ["easy", "medium", "hard"]

_tokenizer = None
def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small")
    return _tokenizer


def load_docs(years, seed):
    """Load raw documents (train+val combined) for the given years."""
    docs_by_tier = {t: [] for t in TIERS}
    for year in years:
        for tier in TIERS:
            try:
                docs = PANDataset(ROOT / "data" / "raw", year=year, tier=tier).load()
            except FileNotFoundError:
                print(f"  [{year}/{tier}] not found, skipping")
                continue
            train, val, _ = split_documents(docs, seed=seed)
            combined = train + val
            docs_by_tier[tier].extend(combined)
            print(f"  [{year}/{tier}] {len(train)} train + {len(val)} val = {len(combined)} total")
    return docs_by_tier


def forward_doc_joint(model, extractor, paragraphs, batch_size=4):
    """
    Encodes a document end-to-end with gradients flowing into DeBERTa.
    Returns per-boundary logits for the whole document.
    """
    tokenizer = get_tokenizer()
    pairs = list(zip(paragraphs[:-1], paragraphs[1:]))
    _, style_diffs = extractor.extract_document(paragraphs)
    n = len(pairs)
    assert style_diffs.shape[0] == n

    fused_list = []
    for i in range(0, n, batch_size):
        batch = pairs[i:i+batch_size]
        a_list = [p[0] for p in batch]
        b_list = [p[1] for p in batch]
        enc = tokenizer(a_list, b_list, padding=True, truncation=True,
                        max_length=256, return_tensors="pt")
        input_ids      = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)
        out = model.deberta(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]  # (batch, 768), gradients flow

        style_batch = torch.tensor(
            style_diffs[i:i+batch_size], dtype=torch.float32
        ).to(DEVICE)

        fused = model.style_fusion(cls, style_batch)  # (batch, fused_dim)
        fused_list.append(fused)

    all_fused = torch.cat(fused_list, dim=0).unsqueeze(0)  # (1, n, fused_dim)
    logits = model.forward_document(all_fused).squeeze(0).reshape(-1)
    return logits


def train(args):
    torch.cuda.empty_cache(); gc.collect()
    print(f"Device: {DEVICE}")
    print(f"Years: {args.years} | Seed: {args.seed}")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    model = DeBERTaFusionE2E(
        style_dim=193, unfreeze_layers=2,
        use_style=True, use_attention_fusion=True,
        use_bilstm=True, use_adversarial=False,
    )
    model.load_state_dict(state_dict, strict=False)
    model = model.to(DEVICE)  # DeBERTa stays on GPU, stays trainable — no freezing this time

    deberta_params = [p for n, p in model.named_parameters() if 'deberta' in n and p.requires_grad]
    other_params   = [p for n, p in model.named_parameters() if 'deberta' not in n and p.requires_grad]
    total_trainable = sum(p.numel() for p in deberta_params) + sum(p.numel() for p in other_params)
    print(f"Trainable: {total_trainable:,} params ({len(deberta_params)} DeBERTa tensors + {len(other_params)} fusion/BiLSTM/head tensors)")

    extractor = StylometricExtractor()

    print("\nLoading raw documents (train+val combined)...")
    docs_by_tier = load_docs(args.years, args.seed)

    weights_map = {"easy": 1.0, "medium": 1.0, "hard": args.hard_weight}
    flat_docs, flat_weights = [], []
    for tier in TIERS:
        for doc in docs_by_tier[tier]:
            if len(doc.sentences) < 2:
                continue
            flat_docs.append(doc)
            flat_weights.append(weights_map[tier])

    if not flat_docs:
        print("\nERROR: No training documents found. Check --years.")
        return

    print(f"\nTotal train docs: {len(flat_docs)}")

    optimizer = torch.optim.AdamW([
        {"params": deberta_params, "lr": args.deberta_lr},
        {"params": other_params,   "lr": args.lr},
    ], weight_decay=0.01)

    total_steps  = len(flat_docs) * args.epochs
    warmup_steps = int(0.06 * total_steps)
    scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler       = GradScaler()
    criterion    = nn.BCEWithLogitsLoss()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, n_steps = 0.0, 0

        order = rng.permutation(len(flat_docs))
        weighted_order = []
        for idx in order:
            reps = int(round(flat_weights[idx]))
            weighted_order.extend([idx] * max(reps, 1))
        rng.shuffle(weighted_order)

        for idx in tqdm(weighted_order, desc=f"Epoch {epoch}", ncols=80):
            doc = flat_docs[idx]
            sents  = doc.sentences
            labels = doc.boundary_labels
            n = min(len(sents) - 1, len(labels))
            if n == 0:
                continue
            lbl_t = torch.tensor(labels[:n], dtype=torch.float32).to(DEVICE)

            with autocast(device_type="cuda"):
                logits = forward_doc_joint(model, extractor, sents, batch_size=args.pair_batch_size)
                logits = logits[:n]
                loss   = criterion(logits.reshape(-1), lbl_t.reshape(-1))

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(deberta_params + other_params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            n_steps += 1

        avg_loss = epoch_loss / max(n_steps, 1)
        print(f"\nEpoch {epoch}/{args.epochs} | Loss={avg_loss:.4f}")
        torch.cuda.empty_cache()

        with open(out_dir / f"epoch_{epoch}.json", "w") as f:
            json.dump({"epoch": epoch, "loss": avg_loss}, f, indent=2)

        # Save checkpoint every epoch (in case of interruption on a long run)
        torch.save(model.state_dict(), out_dir / f"epoch_{epoch}_model.pt")

    save_path = out_dir / "finetune_joint_final.pt"
    torch.save(model.state_dict(), save_path)
    print(f"\nDone. Final model saved -> {save_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",   default="checkpoints/curriculum_228_best.pt")
    p.add_argument("--out-dir",      default="checkpoints/finetune_joint")
    p.add_argument("--epochs",       type=int,   default=5)
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--deberta-lr",   type=float, default=1e-5)
    p.add_argument("--hard-weight",  type=float, default=3.0)
    p.add_argument("--pair-batch-size", type=int, default=4)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--years",        nargs="+",  default=["2025"])
    args = p.parse_args()
    train(args)