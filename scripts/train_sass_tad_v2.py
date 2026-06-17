"""
Train SASS-TAD v2 — with curriculum training.

Usage:
    python scripts/train_sass_tad_v2.py
    python scripts/train_sass_tad_v2.py --year 2024 --tier hard --epochs 50
    python scripts/train_sass_tad_v2.py --no-adversarial
"""

import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from src.utils.data_loader import load_pan_labeled
from src.utils.sbert_cache import EmbeddingCache
from src.features.stylometric import StylometricExtractor
from src.models.sass_tad import SASSTAD
from src.training.losses import SASTADLoss
from src.training.dataset import SASTADDataset, collate_fn
from src.training.trainer_v2 import SASTADTrainer


def main():
    parser = argparse.ArgumentParser(description="Train SASS-TAD v2 (curriculum)")
    parser.add_argument("--year", default="2024")
    parser.add_argument("--tier", default="hard")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--cache-dir", default="data/processed/embeddings")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--no-adversarial", action="store_true")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading PAN {args.year} {args.tier}...")
    train_docs, val_docs, test_docs = load_pan_labeled(args.data_root, args.year, args.tier)

    cache = EmbeddingCache(args.cache_dir)
    source = f"pan{args.year}"
    train_emb, _, _ = cache.load(source, args.tier, "train")
    val_emb, _, _ = cache.load(source, args.tier, "val")

    print("Extracting stylometric features...")
    extractor = StylometricExtractor(use_pos=True)
    style_dim = extractor.feature_dim
    print(f"Stylometric dim: {style_dim}")

    train_dataset = SASTADDataset(train_docs, train_emb, style_extractor=extractor)
    val_dataset = SASTADDataset(val_docs, val_emb, style_extractor=extractor)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True,
    )

    all_labels = [l for doc in train_docs for l in (doc.boundary_labels or [])]
    n_pos = sum(all_labels)
    n_neg = len(all_labels) - n_pos
    pos_weight = n_neg / max(n_pos, 1)
    print(f"Class balance: {n_pos} pos / {n_neg} neg -> pos_weight={pos_weight:.2f}")

    model = SASSTAD(
        sbert_dim=384,
        style_dim=style_dim,
        fusion_dim=256,
        style_embed_dim=128,
        lstm_hidden=128,
        lstm_layers=2,
        dropout=0.3,
        use_adversarial=not args.no_adversarial,
    )
    print(f"Model: {model.count_parameters():,} parameters")
    print(f"Adversarial: {not args.no_adversarial}")

    loss_fn = SASTADLoss(
        alpha=1.0,
        beta=0.3,
        gamma=0.1,
        pos_weight=pos_weight,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6,
    )

    trainer = SASTADTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        max_epochs=args.epochs,
        patience=10,
        grad_accum_steps=4,
        use_amp=(device == "cuda"),
        checkpoint_dir=args.checkpoint_dir,
        boundary_only_epochs=5,
        contrastive_ramp_epochs=5,
        adversarial_ramp_epochs=5,
        max_beta=0.3,
        max_gamma=0.1,
        max_grl_lambda=1.0,
    )

    history = trainer.fit(train_loader, val_loader)

    # Final evaluation on test set
    print("\n" + "=" * 70)
    print("Final evaluation on TEST set:")
    test_emb, _, _ = cache.load(source, args.tier, "test")
    test_dataset = SASTADDataset(test_docs, test_emb, style_extractor=extractor)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    best_ckpt = Path(args.checkpoint_dir) / "best_model.pt"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        print(f"Loaded best model from epoch {ckpt['epoch']}")

    test_metrics = trainer.evaluate(test_loader)
    print(f"  F1:         {test_metrics['f1']:.4f}")
    print(f"  Precision:  {test_metrics['precision']:.4f}")
    print(f"  Recall:     {test_metrics['recall']:.4f}")
    print(f"  WindowDiff: {test_metrics['window_diff']:.4f}")
    print(f"  Accuracy:   {test_metrics['accuracy']:.4f}")

    # Compare with baselines
    print("\n--- Comparison ---")
    print("Fused baseline:  F1=0.604  WD=0.446")
    print(f"SASS-TAD:        F1={test_metrics['f1']:.3f}  WD={test_metrics['window_diff']:.3f}")


if __name__ == "__main__":
    main()
