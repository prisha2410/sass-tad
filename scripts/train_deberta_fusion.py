"""
Training script for DeBERTa + Stylometric Fusion.

Usage:
    # Pair-level (baseline)
    python scripts/train_deberta_fusion.py --finetune --tier hard --year 2025

    # Attention fusion
    python scripts/train_deberta_fusion.py --finetune --tier all --year 2024,2025 --attention-fusion

    # BiLSTM document-level (Step 3)
    python scripts/train_deberta_fusion.py --finetune --tier all --year 2024,2025 --attention-fusion --bilstm

    # BiLSTM + curriculum learning (Step 5)
    python scripts/train_deberta_fusion.py --finetune --year 2024,2025 --attention-fusion --bilstm --curriculum --epochs 12

    # BiLSTM + pseudo-topic adversarial (Step 4)
    # Requires: python scripts/precompute_topics.py --year 2024,2025 --n-topics 20
    python scripts/train_deberta_fusion.py --finetune --tier all --year 2024,2025 --attention-fusion --bilstm --adversarial --epochs 10

    # Ablations
    python scripts/train_deberta_fusion.py --finetune --tier all --year 2025 --no-style
    python scripts/train_deberta_fusion.py --finetune --tier all --year 2025 --unfreeze-layers 4
"""

import argparse
import json
import os
import sys
import time
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

warnings.filterwarnings("ignore", message="Be aware, overflowing tokens")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.data_loader import PANDataset, split_documents, dataset_stats
from src.features.stylometric import StylometricExtractor
from src.evaluation.metrics import binary_metrics, window_diff
from src.models.deberta_fusion import (
    DeBERTaFusionE2E, E2EDataset, e2e_collate_fn, DocLevelDataset,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--finetune", action="store_true")
    p.add_argument("--tier", default="all", help="easy/medium/hard/all (ignored when --curriculum is set)")
    p.add_argument("--year", default="2025")
    p.add_argument("--eval-tier", default=None)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--accum-steps", type=int, default=2)
    p.add_argument("--lr-deberta", type=float, default=1e-5)
    p.add_argument("--lr-head", type=float, default=5e-4)
    p.add_argument("--unfreeze-layers", type=int, default=2)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--pos-weight", type=float, default=2.0)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-style", action="store_true")
    p.add_argument("--attention-fusion", action="store_true")
    p.add_argument("--bilstm", action="store_true")
    p.add_argument("--bilstm-hidden", type=int, default=128)
    p.add_argument("--sub-batch", type=int, default=8)
    p.add_argument("--doc-accum", type=int, default=4)
    p.add_argument("--adversarial", action="store_true")
    p.add_argument("--n-topics", type=int, default=20)
    p.add_argument("--adv-weight", type=float, default=0.3)
    p.add_argument("--topic-cache", type=str, default="data/cache/topic_clusters.pt")
    p.add_argument("--save-checkpoint", type=str, default=None)
    p.add_argument("--curriculum", action="store_true",
                   help="Three-phase curriculum: Easy → Easy+Medium → All tiers")
    p.add_argument("--curriculum-phases", type=str, default="3,3,4",
                   help="Epoch counts per curriculum phase (default: 3,3,4 for 10 epochs)")
    return p.parse_args()


def load_docs(year_str: str, tier: str, seed: int):
    years = [y.strip() for y in year_str.split(",")]
    all_docs = []
    for year in years:
        tiers = ["easy", "medium", "hard"] if tier == "all" else [tier]
        for t in tiers:
            docs = PANDataset(root="data/raw", year=year, tier=t).load()
            all_docs.extend(docs)
            print(f"  Loaded PAN {year} {t}: {len(docs)} docs")
    print(f"  Total: {len(all_docs)} docs")
    train, val, test = split_documents(all_docs, seed=seed)
    return train, val, test


def load_eval_docs(year: str, tier: str, seed: int):
    docs = PANDataset(root="data/raw", year=year, tier=tier).load()
    _, _, test = split_documents(docs, seed=seed)
    return test


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


def build_pair_datasets(train_docs, val_docs, test_docs, extractor, use_style):
    def process(docs):
        sents_list = [d.sentences for d in docs]
        labels_list = [d.boundary_labels for d in docs]
        style_diffs = build_style_diffs(docs, extractor) if use_style and extractor else None
        return E2EDataset(sents_list, labels_list, style_diffs)

    print("Building pair-level datasets...")
    t0 = time.time()
    train_ds = process(train_docs)
    val_ds = process(val_docs)
    test_ds = process(test_docs)
    print(f"  Done in {time.time() - t0:.1f}s")
    print(f"  Train: {len(train_ds)} pairs, Val: {len(val_ds)}, Test: {len(test_ds)}")
    return train_ds, val_ds, test_ds


def build_doc_datasets(train_docs, val_docs, test_docs, extractor, use_style):
    def process(docs):
        sents_list = [d.sentences for d in docs]
        labels_list = [d.boundary_labels for d in docs]
        doc_ids = [d.doc_id for d in docs]
        style_diffs = build_style_diffs(docs, extractor) if use_style and extractor else None
        return DocLevelDataset(sents_list, labels_list, style_diffs, doc_ids=doc_ids)

    print("Building document-level datasets...")
    t0 = time.time()
    train_ds = process(train_docs)
    val_ds = process(val_docs)
    test_ds = process(test_docs)
    print(f"  Done in {time.time() - t0:.1f}s")
    print(f"  Train: {len(train_ds)} docs, Val: {len(val_ds)}, Test: {len(test_ds)}")
    return train_ds, val_ds, test_ds


def load_topic_clusters(path: str):
    cache = Path(path)
    if not cache.exists():
        print(f"ERROR: Topic cache not found at {cache}")
        print("Run: python scripts/precompute_topics.py --year 2024,2025 --n-topics 20")
        sys.exit(1)
    data = torch.load(cache, weights_only=False)
    print(f"Loaded pseudo-topic clusters: {data['n_topics']} topics, {len(data['boundary_topics'])} docs")
    return data["boundary_topics"], data["n_topics"]


def adv_lambda_schedule(epoch: int, total_epochs: int) -> float:
    p = epoch / max(total_epochs, 1)
    return float(2.0 / (1.0 + np.exp(-10 * p)) - 1.0)


# --------------------------------------------------------------------------- #
#  Pair-level training/eval
# --------------------------------------------------------------------------- #

def threshold_search_pairs(model, loader, device, use_style):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            style = batch.get("style_diff")
            if style is not None and use_style:
                style = style.to(device)
            else:
                style = None
            with autocast("cuda", dtype=torch.float16):
                logits = model(ids, mask, style_diff=style)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(batch["labels"].numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.20, 0.81, 0.02):
        preds = (all_probs >= t).astype(int)
        f1 = binary_metrics(all_labels, preds)["f1"]
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def evaluate_pairs(model, loader, device, threshold, use_style, docs):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            style = batch.get("style_diff")
            if style is not None and use_style:
                style = style.to(device)
            else:
                style = None
            with autocast("cuda", dtype=torch.float16):
                logits = model(ids, mask, style_diff=style)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(batch["labels"].numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels, dtype=int)
    all_preds = (all_probs >= threshold).astype(int)
    metrics = binary_metrics(all_labels, all_preds)

    wd_scores = []
    idx = 0
    for doc in docs:
        n = len(doc.sentences) - 1
        if n > 0:
            t_doc = all_labels[idx:idx + n]
            p_doc = all_preds[idx:idx + n]
            wd_scores.append(window_diff(t_doc, p_doc))
            idx += n
    metrics["window_diff"] = float(np.mean(wd_scores)) if wd_scores else 0.0
    metrics["threshold"] = threshold
    return metrics


# --------------------------------------------------------------------------- #
#  Document-level training/eval (BiLSTM)
# --------------------------------------------------------------------------- #

def encode_document(model, doc, tokenizer, device, use_style, max_length, sub_batch_size):
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
        if use_style and style_diffs is not None:
            style = torch.tensor(
                style_diffs[start:end], dtype=torch.float32
            ).to(device)

        with autocast("cuda", dtype=torch.float16):
            fused = model.encode_pairs(
                encoded["input_ids"], encoded["attention_mask"],
                style_diff=style,
            )
        all_fused.append(fused)

    return torch.cat(all_fused, dim=0)


def train_epoch_bilstm(model, dataset, tokenizer, optimizer, scaler, bce, device, args,
                        topic_labels=None, n_topics=None, epoch=1):
    model.train()
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)

    epoch_loss = 0.0
    epoch_bce_loss = 0.0
    epoch_adv_loss = 0.0
    n_docs = 0
    n_adv_docs = 0
    optimizer.zero_grad()

    cur_lambda = adv_lambda_schedule(epoch, args.epochs) if args.adversarial else None

    for doc_idx, idx in enumerate(indices):
        doc = dataset[idx]
        labels = torch.tensor(doc["labels"], dtype=torch.float32).to(device)

        fused = encode_document(
            model, doc, tokenizer, device,
            not args.no_style, args.max_length, args.sub_batch,
        )

        with autocast("cuda", dtype=torch.float16):
            if args.adversarial:
                logits, topic_logits = model.forward_document(
                    fused.unsqueeze(0), topic_lambda=cur_lambda,
                )
            else:
                logits = model.forward_document(fused.unsqueeze(0))
                topic_logits = None

            bce_loss = bce(logits, labels)
            loss = bce_loss

            adv_loss_val = 0.0
            if args.adversarial and topic_logits is not None:
                doc_id = doc["doc_id"]
                doc_topic_ids = topic_labels.get(doc_id) if topic_labels else None
                if doc_topic_ids is not None:
                    n_b = min(len(doc_topic_ids), topic_logits.size(0))
                    if n_b > 0:
                        topic_targets = torch.tensor(
                            doc_topic_ids[:n_b], dtype=torch.long, device=device
                        )
                        adv_loss = nn.functional.cross_entropy(
                            topic_logits[:n_b], topic_targets,
                        )
                        loss = bce_loss + args.adv_weight * adv_loss
                        adv_loss_val = adv_loss.item()
                        n_adv_docs += 1

            loss = loss / args.doc_accum

        scaler.scale(loss).backward()

        if (doc_idx + 1) % args.doc_accum == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        epoch_loss += loss.item() * args.doc_accum
        epoch_bce_loss += bce_loss.item()
        epoch_adv_loss += adv_loss_val
        n_docs += 1

    avg_loss = epoch_loss / max(n_docs, 1)
    avg_bce = epoch_bce_loss / max(n_docs, 1)
    avg_adv = epoch_adv_loss / max(n_adv_docs, 1)

    if args.adversarial:
        print(f"    [adv] lambda={cur_lambda:.3f} bce={avg_bce:.4f} adv_loss={avg_adv:.4f} "
              f"(matched {n_adv_docs}/{n_docs} docs)")

    return avg_loss


def threshold_search_bilstm(model, dataset, tokenizer, device, args):
    model.eval()
    all_probs, all_labels = [], []
    use_style = not args.no_style

    with torch.no_grad():
        for idx in range(len(dataset)):
            doc = dataset[idx]
            labels = doc["labels"]

            fused = encode_document(
                model, doc, tokenizer, device,
                use_style, args.max_length, args.sub_batch,
            )

            with autocast("cuda", dtype=torch.float16):
                logits = model.forward_document(fused.unsqueeze(0))

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    best_f1, best_t = 0, 0.5
    for t in np.arange(0.20, 0.81, 0.02):
        preds = (all_probs >= t).astype(int)
        f1 = binary_metrics(all_labels, preds)["f1"]
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def evaluate_bilstm(model, dataset, tokenizer, device, threshold, args):
    model.eval()
    all_probs, all_labels = [], []
    doc_boundaries = []
    use_style = not args.no_style

    with torch.no_grad():
        for idx in range(len(dataset)):
            doc = dataset[idx]
            labels = doc["labels"]
            n = len(labels)
            doc_boundaries.append(n)

            fused = encode_document(
                model, doc, tokenizer, device,
                use_style, args.max_length, args.sub_batch,
            )

            with autocast("cuda", dtype=torch.float16):
                logits = model.forward_document(fused.unsqueeze(0))

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels, dtype=int)
    all_preds = (all_probs >= threshold).astype(int)

    metrics = binary_metrics(all_labels, all_preds)

    wd_scores = []
    idx = 0
    for n in doc_boundaries:
        if n > 0:
            t_doc = all_labels[idx:idx + n]
            p_doc = all_preds[idx:idx + n]
            wd_scores.append(window_diff(t_doc, p_doc))
            idx += n
    metrics["window_diff"] = float(np.mean(wd_scores)) if wd_scores else 0.0
    metrics["threshold"] = threshold
    return metrics


# --------------------------------------------------------------------------- #
#  Curriculum helpers
# --------------------------------------------------------------------------- #

def parse_curriculum_phases(phases_str: str, total_epochs: int):
    """Parse phase epoch counts, scale to total_epochs if needed."""
    counts = [int(x) for x in phases_str.split(",")]
    assert len(counts) == 3, "--curriculum-phases must have exactly 3 values"
    if sum(counts) != total_epochs:
        # Scale proportionally to total_epochs
        ratios = [c / sum(counts) for c in counts]
        counts = [max(1, round(r * total_epochs)) for r in ratios]
        counts[-1] = total_epochs - sum(counts[:-1])
    return counts


def build_curriculum_datasets(year_str: str, seed: int, extractor, use_style: bool):
    """Load and build datasets for all three curriculum phases."""
    years = [y.strip() for y in year_str.split(",")]

    phase_docs = {"easy": [], "easy_medium": [], "all": []}
    tier_test_docs = {}

    for year in years:
        for t in ["easy", "medium", "hard"]:
            docs = PANDataset(root="data/raw", year=year, tier=t).load()
            print(f"  Loaded PAN {year} {t}: {len(docs)} docs")
            phase_docs["easy"].extend(docs if t == "easy" else [])
            phase_docs["easy_medium"].extend(docs if t in ["easy", "medium"] else [])
            phase_docs["all"].extend(docs)

    # Use all-tier split for val/test (consistent across phases)
    _, val_docs, test_docs = split_documents(phase_docs["all"], seed=seed)

    print(f"  Val: {len(val_docs)} docs | Test: {len(test_docs)} docs")

    def build_train_ds(docs):
        train_d, _, _ = split_documents(docs, seed=seed)
        sents_list = [d.sentences for d in train_d]
        labels_list = [d.boundary_labels for d in train_d]
        doc_ids = [d.doc_id for d in train_d]
        style_diffs = build_style_diffs(train_d, extractor) if use_style and extractor else None
        return DocLevelDataset(sents_list, labels_list, style_diffs, doc_ids=doc_ids), len(train_d)

    def build_eval_ds(docs):
        sents_list = [d.sentences for d in docs]
        labels_list = [d.boundary_labels for d in docs]
        doc_ids = [d.doc_id for d in docs]
        style_diffs = build_style_diffs(docs, extractor) if use_style and extractor else None
        return DocLevelDataset(sents_list, labels_list, style_diffs, doc_ids=doc_ids)

    print("\nBuilding curriculum datasets...")
    t0 = time.time()
    easy_train_ds, n_easy = build_train_ds(phase_docs["easy"])
    easy_med_train_ds, n_easy_med = build_train_ds(phase_docs["easy_medium"])
    all_train_ds, n_all = build_train_ds(phase_docs["all"])
    val_ds = build_eval_ds(val_docs)
    test_ds = build_eval_ds(test_docs)
    print(f"  Done in {time.time() - t0:.1f}s")
    print(f"  Phase 1 train (Easy): {n_easy} docs")
    print(f"  Phase 2 train (Easy+Medium): {n_easy_med} docs")
    print(f"  Phase 3 train (All): {n_all} docs")

    return (easy_train_ds, easy_med_train_ds, all_train_ds), val_ds, test_ds, val_docs, test_docs


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    args = parse_args()

    if not args.finetune:
        print("Only --finetune mode is supported.")
        return

    if args.adversarial and not args.bilstm:
        print("ERROR: --adversarial requires --bilstm")
        sys.exit(1)

    if args.curriculum and not args.bilstm:
        print("ERROR: --curriculum requires --bilstm")
        sys.exit(1)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_style = not args.no_style

    print("=" * 60)
    print("DeBERTa + Stylometric Fusion — E2E Fine-tuning")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Year(s): {args.year} | Tier: {'curriculum' if args.curriculum else args.tier}")
    print(f"Unfreeze layers: {args.unfreeze_layers} | Style: {use_style}")
    print(f"Attention fusion: {args.attention_fusion}")
    print(f"BiLSTM: {args.bilstm}")
    print(f"Curriculum: {args.curriculum}")
    print(f"Adversarial: {args.adversarial}")
    print()

    topic_labels, n_topics = (None, args.n_topics)
    if args.adversarial:
        topic_labels, n_topics = load_topic_clusters(args.topic_cache)

    extractor = StylometricExtractor() if use_style else None
    style_dim = extractor.feature_dim if extractor else 0

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(DeBERTaFusionE2E.DEBERTA_MODEL)

    model = DeBERTaFusionE2E(
        style_dim=style_dim,
        unfreeze_layers=args.unfreeze_layers,
        use_style=use_style,
        use_attention_fusion=args.attention_fusion,
        use_bilstm=args.bilstm,
        bilstm_hidden=args.bilstm_hidden,
        dropout=args.dropout,
        use_adversarial=args.adversarial,
        n_topics=n_topics,
    ).to(device)

    param_groups = [
        {"params": model.get_deberta_params(), "lr": args.lr_deberta},
        {"params": model.get_head_params(), "lr": args.lr_head},
    ]
    if args.adversarial:
        param_groups.append({"params": model.get_adversary_params(), "lr": args.lr_head})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    total_steps = (21420 * args.epochs) // args.doc_accum
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))
    scaler = GradScaler("cuda")
    pos_weight = torch.tensor([args.pos_weight]).to(device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_f1, best_state, best_threshold = 0, None, 0.5

    # ----------------------------------------------------------------------- #
    #  Curriculum training
    # ----------------------------------------------------------------------- #
    if args.curriculum:
        phase_epochs = parse_curriculum_phases(args.curriculum_phases, args.epochs)
        print(f"Curriculum phases: Easy={phase_epochs[0]}e, "
              f"Easy+Medium={phase_epochs[1]}e, All={phase_epochs[2]}e")

        print("\nLoading data...")
        (easy_ds, easy_med_ds, all_ds), val_ds, test_ds, val_docs, test_docs = \
            build_curriculum_datasets(args.year, args.seed, extractor, use_style)

        phase_names = ["Easy only", "Easy + Medium", "All tiers"]
        phase_datasets = [easy_ds, easy_med_ds, all_ds]
        global_epoch = 0
        patience_counter = 0

        for phase_idx, (phase_name, train_ds, n_epochs) in enumerate(
            zip(phase_names, phase_datasets, phase_epochs)
        ):
            print(f"\n{'='*60}")
            print(f"CURRICULUM PHASE {phase_idx+1}: {phase_name} ({n_epochs} epochs)")
            print(f"{'='*60}")

            for local_epoch in range(1, n_epochs + 1):
                global_epoch += 1
                avg_loss = train_epoch_bilstm(
                    model, train_ds, tokenizer, optimizer, scaler, bce, device, args,
                    topic_labels=topic_labels, n_topics=n_topics, epoch=global_epoch,
                )
                scheduler.step()
                val_thresh, val_f1 = threshold_search_bilstm(model, val_ds, tokenizer, device, args)
                val_metrics = evaluate_bilstm(model, val_ds, tokenizer, device, val_thresh, args)

                print(f"[Phase {phase_idx+1} Epoch {local_epoch}/{n_epochs}] "
                      f"loss={avg_loss:.4f} | val_F1={val_metrics['f1']:.4f} "
                      f"val_WD={val_metrics['window_diff']:.4f} thr={val_thresh:.2f}")

                if val_metrics["f1"] > best_val_f1:
                    best_val_f1 = val_metrics["f1"]
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    best_threshold = val_thresh
                    print(f"  -> New best val F1: {best_val_f1:.4f}")
                else:
                    patience_counter += 1
                    if patience_counter >= args.patience and phase_idx == 2:
                        print(f"  Early stopping in final phase (patience={args.patience})")
                        break

        variant = "deberta+style+attn+bilstm+curriculum"
        tier = "curriculum"

    # ----------------------------------------------------------------------- #
    #  Standard training (non-curriculum)
    # ----------------------------------------------------------------------- #
    else:
        print("Loading data...")
        train_docs, val_docs, test_docs = load_docs(args.year, args.tier, args.seed)
        stats = dataset_stats(train_docs + val_docs + test_docs)
        print(f"  Stats: {stats}\n")

        if args.bilstm:
            train_ds, val_ds, test_ds = build_doc_datasets(
                train_docs, val_docs, test_docs, extractor, use_style,
            )
        else:
            train_ds, val_ds, test_ds = build_pair_datasets(
                train_docs, val_docs, test_docs, extractor, use_style,
            )

        patience_counter = 0

        for epoch in range(1, args.epochs + 1):
            if args.bilstm:
                avg_loss = train_epoch_bilstm(
                    model, train_ds, tokenizer, optimizer, scaler, bce, device, args,
                    topic_labels=topic_labels, n_topics=n_topics, epoch=epoch,
                )
                scheduler.step()
                val_thresh, val_f1 = threshold_search_bilstm(model, val_ds, tokenizer, device, args)
                val_metrics = evaluate_bilstm(model, val_ds, tokenizer, device, val_thresh, args)
            else:
                collate = partial(e2e_collate_fn, tokenizer=tokenizer, max_length=args.max_length)
                train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                          collate_fn=collate, num_workers=0, pin_memory=True)
                val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                                        collate_fn=collate, num_workers=0, pin_memory=True)

                model.train()
                epoch_loss, n_batches = 0.0, 0
                optimizer.zero_grad()

                for step, batch in enumerate(train_loader):
                    ids = batch["input_ids"].to(device)
                    mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    style = batch.get("style_diff")
                    if style is not None and use_style:
                        style = style.to(device)
                    else:
                        style = None

                    with autocast("cuda", dtype=torch.float16):
                        logits = model(ids, mask, style_diff=style)
                        loss = bce(logits, labels) / args.accum_steps

                    scaler.scale(loss).backward()

                    if (step + 1) % args.accum_steps == 0:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        scheduler.step()

                    epoch_loss += loss.item() * args.accum_steps
                    n_batches += 1

                avg_loss = epoch_loss / max(n_batches, 1)
                val_thresh, val_f1 = threshold_search_pairs(model, val_loader, device, use_style)
                val_metrics = evaluate_pairs(model, val_loader, device, val_thresh, use_style, val_docs)

            print(f"Epoch {epoch}/{args.epochs} | loss={avg_loss:.4f} | "
                  f"val_F1={val_metrics['f1']:.4f} val_WD={val_metrics['window_diff']:.4f} "
                  f"thr={val_thresh:.2f}")

            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_threshold = val_thresh
                print(f"  -> New best val F1: {best_val_f1:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"  Early stopping (patience={args.patience})")
                    break

        if args.adversarial:
            variant = "deberta+style+attn+bilstm+adv" if args.attention_fusion else "deberta+style+bilstm+adv"
        elif args.attention_fusion and args.bilstm:
            variant = "deberta+style+attn+bilstm"
        elif args.bilstm:
            variant = "deberta+style+bilstm"
        elif args.attention_fusion:
            variant = "deberta+style+attn"
        elif use_style:
            variant = "deberta+style"
        else:
            variant = "deberta-only"

        tier = args.tier

    # ----------------------------------------------------------------------- #
    #  Test evaluation
    # ----------------------------------------------------------------------- #
    print("\n" + "=" * 60)
    print("TEST EVALUATION")
    print("=" * 60)

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    if args.save_checkpoint:
        os.makedirs(os.path.dirname(args.save_checkpoint) or ".", exist_ok=True)
        torch.save(best_state, args.save_checkpoint)
        print(f"Checkpoint saved to {args.save_checkpoint}")

    if args.bilstm:
        test_metrics = evaluate_bilstm(model, test_ds, tokenizer, device, best_threshold, args)
    else:
        collate = partial(e2e_collate_fn, tokenizer=tokenizer, max_length=args.max_length)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False,
                                 collate_fn=collate, num_workers=0, pin_memory=True)
        test_metrics = evaluate_pairs(model, test_loader, device, best_threshold, use_style, test_docs)

    print(f"\nVariant: {variant} | Year(s): {args.year} | Tier: {tier}")
    print(f"  Accuracy:    {test_metrics['accuracy']:.4f}")
    print(f"  Precision:   {test_metrics['precision']:.4f}")
    print(f"  Recall:      {test_metrics['recall']:.4f}")
    print(f"  F1:          {test_metrics['f1']:.4f}")
    print(f"  WindowDiff:  {test_metrics['window_diff']:.4f}")
    print(f"  Threshold:   {test_metrics['threshold']:.2f}")

    # --- Per-tier breakdown ---
    per_tier_results = {}
    print(f"\n--- Per-tier breakdown ---")
    eval_year = args.year.split(",")[-1]
    for t in ["easy", "medium", "hard"]:
        eval_docs = load_eval_docs(eval_year, t, args.seed)
        sents_list = [d.sentences for d in eval_docs]
        labels_list = [d.boundary_labels for d in eval_docs]
        eval_doc_ids = [d.doc_id for d in eval_docs]
        style_diffs = build_style_diffs(eval_docs, extractor) if use_style else None
        eval_ds = DocLevelDataset(sents_list, labels_list, style_diffs, doc_ids=eval_doc_ids)
        eval_metrics = evaluate_bilstm(model, eval_ds, tokenizer, device, best_threshold, args)
        per_tier_results[t] = eval_metrics
        print(f"  {t:6s} | F1={eval_metrics['f1']:.4f}  WD={eval_metrics['window_diff']:.4f}  (n={len(eval_docs)})")

    # --- Baseline comparison ---
    baselines = {
        "easy":   {"wqd": 0.926, "SCL-DeBERTa": 0.932},
        "medium": {"wqd": 0.753, "SCL-DeBERTa": 0.776},
        "hard":   {"wqd": 0.830, "SCL-DeBERTa": 0.829},
    }
    print(f"\nPAN 2025 published baselines (per tier):")
    for t, b in baselines.items():
        if t in per_tier_results:
            ours_f1 = per_tier_results[t]["f1"]
            for name, f1 in b.items():
                delta = ours_f1 - f1
                marker = "+" if delta >= 0 else ""
                print(f"  {t}/{name}: F1={f1:.3f}  (ours {marker}{delta:.3f})")

    # --- Save results ---
    os.makedirs("results", exist_ok=True)
    result = {
        "variant": variant,
        "year": args.year,
        "tier": tier,
        "curriculum": args.curriculum,
        "curriculum_phases": args.curriculum_phases if args.curriculum else None,
        "unfreeze_layers": args.unfreeze_layers,
        "attention_fusion": args.attention_fusion,
        "bilstm": args.bilstm,
        "adversarial": args.adversarial,
        "n_topics": args.n_topics if args.adversarial else None,
        "adv_weight": args.adv_weight if args.adversarial else None,
        "pos_weight": args.pos_weight,
        "test_metrics": test_metrics,
        "per_tier_results": per_tier_results,
        "best_val_f1": best_val_f1,
        "best_threshold": best_threshold,
        "epochs_trained": args.epochs,
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