#!/usr/bin/env python3
"""
Precompute DeBERTa CLS + stylometric diffs using fine-tuned DeBERTa
from curriculum_228_best.pt.

Usage:
    python scripts/precompute_finetuned_features.py           # seed 42 (default)
    python scripts/precompute_finetuned_features.py --seed 0
    python scripts/precompute_finetuned_features.py --seed 1
"""
import sys, argparse, numpy as np, torch
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.deberta_fusion import DeBERTaFusionE2E
from src.features.stylometric import StylometricExtractor
from src.utils.data_loader import PANDataset, split_documents

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = ROOT / "checkpoints" / "curriculum_228_best.pt"

DATASETS = [
    ("2024", "easy"),
    ("2024", "medium"),
    ("2024", "hard"),
    ("2025", "easy"),
    ("2025", "medium"),
    ("2025", "hard"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_model():
    ckpt  = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model = DeBERTaFusionE2E(
        style_dim=193, unfreeze_layers=2,
        use_style=True, use_attention_fusion=True,
        use_bilstm=True, use_adversarial=False,
    )
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


_tokenizer = None

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small")
    return _tokenizer


@torch.no_grad()
def encode_sentences(model, sentences, batch_size=8):
    tokenizer = get_tokenizer()
    model.deberta = model.deberta.to(DEVICE)
    pairs = list(zip(sentences[:-1], sentences[1:]))
    all_cls = []
    for i in tqdm(range(0, len(pairs), batch_size), desc="      pairs", leave=False, ncols=70):
        batch  = pairs[i : i + batch_size]
        a_list = [p[0] for p in batch]
        b_list = [p[1] for p in batch]
        enc = get_tokenizer()(a_list, b_list, padding=True, truncation=True,
                              max_length=256, return_tensors="pt")
        input_ids      = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)
        out = model.deberta(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        all_cls.append(cls.cpu().numpy())
    model.deberta = model.deberta.cpu()
    return np.concatenate(all_cls, axis=0) if all_cls else np.zeros((0, 768), dtype=np.float32)


def process_split(model, extractor, docs, split_name, out_path):
    if out_path.exists():
        print(f"  [Skip] {out_path.name} already exists")
        return
    all_cls, all_style, all_labels, lengths = [], [], [], []
    for doc in tqdm(docs, desc=f"    {split_name}", ncols=70):
        sents = doc.sentences
        if len(sents) < 2:
            continue
        cls_block = encode_sentences(model, sents)
        _, diffs  = extractor.extract_document(sents)
        n = len(sents) - 1
        assert cls_block.shape[0] == n
        assert diffs.shape[0]     == n
        labels = np.array(doc.boundary_labels[:n], dtype=np.float32)
        all_cls.append(cls_block)
        all_style.append(diffs)
        all_labels.append(labels)
        lengths.append(n)
    if not all_cls:
        print(f"  [Skip] No valid docs")
        return
    np.savez_compressed(
        out_path,
        cls_reprs   = np.concatenate(all_cls,   axis=0),
        style_diffs = np.concatenate(all_style, axis=0),
        labels      = np.concatenate(all_labels, axis=0),
        doc_lengths = np.array(lengths, dtype=np.int32),
    )
    print(f"  {split_name}: {len(lengths)} docs → {out_path.name} ({out_path.stat().st_size/1e6:.1f} MB)")


def main():
    args    = parse_args()
    out_dir = ROOT / "data" / "processed" / f"features_finetuned_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Seed: {args.seed}  →  {out_dir}")

    model     = load_model()
    extractor = StylometricExtractor()

    for year, tier in DATASETS:
        print(f"\n{'='*50}\nPAN {year} — {tier}\n{'='*50}")
        try:
            docs = PANDataset(ROOT / "data" / "raw", year=year, tier=tier).load()
        except FileNotFoundError as e:
            print(f"[Skip] {e}")
            continue
        train, val, test = split_documents(docs, seed=args.seed)
        for split_name, split_docs in [("train", train), ("val", val), ("test", test)]:
            process_split(model, extractor, split_docs, split_name,
                          out_dir / f"{year}_{tier}_{split_name}.npz")

    print("\nDone.")


if __name__ == "__main__":
    main()