#!/usr/bin/env python3
"""
Precomputes DeBERTa CLS + stylometric diffs for the REAL official PAN 2025
test set (data/raw/pan2025_full/{tier}/test/), using the same fine-tuned
DeBERTa encoder (curriculum_228_best.pt) as precompute_finetuned_features.py.

Output goes to a separate dir so it never collides with the internal-split
"test" npz files. Filenames match the {year}_{tier}_test.npz convention so
eval_test_npz.py works unmodified.

Usage:
    python scripts/precompute_official_test.py
"""
import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.deberta_fusion import DeBERTaFusionE2E
from src.features.stylometric import StylometricExtractor

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = ROOT / "checkpoints" / "curriculum_228_best.pt"
TEST_ROOT  = ROOT / "data" / "raw" / "pan2025_full"
OUT_DIR    = ROOT / "data" / "processed" / "features_finetuned_official_test"
TIERS      = ["easy", "medium", "hard"]
YEAR_LABEL = "2025"  # matches naming convention for eval_test_npz.py


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
    for i in range(0, len(pairs), batch_size):
        batch  = pairs[i : i + batch_size]
        a_list = [p[0] for p in batch]
        b_list = [p[1] for p in batch]
        enc = tokenizer(a_list, b_list, padding=True, truncation=True,
                        max_length=256, return_tensors="pt")
        input_ids      = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)
        out = model.deberta(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        all_cls.append(cls.cpu().numpy())
    model.deberta = model.deberta.cpu()
    return np.concatenate(all_cls, axis=0) if all_cls else np.zeros((0, 768), dtype=np.float32)


def parse_problem(txt_path):
    text = txt_path.read_text(encoding="utf-8", errors="replace")
    return [p.strip() for p in text.strip().split("\n") if p.strip()]


def parse_truth(truth_path):
    import json
    data = json.loads(truth_path.read_text(encoding="utf-8"))
    return data.get("changes", [])


def load_official_test_docs(tier):
    test_dir = TEST_ROOT / tier / "test"
    if not test_dir.exists():
        return []
    truth_files = {p.stem.replace("truth-problem-", ""): p
                   for p in test_dir.glob("truth-problem-*.json")}
    prob_files  = {p.stem.replace("problem-", ""): p
                   for p in test_dir.glob("problem-*.txt")}
    common_ids = sorted(set(truth_files) & set(prob_files))

    docs = []
    skipped = 0
    for pid in common_ids:
        paragraphs = parse_problem(prob_files[pid])
        labels     = parse_truth(truth_files[pid])
        if len(paragraphs) < 2 or len(labels) != len(paragraphs) - 1:
            skipped += 1
            continue
        docs.append((paragraphs, labels))
    if skipped:
        print(f"  [{tier}] skipped {skipped} docs (length mismatch)")
    return docs


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}")
    print(f"Test root: {TEST_ROOT}")
    print(f"Output: {OUT_DIR}")

    model     = load_model()
    extractor = StylometricExtractor()

    for tier in TIERS:
        out_path = OUT_DIR / f"{YEAR_LABEL}_{tier}_test.npz"
        if out_path.exists():
            print(f"[Skip] {out_path.name} already exists")
            continue

        print(f"\n{'='*50}\nOfficial PAN 2025 test — {tier}\n{'='*50}")
        docs = load_official_test_docs(tier)
        print(f"  {len(docs)} valid documents")

        all_cls, all_style, all_labels, lengths = [], [], [], []
        for paragraphs, labels in tqdm(docs, desc=f"  {tier}", ncols=70):
            cls_block = encode_sentences(model, paragraphs)
            _, diffs  = extractor.extract_document(paragraphs)
            n = len(paragraphs) - 1
            assert cls_block.shape[0] == n
            assert diffs.shape[0]     == n
            lbl_arr = np.array(labels[:n], dtype=np.float32)
            all_cls.append(cls_block)
            all_style.append(diffs)
            all_labels.append(lbl_arr)
            lengths.append(n)

        if not all_cls:
            print(f"  [Skip] No valid docs for {tier}")
            continue

        np.savez_compressed(
            out_path,
            cls_reprs   = np.concatenate(all_cls,   axis=0),
            style_diffs = np.concatenate(all_style, axis=0),
            labels      = np.concatenate(all_labels, axis=0),
            doc_lengths = np.array(lengths, dtype=np.int32),
        )
        print(f"  Saved {len(lengths)} docs -> {out_path.name} ({out_path.stat().st_size/1e6:.1f} MB)")
        torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()