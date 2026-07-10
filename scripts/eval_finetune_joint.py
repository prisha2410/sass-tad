#!/usr/bin/env python3
"""
Evaluates the joint fine-tuned model (finetune_joint_final.pt) on the
real official PAN 2025 test set. Computes CLS live from raw text since
this model does not use precomputed frozen features.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.deberta_fusion import DeBERTaFusionE2E
from src.features.stylometric import StylometricExtractor
from transformers import AutoTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TIERS  = ["easy", "medium", "hard"]
TEST_ROOT = ROOT / "data" / "raw" / "pan2025_full"

_tokenizer = None
def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small")
    return _tokenizer

def parse_problem(txt_path):
    text = txt_path.read_text(encoding="utf-8", errors="replace")
    return [p.strip() for p in text.strip().split("\n") if p.strip()]

def parse_truth(truth_path):
    data = json.loads(truth_path.read_text(encoding="utf-8"))
    return data.get("changes", [])

def load_test_docs(tier):
    test_dir = TEST_ROOT / tier / "test"
    truth_files = {p.stem.replace("truth-problem-", ""): p
                   for p in test_dir.glob("truth-problem-*.json")}
    prob_files  = {p.stem.replace("problem-", ""): p
                   for p in test_dir.glob("problem-*.txt")}
    common_ids = sorted(set(truth_files) & set(prob_files))
    docs = []
    for pid in common_ids:
        paragraphs = parse_problem(prob_files[pid])
        labels     = parse_truth(truth_files[pid])
        if len(paragraphs) < 2 or len(labels) != len(paragraphs) - 1:
            continue
        docs.append((paragraphs, labels))
    return docs

@torch.no_grad()
def forward_doc_joint_eval(model, extractor, paragraphs, batch_size=4):
    tokenizer = get_tokenizer()
    pairs = list(zip(paragraphs[:-1], paragraphs[1:]))
    _, style_diffs = extractor.extract_document(paragraphs)
    n = len(pairs)
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
        cls = out.last_hidden_state[:, 0, :]
        style_batch = torch.tensor(style_diffs[i:i+batch_size], dtype=torch.float32).to(DEVICE)
        fused = model.style_fusion(cls, style_batch)
        fused_list.append(fused)
    all_fused = torch.cat(fused_list, dim=0).unsqueeze(0)
    logits = model.forward_document(all_fused).squeeze(0).reshape(-1)
    return logits

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

def main(args):
    print(f"Device: {DEVICE}")
    print(f"Loading checkpoint: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = DeBERTaFusionE2E(
        style_dim=193, unfreeze_layers=2,
        use_style=True, use_attention_fusion=True,
        use_bilstm=True, use_adversarial=False,
    )
    model.load_state_dict(state_dict, strict=False)
    model = model.to(DEVICE)
    model.eval()
    print("Model loaded.\n")
    extractor = StylometricExtractor()
    results = {}
    pred_dump = {}
    for tier in TIERS:
        docs = load_test_docs(tier)
        print(f"[{tier}] {len(docs)} test documents")
        all_preds, all_labels, doc_lengths = [], [], []
        for paragraphs, labels in tqdm(docs, desc=f"  {tier}", ncols=70):
            logits = forward_doc_joint_eval(model, extractor, paragraphs, batch_size=args.pair_batch_size)
            probs  = torch.sigmoid(logits).cpu().numpy()
            preds  = (probs >= args.threshold).astype(int).flatten().tolist()
            all_preds.append(preds)
            all_labels.append(labels)
            doc_lengths.append(len(preds))
        key = f"2025_{tier}"
        f1 = compute_f1(all_preds, all_labels)
        results[key] = f1
        print(f"  [{key}] F1 = {f1:.4f}")
        pred_dump[f"{key}__preds"]   = np.array([p for doc in all_preds for p in doc], dtype=np.int8)
        pred_dump[f"{key}__labels"]  = np.array([l for doc in all_labels for l in doc], dtype=np.int8)
        pred_dump[f"{key}__lengths"] = np.array(doc_lengths, dtype=np.int32)
    print("\n" + "="*50)
    print("JOINT FINE-TUNED MODEL - TEST SET RESULTS")
    print("="*50)
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
    if results:
        print(f"  Macro avg: {np.mean(list(results.values())):.4f}")
    out_dir = Path(args.checkpoint).parent
    with open(out_dir / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    np.savez_compressed(out_dir / "test_predictions.npz", **pred_dump)
    print(f"\nSaved -> {out_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/finetune_joint/finetune_joint_final.pt")
    p.add_argument("--threshold", type=float, default=0.40)
    p.add_argument("--pair-batch-size", type=int, default=4)
    args = p.parse_args()
    main(args)
