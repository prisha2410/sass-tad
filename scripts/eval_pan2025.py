#!/usr/bin/env python3
"""
Evaluates a saved DeSAF checkpoint on the real PAN 2025 test set.
Reads raw .txt problem files + truth-*.json ground truth labels.

Usage:
    python scripts/eval_pan2025_test.py \
        --checkpoint checkpoints/fulldata/fulldata_final.pt \
        --test-dir data/raw/pan2025_full
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
from transformers import AutoTokenizer, AutoModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TIERS  = ["easy", "medium", "hard"]


def parse_problem(txt_path):
    """Read paragraphs separated by blank lines."""
    text = txt_path.read_text(encoding="utf-8", errors="replace")
    paragraphs = [p.strip() for p in text.strip().split("\n\n") if p.strip()]
    return paragraphs


def parse_truth(truth_path):
    """Read ground truth labels from truth-problem-*.json."""
    data = json.loads(truth_path.read_text(encoding="utf-8"))
    # PAN format: {"changes": [0, 1, 0, ...]}
    return data.get("changes", [])


def get_cls(deberta, tokenizer, text, max_length=512):
    enc = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=max_length, padding=False).to(DEVICE)
    with torch.no_grad():
        out = deberta(**enc)
    return out.last_hidden_state[:, 0, :].squeeze(0).cpu().float()


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


def evaluate_tier(model, deberta, tokenizer, extractor, tier_dir, threshold, tier_name):
    test_dir = tier_dir / "test"
    if not test_dir.exists():
        print(f"  [{tier_name}] test dir not found: {test_dir}")
        return None

    # Match problem files to truth files by problem ID
    truth_files = {p.stem.replace("truth-problem-", ""): p
                   for p in test_dir.glob("truth-problem-*.json")}
    prob_files  = {p.stem.replace("problem-", ""): p
                   for p in test_dir.glob("problem-*.txt")}

    common_ids = sorted(set(truth_files) & set(prob_files))
    print(f"  [{tier_name}] {len(common_ids)} test problems found")

    all_preds, all_labels = [], []
    model.eval()

    for pid in tqdm(common_ids, desc=f"  {tier_name}", ncols=70):
        paragraphs = parse_problem(prob_files[pid])
        labels     = parse_truth(truth_files[pid])

        if len(paragraphs) < 2:
            continue

        n_pairs = len(paragraphs) - 1
        if len(labels) != n_pairs:
            continue

        # Get DeBERTa CLS for each paragraph pair
        cls_list   = []
        style_list = []

        para_vecs = [get_cls(deberta, tokenizer, p) for p in paragraphs]
        style_vecs = [torch.tensor(extractor.extract_vector(p), dtype=torch.float32)
                      for p in paragraphs]

        for j in range(n_pairs):
            # Pair representation: concat of consecutive paragraph CLS
            pair_text = paragraphs[j] + " [SEP] " + paragraphs[j+1]
            cls_pair  = get_cls(deberta, tokenizer, pair_text)
            style_diff = torch.abs(style_vecs[j+1] - style_vecs[j])
            cls_list.append(cls_pair)
            style_list.append(style_diff)

        cls_block   = torch.stack(cls_list)
        style_block = torch.stack(style_list)

        with torch.no_grad():
            logits = forward_doc(model, cls_block, style_block)
            probs  = torch.sigmoid(logits).cpu().numpy()

        preds = (probs >= threshold).astype(int).flatten().tolist()
        all_preds.append(preds)
        all_labels.append(labels)

    f1 = compute_f1(all_preds, all_labels)
    print(f"  [{tier_name}] Test F1 = {f1:.4f}")
    return f1


def main(args):
    test_root = Path(args.test_dir)
    print(f"Loading checkpoint: {args.checkpoint}")

    # Load model
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    model = DeBERTaFusionE2E(
        style_dim=193, unfreeze_layers=2,
        use_style=True, use_attention_fusion=True,
        use_bilstm=True, use_adversarial=False,
    )
    model.load_state_dict(state_dict, strict=False)
    model = model.to(DEVICE)
    model.eval()
    print("Model loaded.")

    # Load DeBERTa for feature extraction
    print("Loading DeBERTa tokenizer + encoder...")
    model_name = "microsoft/deberta-v3-base"
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    deberta    = AutoModel.from_pretrained(model_name).to(DEVICE)
    deberta.eval()

    # Stylometric extractor
    extractor = StylometricExtractor()

    print(f"\nEvaluating on PAN 2025 test set (threshold={args.threshold})...")
    results = {}
    for tier in TIERS:
        tier_dir = test_root / tier
        f1 = evaluate_tier(model, deberta, tokenizer, extractor,
                            tier_dir, args.threshold, f"2025_{tier}")
        if f1 is not None:
            results[f"2025_{tier}"] = f1

    print("\n" + "="*50)
    print("PAN 2025 TEST SET RESULTS")
    print("="*50)
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
    if results:
        macro = np.mean(list(results.values()))
        print(f"  Macro avg: {macro:.4f}")

    # Save results
    out_path = Path(args.checkpoint).parent / "test_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved -> {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/fulldata/fulldata_final.pt")
    p.add_argument("--test-dir",   default="data/raw/pan2025_full")
    p.add_argument("--threshold",  type=float, default=0.40)
    args = p.parse_args()
    main(args)