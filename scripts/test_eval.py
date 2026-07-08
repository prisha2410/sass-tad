#!/usr/bin/env python3
"""Final test set evaluation with optimal threshold."""
import sys, numpy as np, torch
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.deberta_fusion import DeBERTaFusionE2E

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEAT_DIR = ROOT / "data" / "processed" / "features_finetuned"
YEARS    = ["2024", "2025"]
TIERS    = ["easy", "medium", "hard"]


def load_model(ckpt_path):
    model = DeBERTaFusionE2E(
        style_dim=193, unfreeze_layers=2,
        use_style=True, use_attention_fusion=True,
        use_bilstm=True, use_adversarial=False,
    )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    model.deberta = model.deberta.cpu()
    return model.eval().to(DEVICE)


def load_cached(year, tier, split):
    path = FEAT_DIR / f"{year}_{tier}_{split}.npz"
    if not path.exists():
        return []
    data = np.load(path)
    cls, style, labels = data["cls_reprs"], data["style_diffs"], data["labels"]
    lengths = data["doc_lengths"]
    items, idx = [], 0
    for n in lengths.tolist():
        if n > 0:
            items.append((cls[idx:idx+n].astype(np.float32),
                          style[idx:idx+n].astype(np.float32),
                          labels[idx:idx+n].astype(np.int8)))
        idx += n
    return items


@torch.no_grad()
def forward_doc(model, cls_np, style_np):
    pair_reprs = []
    for j in range(cls_np.shape[0]):
        cls   = torch.tensor(cls_np[j]).unsqueeze(0).to(DEVICE)
        style = torch.tensor(style_np[j]).unsqueeze(0).to(DEVICE)
        fused = model.style_fusion(cls, style)
        pair_reprs.append(fused)
    pair_reprs_t = torch.cat(pair_reprs, dim=0).unsqueeze(0)
    return model.forward_document(pair_reprs_t).squeeze(0).reshape(-1)


def compute_f1(preds_list, labels_list):
    tp = fp = fn = 0
    for preds, lbls in zip(preds_list, labels_list):
        for p, l in zip(preds, lbls):
            if p == 1 and l == 1: tp += 1
            elif p == 1 and l == 0: fp += 1
            elif p == 0 and l == 1: fn += 1
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def evaluate_split(model, year, tier, split, threshold):
    items = load_cached(year, tier, split)
    if not items:
        return None
    all_preds, all_labels = [], []
    for cls_np, style_np, lbl_np in items:
        logits = forward_doc(model, cls_np, style_np)
        probs  = torch.sigmoid(logits).cpu().numpy()
        all_preds.append((probs >= threshold).astype(int).flatten().tolist())
        all_labels.append(lbl_np.astype(int).flatten().tolist())
    return compute_f1(all_preds, all_labels)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/hard_oversample/hard_oversample_best.pt")
    p.add_argument("--threshold",  type=float, default=0.40)
    args = p.parse_args()

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Threshold:  {args.threshold}")
    model = load_model(args.checkpoint)

    print(f"\n{'Dataset':<20} {'Val F1':>8}  {'Test F1':>8}")
    print("-" * 40)

    easy_f1s = {"val": [], "test": []}
    med_f1s  = {"val": [], "test": []}
    hard_f1s = {"val": [], "test": []}

    for year in YEARS:
        for tier in TIERS:
            val_f1  = evaluate_split(model, year, tier, "val",  args.threshold)
            test_f1 = evaluate_split(model, year, tier, "test", args.threshold)
            if val_f1 is None:
                continue
            label = f"{year}_{tier}"
            print(f"{label:<20} {val_f1:>8.4f}  {test_f1:>8.4f}")
            bucket = easy_f1s if tier == "easy" else (med_f1s if tier == "medium" else hard_f1s)
            bucket["val"].append(val_f1)
            bucket["test"].append(test_f1)

    print("-" * 40)
    for name, bucket in [("Easy", easy_f1s), ("Medium", med_f1s), ("Hard", hard_f1s)]:
        vf = np.mean(bucket["val"])  if bucket["val"]  else 0
        tf = np.mean(bucket["test"]) if bucket["test"] else 0
        print(f"{'Avg '+name:<20} {vf:>8.4f}  {tf:>8.4f}")

    all_val  = easy_f1s["val"]  + med_f1s["val"]  + hard_f1s["val"]
    all_test = easy_f1s["test"] + med_f1s["test"] + hard_f1s["test"]
    print(f"{'Macro avg':<20} {np.mean(all_val):>8.4f}  {np.mean(all_test):>8.4f}")


if __name__ == "__main__":
    main()