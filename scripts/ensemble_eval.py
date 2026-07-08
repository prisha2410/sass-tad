#!/usr/bin/env python3
"""
Ensemble: average probabilities from curriculum_228_best.pt + hard_ft_best.pt
Evaluates on val set to see if ensemble beats either model alone.
"""
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

CHECKPOINTS = {
    "curriculum": ROOT / "checkpoints" / "curriculum_228_best.pt",
    "hard_ft":    ROOT / "checkpoints" / "hard_ft_best.pt",
}


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
def get_probs(model, cls_np, style_np):
    pair_reprs = []
    for j in range(cls_np.shape[0]):
        cls   = torch.tensor(cls_np[j]).unsqueeze(0).to(DEVICE)
        style = torch.tensor(style_np[j]).unsqueeze(0).to(DEVICE)
        fused = model.style_fusion(cls, style)
        pair_reprs.append(fused)
    pair_reprs_t = torch.cat(pair_reprs, dim=0).unsqueeze(0)
    logits = model.forward_document(pair_reprs_t).squeeze(0).reshape(-1)
    return torch.sigmoid(logits).cpu().numpy()


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


def main():
    models = {name: load_model(path) for name, path in CHECKPOINTS.items()
              if path.exists()}
    print(f"Loaded: {list(models.keys())}")

    for year in YEARS:
        for tier in TIERS:
            items = load_cached(year, tier, "val")
            if not items:
                continue

            all_probs = {name: [] for name in models}
            all_labels = []

            for cls_np, style_np, lbl_np in items:
                for name, model in models.items():
                    all_probs[name].append(get_probs(model, cls_np, style_np))
                all_labels.append(lbl_np.astype(int).flatten().tolist())

            # Individual F1s
            for name in models:
                preds = [(p >= 0.5).astype(int).flatten().tolist() for p in all_probs[name]]
                f1 = compute_f1(preds, all_labels)
                print(f"  [{year}_{tier}] {name}: F1={f1:.4f}")

            # Ensemble
            if len(models) > 1:
                names = list(models.keys())
                avg_probs = [(all_probs[names[0]][i] + all_probs[names[1]][i]) / 2
                             for i in range(len(items))]
                preds = [(p >= 0.5).astype(int).flatten().tolist() for p in avg_probs]
                f1 = compute_f1(preds, all_labels)
                print(f"  [{year}_{tier}] ENSEMBLE: F1={f1:.4f}")
            print()


if __name__ == "__main__":
    main()