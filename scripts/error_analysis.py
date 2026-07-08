#!/usr/bin/env python3
"""
Error analysis on Hard tier boundaries.
Saves results to results/error_analysis/
"""
import sys, json, numpy as np, torch
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.deberta_fusion import DeBERTaFusionE2E

DEVICE   = torch.device("cpu")
FEAT_DIR = ROOT / "data" / "processed" / "features_finetuned"
OUT_DIR  = ROOT / "results" / "error_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CKPT      = ROOT / "checkpoints" / "hard_oversample" / "hard_oversample_best.pt"
THRESHOLD = 0.40
YEARS     = ["2024", "2025"]


def load_model():
    model = DeBERTaFusionE2E(
        style_dim=193, unfreeze_layers=2,
        use_style=True, use_attention_fusion=True,
        use_bilstm=True, use_adversarial=False,
    )
    state = torch.load(CKPT, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    model.deberta = model.deberta.cpu()
    return model.eval()


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
        cls_t   = torch.tensor(cls_np[j]).unsqueeze(0)
        style_t = torch.tensor(style_np[j]).unsqueeze(0)
        fused   = model.style_fusion(cls_t, style_t)
        pair_reprs.append(fused)
    pair_reprs_t = torch.cat(pair_reprs, dim=0).unsqueeze(0)
    return model.forward_document(pair_reprs_t).squeeze(0).reshape(-1)


def main():
    print("Loading model...")
    model = load_model()

    stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0,
                                  "doc_lengths": [], "n_changes_per_doc": [],
                                  "fp_confidences": [], "fn_confidences": [],
                                  "boundary_positions": {"fp": [], "fn": []}})

    for year in YEARS:
        key = f"{year}_hard"
        items = load_cached(year, "hard", "test")
        if not items:
            continue

        print(f"Analyzing {key} ({len(items)} docs)...")
        for cls_np, style_np, lbl_np in tqdm(items, ncols=70):
            logits = forward_doc(model, cls_np, style_np)
            probs  = torch.sigmoid(logits).numpy()
            preds  = (probs >= THRESHOLD).astype(int)
            labels = lbl_np.astype(int).flatten()
            n      = len(labels)

            stats[key]["doc_lengths"].append(n)
            stats[key]["n_changes_per_doc"].append(int(labels.sum()))

            for i, (p, l, prob) in enumerate(zip(preds, labels, probs)):
                rel_pos = i / max(n - 1, 1)  # relative position in doc
                if p == 1 and l == 1:
                    stats[key]["tp"] += 1
                elif p == 1 and l == 0:
                    stats[key]["fp"] += 1
                    stats[key]["fp_confidences"].append(float(prob))
                    stats[key]["boundary_positions"]["fp"].append(rel_pos)
                elif p == 0 and l == 1:
                    stats[key]["fn"] += 1
                    stats[key]["fn_confidences"].append(float(prob))
                    stats[key]["boundary_positions"]["fn"].append(rel_pos)
                else:
                    stats[key]["tn"] += 1

    print("\n" + "="*60)
    print("ERROR ANALYSIS — Hard Tier")
    print("="*60)

    for key in sorted(stats):
        s = stats[key]
        tp, fp, fn, tn = s["tp"], s["fp"], s["fn"], s["tn"]
        total = tp + fp + fn + tn
        prec  = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec   = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1    = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0

        doc_lengths     = np.array(s["doc_lengths"])
        n_changes       = np.array(s["n_changes_per_doc"])
        fp_conf         = np.array(s["fp_confidences"])
        fn_conf         = np.array(s["fn_confidences"])
        fp_pos          = np.array(s["boundary_positions"]["fp"])
        fn_pos          = np.array(s["boundary_positions"]["fn"])

        no_change_docs  = (n_changes == 0).sum()
        multi_change    = (n_changes > 1).sum()

        print(f"\n{key}")
        print(f"  F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}")
        print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
        print(f"  Total boundaries: {total}")
        print(f"  Docs: {len(doc_lengths)} | avg length: {doc_lengths.mean():.1f} boundaries")
        print(f"  No-change docs: {no_change_docs} ({100*no_change_docs/len(doc_lengths):.1f}%)")
        print(f"  Multi-change docs: {multi_change} ({100*multi_change/len(doc_lengths):.1f}%)")
        if len(fp_conf):
            print(f"  FP avg confidence: {fp_conf.mean():.3f} (range {fp_conf.min():.3f}-{fp_conf.max():.3f})")
            print(f"  FP position in doc: early={( fp_pos < 0.33).mean():.2f} mid={(( fp_pos>=0.33)&(fp_pos<0.67)).mean():.2f} late={(fp_pos>=0.67).mean():.2f}")
        if len(fn_conf):
            print(f"  FN avg confidence: {fn_conf.mean():.3f} (range {fn_conf.min():.3f}-{fn_conf.max():.3f})")
            print(f"  FN position in doc: early={(fn_pos < 0.33).mean():.2f} mid={((fn_pos>=0.33)&(fn_pos<0.67)).mean():.2f} late={(fn_pos>=0.67).mean():.2f}")

    # Save
    out = {}
    for key, s in stats.items():
        out[key] = {k: v for k, v in s.items()
                    if not isinstance(v, list) or len(v) < 10000}
        out[key]["doc_lengths_mean"] = float(np.mean(s["doc_lengths"])) if s["doc_lengths"] else 0
        out[key]["fp_conf_mean"]     = float(np.mean(s["fp_confidences"])) if s["fp_confidences"] else 0
        out[key]["fn_conf_mean"]     = float(np.mean(s["fn_confidences"])) if s["fn_confidences"] else 0

    (OUT_DIR / "hard_error_analysis.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nSaved to {OUT_DIR / 'hard_error_analysis.json'}")


if __name__ == "__main__":
    main()