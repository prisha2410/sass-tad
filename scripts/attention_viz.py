#!/usr/bin/env python3
"""
Attention weight visualization — which stylometric features matter most.
Runs on CPU using existing checkpoint + finetuned features.
Saves plots to results/attention/
"""
import sys, numpy as np, torch, json
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.deberta_fusion import DeBERTaFusionE2E
from src.features.stylometric import StylometricExtractor

DEVICE   = torch.device("cpu")
FEAT_DIR = ROOT / "data" / "processed" / "features_finetuned"
OUT_DIR  = ROOT / "results" / "attention"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CKPT = ROOT / "checkpoints" / "hard_oversample" / "hard_oversample_best.pt"
YEARS = ["2024", "2025"]
TIERS = ["easy", "medium", "hard"]


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
def get_attention_gates(model, cls_np, style_np):
    """Extract gate values from StyleAttentionFusion for each pair."""
    gates = []
    for j in range(cls_np.shape[0]):
        cls_t   = torch.tensor(cls_np[j]).unsqueeze(0)
        style_t = torch.tensor(style_np[j]).unsqueeze(0)
        # StyleAttentionFusion: gate = sigmoid(linear(cls))
        gate = torch.sigmoid(model.style_fusion.gate(cls_t))  # (1, style_dim)
        gates.append(gate.squeeze(0).numpy())
    return np.stack(gates)  # (n, 193)


def main():
    print("Loading model...")
    model     = load_model()
    extractor = StylometricExtractor()
    feat_names = extractor.feature_names  # 193 names

    # Collect gates split by: change boundary vs no-change boundary
    gates_change    = []  # gates at actual style change points
    gates_nochange  = []  # gates at non-change points

    for year in YEARS:
        for tier in TIERS:
            items = load_cached(year, tier, "test")
            if not items:
                continue
            print(f"  Processing {year}_{tier}...")
            for cls_np, style_np, lbl_np in tqdm(items, ncols=70, leave=False):
                gates = get_attention_gates(model, cls_np, style_np)  # (n, 193)
                labels = lbl_np.astype(int).flatten()
                for i, lbl in enumerate(labels):
                    if lbl == 1:
                        gates_change.append(gates[i])
                    else:
                        gates_nochange.append(gates[i])

    gates_change   = np.array(gates_change)    # (N_change, 193)
    gates_nochange = np.array(gates_nochange)  # (N_nochange, 193)

    print(f"\nChange boundaries:    {len(gates_change)}")
    print(f"No-change boundaries: {len(gates_nochange)}")

    # Mean gate per feature
    mean_change   = gates_change.mean(axis=0)
    mean_nochange = gates_nochange.mean(axis=0)
    delta         = mean_change - mean_nochange  # positive = more active at change

    # Top 20 most discriminative features
    top20_idx  = np.argsort(np.abs(delta))[::-1][:20]
    top20_names = [feat_names[i] for i in top20_idx]
    top20_delta = delta[top20_idx]
    top20_change   = mean_change[top20_idx]
    top20_nochange = mean_nochange[top20_idx]

    print("\nTop 20 most discriminative stylometric features (by gate delta):")
    print(f"{'Feature':<35} {'Change':>8}  {'No-change':>10}  {'Delta':>8}")
    print("-" * 65)
    for name, ch, nc, d in zip(top20_names, top20_change, top20_nochange, top20_delta):
        print(f"{name:<35} {ch:>8.4f}  {nc:>10.4f}  {d:>+8.4f}")

    # Save raw data
    results = {
        "feature_names": feat_names,
        "mean_gate_change":    mean_change.tolist(),
        "mean_gate_nochange":  mean_nochange.tolist(),
        "delta":               delta.tolist(),
        "top20_features":      top20_names,
        "top20_delta":         top20_delta.tolist(),
        "n_change":            int(len(gates_change)),
        "n_nochange":          int(len(gates_nochange)),
    }
    out_json = OUT_DIR / "attention_gates.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out_json}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Left: top 20 delta bar chart
        ax = axes[0]
        colors = ["#d62728" if d > 0 else "#1f77b4" for d in top20_delta]
        bars = ax.barh(range(20), top20_delta[::-1], color=colors[::-1])
        ax.set_yticks(range(20))
        ax.set_yticklabels(top20_names[::-1], fontsize=9)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Gate delta (change − no-change)")
        ax.set_title("Top 20 Stylometric Features\nby Attention Gate Discriminability")
        ax.invert_yaxis()

        # Right: change vs no-change gate comparison for top 10
        ax2 = axes[1]
        x = np.arange(10)
        w = 0.35
        ax2.bar(x - w/2, top20_change[:10],   w, label="Change boundary",    color="#d62728", alpha=0.8)
        ax2.bar(x + w/2, top20_nochange[:10], w, label="No-change boundary", color="#1f77b4", alpha=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(top20_names[:10], rotation=45, ha="right", fontsize=8)
        ax2.set_ylabel("Mean attention gate value")
        ax2.set_title("Top 10 Features:\nGate Values at Change vs No-Change")
        ax2.legend()

        plt.tight_layout()
        plot_path = OUT_DIR / "attention_gates.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {plot_path}")
    except ImportError:
        print("matplotlib not installed — skipping plot. Install with: pip install matplotlib")


if __name__ == "__main__":
    main()