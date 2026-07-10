#!/usr/bin/env python3
"""
Attention weight visualization — per-feature AND per-tier analysis.
Tests two things:
  1. Which stylometric features matter most at change vs no-change (original analysis)
  2. Whether overall gate reliance on stylometric shifts by difficulty tier (new)
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

CKPT = ROOT / "checkpoints" / "fulldata" / "fulldata_final.pt"
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
    gates = []
    for j in range(cls_np.shape[0]):
        cls_t   = torch.tensor(cls_np[j]).unsqueeze(0)
        style_t = torch.tensor(style_np[j]).unsqueeze(0)
        gate = torch.sigmoid(model.style_fusion.gate(cls_t))
        gates.append(gate.squeeze(0).numpy())
    return np.stack(gates)


def main():
    print("Loading model...")
    model     = load_model()
    extractor = StylometricExtractor()
    feat_names = extractor.feature_names

    gates_change, gates_nochange = [], []
    gates_by_tier = {t: [] for t in TIERS}  # NEW: per-tier gate collection

    for year in YEARS:
        for tier in TIERS:
            items = load_cached(year, tier, "test")
            if not items:
                continue
            print(f"  Processing {year}_{tier}...")
            for cls_np, style_np, lbl_np in tqdm(items, ncols=70, leave=False):
                gates = get_attention_gates(model, cls_np, style_np)
                labels = lbl_np.astype(int).flatten()
                gates_by_tier[tier].append(gates)  # NEW
                for i, lbl in enumerate(labels):
                    if lbl == 1:
                        gates_change.append(gates[i])
                    else:
                        gates_nochange.append(gates[i])

    gates_change   = np.array(gates_change)
    gates_nochange = np.array(gates_nochange)

    print(f"\nChange boundaries:    {len(gates_change)}")
    print(f"No-change boundaries: {len(gates_nochange)}")

    mean_change   = gates_change.mean(axis=0)
    mean_nochange = gates_nochange.mean(axis=0)
    delta         = mean_change - mean_nochange

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

    # --- NEW: per-tier overall gate magnitude ---
    print("\n" + "="*60)
    print("PER-TIER GATE RELIANCE (mean gate value, averaged across all 193 features)")
    print("="*60)
    tier_means = {}
    for tier in TIERS:
        if not gates_by_tier[tier]:
            continue
        all_gates_tier = np.concatenate(gates_by_tier[tier], axis=0)  # (N_pairs_tier, 193)
        overall_mean = all_gates_tier.mean()
        overall_std  = all_gates_tier.std()
        tier_means[tier] = {"mean": float(overall_mean), "std": float(overall_std), "n": int(all_gates_tier.shape[0])}
        print(f"  [{tier}] mean gate = {overall_mean:.4f} (+/- {overall_std:.4f}), n={all_gates_tier.shape[0]}")

    if len(tier_means) == 3:
        easy_mean = tier_means["easy"]["mean"]
        hard_mean = tier_means["hard"]["mean"]
        print(f"\n  Easy - Hard gate difference: {easy_mean - hard_mean:+.4f}")
        if easy_mean > hard_mean:
            print("  --> CONFIRMS hypothesis: gate relies MORE on stylometric on Easy than Hard")
        else:
            print("  --> DOES NOT confirm hypothesis: gate does not rely more on stylometric on Easy than Hard")

    results = {
        "feature_names": feat_names,
        "mean_gate_change":    mean_change.tolist(),
        "mean_gate_nochange":  mean_nochange.tolist(),
        "delta":               delta.tolist(),
        "top20_features":      top20_names,
        "top20_delta":         top20_delta.tolist(),
        "n_change":            int(len(gates_change)),
        "n_nochange":          int(len(gates_nochange)),
        "per_tier_gate_reliance": tier_means,  # NEW
    }
    out_json = OUT_DIR / "attention_gates.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out_json}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(22, 7))

        ax = axes[0]
        colors = ["#d62728" if d > 0 else "#1f77b4" for d in top20_delta]
        ax.barh(range(20), top20_delta[::-1], color=colors[::-1])
        ax.set_yticks(range(20))
        ax.set_yticklabels(top20_names[::-1], fontsize=9)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Gate delta (change − no-change)")
        ax.set_title("Top 20 Stylometric Features\nby Attention Gate Discriminability")
        ax.invert_yaxis()

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

        ax3 = axes[2]
        if len(tier_means) == 3:
            tiers_order = ["easy", "medium", "hard"]
            means = [tier_means[t]["mean"] for t in tiers_order]
            stds  = [tier_means[t]["std"] for t in tiers_order]
            ax3.bar(tiers_order, means, yerr=stds, color=["#2ca02c", "#ff7f0e", "#d62728"], alpha=0.8, capsize=5)
            ax3.set_ylabel("Mean attention gate value")
            ax3.set_title("Stylometric Gate Reliance\nby Difficulty Tier")

        plt.tight_layout()
        plot_path = OUT_DIR / "attention_gates.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {plot_path}")
    except ImportError:
        print("matplotlib not installed — skipping plot. Install with: pip install matplotlib")


if __name__ == "__main__":
    main()