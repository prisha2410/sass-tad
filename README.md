# DeSAF

**DeBERTa-Stylometric Attention Fusion for Style Change Detection in Topic-Controlled Multi-Author Documents**

---

## The Problem

Style change detection breaks down when multiple authors write about the **same topic** — semantic embeddings conflate topic similarity with authorial similarity. The PAN 2024/2025 Hard tier formally quantifies this: systems strong on Easy (topic-diverse) documents experience major F1 degradation on Hard (topic-controlled) documents.

**DeSAF directly attacks this gap.**

---

## Architecture

```
Document (N sentences)
        │
        ▼
┌─────────────────────────────────────────┐
│         DeBERTa-v3-small (frozen)       │
│   Sentence-pair → CLS vector (768-dim)  │
└──────────────────┬──────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
   CLS (768)           Stylometric diff
                          (193-dim)
        │                     │
        └──────────┬──────────┘
                   ▼
      StyleAttentionFusion
      (gated attention: style weighted
       by CLS context → 961-dim fused)
                   │
                   ▼
         BiLSTM over boundary sequence
                   │
                   ▼
          Per-boundary logit → sigmoid
```

**Key design choices:**
- DeBERTa-v3-small fine-tuned end-to-end with curriculum learning (Easy→Medium→Hard, 2-2-8 epoch schedule)
- 193-dim stylometric features: POS ratios, function word frequencies, readability scores, surface features
- Gated attention fusion: stylometric features weighted by DeBERTa CLS context
- Hard-tier oversampling (3×) via WeightedRandomSampler to address class imbalance
- Decision threshold optimised on validation set (best = 0.40)

---

## Results

All numbers are on the **test set** with threshold = 0.40.

| Dataset | Val F1 | Test F1 |
|---------|--------|---------|
| 2024 Easy | 0.9898 | 0.9895 |
| 2024 Medium | 0.8849 | 0.8854 |
| 2024 Hard | 0.8775 | 0.8629 |
| 2025 Easy | 0.9469 | 0.9550 |
| 2025 Medium | 0.7901 | 0.7901 |
| 2025 Hard | 0.7999 | 0.8218 |
| **Avg Easy** | **0.9684** | **0.9723** |
| **Avg Medium** | **0.8375** | **0.8378** |
| **Avg Hard** | **0.8387** | **0.8424** |
| **Macro** | **0.8815** | **0.8841** |

### Comparison with PAN 2025 Top Systems

| Tier | DeSAF (ours) | Team wqd [1] | SCL-DeBERTa [2] | Δ vs best |
|------|-------------|--------------|-----------------|-----------|
| Easy | **0.9723** | 0.958 | 0.955 | +0.014 |
| Medium | **0.8378** | 0.823 | 0.825 | +0.013 |
| Hard | **0.8424** | 0.830 | 0.829 | +0.012 |
| Macro | **0.8841** | 0.836 | 0.846 | +0.038 |

> [1] Lin et al. (2025). *Team wqd at Style Change Detection in Multi-Author Writing.* PAN@CLEF 2025.
> [2] Lin et al. (2025). *SCL-DeBERTa: Multi-Author Writing Style Change Detection Enhanced by Supervised Contrastive Learning.* PAN@CLEF 2025.
> Note: Team wqd and SCL-DeBERTa report scores on the official TIRA test set. DeSAF evaluates on the same dataset split using independently computed predictions.

### Negative Results

| Approach | Result |
|----------|--------|
| Ensemble (curriculum + hard-ft) | Hard-ft forgot Easy tier (F1: 0.9550 → 0.5738) |
| Gradient Reversal Layer (topic adversarial) | No improvement over baseline fusion |
| CRF output layer | Marginal gain, not worth complexity |

---

## What We Found

**Attention gate analysis:** Features most active at style change boundaries:
- **Higher at boundaries:** `fw_do`, `fw_your`, `fw_which`, `fw_where`, `fw_since` — direct address words and relative clause markers signal author switches
- **Lower at boundaries:** `fw_some`, `fw_many`, `fw_too`, `punct_question` — hedging/quantifier words and question marks indicate same-author continuity

This is consistent with stylometry literature: syntactic and function-word features are more topic-independent than lexical features.

**Error analysis (Hard tier):**
- False positives: overconfident (mean prob 0.73) — high lexical shift but same author
- False negatives: uncertain (mean prob 0.17) — authors deliberately mirror each other's style
- No positional bias — errors distributed uniformly across document positions

---

## Datasets

| Dataset | Granularity | Tiers | Notes |
|---------|-------------|-------|-------|
| PAN 2024 | Sentence-level | Easy / Medium / Hard | Primary benchmark |
| PAN 2025 | Sentence-level | Easy / Medium / Hard | Primary benchmark |

See `docs/data_acquisition.md` for download instructions. Data is **not committed** to this repo.

---

## Project Structure

```
desaf/
├── src/
│   ├── features/
│   │   ├── stylometric.py              # 193-dim feature extractor
│   │   ├── pos_features.py
│   │   ├── surface_features.py
│   │   ├── function_words.py
│   │   └── readability.py
│   ├── models/
│   │   └── deberta_fusion.py           # DeBERTaFusionE2E + StyleAttentionFusion
│   ├── training/
│   │   └── losses.py
│   ├── evaluation/
│   │   └── metrics.py
│   └── utils/
│       ├── data_loader.py
│       └── sbert_cache.py
├── scripts/
│   ├── precompute_finetuned_features.py   # Step 1
│   ├── finetune_hard_oversample.py        # Step 2
│   ├── threshold_sweep.py                 # Step 3
│   ├── test_eval.py                       # Step 4
│   ├── attention_viz.py                   # Attention gate analysis
│   ├── error_analysis.py                  # FP/FN pattern analysis
│   └── ensemble_eval.py                   # Negative result — kept for reference
├── scripts/archive/                        # Old SBERT-based experiments
├── docs/
│   └── data_acquisition.md
├── tests/
├── checkpoints/                            # gitignored
├── data/                                   # gitignored
├── results/                                # gitignored
├── requirements.txt
└── README.md
```

---

## Reproduce

**Step 1 — Precompute features using fine-tuned DeBERTa**
```bash
python scripts/precompute_finetuned_features.py --checkpoint checkpoints/curriculum_228_best.pt
```

**Step 2 — Fine-tune fusion head with Hard oversampling**
```bash
python scripts/finetune_hard_oversample.py --checkpoint checkpoints/curriculum_228_best.pt
```

**Step 3 — Find best threshold**
```bash
python scripts/threshold_sweep.py
```

**Step 4 — Evaluate on test set**
```bash
python scripts/test_eval.py --threshold 0.40
```

---

## Setup

```bash
git clone https://github.com/prisha2410/sass-tad.git
cd sass-tad
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt averaged_perceptron_tagger stopwords
```

---

## Key References

- Bevendorff et al. (2024). *Overview of PAN 2024 Style Change Detection.* LNCS 14613.
- Bevendorff et al. (2025). *Overview of PAN 2025 Style Change Detection.*
- Lin et al. (2025). *Team wqd at Style Change Detection in Multi-Author Writing.* PAN@CLEF 2025.
- Lin et al. (2025). *SCL-DeBERTa: Multi-Author Writing Style Change Detection Enhanced by Supervised Contrastive Learning.* PAN@CLEF 2025.
- He et al. (2021). *DeBERTaV3: Improving DeBERTa using ELECTRA-style pre-training.* ICLR.
- Stamatatos (2009). *A survey of modern authorship attribution methods.* JASIST.