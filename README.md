# SASS-TAD

**Sequence-Aware Stylometric-Semantic Fusion for Topic-Controlled Authorship Change Detection in Multi-Author Documents**

*NLP / Stylometry*

---

## The Problem

Style change detection systems work well when different authors write about different topics. They break down when multiple authors write about the **same topic** — because semantic embeddings like SBERT conflate topic similarity with authorial similarity, making it impossible to distinguish a genuine authorship boundary from a mere topic continuation.

The PAN 2023/24 benchmarks formally quantify this: systems strong on Easy-tier (topic-diverse) documents experience major F1 degradation on Hard-tier (strictly topic-controlled) documents.

**SASS-TAD directly attacks this gap.**

---

## Proposed Architecture

```
Document (N sentences)
        │
        ▼
┌───────────────────────────────────────────────────┐
│              Dual-Channel Extraction               │
│   SBERT embeddings (GPU)  │  Stylometric (CPU)    │
└───────────────┬───────────┴──────────┬────────────┘
                │                      │
                └──────────┬───────────┘
                           ▼
                  Feature Fusion Layer
                           │
                           ▼
               Contrastive Style Encoder
          (same-author pairs ↔ diff-author pairs)
                           │
                           ▼
              Topic-Adversarial Layer (GRL)
            (style encoder learns to forget topic)
                           │
                           ▼
              Sequence-Aware BiLSTM Classifier
                           │
                           ▼
             Per-boundary probability + SHAP / IG
```

Three core innovations over existing PAN systems:

1. **Contrastive Style Encoder** — pulls same-author representations together, pushes different-author ones apart in a topic-agnostic embedding space.
2. **Topic-Adversarial Disentanglement** — a Gradient Reversal Layer forces the style encoder to maximally predict authorship boundaries while minimally predicting topic labels.
3. **Sequence-Aware BiLSTM** — models boundary patterns across neighbouring sentences rather than classifying each boundary independently.

---

## Research Questions

| # | Question |
|---|----------|
| RQ1 | Do topic-independent stylometric features outperform SBERT alone on Medium/Hard PAN tiers? |
| RQ2 | Does sequence-aware BiLSTM outperform independent boundary classifiers (LR, SVM)? |
| RQ3 | Which stylometric features remain stable under strict topic control? |
| RQ4 | Does explicit topic-adversarial disentanglement improve Hard-tier detection over non-adversarial baselines? |

---

## Datasets

| Dataset | Granularity | Tiers | Access | DOI / Link |
|---------|-------------|-------|--------|------------|
| PAN 2023 | Paragraph-level | Easy / Medium / Hard | TIRA registration → Zenodo request | [zenodo.org/records/7729178](https://zenodo.org/records/7729178) |
| PAN 2024 | Sentence-level | Easy / Medium / Hard | TIRA registration → Zenodo request | [pan.webis.de/clef24](https://pan.webis.de/clef24/pan24-web/style-change-detection.html) |
| PAN 2025 | Sentence-level | Easy / Medium / Hard | TIRA registration → Zenodo request | [zenodo.org/records/14891299](https://zenodo.org/records/14891299) |
| MuLD AO3 | Paragraph-level | Cross-domain | HuggingFace (open, no registration) | [ghomasHudson/muld](https://huggingface.co/datasets/ghomasHudson/muld) |

> **Access note for PAN datasets:** Register at [tira.io](https://www.tira.io), then request dataset access on Zenodo using the **same email address**. Datasets contain copyrighted material — research use only, no redistribution.

```python
# MuLD AO3 — loads directly, no registration needed
from datasets import load_dataset
ds = load_dataset("ghomasHudson/muld", "AO3 Style Change Detection")
```

> **Note on granularity:** PAN 2023 operates at paragraph level; PAN 2024/2025 advance to sentence level — matching our task formulation. All three are used: 2023 for cross-granularity analysis, 2024/2025 as primary benchmarks.

Data is **not committed** to this repo. See `docs/data_acquisition.md` for full download instructions.

---

## Project Structure

```
sass-tad/
├── src/
│   ├── features/
│   │   ├── stylometric.py        # Main extractor — call this
│   │   ├── pos_features.py       # POS ratios, trigram entropy
│   │   ├── surface_features.py   # Length, TTR, punctuation
│   │   ├── function_words.py     # Top-150 function word frequencies
│   │   └── readability.py        # Flesch, Gunning Fog, Coleman-Liau
│   ├── models/
│   │   ├── fusion.py             # Feature fusion layer
│   │   ├── contrastive.py        # Contrastive style encoder
│   │   ├── adversarial.py        # GRL + topic classifier head
│   │   ├── bilstm.py             # Sequence-aware BiLSTM
│   │   └── sass_tad.py           # Full end-to-end model
│   ├── training/
│   │   ├── train.py              # Main training loop
│   │   ├── losses.py             # Contrastive + adversarial losses
│   │   └── scheduler.py          # λ schedule for GRL
│   ├── evaluation/
│   │   ├── metrics.py            # Macro F1, WindowDiff
│   │   └── evaluate.py           # Evaluation runner
│   ├── explainability/
│   │   ├── shap_analysis.py      # SHAP on baselines (CPU)
│   │   ├── integrated_gradients.py  # IG on SASS-TAD (GPU)
│   │   └── attention_viz.py      # Attention weight visualisation
│   └── utils/
│       ├── data_loader.py        # PAN + MuLD dataset loaders
│       ├── sbert_cache.py        # Precompute + cache embeddings
│       └── config.py             # Config loading
├── scripts/
│   ├── precompute_embeddings.py  # Run once before Phase 3
│   └── run_baselines.py          # Phase 3 baseline evaluation
├── configs/
│   ├── base.yaml                 # Shared hyperparameters
│   ├── phase3_baselines.yaml
│   └── phase4_sass_tad.yaml
├── tests/
│   └── test_stylometric.py       # Unit tests (no dataset needed)
├── notebooks/                    # Exploratory analysis
├── data/                         # gitignored — local only
├── results/                      # gitignored — local only
├── docs/
│   └── data_acquisition.md
├── requirements.txt
└── README.md
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

- Altakrori et al. (2021). *The topic confusion task.* EMNLP Findings.
- Bevendorff et al. (2024). *Overview of PAN 2024.* LNCS 14613.
- van Leeuwen et al. (2025). *Combining style and semantics for robust authorship verification.* MLWA.
- Stamatatos (2009). *A survey of modern authorship attribution methods.* JASIST.
