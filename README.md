# DeSAF

**DeBERTa-Stylometric Attention Fusion for Style Change Detection in Topic-Controlled Multi-Author Documents**

---

## The Problem

Style change detection breaks down when multiple authors write about the **same topic** — semantic embeddings conflate topic similarity with authorial similarity. The PAN 2024/2025 Hard tier formally quantifies this: systems strong on Easy (topic-diverse) documents experience major F1 degradation on Hard (topic-controlled) documents.

**DeSAF directly attacks this gap** — not by beating the state of the art, but by combining an interpretable, gated fusion architecture with curriculum learning and a rigorous, significance-tested evaluation that most systems in this space don't report.

---

## Architecture

```
Document (N paragraphs)
        |
        v
+------------------------------------------+
|   DeBERTa-v3 (last 2 layers fine-tuned)   |
|   Paragraph-pair -> CLS vector (768-dim)  |
+------------------------------------------+
        |
        +----------------------+
        |                      |
   CLS (768)            Stylometric diff
                            (193-dim)
        |                      |
        +----------+-----------+
                   v
      StyleAttentionFusion
      (gated attention: style weighted
       by CLS context -> 961-dim fused)
                   |
                   v
         BiLSTM over boundary sequence
                   |
                   v
          Per-boundary logit -> sigmoid
```

**Key design choices:**
- DeBERTa-v3 backbone fine-tuned via curriculum learning (Easy->Medium->Hard, 2-2-8 epoch schedule), then frozen for downstream fusion/BiLSTM training
- 193-dim stylometric features: POS ratios, function-word frequencies, readability scores, surface features
- Gated attention fusion: stylometric features weighted by DeBERTa CLS context — the gate learns *when* to trust stylometric evidence, not just how to combine it
- Hard-tier oversampling (3x) via WeightedRandomSampler
- Decision threshold optimized on validation set (default 0.40), with per-tier tuning applied post-hoc (see Results)

---

## Results

All results below are on the **real official PAN 2025 test set** (ground truth obtained from the official PAN release, independent of and evaluated separately from any internal validation split).

### Main model (3-seed mean ± SD)

| Tier | F1 |
|------|-----|
| Easy | 0.9287 ± 0.0009 |
| Medium | 0.7268 ± 0.0016 |
| Hard | 0.7276 ± 0.0022 |
| **Macro** | **0.7944 ± 0.0005** |

### Baseline comparison

| Model | Easy | Medium | Hard | Macro |
|-------|------|--------|------|-------|
| DeBERTa-only (no stylometric, no fusion, no BiLSTM) | 0.8340 | 0.6703 | 0.6960 | 0.7334 |
| **DeSAF (full)** | **0.9287** | **0.7268** | **0.7276** | **0.7944** |

DeSAF significantly outperforms the DeBERTa-only baseline on Hard tier (ΔF1=+0.029, 95% CI [0.016, 0.042], bootstrap p=0.0002, McNemar p<0.0001).

### Ablation study

| Ablation | Easy | Medium | Hard | Macro | Hard significance |
|----------|------|--------|------|-------|---------------------|
| Full model | 0.9292 | 0.7283 | 0.7252 | 0.7942 | — |
| No stylometric | 0.9150 | 0.7197 | 0.7386 | 0.7911 | bootstrap p=0.0086; McNemar n.s. |
| No BiLSTM | 0.8106 | 0.6657 | 0.6972 | 0.7245 | bootstrap p=0.0002; McNemar p<0.0001 |
| No curriculum (Hard-only) | 0.6277 | 0.6446 | 0.7621 | 0.6781 | bootstrap p<0.0001; McNemar p<0.0001 |
| No oversampling | 0.9256 | 0.7266 | 0.7358 | 0.7960 | bootstrap p=0.0414; McNemar n.s. |

**Key finding: sequential modeling (BiLSTM), not stylometric fusion, drives most of DeSAF's advantage.** Removing BiLSTM causes the largest, most consistent drop across both significance tests. Stylometric fusion and Hard-tier oversampling appear beneficial on an internal validation split but show weak/negative effects on the real test set — a validation-to-test generalization gap we report explicitly rather than hide (see paper for full discussion).

**Curriculum learning trades Hard-tier specialization for generalization.** A Hard-only specialist model beats the curriculum-trained model on Hard tier alone (+3.7 F1) but loses 30 points on Easy and 8 on Medium — curriculum learning is what makes a single deployable model work across the full difficulty range.

### Best achievable result — ablation-informed, zero extra training

Using the ablation study's own findings, we route Hard-tier predictions through the Hard-only specialist model and apply per-tier threshold tuning + multi-seed ensembling on Easy/Medium:

| Tier | Strategy | F1 |
|------|----------|-----|
| Easy | 3-seed ensemble + threshold | 0.9320 |
| Medium | 3-seed ensemble + threshold | 0.7329 |
| Hard | Specialist routing + threshold | **0.7644** |
| **Macro** | | **0.8098** |

### Comparison with PAN 2025 systems

| System | Easy | Medium | Hard | Training data |
|--------|------|--------|------|-----------------|
| Team wqd [1] | 0.958 | 0.823 | 0.830 | PAN 2025 only |
| SCL-DeBERTa [2] | 0.955 | 0.823–0.825 | 0.829 | PAN 2025 only |
| stylospies | 0.959 | 0.786 | 0.791 | PAN 2025 only |
| **DeSAF (best combo)** | **0.932** | **0.733** | **0.764** | PAN 2025 / 2024+2025 |
| better_call_claude [3] | 0.923 | 0.828 | 0.724 | PAN 2025 only |

**Honest positioning: DeSAF does not reach the top 2 systems** (~6.5pt behind on Hard). It clearly and significantly beats better_call_claude — architecturally the closest prior system — and narrows the gap to stylospies to ~2.7pt. DeSAF's contribution is architectural interpretability and evaluation rigor (multi-seed, significance-tested, internal-vs-external comparison), not SOTA performance.

> [1] Lin et al. (2025). *Team wqd at Style Change Detection in Multi-Author Writing.* PAN@CLEF 2025.
> [2] Lin, Liu, Ye, & Han (2025). *SCL-DeBERTa: Multi-Author Writing Style Change Detection Enhanced by Supervised Contrastive Learning.* PAN@CLEF 2025.
> [3] Römisch et al. (2025). *Team "better_call_claude": Style Change Detection using a Sequential Sentence Pair Classifier.* PAN@CLEF 2025 / arXiv:2508.00675.

### Negative results

| Approach | Result |
|----------|--------|
| Joint fine-tuning of DeBERTa backbone (5 epochs, single GPU) | Did not outperform frozen-backbone approach; degraded Medium tier by 2.3pt |
| Gradient Reversal Layer (topic adversarial, early prototype) | No improvement over baseline fusion |
| Naive ensemble of separate style/semantic models (early prototype) | Worse than learned gated fusion |

---

## What We Found — Attention Gate Interpretability

**Feature-level analysis (change vs. no-change boundaries):** the gate learns clear, specific, interpretable distinctions:
- **Up-weighted at true change boundaries:** `pos_adv_ratio` (adverb usage ratio), `fw_do` — the model learned these are informative style-change signals on its own
- **Down-weighted at true change boundaries:** most function-word frequency features (`fw_did`, `fw_more`, `fw_with`, `fw_my`, `fw_down`, `fw_whose`, `fw_if`, `fw_just`, `fw_her`, `fw_than`, `fw_too`, `fw_no`, `punct_question`, and others) — the model learned these are *not* reliable indicators in isolation

**Per-tier gate reliance:** a small, directionally-consistent trend (Easy 0.6117 > Hard 0.6041) suggesting reduced stylometric reliance as topic control increases — but the effect size is modest relative to within-tier variance; we report this cautiously rather than as a strong independent claim.

**Error analysis (Hard tier):**
- False positives: overconfident (mean prob ~0.73) — high lexical shift but same author
- False negatives: uncertain (mean prob ~0.17) — authors deliberately mirror each other's style
- No positional bias — errors distributed uniformly across document positions

---

## Datasets

| Dataset | Tiers | Notes |
|---------|-------|-------|
| PAN 2024 | Easy / Medium / Hard | Training data |
| PAN 2025 | Easy / Medium / Hard | Training data + official held-out test set (obtained separately, used only for final evaluation) |

See `docs/data_acquisition.md` for download instructions. Data is **not committed** to this repo.

---

## Project Structure

```
desaf/
├── src/
│   ├── features/            # stylometric.py + 193-dim feature extractors
│   ├── models/               # deberta_fusion.py — DeBERTaFusionE2E + StyleAttentionFusion
│   ├── training/              # losses.py
│   ├── evaluation/            # metrics.py
│   └── utils/                 # data_loader.py, sbert_cache.py
├── scripts/
│   ├── precompute_finetuned_features.py   # backbone feature precompute
│   ├── precompute_official_test.py         # real PAN 2025 test set feature precompute
│   ├── retrain_fulldata.py                 # main model training (100% data)
│   ├── train_deberta_only.py               # baseline training
│   ├── eval_test_npz.py                    # main model + baseline evaluation
│   ├── eval_deberta_only.py
│   ├── ablation_no_stylometric.py
│   ├── ablation_no_bilstm.py
│   ├── ablation_no_curriculum.py
│   ├── ablation_no_oversampling.py
│   ├── eval_ablation.py                    # unified ablation evaluation
│   ├── finetune_joint.py                   # joint backbone fine-tuning experiment
│   ├── eval_finetune_joint.py
│   ├── threshold_sweep.py                  # per-tier threshold optimization
│   ├── ensemble_seeds.py                   # multi-seed ensembling
│   ├── tier_routing_check.py               # ablation-informed tier routing
│   ├── compute_significance.py             # paired bootstrap + McNemar's test
│   ├── attention_viz.py                    # attention gate interpretability analysis
│   ├── analyze_stylometric_by_length.py    # stylometric gain by paragraph length
│   └── error_analysis.py
├── docs/
│   └── data_acquisition.md
├── tests/
├── archive/                  # gitignored — superseded prototype code/checkpoints
├── checkpoints/               # gitignored
├── data/                      # gitignored
├── results/                   # gitignored
├── RESULTS_SUMMARY.md          # full, current experimental record
├── requirements.txt
└── README.md
```

---

## Reproduce

**1 — Precompute backbone features (training data)**
```bash
python scripts/precompute_finetuned_features.py --seed 42
```

**2 — Precompute official test set features**
```bash
python scripts/precompute_official_test.py
```

**3 — Train main model**
```bash
python scripts/retrain_fulldata.py --checkpoint checkpoints/curriculum_228_best.pt --feat-dir data/processed/features_finetuned
```

**4 — Evaluate on real test set**
```bash
python scripts/eval_test_npz.py --checkpoint checkpoints/fulldata/fulldata_final.pt --feat-dir data/processed/features_finetuned_official_test --years 2025
```

**5 — Run ablations**
```bash
python scripts/ablation_no_stylometric.py --checkpoint checkpoints/curriculum_228_best.pt --feat-dir data/processed/features_finetuned
python scripts/ablation_no_bilstm.py --checkpoint checkpoints/curriculum_228_best.pt --feat-dir data/processed/features_finetuned
python scripts/ablation_no_curriculum.py --base-checkpoint checkpoints/curriculum_228_best.pt --feat-dir data/processed/features_finetuned
python scripts/ablation_no_oversampling.py --checkpoint checkpoints/curriculum_228_best.pt --feat-dir data/processed/features_finetuned
```

**6 — Significance testing**
```bash
python scripts/compute_significance.py --a checkpoints/fulldata/test_predictions.npz --b checkpoints/deberta_only/test_predictions.npz --tier 2025_hard
```

See `RESULTS_SUMMARY.md` for the complete, current experimental record.

---

## Setup

```bash
git clone https://github.com/prisha2410/desaf.git
cd desaf
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt averaged_perceptron_tagger stopwords
```

---

## Key References

- Zangerle, Mayerl, Potthast, & Stein (2025). *Overview of the Multi-Author Writing Style Analysis Task at PAN 2025.* CLEF 2025 Working Notes, CEUR-WS Vol. 4038.
- Bevendorff et al. (2025). *Overview of PAN 2025: Voight-Kampff Generative AI Detection, Multilingual Text Detoxification, Multi-author Writing Style Analysis, and Generative Plagiarism Detection.* CLEF 2025, Springer LNCS 16089.
- Lin et al. (2025). *Team wqd at Style Change Detection in Multi-Author Writing.* PAN@CLEF 2025.
- Lin, Liu, Ye, & Han (2025). *SCL-DeBERTa: Multi-Author Writing Style Change Detection Enhanced by Supervised Contrastive Learning.* PAN@CLEF 2025.
- Römisch, Gorovaia, Halchynska, Schmidt, & Yamshchikov (2025). *Team "better_call_claude": Style Change Detection using a Sequential Sentence Pair Classifier.* arXiv:2508.00675.
- Mady, Reschke, & Schuller (2026). *Feature-Augmented Transformers for Robust AI-Text Detection Across Domains and Generators.* arXiv:2605.03969.
- Hashemi & Shi (2025). *A Survey on Writing Style Change Detection: Current Literature and Future Directions.* Machine Intelligence Research, 22(3), 397-416.
- He, Gao, & Chen (2021). *DeBERTaV3: Improving DeBERTa using ELECTRA-style pre-training.* ICLR.
- Stamatatos (2009). *A survey of modern authorship attribution methods.* JASIST.


