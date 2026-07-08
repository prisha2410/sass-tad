# Data Acquisition Guide

## Overview

This project uses **PAN 2024** and **PAN 2025** as primary benchmarks — both sentence-level style change detection datasets with three tiers (Easy / Medium / Hard) based on topic control.

---

## PAN 2024

**Granularity:** Sentence-level
**Tiers:** Easy (topic-diverse), Medium (partial topic control), Hard (strict topic control)
**Format:** `problem-X.txt` (one sentence per line) + `truth-problem-X.json` (with a `changes` list)

**Access (two steps):**
1. Register at [tira.io](https://www.tira.io)
2. Request access using the **same email** as your TIRA account:
   [pan.webis.de/clef24/pan24-web/style-change-detection.html](https://pan.webis.de/clef24/pan24-web/style-change-detection.html)

**Place files at:**
```
data/raw/pan2024/
├── easy/
│   ├── train/
│   └── validation/
├── medium/
│   ├── train/
│   └── validation/
└── hard/
    ├── train/
    └── validation/
```

---

## PAN 2025

**Granularity:** Sentence-level (same format as PAN 2024)
**Tiers:** Easy / Medium / Hard

**Access (two steps):**
1. Register at [tira.io](https://www.tira.io)
2. Request access using the **same email** as your TIRA account:
   [zenodo.org/records/14891299](https://zenodo.org/records/14891299)

**Place files at:**
```
data/raw/pan2025/
├── easy/
│   ├── train/
│   └── validation/
├── medium/
│   ├── train/
│   └── validation/
└── hard/
    ├── train/
    └── validation/
```

---

## Important Notes

- PAN datasets contain copyrighted material — **research use only, no redistribution**
- Do **not** commit data files — `data/` is gitignored
- Test set ground truth is **not publicly released** — reported numbers use the validation split unless stated otherwise
- Precomputed DeBERTa features (`.npz`) go in `data/processed/features_finetuned/` — also gitignored