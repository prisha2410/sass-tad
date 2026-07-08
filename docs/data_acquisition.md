# Data Acquisition Guide

## PAN 2023 / 2024 / 2025

All three PAN datasets require a two-step access process:

1. **Register on TIRA** — [tira.io](https://www.tira.io)
2. **Request access on Zenodo** using the **same email address** as your TIRA account

| Year | Granularity | Zenodo Link |
|------|-------------|-------------|
| PAN 2023 | Paragraph-level | https://zenodo.org/records/7729178 |
| PAN 2024 | Sentence-level | https://pan.webis.de/clef24/pan24-web/style-change-detection.html |
| PAN 2025 | Sentence-level | https://zenodo.org/records/14891299 |

Once downloaded, place files at:
```
data/raw/pan2023/
data/raw/pan2024/
data/raw/pan2025/
```

Each year contains three sub-datasets: `easy/`, `medium/`, `hard/`
Each problem instance is a `.txt` file; ground truth is a `.json` file.

---

## MuLD AO3

No registration required. Load directly via HuggingFace:

```python
from datasets import load_dataset
ds = load_dataset("ghomasHudson/muld", "AO3 Style Change Detection")
```

Or save locally:
```python
ds.save_to_disk("data/raw/muld_ao3")
```

---

## Important Notes

- PAN datasets contain copyrighted material — research use only, no redistribution.
- Do **not** commit any data files. The `data/` directory is gitignored.
- SBERT embeddings (`.pt` files) go in `data/processed/embeddings/` — also gitignored.