"""
SASS-TAD Data Loader
====================

Handles PAN 2023 / 2024 / 2025 and MuLD AO3 datasets.

PAN split strategy (minor project)
-----------------------------------
    PAN provides train (70%) + validation (15%) with ground truth.
    The official test split has NO labels (withheld for competition only).

    Since this is a minor project (not a PAN submission), we ignore the
    official test split and build our own clean 70/15/15:

        Step 1: Load PAN train + validation (both labeled) — pool them
        Step 2: Call split_documents(docs, seed=42) → train / val / test
        Step 3: You now have a real held-out test set with ground truth

    Use load_pan_labeled() to do steps 1+2 in one call.

PAN folder structure (per year, per tier)
------------------------------------------
    data/raw/pan{year}/{tier}/
        train/
            problem-1.txt
            truth-problem-1.json
            ...
        validation/
            problem-1.txt
            truth-problem-1.json
            ...

PAN JSON ground truth format
-----------------------------
    { "authors": <int>, "changes": [0, 1, 0, ...] }

MuLD AO3
---------
    Full ground truth, no split issues.
    load_muld() returns all labeled docs — pass through split_documents() too.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Literal, Optional, Tuple


# --------------------------------------------------------------------------- #
#  Data structures
# --------------------------------------------------------------------------- #

@dataclass
class Document:
    """
    A single document ready for the feature pipeline.

    Attributes
    ----------
    doc_id          : str
    sentences       : list of str
    boundary_labels : list of int, length = len(sentences) - 1
                      1 = style change between sentence i and i+1, else 0
                      None only for official PAN test split (we don't use it)
    n_authors       : int or None
    source          : 'pan2023' | 'pan2024' | 'pan2025' | 'muld_ao3'
    tier            : 'easy' | 'medium' | 'hard' | None
    split           : 'train' | 'val' | 'test'  (our split, not PAN's)
    """
    doc_id:          str
    sentences:       List[str]
    boundary_labels: Optional[List[int]]
    n_authors:       Optional[int]
    source:          str
    tier:            Optional[str]
    split:           str = "unsplit"

    def __len__(self) -> int:
        return len(self.sentences)

    @property
    def n_boundaries(self) -> int:
        return max(0, len(self.sentences) - 1)

    @property
    def has_labels(self) -> bool:
        return self.boundary_labels is not None

    @property
    def n_changes(self) -> int:
        if not self.has_labels:
            return 0
        return sum(self.boundary_labels)

    def __repr__(self) -> str:
        label_str = f"{self.n_changes} changes" if self.has_labels else "no labels"
        return (
            f"Document(id={self.doc_id!r}, sents={len(self.sentences)}, "
            f"{label_str}, source={self.source}, tier={self.tier}, split={self.split})"
        )


# --------------------------------------------------------------------------- #
#  PAN loader
# --------------------------------------------------------------------------- #

Tier  = Literal["easy", "medium", "hard"]

# PAN 2023 uses dataset1/2/3; 2024 and 2025 use easy/medium/hard directly
_PAN_TIER_DIRS = {
    "pan2023": {"easy": "dataset1", "medium": "dataset2", "hard": "dataset3"},
    "pan2024": {"easy": "easy",     "medium": "medium",   "hard": "hard"},
    "pan2025": {"easy": "easy",     "medium": "medium",   "hard": "hard"},
}


def _read_pan_document(
    txt_path: Path,
    truth_path: Optional[Path],
    doc_id: str,
    source: str,
    tier: str,
) -> Document:
    raw = txt_path.read_text(encoding="utf-8").strip()
    sentences = [s.strip() for s in raw.split("\n") if s.strip()]

    boundary_labels = None
    n_authors = None

    if truth_path is not None and truth_path.exists():
        truth = json.loads(truth_path.read_text(encoding="utf-8"))
        boundary_labels = truth.get("changes", None)
        n_authors = truth.get("authors", None)

        if boundary_labels is not None:
            expected = max(0, len(sentences) - 1)
            boundary_labels = (boundary_labels + [0] * expected)[:expected]

    return Document(
        doc_id=doc_id,
        sentences=sentences,
        boundary_labels=boundary_labels,
        n_authors=n_authors,
        source=source,
        tier=tier,
    )


class PANDataset:
    """
    Loads PAN labeled documents (train + validation splits) for one year/tier.

    Parameters
    ----------
    root : str or Path
        Path to data/raw/
    year : '2023' | '2024' | '2025'
    tier : 'easy' | 'medium' | 'hard'

    Note: We only load train + validation (both have ground truth).
          The official PAN test split has no labels — we ignore it.

    Example
    -------
        ds = PANDataset("data/raw", year="2024", tier="hard")
        docs = ds.load()   # list of all labeled Documents
    """

    def __init__(
        self,
        root: str | Path,
        year: Literal["2023", "2024", "2025"],
        tier: Tier,
    ):
        self.root = Path(root)
        self.year = year
        self.tier = tier

        source_key = f"pan{year}"
        if source_key not in _PAN_TIER_DIRS:
            raise ValueError(f"Unsupported year: {year}. Choose 2023, 2024, or 2025.")

        tier_dir = _PAN_TIER_DIRS[source_key].get(tier)
        if tier_dir is None:
            raise ValueError(f"Unknown tier '{tier}' for {source_key}.")

        self._source   = source_key
        self._tier_dir = self.root / source_key / tier_dir

        if not self._tier_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self._tier_dir}\n"
                f"See docs/data_acquisition.md for download instructions."
            )

    def _load_split(self, pan_split: str) -> List[Document]:
        split_dir = self._tier_dir / pan_split
        if not split_dir.exists():
            return []

        docs = []
        for txt_path in sorted(split_dir.glob("problem-*.txt")):
            problem_id = txt_path.stem
            truth_path = split_dir / f"truth-{problem_id}.json"
            doc_id = f"{self._source}_{self.tier}_{pan_split}_{problem_id}"
            doc = _read_pan_document(
                txt_path=txt_path,
                truth_path=truth_path,
                doc_id=doc_id,
                source=self._source,
                tier=self.tier,
            )
            if doc.has_labels:   # only keep labeled docs
                docs.append(doc)
        return docs

    def load(self) -> List[Document]:
        """Load all labeled documents (train + validation combined)."""
        docs = self._load_split("train") + self._load_split("validation")
        if not docs:
            raise FileNotFoundError(
                f"No labeled documents found in {self._tier_dir}\n"
                f"Make sure train/ and validation/ subdirectories exist."
            )
        return docs

    def __repr__(self) -> str:
        return f"PANDataset(year={self.year}, tier={self.tier}, root={self.root})"


# --------------------------------------------------------------------------- #
#  MuLD AO3 loader
# --------------------------------------------------------------------------- #

class MuLDDataset:
    """
    Loads MuLD AO3 Style Change Detection dataset.

    Parameters
    ----------
    local_path : str or Path, optional
        Path to a directory saved via ds.save_to_disk("data/raw/muld_ao3").
        If None or path doesn't exist, loads from HuggingFace (needs internet).

    Example
    -------
        # First time — saves locally
        ds = MuLDDataset()
        # Thereafter
        ds = MuLDDataset("data/raw/muld_ao3")
        docs = ds.load()
    """

    def __init__(self, local_path: Optional[str | Path] = None):
        self._local_path = Path(local_path) if local_path else None
        self._documents: List[Document] = []

    def load(self) -> List[Document]:
        if self._local_path and self._local_path.exists():
            self._load_from_disk(self._local_path)
        else:
            print("[MuLD] Downloading from HuggingFace (this only happens once)...")
            self._load_from_hub()
        return self._documents

    def _load_from_disk(self, path: Path) -> None:
        from datasets import load_from_disk
        hf_ds = load_from_disk(str(path))
        for split_name in hf_ds.keys():
            self._parse_split(hf_ds[split_name], split_name)

    def _load_from_hub(self) -> None:
        from datasets import load_dataset as hf_load
        hf_ds = hf_load("ghomasHudson/muld", "AO3 Style Change Detection")
        for split_name in hf_ds.keys():
            self._parse_split(hf_ds[split_name], split_name)

    def _parse_split(self, split_ds, split_name: str) -> None:
        for idx, row in enumerate(split_ds):
            doc_id    = f"muld_ao3_{split_name}_{idx}"
            text      = row.get("document", "") or row.get("text", "")
            sentences = [s.strip() for s in text.split("\n") if s.strip()]

            boundary_labels = None
            label_raw = row.get("summary", None) or row.get("label", None)
            if label_raw is not None:
                try:
                    parsed = json.loads(label_raw) if isinstance(label_raw, str) else label_raw
                    if isinstance(parsed, list):
                        boundary_labels = [int(x) for x in parsed]
                    elif isinstance(parsed, dict):
                        boundary_labels = parsed.get("changes", None)
                except (json.JSONDecodeError, TypeError):
                    boundary_labels = None

            if boundary_labels is not None:
                self._documents.append(Document(
                    doc_id=doc_id,
                    sentences=sentences,
                    boundary_labels=boundary_labels,
                    n_authors=None,
                    source="muld_ao3",
                    tier=None,
                ))

    def __repr__(self) -> str:
        return f"MuLDDataset(local_path={self._local_path})"


# --------------------------------------------------------------------------- #
#  Our own 70 / 15 / 15 splitter
# --------------------------------------------------------------------------- #

def split_documents(
    docs: List[Document],
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    seed:        int   = 42,
) -> Tuple[List[Document], List[Document], List[Document]]:
    """
    Split a list of labeled Documents into train / val / test.

    Parameters
    ----------
    docs        : list of Document (all must have ground truth labels)
    train_ratio : default 0.70
    val_ratio   : default 0.15
    seed        : random seed for reproducibility (default 42)

    Returns
    -------
    (train_docs, val_docs, test_docs)

    Example
    -------
        all_docs = PANDataset("data/raw", "2024", "hard").load()
        train, val, test = split_documents(all_docs)
        print(len(train), len(val), len(test))
    """
    labeled = [d for d in docs if d.has_labels]
    if len(labeled) < len(docs):
        print(f"[split_documents] Warning: {len(docs) - len(labeled)} unlabeled docs dropped.")

    rng = random.Random(seed)
    shuffled = labeled[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train_docs = shuffled[:n_train]
    val_docs   = shuffled[n_train : n_train + n_val]
    test_docs  = shuffled[n_train + n_val :]

    # tag each doc with its split
    for d in train_docs: d.split = "train"
    for d in val_docs:   d.split = "val"
    for d in test_docs:  d.split = "test"

    print(
        f"[split_documents] {n} docs → "
        f"train={len(train_docs)}, val={len(val_docs)}, test={len(test_docs)}"
    )
    return train_docs, val_docs, test_docs


# --------------------------------------------------------------------------- #
#  Convenience loaders
# --------------------------------------------------------------------------- #

def load_pan_labeled(
    root: str | Path = "data/raw",
    year: Literal["2023", "2024", "2025"] = "2024",
    tier: Tier = "hard",
    seed: int = 42,
) -> Tuple[List[Document], List[Document], List[Document]]:
    """
    One-call loader: load PAN labeled docs and return our own train/val/test split.

    Example
    -------
        train, val, test = load_pan_labeled("data/raw", year="2024", tier="hard")
    """
    docs = PANDataset(root=root, year=year, tier=tier).load()
    return split_documents(docs, seed=seed)


def load_pan_all_tiers(
    root: str | Path = "data/raw",
    year: Literal["2023", "2024", "2025"] = "2024",
    seed: int = 42,
) -> Dict[str, Tuple[List[Document], List[Document], List[Document]]]:
    """
    Load all three tiers for a given year, each with its own 70/15/15 split.

    Returns
    -------
    {
        "easy":   (train, val, test),
        "medium": (train, val, test),
        "hard":   (train, val, test),
    }
    """
    return {
        tier: load_pan_labeled(root=root, year=year, tier=tier, seed=seed)
        for tier in ("easy", "medium", "hard")
    }


# --------------------------------------------------------------------------- #
#  Dataset statistics
# --------------------------------------------------------------------------- #

def dataset_stats(docs: List[Document]) -> dict:
    """Summary statistics for a list of Documents."""
    if not docs:
        return {}

    labeled = [d for d in docs if d.has_labels]
    n = len(docs)

    avg_sents  = sum(len(d.sentences) for d in docs) / n
    avg_bounds = sum(d.n_boundaries   for d in docs) / n

    if labeled:
        avg_changes   = sum(d.n_changes for d in labeled) / len(labeled)
        total_bounds  = sum(d.n_boundaries for d in labeled)
        total_changes = sum(d.n_changes    for d in labeled)
        change_rate   = total_changes / total_bounds if total_bounds else 0.0
        pct_no_change = sum(1 for d in labeled if d.n_changes == 0) / len(labeled)
    else:
        avg_changes = change_rate = pct_no_change = 0.0

    return {
        "n_docs":             n,
        "n_labeled":          len(labeled),
        "avg_sentences":      round(avg_sents,  2),
        "avg_boundaries":     round(avg_bounds, 2),
        "avg_changes":        round(avg_changes, 2),
        "change_rate":        round(change_rate, 4),
        "pct_no_change_docs": round(pct_no_change, 4),
    }