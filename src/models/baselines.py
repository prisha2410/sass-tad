"""
Baseline classifiers for style change detection (Phase 3).

Three baselines:
  1. StylometricBaseline — logistic regression on hand-crafted feature diffs
  2. SBERTBaseline       — logistic regression on SBERT embedding diffs
  3. FusedBaseline       — logistic regression on concatenated features

All are sentence-pair-level binary classifiers: given sentence i and i+1,
predict whether there's a style change boundary between them (1) or not (0).

These are intentionally simple (sklearn LogisticRegression) — they exist
to give "before SASS-TAD" reference numbers, not to compete with the
final architecture.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from src.features.stylometric import StylometricExtractor
from src.utils.data_loader import Document


@dataclass
class BaselineResult:
    name: str
    y_true: List[int]
    y_pred: List[int]
    doc_true: List[List[int]]   # per-document boundary sequences (for WindowDiff)
    doc_pred: List[List[int]]


def _build_stylometric_diffs(docs: List[Document], extractor: StylometricExtractor) -> Tuple[np.ndarray, List[int], List[int]]:
    """
    For each document, extract per-sentence stylometric features and compute
    consecutive-sentence diff vectors. Returns (X, y, doc_lengths).
    """
    X_rows = []
    y = []
    doc_lengths = []

    for doc in docs:
        if len(doc.sentences) < 2:
            doc_lengths.append(0)
            continue

        vecs = [extractor.extract_vector(s) for s in doc.sentences]

        n_pairs = len(doc.sentences) - 1
        doc_lengths.append(n_pairs)

        for i in range(n_pairs):
            diff = np.abs(np.asarray(vecs[i + 1]) - np.asarray(vecs[i]))
            X_rows.append(diff)
            y.append(int(doc.boundary_labels[i]))

    X = np.vstack(X_rows) if X_rows else np.zeros((0, 0))
    return X, y, doc_lengths


def _build_sbert_diffs(cache_path: str) -> Tuple[np.ndarray, List[int], List[int]]:
    """
    Load cached SBERT embeddings and compute consecutive-sentence diff vectors.
    Cache format (see sbert_cache.py):
        {"doc_id": [...], "embeddings": List[Tensor(n_sent, 384)], "labels": [[...], ...]}
    """
    import torch
    cache = torch.load(cache_path)

    X_rows = []
    y = []
    doc_lengths = []

    for emb, labels in zip(cache["embeddings"], cache["labels"]):
        emb = emb.numpy() if hasattr(emb, "numpy") else np.asarray(emb)
        n_pairs = emb.shape[0] - 1
        doc_lengths.append(max(n_pairs, 0))

        for i in range(n_pairs):
            diff = np.abs(emb[i + 1] - emb[i])
            X_rows.append(diff)
            y.append(int(labels[i]))

    X = np.vstack(X_rows) if X_rows else np.zeros((0, 0))
    return X, y, doc_lengths


def _split_by_doc(y: List[int], doc_lengths: List[int]) -> List[List[int]]:
    """Re-chunk a flat label list back into per-document sequences."""
    out = []
    idx = 0
    for n in doc_lengths:
        out.append(y[idx:idx + n])
        idx += n
    return out


class StylometricBaseline:
    """Logistic regression on stylometric feature-diff vectors."""

    name = "stylometric"

    def __init__(self):
        self.extractor = StylometricExtractor()
        self.pipeline = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        )

    def fit(self, train_docs: List[Document]):
        X, y, _ = _build_stylometric_diffs(train_docs, self.extractor)
        self.pipeline.fit(X, y)
        return self

    def evaluate(self, docs: List[Document]) -> BaselineResult:
        X, y_true, doc_lengths = _build_stylometric_diffs(docs, self.extractor)
        y_pred = self.pipeline.predict(X).tolist()
        return BaselineResult(
            name=self.name,
            y_true=y_true,
            y_pred=y_pred,
            doc_true=_split_by_doc(y_true, doc_lengths),
            doc_pred=_split_by_doc(y_pred, doc_lengths),
        )


class SBERTBaseline:
    """Logistic regression on SBERT embedding-diff vectors (precomputed cache)."""

    name = "sbert"

    def __init__(self):
        self.pipeline = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        )

    def fit(self, train_cache_path: str):
        X, y, _ = _build_sbert_diffs(train_cache_path)
        self.pipeline.fit(X, y)
        return self

    def evaluate(self, cache_path: str) -> BaselineResult:
        X, y_true, doc_lengths = _build_sbert_diffs(cache_path)
        y_pred = self.pipeline.predict(X).tolist()
        return BaselineResult(
            name=self.name,
            y_true=y_true,
            y_pred=y_pred,
            doc_true=_split_by_doc(y_true, doc_lengths),
            doc_pred=_split_by_doc(y_pred, doc_lengths),
        )


class FusedBaseline:
    """Logistic regression on concatenated [stylometric_diff, sbert_diff] vectors."""

    name = "fused"

    def __init__(self):
        self.extractor = StylometricExtractor()
        self.pipeline = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        )

    def _build(self, docs: List[Document], cache_path: str):
        import torch
        cache = torch.load(cache_path)

        X_rows, y, doc_lengths = [], [], []

        for doc, emb, labels in zip(docs, cache["embeddings"], cache["labels"]):
            emb = emb.numpy() if hasattr(emb, "numpy") else np.asarray(emb)
            n_pairs = min(len(doc.sentences) - 1, emb.shape[0] - 1)
            doc_lengths.append(max(n_pairs, 0))

            if n_pairs <= 0:
                continue

            vecs = [self.extractor.extract_vector(s) for s in doc.sentences]

            for i in range(n_pairs):
                style_diff = np.abs(np.asarray(vecs[i + 1]) - np.asarray(vecs[i]))
                sbert_diff = np.abs(emb[i + 1] - emb[i])
                X_rows.append(np.concatenate([style_diff, sbert_diff]))
                y.append(int(labels[i]))

        X = np.vstack(X_rows) if X_rows else np.zeros((0, 0))
        return X, y, doc_lengths

    def fit(self, train_docs: List[Document], train_cache_path: str):
        X, y, _ = self._build(train_docs, train_cache_path)
        self.pipeline.fit(X, y)
        return self

    def evaluate(self, docs: List[Document], cache_path: str) -> BaselineResult:
        X, y_true, doc_lengths = self._build(docs, cache_path)
        y_pred = self.pipeline.predict(X).tolist()
        return BaselineResult(
            name=self.name,
            y_true=y_true,
            y_pred=y_pred,
            doc_true=_split_by_doc(y_true, doc_lengths),
            doc_pred=_split_by_doc(y_pred, doc_lengths),
        )