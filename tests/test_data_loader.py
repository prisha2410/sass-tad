"""
Unit tests for data_loader.py — synthetic fixtures, no real dataset needed.

Run with:
    pytest tests/test_data_loader.py -v
"""

import json
import pytest
from pathlib import Path

from src.utils.data_loader import (
    Document,
    PANDataset,
    split_documents,
    load_pan_labeled,
    dataset_stats,
)


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def _make_pan_dir(tmp_path, year, tier, n_docs=10):
    """
    Synthetic PAN structure with both train and validation splits.
    train gets 70%, validation gets 30% of n_docs.
    """
    tier_key  = {"easy": "easy", "medium": "medium", "hard": "hard",
                 "pan2023_easy": "dataset1", "pan2023_medium": "dataset2",
                 "pan2023_hard": "dataset3"}.get(tier, tier)

    # PAN 2023 uses dataset1/2/3
    if year == "2023":
        tier_key = {"easy": "dataset1", "medium": "dataset2", "hard": "dataset3"}[tier]

    n_train = max(1, int(n_docs * 0.7))
    n_val   = n_docs - n_train

    for split, count in [("train", n_train), ("validation", n_val)]:
        split_dir = tmp_path / f"pan{year}" / tier_key / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for i in range(1, count + 1):
            sents  = [f"Sentence {j} by author {'A' if j%2==0 else 'B'}." for j in range(1, 5)]
            labels = [0, 1, 0]
            (split_dir / f"problem-{i}.txt").write_text("\n".join(sents), encoding="utf-8")
            (split_dir / f"truth-problem-{i}.json").write_text(
                json.dumps({"authors": 2, "changes": labels}), encoding="utf-8"
            )
    return tmp_path


# --------------------------------------------------------------------------- #
#  Document
# --------------------------------------------------------------------------- #

class TestDocument:

    def _doc(self, sents, labels):
        return Document("d1", sents, labels, 2, "pan2024", "hard")

    def test_len(self):
        assert len(self._doc(["a","b","c"], [0,1])) == 3

    def test_n_boundaries(self):
        assert self._doc(["a","b","c"], [0,1]).n_boundaries == 2

    def test_has_labels_true(self):
        assert self._doc(["a","b"], [0]).has_labels is True

    def test_has_labels_false(self):
        doc = Document("x", ["a","b"], None, None, "pan2024", "easy")
        assert doc.has_labels is False

    def test_n_changes(self):
        assert self._doc(["a","b","c","d"], [0,1,1]).n_changes == 2

    def test_n_changes_no_labels(self):
        doc = Document("x", ["a","b"], None, None, "pan2024", "easy")
        assert doc.n_changes == 0

    def test_repr_contains_id(self):
        doc = Document("my_doc", ["a","b"], [0], 1, "pan2024", "hard")
        assert "my_doc" in repr(doc)

    def test_default_split(self):
        doc = Document("x", ["a"], None, None, "pan2024", "easy")
        assert doc.split == "unsplit"


# --------------------------------------------------------------------------- #
#  PANDataset
# --------------------------------------------------------------------------- #

class TestPANDataset:

    def test_loads_all_labeled_docs(self, tmp_path):
        root = _make_pan_dir(tmp_path, "2024", "hard", n_docs=10)
        docs = PANDataset(root, "2024", "hard").load()
        assert len(docs) == 10

    def test_all_docs_have_labels(self, tmp_path):
        root = _make_pan_dir(tmp_path, "2024", "hard", n_docs=6)
        docs = PANDataset(root, "2024", "hard").load()
        assert all(d.has_labels for d in docs)

    def test_labels_length_correct(self, tmp_path):
        root = _make_pan_dir(tmp_path, "2024", "hard", n_docs=6)
        docs = PANDataset(root, "2024", "hard").load()
        for doc in docs:
            assert len(doc.boundary_labels) == doc.n_boundaries

    def test_source_field(self, tmp_path):
        root = _make_pan_dir(tmp_path, "2025", "easy", n_docs=4)
        docs = PANDataset(root, "2025", "easy").load()
        assert all(d.source == "pan2025" for d in docs)

    def test_pan2023_tier_mapping(self, tmp_path):
        root = _make_pan_dir(tmp_path, "2023", "hard", n_docs=4)
        docs = PANDataset(root, "2023", "hard").load()
        assert len(docs) == 4

    def test_invalid_year_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unsupported year"):
            PANDataset(tmp_path, "2020", "hard")

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            PANDataset(tmp_path / "nonexistent", "2024", "hard").load()

    def test_doc_ids_unique(self, tmp_path):
        root = _make_pan_dir(tmp_path, "2024", "hard", n_docs=8)
        docs = PANDataset(root, "2024", "hard").load()
        ids  = [d.doc_id for d in docs]
        assert len(ids) == len(set(ids))

    def test_all_years_load(self, tmp_path):
        for year in ["2023", "2024", "2025"]:
            root = _make_pan_dir(tmp_path / year, year, "easy", n_docs=4)
            docs = PANDataset(root, year, "easy").load()
            assert len(docs) == 4


# --------------------------------------------------------------------------- #
#  split_documents
# --------------------------------------------------------------------------- #

class TestSplitDocuments:

    def _make_docs(self, n=100):
        return [
            Document(f"d{i}", ["a","b","c"], [0,1], 2, "pan2024", "hard")
            for i in range(n)
        ]

    def test_total_count(self):
        train, val, test = split_documents(self._make_docs(100))
        assert len(train) + len(val) + len(test) == 100

    def test_approximate_ratios(self):
        train, val, test = split_documents(self._make_docs(100))
        assert 65 <= len(train) <= 75
        assert 10 <= len(val)   <= 20
        assert 10 <= len(test)  <= 20

    def test_split_tags_assigned(self):
        train, val, test = split_documents(self._make_docs(60))
        assert all(d.split == "train" for d in train)
        assert all(d.split == "val"   for d in val)
        assert all(d.split == "test"  for d in test)

    def test_no_overlap(self):
        train, val, test = split_documents(self._make_docs(90))
        train_ids = {d.doc_id for d in train}
        val_ids   = {d.doc_id for d in val}
        test_ids  = {d.doc_id for d in test}
        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)

    def test_reproducible_with_seed(self):
        docs = self._make_docs(50)
        t1, v1, e1 = split_documents(docs[:], seed=42)
        t2, v2, e2 = split_documents(docs[:], seed=42)
        assert [d.doc_id for d in t1] == [d.doc_id for d in t2]

    def test_different_seeds_differ(self):
        docs = self._make_docs(50)
        t1, _, _ = split_documents(docs[:], seed=42)
        t2, _, _ = split_documents(docs[:], seed=99)
        assert [d.doc_id for d in t1] != [d.doc_id for d in t2]

    def test_unlabeled_docs_dropped(self):
        docs = self._make_docs(10)
        docs[0].boundary_labels = None   # one unlabeled
        train, val, test = split_documents(docs)
        assert len(train) + len(val) + len(test) == 9

    def test_load_pan_labeled_convenience(self, tmp_path):
        root = _make_pan_dir(tmp_path, "2024", "hard", n_docs=20)
        train, val, test = load_pan_labeled(root=root, year="2024", tier="hard")
        assert len(train) + len(val) + len(test) == 20
        assert all(d.split == "train" for d in train)
        assert all(d.split == "test"  for d in test)


# --------------------------------------------------------------------------- #
#  dataset_stats
# --------------------------------------------------------------------------- #

class TestDatasetStats:

    def test_n_docs(self):
        docs = [Document(f"d{i}", ["a","b","c"], [0,1], 2, "pan2024", "hard") for i in range(5)]
        assert dataset_stats(docs)["n_docs"] == 5

    def test_change_rate_range(self):
        docs = [Document(f"d{i}", ["a","b","c"], [0,1], 2, "pan2024", "hard") for i in range(5)]
        stats = dataset_stats(docs)
        assert 0.0 <= stats["change_rate"] <= 1.0

    def test_empty_list(self):
        assert dataset_stats([]) == {}

    def test_all_unlabeled(self):
        docs = [Document("d1", ["a","b"], None, None, "pan2024", "easy")]
        stats = dataset_stats(docs)
        assert stats["n_labeled"] == 0
        assert stats["change_rate"] == 0.0