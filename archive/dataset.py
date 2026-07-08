"""
PyTorch Dataset and collation for SASS-TAD training.

Handles variable-length documents by padding to the longest
document in each batch, with masks for valid positions.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Optional, Tuple
import numpy as np

from src.utils.data_loader import Document
from src.features.stylometric import StylometricExtractor


class SASTADDataset(Dataset):
    """
    Dataset that pairs precomputed SBERT embeddings with
    on-the-fly stylometric features for each document.

    Args:
        docs: list of Document objects
        sbert_embeddings: list of tensors, shape (n_sents, 384) per doc
        style_extractor: StylometricExtractor instance (shared, extracts on init)
        topic_labels: optional list of per-sentence topic IDs
    """
    def __init__(
        self,
        docs: List[Document],
        sbert_embeddings: List[torch.Tensor],
        style_extractor: Optional[StylometricExtractor] = None,
        precomputed_style: Optional[List[np.ndarray]] = None,
        topic_labels: Optional[List[List[int]]] = None,
    ):
        assert len(docs) == len(sbert_embeddings)
        self.docs = docs
        self.sbert_embeddings = sbert_embeddings
        self.topic_labels = topic_labels

        if precomputed_style is not None:
            self.style_vectors = precomputed_style
        elif style_extractor is not None:
            self.style_vectors = []
            for doc in docs:
                vecs = np.stack([style_extractor.extract_vector(s) for s in doc.sentences])
                self.style_vectors.append(vecs)
        else:
            raise ValueError("Provide either style_extractor or precomputed_style")

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        doc = self.docs[idx]
        sbert = self.sbert_embeddings[idx]  # (N, 384)
        style = torch.tensor(self.style_vectors[idx], dtype=torch.float32)  # (N, style_dim)

        boundary = torch.tensor(doc.boundary_labels or [], dtype=torch.float32)
        n_sents = len(doc.sentences)

        item = {
            "sbert": sbert,
            "style": style,
            "boundary_labels": boundary,
            "length": n_sents,
            "doc_id": doc.doc_id,
        }

        if self.topic_labels is not None:
            item["topic_labels"] = torch.tensor(self.topic_labels[idx], dtype=torch.long)

        return item


def collate_fn(batch):
    """
    Collates variable-length documents into padded tensors.
    Returns masks for valid positions.
    """
    max_sents = max(item["length"] for item in batch)
    B = len(batch)
    sbert_dim = batch[0]["sbert"].shape[-1]
    style_dim = batch[0]["style"].shape[-1]

    sbert_padded = torch.zeros(B, max_sents, sbert_dim)
    style_padded = torch.zeros(B, max_sents, style_dim)
    boundary_padded = torch.zeros(B, max_sents - 1)
    sentence_mask = torch.zeros(B, max_sents, dtype=torch.bool)
    boundary_mask = torch.zeros(B, max_sents - 1, dtype=torch.bool)
    lengths = torch.zeros(B, dtype=torch.long)

    has_topics = "topic_labels" in batch[0]
    if has_topics:
        topic_padded = torch.zeros(B, max_sents, dtype=torch.long)

    doc_ids = []

    for i, item in enumerate(batch):
        n = item["length"]
        lengths[i] = n
        sbert_padded[i, :n] = item["sbert"][:n]
        style_padded[i, :n] = item["style"][:n]
        sentence_mask[i, :n] = True

        n_boundaries = min(len(item["boundary_labels"]), max_sents - 1)
        if n_boundaries > 0:
            boundary_padded[i, :n_boundaries] = item["boundary_labels"][:n_boundaries]
            boundary_mask[i, :n_boundaries] = True

        if has_topics:
            topic_padded[i, :n] = item["topic_labels"][:n]

        doc_ids.append(item["doc_id"])

    result = {
        "sbert": sbert_padded,
        "style": style_padded,
        "boundary_labels": boundary_padded,
        "boundary_mask": boundary_mask,
        "sentence_mask": sentence_mask,
        "lengths": lengths,
        "doc_ids": doc_ids,
    }

    if has_topics:
        result["topic_labels"] = topic_padded

    return result
