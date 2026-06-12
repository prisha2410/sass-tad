"""
SASS-TAD SBERT Embedding Cache
================================

Precomputes and caches SBERT sentence embeddings to disk.
Run once — loads in milliseconds every time after.

Model: all-MiniLM-L6-v2
    - 384-dim embeddings
    - ~90MB VRAM (frozen, inference only)
    - Fast and accurate for sentence-level tasks

Cache format (per .pt file):
    {
        "doc_ids":    List[str],                    # document identifiers
        "embeddings": List[Tensor(n_sents, 384)],   # one tensor per doc
        "labels":     List[Optional[List[int]]],    # boundary labels
        "meta": {
            "model":   "all-MiniLM-L6-v2",
            "source":  "pan2024",
            "tier":    "hard",
            "split":   "train",
            "n_docs":  int,
        }
    }

Cache filenames:
    pan2024_hard_train.pt
    pan2024_hard_val.pt
    pan2024_hard_test.pt
    muld_ao3_train.pt
    ...

Usage
-----
    # Precompute (run once)
    python scripts/precompute_embeddings.py

    # Load cached embeddings
    from src.utils.sbert_cache import EmbeddingCache
    cache = EmbeddingCache("data/processed/embeddings")
    embeddings, labels, doc_ids = cache.load("pan2024", "hard", "train")
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.utils.data_loader import Document


# --------------------------------------------------------------------------- #
#  Constants
# --------------------------------------------------------------------------- #

SBERT_MODEL   = "all-MiniLM-L6-v2"
EMBED_DIM     = 384
BATCH_SIZE    = 64     # safe for 4GB VRAM with MiniLM


# --------------------------------------------------------------------------- #
#  Encoder
# --------------------------------------------------------------------------- #

class SBERTEncoder:
    """
    Thin wrapper around SentenceTransformer.
    Loads model once, encodes sentence lists in batches.

    Parameters
    ----------
    model_name : str
    device     : 'cuda' | 'cpu' | None (auto-detect)
    batch_size : int
    """

    def __init__(
        self,
        model_name: str = SBERT_MODEL,
        device:     Optional[str] = None,
        batch_size: int = BATCH_SIZE,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device     = device
        self.batch_size = batch_size
        self.model_name = model_name

        print(f"[SBERTEncoder] Loading {model_name} on {device}...")
        self.model = SentenceTransformer(model_name, device=device)
        self.model.eval()
        print(f"[SBERTEncoder] Ready — embedding dim: {self.model.get_sentence_embedding_dimension()}")

    @torch.no_grad()
    def encode_sentences(self, sentences: List[str]) -> torch.Tensor:
        """
        Encode a list of sentences.

        Returns
        -------
        torch.Tensor  shape (n_sentences, embed_dim)  dtype float32  on CPU
        """
        embeddings = self.model.encode(
            sentences,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_tensor=True,
            device=self.device,
            normalize_embeddings=True,   # cosine similarity friendly
        )
        return embeddings.cpu()

    @torch.no_grad()
    def encode_documents(
        self,
        docs: List[Document],
        desc: str = "Encoding",
    ) -> List[torch.Tensor]:
        """
        Encode all sentences across a list of documents.

        Parameters
        ----------
        docs : list of Document
        desc : tqdm description string

        Returns
        -------
        List of tensors, one per document, shape (n_sents_i, embed_dim)
        """
        result = []
        for doc in tqdm(docs, desc=desc, unit="doc"):
            if not doc.sentences:
                result.append(torch.zeros(0, EMBED_DIM))
                continue
            emb = self.encode_sentences(doc.sentences)
            result.append(emb)
        return result


# --------------------------------------------------------------------------- #
#  Cache writer
# --------------------------------------------------------------------------- #

def save_cache(
    path: Path,
    doc_ids:    List[str],
    embeddings: List[torch.Tensor],
    labels:     List[Optional[List[int]]],
    meta:       Dict,
) -> None:
    """Save embeddings to a .pt cache file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "doc_ids":    doc_ids,
        "embeddings": embeddings,
        "labels":     labels,
        "meta":       meta,
    }, path)
    size_mb = path.stat().st_size / 1e6
    print(f"[Cache] Saved {path.name} — {len(doc_ids)} docs, {size_mb:.1f} MB")


def compute_and_save(
    docs:       List[Document],
    cache_path: Path,
    encoder:    SBERTEncoder,
    source:     str,
    tier:       Optional[str],
    split:      str,
) -> None:
    """Encode a list of documents and save to cache."""
    if cache_path.exists():
        print(f"[Cache] Already exists, skipping: {cache_path.name}")
        return

    desc = f"{source} {tier or ''} {split}".strip()
    t0 = time.time()
    embeddings = encoder.encode_documents(docs, desc=desc)
    elapsed = time.time() - t0

    doc_ids = [d.doc_id for d in docs]
    labels  = [d.boundary_labels for d in docs]
    meta    = {
        "model":   encoder.model_name,
        "source":  source,
        "tier":    tier,
        "split":   split,
        "n_docs":  len(docs),
        "elapsed_seconds": round(elapsed, 1),
    }

    save_cache(cache_path, doc_ids, embeddings, labels, meta)


# --------------------------------------------------------------------------- #
#  Cache loader
# --------------------------------------------------------------------------- #

class EmbeddingCache:
    """
    Loads precomputed SBERT embeddings from disk.

    Parameters
    ----------
    cache_dir : str or Path
        Directory containing .pt cache files.

    Example
    -------
        cache = EmbeddingCache("data/processed/embeddings")

        # Load PAN split
        embeddings, labels, doc_ids = cache.load("pan2024", "hard", "train")

        # embeddings: List[Tensor(n_sents, 384)]
        # labels:     List[List[int]]  (boundary labels per doc)
        # doc_ids:    List[str]
    """

    def __init__(self, cache_dir: str | Path = "data/processed/embeddings"):
        self.cache_dir = Path(cache_dir)

    def _cache_path(self, source: str, tier: Optional[str], split: str) -> Path:
        if tier:
            name = f"{source}_{tier}_{split}.pt"
        else:
            name = f"{source}_{split}.pt"
        return self.cache_dir / name

    def exists(self, source: str, tier: Optional[str], split: str) -> bool:
        return self._cache_path(source, tier, split).exists()

    def load(
        self,
        source: str,
        tier:   Optional[str],
        split:  str,
    ) -> Tuple[List[torch.Tensor], List[Optional[List[int]]], List[str]]:
        """
        Load cached embeddings.

        Returns
        -------
        (embeddings, labels, doc_ids)
            embeddings : List[Tensor(n_sents, 384)]
            labels     : List[Optional[List[int]]]
            doc_ids    : List[str]
        """
        path = self._cache_path(source, tier, split)
        if not path.exists():
            raise FileNotFoundError(
                f"Cache not found: {path}\n"
                f"Run: python scripts/precompute_embeddings.py"
            )

        data = torch.load(path, map_location="cpu", weights_only=False)
        return data["embeddings"], data["labels"], data["doc_ids"]

    def load_meta(self, source: str, tier: Optional[str], split: str) -> Dict:
        path = self._cache_path(source, tier, split)
        data = torch.load(path, map_location="cpu", weights_only=False)
        return data["meta"]

    def list_available(self) -> List[str]:
        """List all available cache files."""
        return [p.name for p in sorted(self.cache_dir.glob("*.pt"))]
