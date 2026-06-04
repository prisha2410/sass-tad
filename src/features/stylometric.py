"""
StylometricExtractor — main entry point for Phase 1 feature extraction.

Wires together:
  - SurfaceFeatures   (14 dims)
  - ReadabilityFeatures (5 dims)
  - FunctionWordFeatures (~152 dims)
  - POSFeatures        (11 dims)

Total raw vector: ~182 dims per text segment.

For boundary detection the extractor also produces a *difference vector*
(element-wise absolute difference) between two adjacent sentence vectors,
which is what the downstream BiLSTM/classifier actually consumes.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from .surface_features import extract_surface_features
from .readability import extract_readability_features
from .function_words import extract_function_word_features
from .pos_features import extract_pos_features, SPACY_AVAILABLE


class StylometricExtractor:
    """
    Extracts stylometric feature vectors from text segments.

    Args:
        use_pos (bool): Whether to include POS features (requires spaCy).
                        Set False for quick CPU-only runs or when spaCy
                        is not installed. Default: True.
    """

    def __init__(self, use_pos: bool = True):
        self.use_pos = use_pos and SPACY_AVAILABLE
        if use_pos and not SPACY_AVAILABLE:
            import warnings
            warnings.warn(
                "spaCy not available — POS features disabled. "
                "Install with: pip install spacy && python -m spacy download en_core_web_sm",
                RuntimeWarning,
            )
        self._feature_names: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, text: str) -> Dict[str, float]:
        """
        Extract a flat feature dict for a single text segment.

        Args:
            text: Raw text (one sentence or short paragraph).

        Returns:
            Dict mapping feature_name → float value.
        """
        feats: Dict[str, float] = {}
        feats.update(extract_surface_features(text))
        feats.update(extract_readability_features(text))
        feats.update(extract_function_word_features(text))
        if self.use_pos:
            feats.update(extract_pos_features(text))
        return feats

    def extract_vector(self, text: str) -> np.ndarray:
        """
        Extract a numpy feature vector for a single text segment.
        Feature order is determined by self.feature_names.
        """
        feats = self.extract(text)
        names = self.feature_names
        return np.array([feats[n] for n in names], dtype=np.float32)

    def extract_diff(
        self,
        text_a: str,
        text_b: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract feature vectors for two adjacent segments and their
        absolute difference vector.

        Args:
            text_a: First segment (e.g. sentence i).
            text_b: Second segment (e.g. sentence i+1).

        Returns:
            (vec_a, vec_b, diff)  — all numpy float32 arrays of shape (D,)
        """
        vec_a = self.extract_vector(text_a)
        vec_b = self.extract_vector(text_b)
        diff = np.abs(vec_a - vec_b)
        return vec_a, vec_b, diff

    def extract_document(
        self,
        sentences: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features for a full document (list of sentences).

        Args:
            sentences: Ordered list of sentence strings.

        Returns:
            vectors: (N, D) array of per-sentence feature vectors.
            diffs:   (N-1, D) array of consecutive difference vectors.
                     Empty array of shape (0, D) if len(sentences) < 2.
        """
        if not sentences:
            D = len(self.feature_names)
            return np.zeros((0, D), dtype=np.float32), np.zeros((0, D), dtype=np.float32)

        vectors = np.stack([self.extract_vector(s) for s in sentences])  # (N, D)
        if len(sentences) < 2:
            D = vectors.shape[1]
            return vectors, np.zeros((0, D), dtype=np.float32)

        diffs = np.abs(vectors[1:] - vectors[:-1])  # (N-1, D)
        return vectors, diffs

    # ------------------------------------------------------------------
    # Feature name registry
    # ------------------------------------------------------------------

    @property
    def feature_names(self) -> List[str]:
        """Ordered list of feature names. Computed once and cached."""
        if self._feature_names is None:
            self._feature_names = self._build_feature_names()
        return self._feature_names

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)

    def _build_feature_names(self) -> List[str]:
        # Extract from a dummy string to get consistent key order
        dummy = "The quick brown fox jumps over the lazy dog."
        names: List[str] = []
        names += list(extract_surface_features(dummy).keys())
        names += list(extract_readability_features(dummy).keys())
        names += list(extract_function_word_features(dummy).keys())
        if self.use_pos:
            names += list(extract_pos_features(dummy).keys())
        return names