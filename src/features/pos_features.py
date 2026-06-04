"""
POS-based stylometric features using spaCy.
Requires: pip install spacy && python -m spacy download en_core_web_sm
"""

import math
from collections import Counter
from typing import Dict, Optional

try:
    import spacy
    _NLP: Optional[object] = None  # lazy-loaded

    def _get_nlp():
        global _NLP
        if _NLP is None:
            _NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        return _NLP

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


# POS tags we care about (Universal POS tagset)
_CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV"}
_ALL_TRACKED = {"NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP",
                "CONJ", "CCONJ", "SCONJ", "AUX", "PART", "INTJ", "NUM"}


def extract_pos_features(text: str) -> Dict[str, float]:
    """
    Extract POS-based features from text.

    Features:
        - pos_noun_ratio: NOUN / total tokens
        - pos_verb_ratio: VERB / total tokens
        - pos_adj_ratio:  ADJ  / total tokens
        - pos_adv_ratio:  ADV  / total tokens
        - pos_pron_ratio: PRON / total tokens
        - pos_det_ratio:  DET  / total tokens
        - pos_adp_ratio:  ADP  / total tokens (prepositions)
        - pos_aux_ratio:  AUX  / total tokens
        - pos_content_ratio: (NOUN+VERB+ADJ+ADV) / total
        - pos_trigram_entropy: Shannon entropy over POS trigrams
        - pos_unique_trigram_ratio: unique trigrams / total trigrams
    """
    if not text or not text.strip():
        return _zero_features()

    if not SPACY_AVAILABLE:
        raise RuntimeError(
            "spaCy not installed. Run: pip install spacy && "
            "python -m spacy download en_core_web_sm"
        )

    nlp = _get_nlp()
    doc = nlp(text)

    tokens = [tok for tok in doc if not tok.is_space]
    total = len(tokens)
    if total == 0:
        return _zero_features()

    pos_counts: Counter = Counter(tok.pos_ for tok in tokens)

    def ratio(tag: str) -> float:
        return pos_counts.get(tag, 0) / total

    content_count = sum(pos_counts.get(t, 0) for t in _CONTENT_POS)

    # POS trigrams
    pos_seq = [tok.pos_ for tok in tokens]
    trigrams = list(zip(pos_seq, pos_seq[1:], pos_seq[2:]))
    trigram_total = len(trigrams)

    if trigram_total > 0:
        tri_counts = Counter(trigrams)
        probs = [c / trigram_total for c in tri_counts.values()]
        trigram_entropy = -sum(p * math.log2(p) for p in probs)
        unique_trigram_ratio = len(tri_counts) / trigram_total
    else:
        trigram_entropy = 0.0
        unique_trigram_ratio = 0.0

    return {
        "pos_noun_ratio":          ratio("NOUN"),
        "pos_verb_ratio":          ratio("VERB"),
        "pos_adj_ratio":           ratio("ADJ"),
        "pos_adv_ratio":           ratio("ADV"),
        "pos_pron_ratio":          ratio("PRON"),
        "pos_det_ratio":           ratio("DET"),
        "pos_adp_ratio":           ratio("ADP"),
        "pos_aux_ratio":           ratio("AUX"),
        "pos_content_ratio":       content_count / total,
        "pos_trigram_entropy":     trigram_entropy,
        "pos_unique_trigram_ratio": unique_trigram_ratio,
    }


def _zero_features() -> Dict[str, float]:
    return {
        "pos_noun_ratio":           0.0,
        "pos_verb_ratio":           0.0,
        "pos_adj_ratio":            0.0,
        "pos_adv_ratio":            0.0,
        "pos_pron_ratio":           0.0,
        "pos_det_ratio":            0.0,
        "pos_adp_ratio":            0.0,
        "pos_aux_ratio":            0.0,
        "pos_content_ratio":        0.0,
        "pos_trigram_entropy":      0.0,
        "pos_unique_trigram_ratio": 0.0,
    }