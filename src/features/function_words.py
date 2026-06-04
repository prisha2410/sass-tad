"""
Function word frequency features.
Pure vocabulary lookup — no model, no external deps.
"""

import math
import string
from typing import Dict, List

# ---------------------------------------------------------------------------
# Top-150 English function words
# (determiners, prepositions, pronouns, conjunctions, auxiliaries, particles)
# ---------------------------------------------------------------------------

FUNCTION_WORDS: List[str] = [
    # Articles / determiners
    "a", "an", "the", "this", "that", "these", "those", "some", "any",
    "each", "every", "both", "all", "few", "many", "much", "more", "most",
    "other", "another", "such", "what", "which", "whatever", "whichever",
    # Personal pronouns
    "i", "me", "my", "myself", "we", "us", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    # Relative / interrogative pronouns
    "who", "whom", "whose", "which", "that",
    # Prepositions
    "in", "on", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below",
    "from", "up", "down", "out", "off", "over", "under", "again",
    "further", "then", "once", "of", "to", "as",
    # Conjunctions
    "and", "but", "or", "nor", "so", "yet", "for", "because", "since",
    "although", "though", "while", "whereas", "if", "unless", "until",
    "when", "whenever", "where", "wherever", "whether", "than", "after",
    "before", "as", "that",
    # Auxiliary verbs
    "be", "is", "am", "are", "was", "were", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did", "doing",
    "will", "would", "shall", "should", "may", "might", "must",
    "can", "could", "ought",
    # Adverbs / particles
    "not", "no", "never", "ever", "always", "often", "sometimes",
    "here", "there", "now", "then", "how", "very", "just", "also",
    "too", "even", "still", "already", "soon", "only", "almost",
    "enough", "rather", "quite", "perhaps", "maybe",
]

# Deduplicate while preserving order
_seen = set()
FUNCTION_WORDS_UNIQUE: List[str] = []
for _w in FUNCTION_WORDS:
    if _w not in _seen:
        _seen.add(_w)
        FUNCTION_WORDS_UNIQUE.append(_w)

FUNCTION_WORD_SET = set(FUNCTION_WORDS_UNIQUE)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_function_word_features(text: str) -> Dict[str, float]:
    """
    Compute function word frequency features.

    Features:
        - fw_{word}: frequency per token for each function word (150 dims)
        - fw_total_ratio: fraction of tokens that are function words
        - fw_entropy: Shannon entropy over function word distribution
    """
    if not text or not text.strip():
        return _zero_features()

    tokens = [w.lower().strip(string.punctuation) for w in text.split()]
    tokens = [t for t in tokens if t]
    total = len(tokens)

    if total == 0:
        return _zero_features()

    # Count each function word
    counts = {fw: 0 for fw in FUNCTION_WORDS_UNIQUE}
    fw_token_count = 0
    for tok in tokens:
        if tok in FUNCTION_WORD_SET:
            counts[tok] += 1
            fw_token_count += 1

    # Frequencies (per total token count)
    freqs = {fw: counts[fw] / total for fw in FUNCTION_WORDS_UNIQUE}

    # Entropy over function word distribution
    fw_counts_only = [counts[fw] for fw in FUNCTION_WORDS_UNIQUE]
    fw_total = sum(fw_counts_only)
    if fw_total > 0:
        probs = [c / fw_total for c in fw_counts_only if c > 0]
        entropy = -sum(p * math.log2(p) for p in probs)
    else:
        entropy = 0.0

    features = {f"fw_{fw}": freqs[fw] for fw in FUNCTION_WORDS_UNIQUE}
    features["fw_total_ratio"] = fw_token_count / total
    features["fw_entropy"] = entropy

    return features


def get_feature_names() -> List[str]:
    """Return ordered list of feature names (useful for downstream alignment)."""
    names = [f"fw_{fw}" for fw in FUNCTION_WORDS_UNIQUE]
    names += ["fw_total_ratio", "fw_entropy"]
    return names


def _zero_features() -> Dict[str, float]:
    features = {f"fw_{fw}": 0.0 for fw in FUNCTION_WORDS_UNIQUE}
    features["fw_total_ratio"] = 0.0
    features["fw_entropy"] = 0.0
    return features