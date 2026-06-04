"""
Readability features — formula-based, CPU only, no model dependencies.
Implements Flesch Reading Ease, Gunning Fog, and Coleman-Liau.
"""

import re
import string
from typing import Dict


# ---------------------------------------------------------------------------
# Syllable counter (approximation — avoids needing CMU dict)
# ---------------------------------------------------------------------------

def _count_syllables(word: str) -> int:
    """Approximate syllable count for an English word."""
    word = word.lower().strip(string.punctuation)
    if not word:
        return 0
    # Remove trailing 'e' (usually silent)
    if word.endswith('e') and len(word) > 2:
        word = word[:-1]
    # Count vowel groups
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    return max(1, count)


def _count_complex_words(words: list) -> int:
    """Words with 3+ syllables (used by Gunning Fog)."""
    return sum(1 for w in words if _count_syllables(w) >= 3)


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

def extract_readability_features(text: str) -> Dict[str, float]:
    """
    Compute readability scores for a text string.

    Features:
        - flesch_reading_ease: 0-100, higher = easier
        - gunning_fog: grade level, lower = simpler
        - coleman_liau: grade level estimate
        - avg_syllables_per_word: mean syllable count
        - complex_word_ratio: fraction of words with 3+ syllables
    """
    if not text or not text.strip():
        return _zero_features()

    # Tokenise
    words = text.split()
    words_clean = [w.strip(string.punctuation) for w in words]
    words_clean = [w for w in words_clean if w]
    word_count = len(words_clean)

    if word_count == 0:
        return _zero_features()

    # Sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = max(1, len(sentences))

    # Syllables
    syllable_counts = [_count_syllables(w) for w in words_clean]
    total_syllables = sum(syllable_counts)
    avg_syllables = total_syllables / word_count

    # Characters (letters only, no spaces/punct) — for Coleman-Liau
    char_count = sum(len(w) for w in words_clean)

    # Complex words
    complex_count = _count_complex_words(words_clean)
    complex_ratio = complex_count / word_count

    # ------------------------------------------------------------------
    # Flesch Reading Ease
    # FRE = 206.835 - 1.015*(words/sentences) - 84.6*(syllables/words)
    # ------------------------------------------------------------------
    fre = (
        206.835
        - 1.015 * (word_count / sentence_count)
        - 84.6 * (total_syllables / word_count)
    )
    fre = max(0.0, min(100.0, fre))  # clamp to [0, 100]

    # ------------------------------------------------------------------
    # Gunning Fog Index
    # GF = 0.4 * ((words/sentences) + 100*(complex/words))
    # ------------------------------------------------------------------
    gf = 0.4 * ((word_count / sentence_count) + 100 * complex_ratio)

    # ------------------------------------------------------------------
    # Coleman-Liau Index
    # CLI = 0.0588*L - 0.296*S - 15.8
    # L = avg letters per 100 words, S = avg sentences per 100 words
    # ------------------------------------------------------------------
    L = (char_count / word_count) * 100
    S = (sentence_count / word_count) * 100
    cli = 0.0588 * L - 0.296 * S - 15.8

    return {
        "flesch_reading_ease":  fre,
        "gunning_fog":          gf,
        "coleman_liau":         cli,
        "avg_syllables_per_word": avg_syllables,
        "complex_word_ratio":   complex_ratio,
    }


def _zero_features() -> Dict[str, float]:
    return {
        "flesch_reading_ease":    0.0,
        "gunning_fog":            0.0,
        "coleman_liau":           0.0,
        "avg_syllables_per_word": 0.0,
        "complex_word_ratio":     0.0,
    }