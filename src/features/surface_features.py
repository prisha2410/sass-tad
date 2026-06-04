"""
Surface and punctuation features.
Zero external dependencies — pure Python string ops.
"""

import re
import string
from typing import Dict


def extract_surface_features(text: str) -> Dict[str, float]:
    """
    Extract surface-level and punctuation features from a text string.

    Features:
        - char_count: total character count
        - word_count: total word count
        - avg_word_length: mean characters per word
        - type_token_ratio: unique words / total words (lexical diversity)
        - avg_sentence_length: mean words per sentence
        - punct_comma: comma frequency per word
        - punct_period: period frequency per word
        - punct_exclaim: exclamation mark frequency per word
        - punct_question: question mark frequency per word
        - punct_semicolon: semicolon frequency per word
        - punct_colon: colon frequency per word
        - punct_dash: dash/hyphen frequency per word
        - uppercase_ratio: proportion of uppercase letters
        - digit_ratio: proportion of digit characters
    """
    if not text or not text.strip():
        return _zero_features()

    words = text.split()
    word_count = len(words)

    if word_count == 0:
        return _zero_features()

    # Basic counts
    char_count = len(text)
    avg_word_length = sum(len(w.strip(string.punctuation)) for w in words) / word_count

    # Type-token ratio (lowercase for fair comparison)
    tokens_lower = [w.lower().strip(string.punctuation) for w in words]
    tokens_lower = [t for t in tokens_lower if t]  # remove empties
    ttr = len(set(tokens_lower)) / len(tokens_lower) if tokens_lower else 0.0

    # Sentence-level: split on . ! ?
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if sentences:
        sent_word_counts = [len(s.split()) for s in sentences]
        avg_sentence_length = sum(sent_word_counts) / len(sent_word_counts)
    else:
        avg_sentence_length = float(word_count)

    # Punctuation frequencies (per word, to normalise for length)
    punct_comma     = text.count(',')  / word_count
    punct_period    = text.count('.')  / word_count
    punct_exclaim   = text.count('!')  / word_count
    punct_question  = text.count('?')  / word_count
    punct_semicolon = text.count(';')  / word_count
    punct_colon     = text.count(':')  / word_count
    punct_dash      = (text.count('-') + text.count('—')) / word_count

    # Character-level ratios
    letters = [c for c in text if c.isalpha()]
    uppercase_ratio = sum(1 for c in letters if c.isupper()) / len(letters) if letters else 0.0
    digit_ratio = sum(1 for c in text if c.isdigit()) / char_count if char_count else 0.0

    return {
        "char_count":           float(char_count),
        "word_count":           float(word_count),
        "avg_word_length":      avg_word_length,
        "type_token_ratio":     ttr,
        "avg_sentence_length":  avg_sentence_length,
        "punct_comma":          punct_comma,
        "punct_period":         punct_period,
        "punct_exclaim":        punct_exclaim,
        "punct_question":       punct_question,
        "punct_semicolon":      punct_semicolon,
        "punct_colon":          punct_colon,
        "punct_dash":           punct_dash,
        "uppercase_ratio":      uppercase_ratio,
        "digit_ratio":          digit_ratio,
    }


def _zero_features() -> Dict[str, float]:
    keys = [
        "char_count", "word_count", "avg_word_length", "type_token_ratio",
        "avg_sentence_length", "punct_comma", "punct_period", "punct_exclaim",
        "punct_question", "punct_semicolon", "punct_colon", "punct_dash",
        "uppercase_ratio", "digit_ratio",
    ]
    return {k: 0.0 for k in keys}