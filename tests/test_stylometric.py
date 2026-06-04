"""
Unit tests for Phase 1: Stylometric Feature Extractor.

Run with:
    pytest tests/test_stylometric.py -v

No dataset required — all tests use synthetic sentences.
"""

import math
import pytest
import numpy as np

# ── import path setup (works whether run from root or tests/) ──────────────
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.features.surface_features import extract_surface_features
from src.features.readability import extract_readability_features
from src.features.function_words import (
    extract_function_word_features,
    FUNCTION_WORDS_UNIQUE,
    FUNCTION_WORD_SET,
)
from src.features.stylometric import StylometricExtractor


# ===========================================================================
# Fixtures
# ===========================================================================

SIMPLE_A = "The cat sat on the mat and it was very comfortable."
SIMPLE_B = "Scientists discovered a new exoplanet with extraordinary atmospheric conditions."
FORMAL   = "Furthermore, the aforementioned methodology demonstrates considerable efficacy in ameliorating systemic inefficiencies."
CASUAL   = "lol yeah ok sure whatever i guess its fine haha"
EMPTY    = ""
SINGLE   = "Hi."


# ===========================================================================
# Surface Features
# ===========================================================================

class TestSurfaceFeatures:

    def test_keys_present(self):
        feats = extract_surface_features(SIMPLE_A)
        expected_keys = [
            "char_count", "word_count", "avg_word_length", "type_token_ratio",
            "avg_sentence_length", "punct_comma", "punct_period", "punct_exclaim",
            "punct_question", "punct_semicolon", "punct_colon", "punct_dash",
            "uppercase_ratio", "digit_ratio",
        ]
        for k in expected_keys:
            assert k in feats, f"Missing key: {k}"

    def test_word_count_correct(self):
        feats = extract_surface_features("Hello world this is a test")
        assert feats["word_count"] == 6.0

    def test_type_token_ratio_range(self):
        feats = extract_surface_features(SIMPLE_A)
        assert 0.0 <= feats["type_token_ratio"] <= 1.0

    def test_ttr_unique_words(self):
        # All unique words → TTR should be 1.0
        feats = extract_surface_features("dog cat bird fish snake")
        assert feats["type_token_ratio"] == pytest.approx(1.0)

    def test_ttr_repeated_words(self):
        # All same word → TTR should be ~1/N → low
        feats = extract_surface_features("the the the the the")
        assert feats["type_token_ratio"] < 0.5

    def test_punctuation_counts(self):
        text = "Hello, world! How are you? Fine; very fine: indeed."
        feats = extract_surface_features(text)
        word_count = feats["word_count"]
        assert feats["punct_comma"]     == pytest.approx(1 / word_count)
        assert feats["punct_exclaim"]   == pytest.approx(1 / word_count)
        assert feats["punct_question"]  == pytest.approx(1 / word_count)
        assert feats["punct_semicolon"] == pytest.approx(1 / word_count)
        assert feats["punct_colon"]     == pytest.approx(1 / word_count)

    def test_uppercase_ratio(self):
        feats = extract_surface_features("HELLO world")
        # "HELLO" = 5 uppercase, "world" = 5 lowercase → ratio = 0.5
        assert feats["uppercase_ratio"] == pytest.approx(0.5)

    def test_empty_string(self):
        feats = extract_surface_features(EMPTY)
        assert all(v == 0.0 for v in feats.values())

    def test_all_floats(self):
        feats = extract_surface_features(SIMPLE_A)
        assert all(isinstance(v, float) for v in feats.values())


# ===========================================================================
# Readability Features
# ===========================================================================

class TestReadabilityFeatures:

    def test_keys_present(self):
        feats = extract_readability_features(SIMPLE_A)
        for k in ["flesch_reading_ease", "gunning_fog", "coleman_liau",
                  "avg_syllables_per_word", "complex_word_ratio"]:
            assert k in feats

    def test_flesch_range(self):
        feats = extract_readability_features(SIMPLE_A)
        assert 0.0 <= feats["flesch_reading_ease"] <= 100.0

    def test_simpler_text_higher_flesch(self):
        simple = extract_readability_features("The cat sat on the mat.")
        complex_ = extract_readability_features(FORMAL)
        assert simple["flesch_reading_ease"] > complex_["flesch_reading_ease"]

    def test_complex_word_ratio_range(self):
        feats = extract_readability_features(SIMPLE_A)
        assert 0.0 <= feats["complex_word_ratio"] <= 1.0

    def test_avg_syllables_positive(self):
        feats = extract_readability_features(SIMPLE_A)
        assert feats["avg_syllables_per_word"] > 0.0

    def test_empty_string(self):
        feats = extract_readability_features(EMPTY)
        assert all(v == 0.0 for v in feats.values())

    def test_formal_higher_fog(self):
        simple = extract_readability_features("I like cats and dogs very much.")
        formal = extract_readability_features(FORMAL)
        assert formal["gunning_fog"] > simple["gunning_fog"]


# ===========================================================================
# Function Word Features
# ===========================================================================

class TestFunctionWordFeatures:

    def test_keys_present(self):
        feats = extract_function_word_features(SIMPLE_A)
        assert "fw_total_ratio" in feats
        assert "fw_entropy" in feats
        # Check a few spot function words
        for fw in ["the", "and", "it", "was"]:
            assert f"fw_{fw}" in feats

    def test_function_word_counts(self):
        # "the" appears twice in SIMPLE_A, function words total should be nonzero
        feats = extract_function_word_features(SIMPLE_A)
        assert feats["fw_the"] > 0.0
        assert feats["fw_total_ratio"] > 0.0

    def test_entropy_positive(self):
        feats = extract_function_word_features(SIMPLE_A)
        assert feats["fw_entropy"] >= 0.0

    def test_no_function_words(self):
        # Invented words that aren't in our function word list
        feats = extract_function_word_features("Zorp fliggle wumbo snazzle blurp")
        assert feats["fw_total_ratio"] == 0.0
        assert feats["fw_entropy"] == 0.0

    def test_all_function_words_zero_to_one(self):
        feats = extract_function_word_features(SIMPLE_A)
        fw_freqs = [v for k, v in feats.items() if k.startswith("fw_") and k not in ("fw_total_ratio", "fw_entropy")]
        assert all(0.0 <= v <= 1.0 for v in fw_freqs)

    def test_empty_string(self):
        feats = extract_function_word_features(EMPTY)
        assert all(v == 0.0 for v in feats.values())

    def test_function_word_list_no_duplicates(self):
        assert len(FUNCTION_WORDS_UNIQUE) == len(set(FUNCTION_WORDS_UNIQUE))

    def test_function_word_set_matches_list(self):
        assert set(FUNCTION_WORDS_UNIQUE) == FUNCTION_WORD_SET


# ===========================================================================
# StylometricExtractor (no-POS mode to avoid spaCy requirement in CI)
# ===========================================================================

class TestStylometricExtractor:

    @pytest.fixture
    def extractor(self):
        return StylometricExtractor(use_pos=False)

    def test_extract_returns_dict(self, extractor):
        feats = extractor.extract(SIMPLE_A)
        assert isinstance(feats, dict)
        assert len(feats) > 0

    def test_extract_vector_shape(self, extractor):
        vec = extractor.extract_vector(SIMPLE_A)
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (extractor.feature_dim,)
        assert vec.dtype == np.float32

    def test_feature_dim_consistent(self, extractor):
        vec_a = extractor.extract_vector(SIMPLE_A)
        vec_b = extractor.extract_vector(SIMPLE_B)
        assert vec_a.shape == vec_b.shape

    def test_extract_diff_shapes(self, extractor):
        vec_a, vec_b, diff = extractor.extract_diff(SIMPLE_A, SIMPLE_B)
        D = extractor.feature_dim
        assert vec_a.shape == (D,)
        assert vec_b.shape == (D,)
        assert diff.shape == (D,)

    def test_diff_is_absolute(self, extractor):
        _, _, diff = extractor.extract_diff(SIMPLE_A, SIMPLE_B)
        assert np.all(diff >= 0.0), "Diff should be non-negative (absolute)"

    def test_diff_same_text_is_zero(self, extractor):
        _, _, diff = extractor.extract_diff(SIMPLE_A, SIMPLE_A)
        assert np.allclose(diff, 0.0), "Diff of identical texts should be zero"

    def test_diff_different_styles(self, extractor):
        _, _, diff = extractor.extract_diff(FORMAL, CASUAL)
        assert np.sum(diff) > 0.0, "Diff of stylistically different texts should be nonzero"

    def test_extract_document_shapes(self, extractor):
        sentences = [SIMPLE_A, SIMPLE_B, FORMAL, CASUAL]
        vectors, diffs = extractor.extract_document(sentences)
        D = extractor.feature_dim
        assert vectors.shape == (4, D)
        assert diffs.shape == (3, D)

    def test_extract_document_single_sentence(self, extractor):
        vectors, diffs = extractor.extract_document([SIMPLE_A])
        assert vectors.shape[0] == 1
        assert diffs.shape[0] == 0  # No pairs → no diffs

    def test_extract_document_empty(self, extractor):
        vectors, diffs = extractor.extract_document([])
        assert vectors.shape[0] == 0
        assert diffs.shape[0] == 0

    def test_feature_names_length_matches_dim(self, extractor):
        assert len(extractor.feature_names) == extractor.feature_dim

    def test_no_nan_in_vector(self, extractor):
        vec = extractor.extract_vector(SIMPLE_A)
        assert not np.any(np.isnan(vec)), "Feature vector should not contain NaN"

    def test_no_inf_in_vector(self, extractor):
        vec = extractor.extract_vector(SIMPLE_A)
        assert not np.any(np.isinf(vec)), "Feature vector should not contain Inf"

    def test_empty_string_gives_zero_vector(self, extractor):
        vec = extractor.extract_vector(EMPTY)
        assert np.allclose(vec, 0.0)

    def test_all_values_finite(self, extractor):
        for text in [SIMPLE_A, SIMPLE_B, FORMAL, CASUAL, SINGLE]:
            vec = extractor.extract_vector(text)
            assert np.all(np.isfinite(vec)), f"Non-finite values for: {text!r}"


# ===========================================================================
# Integration: style-change detection signal
# ===========================================================================

class TestStyleChangeSignal:
    """
    Sanity-check that the diff vector actually captures style changes.
    This is the core hypothesis of Phase 1.
    """

    @pytest.fixture
    def extractor(self):
        return StylometricExtractor(use_pos=False)

    def test_cross_style_diff_larger_than_same_style(self, extractor):
        """
        Difference between casual and formal text should be larger than
        difference between two formal texts.
        """
        formal_a = "The investigation revealed numerous systemic deficiencies in the regulatory framework."
        formal_b = "Consequently, the committee proposed substantial amendments to existing protocols."
        casual_a = "omg this is so bad lol i cant believe it happened"
        casual_b   = "yeah same i was like wtf when i heard about it haha"

        _, _, diff_same_formal = extractor.extract_diff(formal_a, formal_b)
        _, _, diff_cross       = extractor.extract_diff(formal_a, casual_a)
        _, _, diff_same_casual = extractor.extract_diff(casual_a, casual_b)

        norm_same_formal = float(np.linalg.norm(diff_same_formal))
        norm_cross       = float(np.linalg.norm(diff_cross))
        norm_same_casual = float(np.linalg.norm(diff_same_casual))

        print(f"\nDiff norms — same-formal: {norm_same_formal:.3f}, "
              f"cross: {norm_cross:.3f}, same-casual: {norm_same_casual:.3f}")

        assert norm_cross > norm_same_formal, (
            "Cross-style diff should be larger than same-style diff (formal pair)"
        )

    def test_document_boundary_signal(self, extractor):
        """
        In a document with a known style boundary at position 2,
        the diff at position 2 (boundary) should be the largest.
        """
        doc = [
            "The methodology employed rigorous statistical analysis to validate the hypothesis.",
            "Furthermore, the experimental design incorporated multiple control variables.",
            # ← BOUNDARY HERE ←
            "lol ok so basically i just winged it and hoped for the best tbh",
            "yeah same i never really understood what we were supposed to do",
        ]
        _, diffs = extractor.extract_document(doc)
        # diffs[i] = |vec[i+1] - vec[i]|
        # Boundary is between sentence 1 and 2 → diffs[1]
        norms = [float(np.linalg.norm(d)) for d in diffs]
        print(f"\nDiff norms per position: {[f'{n:.3f}' for n in norms]}")
        boundary_idx = np.argmax(norms)
        assert boundary_idx == 1, (
            f"Expected boundary at diff[1], got diff[{boundary_idx}]. "
            f"Norms: {norms}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])