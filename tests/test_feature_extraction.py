"""Tests for src.feature_extraction."""

import pytest

from src.feature_extraction import FEATURE_NAMES, extract_features

SAMPLE_SCREENPLAY = """
FADE IN:

INT. UNIVERSITY OFFICE - DAY

A dusty office filled with ancient maps and artifacts. PROFESSOR JONES, 50s,
sits behind a desk covered in papers.

              PROFESSOR JONES
    The artifact was last seen in Cairo,
    nearly forty years ago.

              MARCUS
    You can't be serious about going
    after it yourself.

              PROFESSOR JONES
    Someone has to. And I know those
    tombs better than anyone alive.

EXT. AIRPORT RUNWAY - MORNING

A small propeller plane idles on the tarmac. Jones walks toward it carrying
a worn leather satchel.

              PILOT
    Where to, Professor?

              PROFESSOR JONES
    Cairo. And step on it.

INT. CAIRO MARKETPLACE - DAY

Bustling streets filled with vendors. Jones pushes through the crowd.

              SALLAH
    My friend! I did not expect to see
    you here so soon.

              PROFESSOR JONES
    The situation has changed, Sallah.
    We need to move quickly.
"""


def test_extract_features_returns_all_keys():
    features = extract_features(SAMPLE_SCREENPLAY)
    for name in FEATURE_NAMES:
        assert name in features, f"Missing feature: {name}"


def test_extract_features_types():
    features = extract_features(SAMPLE_SCREENPLAY)
    for name in FEATURE_NAMES:
        assert isinstance(features[name], float), f"{name} should be float"


def test_extract_features_positive_values():
    features = extract_features(SAMPLE_SCREENPLAY)
    assert features["script_length"] > 0
    assert features["avg_sentence_length"] > 0


def test_extract_features_empty_string():
    features = extract_features("")
    for name in FEATURE_NAMES:
        assert name in features


def test_extract_features_detects_characters():
    features = extract_features(SAMPLE_SCREENPLAY)
    # The sample has PROFESSOR JONES, MARCUS, PILOT, SALLAH
    assert features["main_character_count"] >= 2


def test_extract_features_detects_dialogue():
    features = extract_features(SAMPLE_SCREENPLAY)
    assert features["dialogue_ratio"] > 0
