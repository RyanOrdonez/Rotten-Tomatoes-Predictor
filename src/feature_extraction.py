"""Extract numeric features from screenplay text.

Ports the extract_features() logic from the original Jupyter notebook
so that features stay consistent with the training data.
"""

import re

import spacy
from textstat import flesch_kincaid_grade

# Load spaCy model once at module level
nlp = spacy.load("en_core_web_sm")

FEATURE_NAMES = [
    "script_length",
    "avg_sentence_length",
    "readability_score",
    "main_character_count",
    "dialogue_ratio",
]


def _spacy_tokenize_sentences(text: str) -> list[str]:
    doc = nlp(text)
    return [sent.text for sent in doc.sents]


def _spacy_tokenize_words(text: str) -> list[str]:
    doc = nlp(text)
    return [token.text for token in doc if not token.is_space]


def extract_features(text: str) -> dict[str, float]:
    """Extract the 5 numeric features used by the prediction model.

    Parameters
    ----------
    text : str
        Raw screenplay / script-format text.

    Returns
    -------
    dict with keys matching FEATURE_NAMES.
    """
    # Speaker names in uppercase, center-aligned (screenplay convention)
    speakers = re.findall(r"\n\s{10,}([A-Z][A-Z\s]+)\n", text)

    # Dialogue lines are typically indented
    dialogue_lines = re.findall(r"\n\s{10,}.+?\n", text)

    # Tokenize
    sentences = _spacy_tokenize_sentences(text)
    words = _spacy_tokenize_words(text)

    try:
        readability = flesch_kincaid_grade(text)
    except Exception:
        readability = 0.0

    num_words = len(words)
    num_sentences = max(1, len(sentences))

    return {
        "script_length": float(num_words),
        "dialogue_ratio": len(dialogue_lines) / num_sentences,
        "main_character_count": float(len(set(speakers))),
        "avg_sentence_length": num_words / num_sentences,
        "readability_score": float(readability),
    }
