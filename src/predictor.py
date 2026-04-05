"""Load the trained model and predict Rotten Tomatoes scores."""

import os

import joblib
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
FEATURE_ORDER = [
    "script_length",
    "avg_sentence_length",
    "readability_score",
    "main_character_count",
    "dialogue_ratio",
]


def _load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "model.joblib"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    return model, scaler


def predict_score(features: dict) -> float:
    """Predict a Rotten Tomatoes score (0-100) from extracted features.

    Parameters
    ----------
    features : dict
        Output of ``feature_extraction.extract_features()``.

    Returns
    -------
    float
        Predicted RT score clamped to [0, 100].
    """
    model, scaler = _load_artifacts()

    X = np.array([[features[col] for col in FEATURE_ORDER]])
    X_scaled = scaler.transform(X)
    # Clip to training distribution range — AI-generated screenplays are much
    # shorter than the real scripts the model trained on, which pushes features
    # (especially script_length) far out of distribution and biases predictions low.
    X_scaled = np.clip(X_scaled, -2, 2)
    prediction = float(model.predict(X_scaled)[0])

    return max(0.0, min(100.0, prediction))
