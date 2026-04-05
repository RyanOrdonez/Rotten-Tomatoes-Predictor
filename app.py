"""Rotten Tomatoes Score Predictor — Streamlit Web App.

Run with:
    streamlit run app.py
"""

import hashlib
import json
import os
import time
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="RT Score Predictor",
    page_icon="🍅",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Rate limiting — track usage by IP, max 3 AI predictions per day
# ---------------------------------------------------------------------------
RATE_LIMIT_DIR = Path("/tmp/rt_predictor_rate_limits")
MAX_AI_PREDICTIONS_PER_DAY = 3


def _get_user_ip() -> str:
    """Get a hashed identifier for the current user."""
    # Streamlit Cloud exposes headers via st.context
    try:
        headers = st.context.headers
        ip = headers.get("X-Forwarded-For", headers.get("Remote-Addr", "unknown"))
        # Take the first IP if there are multiple (proxy chain)
        ip = ip.split(",")[0].strip()
    except Exception:
        ip = "unknown"
    return hashlib.sha256(ip.encode()).hexdigest()[:16]


def _check_rate_limit() -> bool:
    """Return True if the user is within their daily AI prediction limit."""
    RATE_LIMIT_DIR.mkdir(parents=True, exist_ok=True)
    user_hash = _get_user_ip()
    today = time.strftime("%Y-%m-%d")
    limit_file = RATE_LIMIT_DIR / f"{user_hash}.json"

    data = {"date": today, "count": 0}
    if limit_file.exists():
        try:
            data = json.loads(limit_file.read_text())
        except Exception:
            pass

    # Reset count if it's a new day
    if data.get("date") != today:
        data = {"date": today, "count": 0}

    return data["count"] < MAX_AI_PREDICTIONS_PER_DAY


def _increment_usage():
    """Record one AI prediction for the current user."""
    RATE_LIMIT_DIR.mkdir(parents=True, exist_ok=True)
    user_hash = _get_user_ip()
    today = time.strftime("%Y-%m-%d")
    limit_file = RATE_LIMIT_DIR / f"{user_hash}.json"

    data = {"date": today, "count": 0}
    if limit_file.exists():
        try:
            data = json.loads(limit_file.read_text())
        except Exception:
            pass

    if data.get("date") != today:
        data = {"date": today, "count": 0}

    data["count"] += 1
    limit_file.write_text(json.dumps(data))


def _get_remaining_uses() -> int:
    """Return how many AI predictions the user has left today."""
    RATE_LIMIT_DIR.mkdir(parents=True, exist_ok=True)
    user_hash = _get_user_ip()
    today = time.strftime("%Y-%m-%d")
    limit_file = RATE_LIMIT_DIR / f"{user_hash}.json"

    data = {"date": today, "count": 0}
    if limit_file.exists():
        try:
            data = json.loads(limit_file.read_text())
        except Exception:
            pass

    if data.get("date") != today:
        return MAX_AI_PREDICTIONS_PER_DAY

    return max(0, MAX_AI_PREDICTIONS_PER_DAY - data["count"])


def _get_api_key() -> str | None:
    """Get the Anthropic API key from Streamlit secrets or environment."""
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        return os.environ.get("ANTHROPIC_API_KEY")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("How it works")
    st.markdown(
        """
        1. **Describe** your movie idea in the text box
        2. **AI expands** your idea into a screenplay-format script (~2000 words)
        3. **Features** like dialogue ratio, readability, and script length are extracted
        4. A **trained model** predicts the Rotten Tomatoes score

        The model was trained on **739 real screenplays** and their actual
        Rotten Tomatoes critic scores.
        """
    )
    st.divider()
    remaining = _get_remaining_uses()
    st.info(f"AI predictions remaining today: **{remaining}/{MAX_AI_PREDICTIONS_PER_DAY}**")
    st.divider()
    st.caption("Built by Ryan Ordonez")

# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
st.title("🍅 Rotten Tomatoes Score Predictor")
st.markdown(
    "Describe a movie idea and get a predicted Rotten Tomatoes critic score. "
    "AI will expand your idea into a screenplay, then the model predicts how "
    "critics might rate it."
)

st.warning(
    "**Disclaimer:** This prediction is experimental. The model was trained on "
    "739 screenplays and should be treated as a fun demo, not a reliable forecast.",
    icon="⚠️",
)

# ---------------------------------------------------------------------------
# Text input
# ---------------------------------------------------------------------------
user_text = st.text_area(
    "Describe your movie idea",
    height=180,
    placeholder=(
        "e.g., A prequel to Indiana Jones where he is a teenage boy "
        "following in his father's footsteps to find lost treasures. "
        "He has a love interest that never pans out, which drives him "
        "to search for a mythical artifact that will win her heart. "
        "Along the way he falls in love with his colleague instead..."
    ),
)

# ---------------------------------------------------------------------------
# Predict button
# ---------------------------------------------------------------------------
if st.button("🎬 Predict Score", type="primary", use_container_width=True):
    if not user_text.strip():
        st.error("Please enter a movie idea first.")
        st.stop()

    # Check API key
    api_key = _get_api_key()
    if not api_key:
        st.error("AI expansion is not configured. The app owner needs to set the ANTHROPIC_API_KEY.")
        st.stop()

    # Check rate limit
    if not _check_rate_limit():
        st.error(
            f"You've used all {MAX_AI_PREDICTIONS_PER_DAY} AI predictions for today. "
            "Come back tomorrow!"
        )
        st.stop()

    # --- AI Expansion ---
    with st.spinner("Expanding your idea into a screenplay with AI..."):
        try:
            from src.script_expander import expand_plot_to_script

            script_text = expand_plot_to_script(user_text, api_key=api_key)
            _increment_usage()
        except Exception as e:
            st.error(f"AI expansion failed: {e}")
            st.stop()

    with st.expander("View generated screenplay", expanded=False):
        st.text(script_text)

    # --- Feature Extraction ---
    with st.spinner("Extracting script features..."):
        try:
            from src.feature_extraction import extract_features

            features = extract_features(script_text)
        except Exception as e:
            st.error(f"Feature extraction failed: {e}")
            st.stop()

    st.subheader("Extracted Features")
    col1, col2, col3 = st.columns(3)
    col1.metric("Script Length", f"{features['script_length']:,.0f} words")
    col2.metric("Avg Sentence Length", f"{features['avg_sentence_length']:.1f}")
    col3.metric("Readability Score", f"{features['readability_score']:.1f}")

    col4, col5 = st.columns(2)
    col4.metric("Main Characters", f"{features['main_character_count']:.0f}")
    col5.metric("Dialogue Ratio", f"{features['dialogue_ratio']:.2f}")

    # --- Prediction ---
    with st.spinner("Predicting score..."):
        try:
            from src.predictor import predict_score

            score = predict_score(features)
        except FileNotFoundError:
            st.error("Model not found. The app is not configured correctly.")
            st.stop()
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

    # --- Display Score ---
    st.divider()
    st.subheader("Predicted Rotten Tomatoes Score")

    # Color-code the score
    if score >= 60:
        color = "🟢"
        label = "Fresh"
    elif score >= 40:
        color = "🟡"
        label = "Mixed"
    else:
        color = "🔴"
        label = "Rotten"

    st.markdown(
        f"<h1 style='text-align: center; font-size: 4rem;'>"
        f"{color} {score:.0f}%</h1>"
        f"<p style='text-align: center; font-size: 1.3rem;'>{label}</p>",
        unsafe_allow_html=True,
    )

    st.caption(
        "Score range is 0-100. The model has a margin of error of ~20 points. "
        "This is for entertainment purposes only."
    )
