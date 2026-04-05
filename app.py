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
# Rate limiting — per-user daily limit + global monthly budget cap
# ---------------------------------------------------------------------------
RATE_LIMIT_DIR = Path("/tmp/rt_predictor_rate_limits")
BUDGET_FILE = RATE_LIMIT_DIR / "_monthly_budget.json"
MAX_AI_PREDICTIONS_PER_DAY = 3
MONTHLY_BUDGET_USD = 5.00
EST_COST_PER_PREDICTION = 0.03


def _get_user_ip() -> str:
    try:
        headers = st.context.headers
        ip = headers.get("X-Forwarded-For", headers.get("Remote-Addr", "unknown"))
        ip = ip.split(",")[0].strip()
    except Exception:
        ip = "unknown"
    return hashlib.sha256(ip.encode()).hexdigest()[:16]


def _load_budget() -> dict:
    RATE_LIMIT_DIR.mkdir(parents=True, exist_ok=True)
    current_month = time.strftime("%Y-%m")
    data = {"month": current_month, "spent": 0.0}
    if BUDGET_FILE.exists():
        try:
            data = json.loads(BUDGET_FILE.read_text())
        except Exception:
            pass
    if data.get("month") != current_month:
        data = {"month": current_month, "spent": 0.0}
    return data


def _check_budget() -> bool:
    return _load_budget()["spent"] < MONTHLY_BUDGET_USD


def _record_spend(cost: float):
    data = _load_budget()
    data["spent"] = data.get("spent", 0.0) + cost
    BUDGET_FILE.write_text(json.dumps(data))


def _check_rate_limit() -> bool:
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
    return data["count"] < MAX_AI_PREDICTIONS_PER_DAY


def _increment_usage():
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
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        return os.environ.get("ANTHROPIC_API_KEY")


# ---------------------------------------------------------------------------
# Initialize session state
# ---------------------------------------------------------------------------
if "synopsis" not in st.session_state:
    st.session_state.synopsis = None
if "score" not in st.session_state:
    st.session_state.score = None
if "features" not in st.session_state:
    st.session_state.features = None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("How it works")
    st.markdown(
        """
        1. **Describe** your movie idea in the text box
        2. **AI generates** a polished movie synopsis
        3. **Features** like readability and structure are extracted
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
# Header
# ---------------------------------------------------------------------------
st.title("🍅 Rotten Tomatoes Score Predictor")
st.markdown(
    "Describe a movie idea, AI writes the synopsis, and a trained model "
    "predicts how critics might rate it."
)

st.warning(
    "**Disclaimer:** This prediction is experimental. The model was trained on "
    "739 screenplays and should be treated as a fun demo, not a reliable forecast.",
    icon="⚠️",
)

# ---------------------------------------------------------------------------
# Step 1: Movie idea input + Generate Synopsis
# ---------------------------------------------------------------------------
user_text = st.text_area(
    "Describe your movie idea",
    height=120,
    placeholder=(
        "e.g., A prequel to Indiana Jones where he is a teenage boy "
        "following in his father's footsteps to find lost treasures. "
        "He has a love interest that never pans out, which drives him "
        "to search for a mythical artifact that will win her heart. "
        "Along the way he falls in love with his colleague instead..."
    ),
)

if st.button("✨ Generate Synopsis", type="primary", use_container_width=True):
    if not user_text.strip():
        st.error("Please enter a movie idea first.")
        st.stop()

    api_key = _get_api_key()
    if not api_key:
        st.error("AI is not configured. The app owner needs to set the ANTHROPIC_API_KEY.")
        st.stop()

    if not _check_budget():
        st.error("The monthly AI budget has been reached. Please check back next month!")
        st.stop()

    if not _check_rate_limit():
        st.error(
            f"You've used all {MAX_AI_PREDICTIONS_PER_DAY} AI predictions for today. "
            "Come back tomorrow!"
        )
        st.stop()

    with st.spinner("AI is writing your movie synopsis..."):
        try:
            from src.script_expander import expand_plot_to_synopsis

            st.session_state.synopsis = expand_plot_to_synopsis(user_text, api_key=api_key)
            st.session_state.score = None
            st.session_state.features = None
            _increment_usage()
            _record_spend(EST_COST_PER_PREDICTION)
        except Exception as e:
            st.error(f"Synopsis generation failed: {e}")
            st.stop()

    st.rerun()

# ---------------------------------------------------------------------------
# Step 2: Show synopsis + Predict Score button
# ---------------------------------------------------------------------------
if st.session_state.synopsis:
    st.divider()

    col_synopsis, col_score = st.columns([2, 1])

    with col_synopsis:
        st.subheader("📖 Movie Synopsis")
        st.markdown(st.session_state.synopsis)

        if st.session_state.score is None:
            if st.button("🎬 Predict My Score", type="primary", use_container_width=True):
                with st.spinner("Analyzing and predicting..."):
                    try:
                        from src.feature_extraction import extract_features
                        from src.predictor import predict_score

                        st.session_state.features = extract_features(st.session_state.synopsis)
                        st.session_state.score = predict_score(st.session_state.features)
                    except FileNotFoundError:
                        st.error("Model not found. The app is not configured correctly.")
                        st.stop()
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
                        st.stop()

                st.rerun()

    with col_score:
        if st.session_state.score is not None:
            score = st.session_state.score
            features = st.session_state.features

            # Color-coded score
            if score >= 60:
                color, label = "🟢", "Fresh"
            elif score >= 40:
                color, label = "🟡", "Mixed"
            else:
                color, label = "🔴", "Rotten"

            st.markdown(
                f"<div style='text-align: center; padding-top: 0.5rem;'>"
                f"<h1 style='font-size: 3.5rem; margin-bottom: 0;'>{color}</h1>"
                f"<h1 style='font-size: 3rem; margin-top: 0;'>{score:.0f}%</h1>"
                f"<p style='font-size: 1.2rem; font-weight: bold;'>{label}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

            st.divider()
            st.caption("**Script Analysis**")
            st.metric("Words", f"{features['script_length']:,.0f}")
            st.metric("Readability", f"{features['readability_score']:.1f}")
            st.metric("Avg Sentence", f"{features['avg_sentence_length']:.1f}")
            st.metric("Characters", f"{features['main_character_count']:.0f}")
            st.metric("Dialogue Ratio", f"{features['dialogue_ratio']:.2f}")

    if st.session_state.score is not None:
        st.caption(
            "Score range is 0-100. The model has a margin of error of ~20 points. "
            "This is for entertainment purposes only."
        )
