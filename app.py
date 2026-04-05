"""Rotten Tomatoes Score Predictor — Streamlit Web App.

Run with:
    streamlit run app.py
"""

import hashlib
import json
import os
import time
import threading
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="RT Score Predictor",
    page_icon="🍅",
    layout="centered",
)

# Reduce top padding and tighten layout
st.markdown(
    """
    <style>
    .block-container { padding-top: 1rem; padding-bottom: 0rem; }
    header { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Rate limiting — per-user daily limit + global monthly budget cap
# ---------------------------------------------------------------------------
RATE_LIMIT_DIR = Path("/tmp/rt_predictor_rate_limits")
BUDGET_FILE = RATE_LIMIT_DIR / "_monthly_budget.json"
MAX_AI_PREDICTIONS_PER_DAY = 3
MONTHLY_BUDGET_USD = 5.00
# Two API calls per prediction (synopsis + screenplay)
EST_COST_PER_PREDICTION = 0.05


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


def _run_in_thread(func, *args):
    """Run a function in a thread and return the result."""
    result = {"value": None, "error": None}

    def wrapper():
        try:
            result["value"] = func(*args)
        except Exception as e:
            result["error"] = e

    thread = threading.Thread(target=wrapper)
    thread.start()
    return thread, result


def _animate_progress(bar, status, thread, start_pct, end_pct, label):
    """Smoothly animate progress bar while a thread is running."""
    status.text(label)
    current = start_pct
    step = 0.02
    while thread.is_alive():
        if current < end_pct - 0.02:
            current += step
            # Slow down as we approach the target to avoid hitting it before done
            if current > (start_pct + end_pct) / 2:
                step = 0.005
            bar.progress(min(current, end_pct - 0.01))
        time.sleep(0.3)
    bar.progress(end_pct)


# ---------------------------------------------------------------------------
# Initialize session state
# ---------------------------------------------------------------------------
if "synopsis" not in st.session_state:
    st.session_state.synopsis = None
if "screenplay" not in st.session_state:
    st.session_state.screenplay = None
if "score" not in st.session_state:
    st.session_state.score = None
if "features" not in st.session_state:
    st.session_state.features = None
if "vibes" not in st.session_state:
    st.session_state.vibes = None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("How it works")
    st.markdown(
        """
        1. **Describe** your movie idea in the text box
        2. **AI generates** a polished movie synopsis
        3. **Features** are extracted and analyzed
        4. A **trained model** predicts the Rotten Tomatoes score

        The model was trained on **739 real screenplays** and their actual
        Rotten Tomatoes critic scores.
        """
    )
    st.divider()
    remaining = _get_remaining_uses()
    st.info(f"Predictions remaining today: **{remaining}/{MAX_AI_PREDICTIONS_PER_DAY}**")
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

st.caption(
    "⚠️ This prediction is experimental — trained on 739 screenplays. "
    "For entertainment only."
)

# ---------------------------------------------------------------------------
# Step 1: Movie idea input + Generate Synopsis
# ---------------------------------------------------------------------------
user_text = st.text_area(
    "Describe your movie idea",
    height=150,
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
            f"You've used all {MAX_AI_PREDICTIONS_PER_DAY} predictions for today. "
            "Come back tomorrow!"
        )
        st.stop()

    # Reset previous results
    st.session_state.synopsis = None
    st.session_state.screenplay = None
    st.session_state.score = None
    st.session_state.features = None
    st.session_state.vibes = None

    progress_bar = st.progress(0)
    status_text = st.empty()

    # Generate synopsis AND screenplay in parallel
    from src.script_expander import expand_plot_to_synopsis, expand_plot_to_screenplay

    thread1, result1 = _run_in_thread(expand_plot_to_synopsis, user_text, api_key)
    thread2, result2 = _run_in_thread(expand_plot_to_screenplay, user_text, api_key)

    # Animate progress while both threads run
    status_text.text("🎬 Generating synopsis and analyzing structure...")
    current = 0.0
    step = 0.02
    while thread1.is_alive() or thread2.is_alive():
        if current < 0.88:
            current += step
            if current > 0.5:
                step = 0.005
            progress_bar.progress(min(current, 0.89))
        time.sleep(0.3)

    thread1.join()
    thread2.join()

    if result1["error"]:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Synopsis generation failed: {result1['error']}")
        st.stop()

    if result2["error"]:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Screenplay generation failed: {result2['error']}")
        st.stop()

    st.session_state.synopsis = result1["value"]
    st.session_state.screenplay = result2["value"]

    # Phase 3: Done
    progress_bar.progress(1.0)
    status_text.text("✅ Ready to predict!")
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()

    _increment_usage()
    _record_spend(EST_COST_PER_PREDICTION)

    st.rerun()

# ---------------------------------------------------------------------------
# Step 2: Show synopsis + Predict Score button
# ---------------------------------------------------------------------------
if st.session_state.synopsis:
    st.divider()

    col_synopsis, col_score = st.columns([2, 1])

    with col_synopsis:
        st.subheader("📖 Movie Synopsis")
        st.markdown(
            f"<div style='font-size: 0.9rem; line-height: 1.5;'>{st.session_state.synopsis}</div>",
            unsafe_allow_html=True,
        )

        if st.session_state.score is None:
            if st.button("🎬 Predict My Score", type="primary", use_container_width=True):
                with st.spinner("Analyzing and predicting..."):
                    try:
                        from src.feature_extraction import extract_features
                        from src.predictor import predict_score
                        from src.script_expander import rate_vibes

                        # Extract features from the SCREENPLAY (not synopsis)
                        st.session_state.features = extract_features(st.session_state.screenplay)
                        st.session_state.score = predict_score(st.session_state.features)

                        # Get vibe ratings from Claude
                        api_key = _get_api_key()
                        st.session_state.vibes = rate_vibes(
                            st.session_state.synopsis, api_key=api_key
                        )
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
            vibes = st.session_state.vibes or {}

            # Derive audience score (offset from critic score)
            import random
            random.seed(int(score * 100))
            audience_offset = random.randint(-12, 15)
            audience_score = max(0, min(100, score + audience_offset))

            # --- Tomatometer & Popcornmeter ---
            tom_icon = "🍅" if score >= 60 else "🤢"
            tom_label = "Fresh" if score >= 60 else "Rotten"
            pop_icon = "🍿" if audience_score >= 60 else "👎"
            pop_label = "Hot" if audience_score >= 60 else "Meh"

            score_col1, score_col2 = st.columns(2)
            with score_col1:
                st.markdown(
                    f"<div style='text-align: center;'>"
                    f"<p style='font-size: 2.5rem; margin-bottom: 0;'>{tom_icon}</p>"
                    f"<h2 style='margin: 0;'>{score:.0f}%</h2>"
                    f"<p style='font-size: 0.8rem; margin-top: 0;'><b>Tomatometer</b></p>"
                    f"<p style='font-size: 0.75rem; color: gray;'>{tom_label}</p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with score_col2:
                st.markdown(
                    f"<div style='text-align: center;'>"
                    f"<p style='font-size: 2.5rem; margin-bottom: 0;'>{pop_icon}</p>"
                    f"<h2 style='margin: 0;'>{audience_score:.0f}%</h2>"
                    f"<p style='font-size: 0.8rem; margin-top: 0;'><b>Popcornmeter</b></p>"
                    f"<p style='font-size: 0.75rem; color: gray;'>{pop_label}</p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            # --- Vibe Meters ---
            st.divider()
            st.markdown("**The Vibe Check**")

            from src.script_expander import VIBE_CATEGORIES

            for cat_name, endpoints in VIBE_CATEGORIES.items():
                vibe_score = vibes.get(cat_name, 50)
                vibe_score = max(0, min(100, vibe_score))
                st.caption(f"{cat_name}")
                st.progress(vibe_score / 100)
                # Show the endpoint label based on score
                if vibe_score <= 25:
                    label = endpoints["low"]
                elif vibe_score >= 75:
                    label = endpoints["high"]
                else:
                    label = ""
                if label:
                    st.caption(f"*{label}*")

    if st.session_state.score is not None:
        st.caption(
            "Scores are for entertainment only. Tomatometer is predicted by a model "
            "trained on 739 real screenplays. Vibes are rated by AI."
        )
