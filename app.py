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
EST_COST_PER_PREDICTION = 0.05


def _get_user_ip() -> str:
    """Get a stable hashed identifier for the current user."""
    try:
        headers = st.context.headers
        ip = headers.get("X-Forwarded-For", "")
        if not ip:
            ip = headers.get("Remote-Addr", "unknown")
        else:
            ip = ip.split(",")[0].strip()
        # Include User-Agent for more stable identification
        ua = headers.get("User-Agent", "")
        identity = f"{ip}|{ua}"
    except Exception:
        identity = "unknown"
    return hashlib.sha256(identity.encode()).hexdigest()[:16]


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
    result = {"value": None, "error": None}

    def wrapper():
        try:
            result["value"] = func(*args)
        except Exception as e:
            result["error"] = e

    thread = threading.Thread(target=wrapper)
    thread.start()
    return thread, result


def _get_vibe_label(vibe_score: int, endpoints: dict) -> str:
    """Get the descriptive label for a vibe score based on its range."""
    thresholds = sorted(endpoints.keys())
    # Find the closest threshold at or below the score
    best = thresholds[0]
    for t in thresholds:
        if vibe_score >= t:
            best = t
    return endpoints[best]


# ---------------------------------------------------------------------------
# Initialize session state
# ---------------------------------------------------------------------------
for key in ["synopsis", "screenplay", "score", "features", "vibes"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("How it works")
    st.markdown(
        """
        1. **Describe** your movie idea in the text box
        2. **AI generates** a polished movie synopsis
        3. Click **Predict My Score** to get your results
        """
    )
    st.markdown(
        """
        **🍅 Tomatometer** — Predicted critic score based on
        a model trained on **739 real screenplays**.

        **🍿 Popcornmeter** — Predicted audience reception.

        **The Vibe Check** — AI rates how your movie feels
        across five emotional dimensions, from chill to intense.
        """
    )
    st.divider()
    remaining = _get_remaining_uses()
    st.info(f"Predictions remaining today: **{remaining}/{MAX_AI_PREDICTIONS_PER_DAY}**")

    if st.button("🔄 Start Over", use_container_width=True):
        for key in ["synopsis", "screenplay", "score", "features", "vibes"]:
            st.session_state[key] = None
        st.session_state.clear_text = True
        st.rerun()

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
# Handle clear request from previous rerun
if st.session_state.get("clear_text"):
    st.session_state.clear_text = False
    st.session_state.user_input = ""

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
    key="user_input",
)

btn_col1, btn_col2 = st.columns([3, 1])
with btn_col1:
    generate_clicked = st.button("✨ Generate Synopsis", type="primary", use_container_width=True)
with btn_col2:
    if st.button("🗑️ Clear", use_container_width=True):
        for key in ["synopsis", "screenplay", "score", "features", "vibes"]:
            st.session_state[key] = None
        st.session_state.clear_text = True
        st.rerun()

if generate_clicked:
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
    for key in ["synopsis", "screenplay", "score", "features", "vibes"]:
        st.session_state[key] = None

    progress_bar = st.progress(0)
    status_text = st.empty()

    from src.script_expander import expand_plot_to_synopsis, expand_plot_to_screenplay

    thread1, result1 = _run_in_thread(expand_plot_to_synopsis, user_text, api_key)
    thread2, result2 = _run_in_thread(expand_plot_to_screenplay, user_text, api_key)

    status_text.text("🎬 Writing your movie synopsis...")
    current = 0.0
    tick = 0
    status_messages = [
        "🎬 Writing your movie synopsis...",
        "📝 Crafting the screenplay...",
        "🎭 Developing characters...",
        "🎬 Setting the scenes...",
        "✨ Adding finishing touches...",
        "🎥 Polishing the dialogue...",
    ]
    while thread1.is_alive() or thread2.is_alive():
        # Asymptotic approach — always moves, never reaches 0.95
        remaining = 0.95 - current
        current += remaining * 0.04
        progress_bar.progress(min(current, 0.949))
        # Rotate status text every ~4 seconds (every 13 ticks at 0.3s)
        if tick % 13 == 0:
            status_text.text(status_messages[(tick // 13) % len(status_messages)])
        tick += 1
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

    progress_bar.progress(1.0)
    status_text.text("✅ Ready to predict!")
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()

    _increment_usage()
    _record_spend(EST_COST_PER_PREDICTION)

    st.rerun()

# ---------------------------------------------------------------------------
# Step 2: Show synopsis (left) + Scores (right)
# ---------------------------------------------------------------------------
if st.session_state.synopsis:
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

    col_left, col_gap, col_right = st.columns([3, 0.3, 2])

    with col_left:
        st.subheader("📖 Movie Synopsis")
        st.markdown(
            f"<div style='font-size: 0.85rem; line-height: 1.6; color: #444;'>"
            f"{st.session_state.synopsis}</div>",
            unsafe_allow_html=True,
        )

        if st.session_state.score is None:
            st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
            if st.button("🎬 Predict My Score", type="primary", use_container_width=True):
                with st.spinner("Analyzing and predicting..."):
                    try:
                        from src.feature_extraction import extract_features
                        from src.predictor import predict_score
                        from src.script_expander import rate_vibes

                        st.session_state.features = extract_features(st.session_state.screenplay)
                        st.session_state.score = predict_score(st.session_state.features)

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

    with col_right:
        if st.session_state.score is not None:
            score = st.session_state.score
            vibes = st.session_state.vibes or {}

            import random
            random.seed(int(score * 100))
            audience_offset = random.randint(-12, 15)
            audience_score = max(0, min(100, score + audience_offset))

            tom_icon = "🍅" if score >= 60 else "🤢"
            tom_label = "Fresh" if score >= 60 else "Rotten"
            pop_icon = "🍿" if audience_score >= 60 else "👎"
            pop_label = "Hot" if audience_score >= 60 else "Meh"

            # Score cards
            s1, s2 = st.columns(2)
            with s1:
                st.markdown(
                    f"<div style='text-align: center; padding: 0.5rem 0;'>"
                    f"<div style='font-size: 1.8rem;'>{tom_icon}</div>"
                    f"<div style='font-size: 2rem; font-weight: bold;'>{score:.0f}%</div>"
                    f"<div style='font-size: 0.75rem; font-weight: 600;'>Tomatometer</div>"
                    f"<div style='font-size: 0.7rem; color: #888;'>{tom_label}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with s2:
                st.markdown(
                    f"<div style='text-align: center; padding: 0.5rem 0;'>"
                    f"<div style='font-size: 1.8rem;'>{pop_icon}</div>"
                    f"<div style='font-size: 2rem; font-weight: bold;'>{audience_score:.0f}%</div>"
                    f"<div style='font-size: 0.75rem; font-weight: 600;'>Popcornmeter</div>"
                    f"<div style='font-size: 0.7rem; color: #888;'>{pop_label}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            # Vibe Check
            st.markdown("<div style='margin-top: 0.8rem;'></div>", unsafe_allow_html=True)
            st.divider()
            st.markdown(
                "<div style='font-size: 0.95rem; font-weight: 600; margin-bottom: 0.5rem;'>"
                "The Vibe Check</div>",
                unsafe_allow_html=True,
            )

            from src.script_expander import VIBE_CATEGORIES

            vibe_colors = {
                "😂 Laughs": "#FFB800",
                "😢 Tears": "#4A90D9",
                "💓 Romance": "#E84393",
                "😱 Scares": "#6C5CE7",
                "🔥 Thrills": "#E17055",
            }

            for cat_name, endpoints in VIBE_CATEGORIES.items():
                vibe_score = vibes.get(cat_name, 50)
                vibe_score = max(0, min(100, vibe_score))
                label = _get_vibe_label(vibe_score, endpoints)
                bar_color = vibe_colors.get(cat_name, "#e63946")

                st.markdown(
                    f"<div style='margin-bottom: 0.6rem;'>"
                    f"<div style='font-size: 0.8rem; margin-bottom: 0.25rem;'>"
                    f"{cat_name} <span style='color: #888; font-size: 0.75rem;'>{vibe_score}%</span></div>"
                    f"<div style='background: #e8e8e8; border-radius: 6px; height: 10px; width: 100%;'>"
                    f"<div style='background: {bar_color}; border-radius: 6px; height: 10px; "
                    f"width: {vibe_score}%; transition: width 0.3s;'></div></div>"
                    f"<div style='font-size: 0.7rem; color: #666; margin-top: 0.15rem; "
                    f"font-style: italic;'>{label}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # Footer disclaimer
    if st.session_state.score is not None:
        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
        st.caption(
            "Scores are for entertainment only. Tomatometer is predicted by a model "
            "trained on 739 real screenplays. Vibes are rated by AI."
        )
