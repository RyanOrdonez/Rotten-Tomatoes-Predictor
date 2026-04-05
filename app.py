"""Rotten Tomatoes Score Predictor — Streamlit Web App.

Run with:
    streamlit run app.py
"""

import os

import streamlit as st

st.set_page_config(
    page_title="RT Score Predictor",
    page_icon="🍅",
    layout="centered",
)

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
# Mode selection
# ---------------------------------------------------------------------------
mode = st.radio(
    "Input mode",
    ["Expand my idea with AI", "Paste a full script"],
    horizontal=True,
)

# ---------------------------------------------------------------------------
# API key (only for AI expansion mode)
# ---------------------------------------------------------------------------
api_key = None
if mode == "Expand my idea with AI":
    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        help="Required for AI expansion. Get one at console.anthropic.com",
    )

# ---------------------------------------------------------------------------
# Text input
# ---------------------------------------------------------------------------
if mode == "Expand my idea with AI":
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
else:
    user_text = st.text_area(
        "Paste screenplay text",
        height=300,
        placeholder="Paste a full screenplay or script here...",
    )

# ---------------------------------------------------------------------------
# Predict button
# ---------------------------------------------------------------------------
if st.button("Predict Score", type="primary", use_container_width=True):
    if not user_text.strip():
        st.error("Please enter some text first.")
        st.stop()

    script_text = user_text

    # --- AI Expansion ---
    if mode == "Expand my idea with AI":
        if not api_key:
            st.error(
                "Please provide an Anthropic API key to use AI expansion, "
                "or switch to 'Paste a full script' mode."
            )
            st.stop()

        with st.spinner("Expanding your idea into a screenplay with AI..."):
            try:
                from src.script_expander import expand_plot_to_script

                script_text = expand_plot_to_script(user_text, api_key=api_key)
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
            st.error(
                "Model not found. Run `python train_model.py` first to train "
                "and save the model."
            )
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
