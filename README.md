# 🍅 Rotten Tomatoes Score Predictor

**Describe a movie idea and get a predicted Rotten Tomatoes score, audience rating, and emotional vibe breakdown — powered by AI.**

---

## [>>> Try It Now — Launch the App <<<](https://rotten-tomatoes-score-predictor.streamlit.app/)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rotten-tomatoes-score-predictor.streamlit.app/)

**No install required.** Click the link, type a movie idea, and get your results. Each user gets **3 free predictions per day**.

---

## How It Works

1. **Describe your movie idea** — Type a few sentences about your film concept
2. **AI writes a synopsis** — Claude generates a polished movie synopsis from your idea
3. **Get your scores:**
   - 🍅 **Tomatometer** — Predicted critic score from a model trained on 739 real screenplays
   - 🍿 **Popcornmeter** — Predicted audience reception
4. **The Vibe Check** — AI rates your movie across five emotional dimensions

---

## The Vibe Check

Every prediction includes a breakdown of how your movie would make the audience *feel*:

| Vibe | What It Measures | Low End | High End |
|---|---|---|---|
| 😂 **Laughs** | How funny is it? | *Waiting for your Hot Pocket in the microwave* | *Laughing so hard you shart a little* |
| 😢 **Tears** | Will people cry? | *Bone dry, not a tear in sight* | *Ugly crying in public* |
| 💓 **Romance** | Is there a love story? | *Awkward handshake energy* | *Steamy enough to fog your screen* |
| 😱 **Scares** | Is it horror-movie scary? | *Night light off, no big deal* | *Sleeping with every light on for a week* |
| 🔥 **Thrills** | Is it intense and exciting? | *Sunday afternoon nap* | *White-knuckle grip, forgot to breathe* |

---

## About the Model

The Tomatometer prediction comes from a **Gradient Boosting Regressor** trained on **739 real movie screenplays** paired with their actual Rotten Tomatoes critic scores. Behind the scenes, AI generates a full screenplay from your idea, extracts structural features (length, dialogue patterns, readability, character count), and feeds them into the model.

### Disclaimer

This is an experimental demo for entertainment. Predicting critic scores from text alone is extremely hard — real reviews depend on directing, acting, budget, and many factors beyond a script. The original deep learning research (FFNN, BERT, BERT+Fusion) is preserved in [`notebooks/`](notebooks/RottenTomatoesScorePredictor.ipynb).

---

## Self-Hosting

```bash
git clone https://github.com/RyanOrdonez/Rotten-Tomatoes-Predictor.git
cd Rotten-Tomatoes-Predictor
pip install -r requirements.txt
streamlit run app.py
```

The trained model is included. spaCy downloads automatically on first run. You'll need an [Anthropic API key](https://console.anthropic.com/) set as `ANTHROPIC_API_KEY`.

### Deploy Your Own (Free)

1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Select your forked repo, set `app.py` as the main file
4. Add your `ANTHROPIC_API_KEY` in **Settings > Secrets**
5. Deploy

---

## Project Structure

```
├── app.py                     # Streamlit web app
├── train_model.py             # Retrain the model (optional)
├── requirements.txt
├── data/                      # Training dataset (739 screenplays)
├── models/                    # Pre-trained model (included)
├── src/
│   ├── feature_extraction.py  # NLP feature pipeline
│   ├── predictor.py           # Model inference
│   └── script_expander.py     # Claude API: synopsis, screenplay, vibe ratings
├── notebooks/                 # Original research notebook
└── tests/
```

## Tech Stack

Streamlit | Anthropic Claude API | scikit-learn | spaCy | textstat | Pandas

---

**Ryan Ordonez** — MS in Data Science, University of Colorado Boulder
