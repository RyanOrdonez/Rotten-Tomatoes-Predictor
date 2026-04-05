# Rotten Tomatoes Score Predictor

**Describe a movie idea and get a predicted Rotten Tomatoes critic score.**

Type your movie concept into the web app, AI expands it into a screenplay, and a trained model predicts how critics might rate it.

---

## Try It Now

<!-- Once deployed, uncomment and replace with your Streamlit Cloud URL: -->
<!-- [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app) -->

**No install required.** Just click the link above to use the app in your browser.

Each user gets **3 free AI predictions per day**.

---

## How It Works

1. **You describe a movie idea** in a few sentences
2. **Claude AI** expands it into a ~2000-word screenplay with scene headings, dialogue, and action
3. **Features are extracted** -- script length, dialogue ratio, readability, character count, sentence structure
4. **A trained model** predicts the Rotten Tomatoes critic score (0-100)

---

## About the Model

A **Gradient Boosting Regressor** trained on **739 real screenplays** paired with their actual Rotten Tomatoes critic scores.

| Feature | What It Measures |
|---|---|
| Script Length | Total word count |
| Avg Sentence Length | Words per sentence |
| Readability Score | Flesch-Kincaid grade level |
| Main Character Count | Unique speaking characters |
| Dialogue Ratio | Dialogue vs. action lines |

### Disclaimer

This is an experimental demo. Predicting critic scores from screenplay text alone is extremely difficult -- critical reception depends on directing, acting, budget, marketing, and many factors not in a script. The model was trained on only 739 samples. Treat predictions as entertainment, not forecasts.

The original deep learning research (FFNN, BERT, BERT+Fusion) is preserved in [`notebooks/`](notebooks/RottenTomatoesScorePredictor.ipynb).

---

## Self-Hosting / Development

If you want to run this yourself:

```bash
git clone https://github.com/RyanOrdonez/Rotten-Tomatoes-Predictor.git
cd Rotten-Tomatoes-Predictor
pip install -r requirements.txt
streamlit run app.py
```

The trained model is included in the repo. spaCy downloads automatically on first run.

### Deploy Your Own (Free)

1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Select your forked repo, set `app.py` as the main file
4. Add your `ANTHROPIC_API_KEY` in **Settings > Secrets** as:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
5. Click Deploy

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
│   └── script_expander.py     # Claude API integration
├── notebooks/                 # Original research notebook
└── tests/
```

## Tech Stack

Streamlit | Anthropic Claude API | scikit-learn | spaCy | textstat | Pandas

---

**Ryan Ordonez** -- MS in Data Science, University of Colorado Boulder
