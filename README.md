# Rotten Tomatoes Score Predictor

*A Deep Learning Regression Project – Intro to Deep Learning (DTSA 5511)*

## Project Summary

Predict Rotten Tomatoes critic scores directly from a movie's screenplay.\
This notebook-driven pipeline demonstrates end‑to‑end data collection, feature engineering, model training, and critical evaluation – illustrating both the power **and** limits of deep learning when data are sparse.

---

## Graded Steps (rubric‑aligned)

| Step                                                                                  | Rubric focus                            | What was delivered                                                                                                                                    |
| ------------------------------------------------------------------------------------- | --------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1 — Data Collection & Provenance**                                                  | Data provenance • Clear problem framing | 1 200+ scripts scraped from IMSDb & ScriptSlug – matched with RT critic scores via OMDb API; final **739** script‑score pairs retained after cleaning |
| **2 — Problem Definition**                                                            | Supervised task clarity                 | Framed as **regression** (0‑100 score) and motivated by risk mitigation in film production                                                            |
| **3 — EDA & Feature Engineering**                                                     | 34 pts                                  | • Structural & linguistic features extracted (script length, dialogue ratio, char count, readability, etc.)                                           |
| • Outlier handling, distribution plots, correlation heat‑map, and dataset diagnostics |                                         |                                                                                                                                                       |
| **4 — Model Building & Evaluation**                                                   | 65 pts                                  | *Two deep‑learning baselines:*                                                                                                                        |
|  • **FFNN** on 5 engineered features                                                  |                                         |                                                                                                                                                       |
|  • **BERT‑base** on full script text                                                  |                                         |                                                                                                                                                       |
|  • **BERT + numeric** fusion with partial fine‑tuning                                 |                                         |                                                                                                                                                       |
| Metrics reported on held‑out 20 % test set                                            |                                         |                                                                                                                                                       |
| **5 — Deliverables & Conclusion**                                                     | 35 pts                                  | Final notebook, cleaned data, reproducibility assets, metrics table, scatter plots, and this README                                                   |

---

## Key Findings

| Model          | Input features   | MAE       | RMSE  | R²     |
| -------------- | ---------------- | --------- | ----- | ------ |
| Constant mean  | –                | **0.198** | 0.239 | 0.000  |
| FFNN           | 5 numeric        | 0.209     | 0.246 | −0.054 |
| BERT (text)    | Full script      | 0.207     | 0.244 | −0.036 |
| BERT + numeric | Script + numeric | 0.207     | 0.244 | −0.036 |

*Even sophisticated models could not outperform a naïve mean‑prediction baseline.*\
Limited sample size and missing high‑level metadata overshadowed model complexity.

---

## Visual Highlights

- `figures/ffnn_actual_vs_pred.png` – Scatter plot of FFNN predictions vs. actual scores
- `figures/error_histogram.png` – Residual distribution confirming scores cluster around the mean
- `figures/eda_feature_dists/` – Feature histograms & box‑plots

(Images are saved by the notebook when run top‑to‑bottom.)

---

## Repository Structure

```
rt-score-predictor/
├─ data/
│  └─ final_movie_data_with_scripts.csv
├─ notebooks/
│  └─ RottenTomatoesScorePredictor_Final.ipynb
├─ scripts/
│  ├─ scrape_scripts.py
│  ├─ fetch_rt_scores.py
│  └─ feature_engineering.py
├─ figures/
├─ app/                 # (optional Streamlit demo)
├─ requirements.txt
├─ run.sh               # one‑line reproducibility script
└─ README.md            # ← you are here
```

---

## How to Reproduce

1. **Clone** the repo & create environment
   ```bash
   git clone https://github.com/<your‑user>/rt-score-predictor.git
   cd rt-score-predictor
   pip install -r requirements.txt   # or conda env create -f environment.yml
   ```
2. **Add data** – place `final_movie_data_with_scripts.csv` in `data/` or run `scripts/*` to recreate.
3. **Execute**
   ```bash
   bash run.sh                   # non‑interactive
   # – or –
   jupyter notebook notebooks/RottenTomatoesScorePredictor_Final.ipynb
   Run‑All in the UI
   ```
4. **(Optional) Demo** –
   ```bash
   streamlit run app/app.py
   ```

---

## Future Work

- **Scale the dataset** to ≥ 10 000 scripts for meaningful text‑based learning.
- **Add metadata** – genre, budget, cast star‑power, release window.
- **Model hierarchy** – start with classification into buckets (Fresh/Rotten, deciles) before fine‑grained regression.
- **Tree‑based ensembles** on numeric + categorical features (e.g. Gradient Boosting) often outperform neural nets on tabular data.
- **Whole‑encoder fine‑tune** with lower LR when data volume allows.

---

© 2025 Ryan Ordonez – MIT License

