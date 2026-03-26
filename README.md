# Rotten Tomatoes Score Predictor from Screenplays

**Predicting Rotten Tomatoes critic scores directly from movie screenplay text using deep learning and transformer-based NLP models.**

---

## Project Overview

This project investigates whether the text of a movie screenplay contains enough signal to predict its eventual critical reception on Rotten Tomatoes. The core hypothesis is that linguistic patterns, narrative structure, and dialogue quality embedded in scripts correlate with professional critic evaluations -- and that modern NLP architectures can learn to extract these signals.

A dataset of 739 screenplay-score pairs was assembled, and three progressively sophisticated modeling approaches were implemented: a baseline feedforward neural network (FFNN) operating on handcrafted text features, a fine-tuned BERT model processing raw screenplay text, and a BERT + numeric fusion model that combines transformer-based text representations with structured numeric features. This progression allows for a systematic evaluation of how much value raw text understanding adds over traditional feature engineering, and whether multimodal fusion further improves performance.

The project demonstrates end-to-end NLP pipeline construction at scale, including screenplay parsing, feature extraction, transformer fine-tuning, and multi-input neural network design -- skills directly applicable to production NLP systems.

## Dataset

| Property | Value |
|---|---|
| Source | Screenplay corpus + Rotten Tomatoes API |
| Total Samples | 739 script-score pairs |
| Target Variable | Rotten Tomatoes critic score (continuous) |
| Text Input | Full screenplay text |
| Numeric Features | Script-level statistics (length, dialogue ratio, etc.) |

## Methodology

### Data Pipeline

1. **Screenplay Collection** -- Aggregation and parsing of movie scripts into clean text
2. **Score Matching** -- Linking screenplays to Rotten Tomatoes critic scores
3. **Feature Engineering** -- Extraction of numeric features (script length, dialogue density, vocabulary richness)
4. **Text Preprocessing** -- Tokenization and encoding for BERT input (512 token limit with chunking strategy)

### Models

| Model | Architecture | Input |
|---|---|---|
| FFNN | Feedforward neural network | Handcrafted numeric features |
| BERT | Fine-tuned `bert-base-uncased` | Raw screenplay text |
| **BERT + Numeric Fusion** | **BERT encoder + numeric branch, concatenated** | **Text + numeric features** |

### BERT Fine-Tuning

- Pretrained `bert-base-uncased` with a regression head
- Learning rate warmup and decay scheduling
- Gradient accumulation for effective larger batch sizes on limited GPU memory

### Fusion Architecture

The multimodal model processes text through a fine-tuned BERT encoder and numeric features through a separate dense network. The two representation streams are concatenated and passed through final dense layers for score prediction, allowing the model to leverage both linguistic understanding and structural script characteristics.

## Results

| Model | Description | Performance |
|---|---|---|
| FFNN | Numeric features only | Baseline |
| BERT | Screenplay text only | Improved over FFNN |
| **BERT + Numeric Fusion** | **Text + numeric features** | **Best performance** |

### Key Findings

- **Text carries predictive signal** -- BERT's ability to process raw screenplay text yields meaningful improvements over handcrafted features alone, confirming that linguistic patterns in scripts correlate with critical reception.
- **Fusion improves further** -- Combining BERT text representations with structured numeric features outperforms either modality in isolation, demonstrating the value of multimodal approaches.
- **Challenge of small data** -- With only 739 samples, overfitting is a persistent challenge. Transfer learning via BERT pretraining is essential for achieving reasonable performance at this data scale.

## Key Visualizations

- **Predicted vs. Actual Score Plots** -- Regression accuracy for each model
- **Training and Validation Loss Curves** -- Convergence behavior and overfitting diagnostics
- **Feature Importance Analysis** -- Which numeric features contribute most to predictions
- **Error Distribution** -- Residual analysis across score ranges
- **Model Comparison Charts** -- Performance metrics across the three architectures

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Deep Learning | PyTorch, Hugging Face Transformers |
| Transformer Model | BERT (`bert-base-uncased`) |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Environment | GPU-accelerated (Colab / local CUDA) |

## How to Run

```bash
# Clone the repository
git clone https://github.com/RyanOrdonez/RottenTomatoesPredictor.git
cd RottenTomatoesPredictor

# Install dependencies
pip install torch transformers pandas numpy matplotlib seaborn scikit-learn

# Run the notebook
jupyter notebook
```

> **Note:** BERT fine-tuning requires a GPU. Google Colab with GPU runtime or a local CUDA-enabled machine is recommended.

## Author

Ryan Ordonez -- MS in Data Science, University of Colorado Boulder
