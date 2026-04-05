"""Train a Gradient Boosting model on pre-computed script features.

Run once after cloning:
    python train_model.py

Saves model.joblib and scaler.joblib to the models/ directory.
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = os.path.join("data", "final_movie_data_with_scripts.csv")
MODEL_DIR = "models"
FEATURE_COLS = [
    "script_length",
    "avg_sentence_length",
    "readability_score",
    "main_character_count",
    "dialogue_ratio",
]


def main():
    # Load data
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} samples from {DATA_PATH}")

    # Drop rows with missing features or target
    df = df.dropna(subset=FEATURE_COLS + ["rt_score"])
    print(f"After dropping NaNs: {len(df)} samples")

    X = df[FEATURE_COLS].values
    y = df["rt_score"].values  # 0-100 scale

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Train Gradient Boosting Regressor
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n--- Test Set Results ---")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.3f}")

    # Cross-validation on full dataset
    cv_scores = cross_val_score(
        model, X_scaled, y, cv=5, scoring="neg_mean_absolute_error"
    )
    print(f"\n5-Fold CV MAE: {-cv_scores.mean():.2f} (+/- {cv_scores.std():.2f})")

    # Feature importance
    print("\n--- Feature Importance ---")
    for name, importance in sorted(
        zip(FEATURE_COLS, model.feature_importances_), key=lambda x: -x[1]
    ):
        print(f"  {name}: {importance:.3f}")

    # Save model and scaler
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "model.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    print(f"\nModel and scaler saved to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
