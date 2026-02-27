# train_model.py
# -----------------------------------------------
# Trains a Random Forest model on crop yield data,
# evaluates with R² and RMSE, and saves the model.
# -----------------------------------------------

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# --- Paths ---
DATA_PATH  = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed_dataset.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'trained_model.pkl')


def load_processed_data(path):
    """Load the processed dataset."""
    print(f"[INFO] Loading processed data from: {path}")
    df = pd.read_csv(path)
    return df


def train_model(X_train, y_train):
    """Train a Random Forest Regressor."""
    print("\n[INFO] Training Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=100,   # Number of trees
        random_state=42,    # For reproducibility
        max_depth=10        # Limit depth to avoid overfitting
    )
    model.fit(X_train, y_train)
    print("[INFO] Model training complete!")
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model using R² score and RMSE."""
    y_pred = model.predict(X_test)

    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n========== MODEL EVALUATION ==========")
    print(f"  R² Score : {r2:.4f}  (1.0 = perfect)")
    print(f"  RMSE     : {rmse:.4f} tons/hectare")
    print("=======================================")
    return r2, rmse


def save_model(model, path):
    """Save the trained model to disk using joblib."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"\n[INFO] Model saved to: {path}")


if __name__ == "__main__":
    # Step 1: Load data
    df = load_processed_data(DATA_PATH)

    feature_cols = ['Marker1', 'Marker2', 'Marker3', 'Temperature', 'Rainfall']
    target_col   = 'Yield'

    X = df[feature_cols]
    y = df[target_col]

    # Step 2: Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Step 3: Train
    model = train_model(X_train, y_train)

    # Step 4: Evaluate
    evaluate_model(model, X_test, y_test)

    # Step 5: Save
    save_model(model, MODEL_PATH)

    print("\n[DONE] Training pipeline complete!")
