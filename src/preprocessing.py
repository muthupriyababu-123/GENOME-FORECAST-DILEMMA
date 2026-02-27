# preprocessing.py
# -----------------------------------------------
# Loads 2022-2023 maize dataset, handles missing values,
# drops non-numeric columns, splits and saves processed data.
# -----------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

RAW_DATA_PATH       = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw_dataset.csv')
PROCESSED_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed_dataset.csv')

def load_data(path):
    print(f"[INFO] Loading data from: {path}")
    df = pd.read_csv(path)
    print(f"[INFO] Dataset shape: {df.shape}")
    print(df.head())
    return df

def handle_missing_values(df):
    print("\n[INFO] Checking for missing values...")
    print(df.isnull().sum())
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].mean(), inplace=True)
    print("[INFO] Missing values handled.")
    return df

def split_and_save(df, save_path):
    # Keep only numeric feature columns needed for ML
    feature_cols = ['Marker1', 'Marker2', 'Marker3', 'Temperature', 'Rainfall']
    target_col   = 'Yield'

    # Save processed (only ML columns)
    processed_df = df[feature_cols + [target_col]]
    processed_df.to_csv(save_path, index=False)

    X = processed_df[feature_cols]
    y = processed_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"\n[INFO] Train size: {X_train.shape[0]} samples")
    print(f"[INFO] Test size:  {X_test.shape[0]} samples")
    print(f"[INFO] Processed dataset saved to: {save_path}")

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df = load_data(RAW_DATA_PATH)
    df = handle_missing_values(df)
    X_train, X_test, y_train, y_test = split_and_save(df, PROCESSED_DATA_PATH)
    print("\n[DONE] Preprocessing complete!")
