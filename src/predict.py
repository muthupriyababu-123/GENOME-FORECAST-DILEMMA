# predict.py
# -----------------------------------------------
# Loads the trained model and predicts crop yield
# based on new input values provided by the user.
# -----------------------------------------------

import joblib
import numpy as np
import os

# --- Path to trained model ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'trained_model.pkl')


def load_model(path):
    """Load the saved trained model."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at '{path}'. Please run train_model.py first."
        )
    model = joblib.load(path)
    print(f"[INFO] Model loaded from: {path}")
    return model


def predict_yield(marker1, marker2, marker3, temperature, rainfall):
    """
    Predict crop yield given genomic markers and climate inputs.

    Parameters:
        marker1     : int   - SNP Marker 1 value (0, 1, or 2)
        marker2     : int   - SNP Marker 2 value (0, 1, or 2)
        marker3     : int   - SNP Marker 3 value (0, 1, or 2)
        temperature : float - Temperature in °C
        rainfall    : float - Rainfall in mm

    Returns:
        float - Predicted yield in tons/hectare
    """
    model = load_model(MODEL_PATH)

    # Build input array (must match training feature order)
    input_data = np.array([[marker1, marker2, marker3, temperature, rainfall]])

    # Predict
    predicted_yield = model.predict(input_data)[0]

    return round(predicted_yield, 2)


if __name__ == "__main__":
    # --- Example prediction ---
    print("=== Crop Yield Prediction ===\n")

    # Sample inputs
    m1   = 1      # Marker1
    m2   = 2      # Marker2
    m3   = 0      # Marker3
    temp = 28.0   # Temperature in °C
    rain = 500.0  # Rainfall in mm

    result = predict_yield(m1, m2, m3, temp, rain)

    print(f"  Marker1     : {m1}")
    print(f"  Marker2     : {m2}")
    print(f"  Marker3     : {m3}")
    print(f"  Temperature : {temp} °C")
    print(f"  Rainfall    : {rain} mm")
    print(f"\n  Predicted Yield: {result} tons/hectare")
