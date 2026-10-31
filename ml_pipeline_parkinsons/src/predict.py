import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
from src.feature_extractor import extract_vocal_biomarkers_22
from src.data_loader import VOCAL_BIOMARKER_COLUMNS

def predict_voice(input_data, model_path=None, scaler_path=None):
    """
    Predicts whether a voice sample indicates healthy condition or Parkinson's profile.
    
    Args:
        input_data (str or np.ndarray): File path to .wav or 1D array of features.
        model_path (str, optional): Path to trained random_forest_model.pkl
        scaler_path (str, optional): Path to trained scaler.pkl
        
    Returns:
        dict: Prediction results with condition, confidence, risk score, and biomarker stats.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if model_path is None:
        model_path = os.path.join(base_dir, "output", "models", "random_forest_model.pkl")
    if scaler_path is None:
        scaler_path = os.path.join(base_dir, "output", "models", "scaler.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Trained models not found at {model_path}. Please run train.py first.")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # 1. Process Input
    if isinstance(input_data, str) and os.path.exists(input_data):
        features = extract_vocal_biomarkers_22(input_data)
    elif isinstance(input_data, (list, np.ndarray)):
        features = np.array(input_data, dtype=np.float32)
        if len(features) > 22:
            features = features[:22]
    else:
        raise ValueError("Invalid input_data: must be an existing audio file path or numeric feature array")

    # 2. Scale & Predict
    features_2d = features.reshape(1, -1)
    features_scaled = scaler.transform(features_2d)

    pred = int(model.predict(features_scaled)[0])
    probs = model.predict_proba(features_scaled)[0] if hasattr(model, "predict_proba") else [0.5, 0.5]

    condition = "Parkinsons" if pred == 1 else "Normal"
    confidence = float(max(probs))
    risk_score = round(float(probs[1]) * 100, 1)

    # 3. Biomarker Breakdown
    biomarkers = {}
    for i, col in enumerate(VOCAL_BIOMARKER_COLUMNS):
        if i < len(features):
            biomarkers[col] = round(float(features[i]), 5)

    return {
        "prediction": condition,
        "confidence": confidence,
        "risk_score": risk_score,
        "status": "warning" if condition == "Parkinsons" else "normal",
        "biomarkers": biomarkers
    }
