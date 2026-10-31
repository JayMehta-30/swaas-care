import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import json
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from src.data_loader import load_dataset
from src.model import get_model

def train_and_evaluate(model_type='random_forest', output_dir=None):
    if output_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, "output", "models")
        
    os.makedirs(output_dir, exist_ok=True)
    print(f"--- Starting Parkinson's Biomarker Training Pipeline using {model_type} ---")
    
    # 1. Load Data
    X, y, feature_names = load_dataset()
    print(f"Dataset loaded. Total samples: {len(X)}")
    print(f"Healthy (0): {np.sum(y == 0)}, Parkinson's (1): {np.sum(y == 1)}")
    
    # 2. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Fit Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Train Model
    model = get_model(model_type)
    model.fit(X_train_scaled, y_train)
    
    # 5. Evaluate
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n" + "="*50)
    print(f" Evaluation Metrics for {model_type.upper()}")
    print("="*50)
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall:    {rec*100:.2f}%")
    print(f"F1 Score:  {f1*100:.2f}%")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Healthy", "Parkinson's"]))
    
    # Feature Importances if available
    if hasattr(model, "feature_importances_"):
        top_indices = np.argsort(model.feature_importances_)[::-1][:5]
        print("Top 5 Discriminative Vocal Biomarkers:")
        for idx in top_indices:
            print(f" - {feature_names[idx]}: {model.feature_importances_[idx]:.4f}")

    # 6. Save Model Artifacts
    model_path = os.path.join(output_dir, "random_forest_model.pkl")
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    metadata_path = os.path.join(output_dir, "metadata.json")
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    metadata = {
        "model_type": model_type,
        "features": feature_names,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "num_samples": int(len(X)),
        "dataset": "UCI Oxford Parkinson's Disease Dataset"
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"\nModel and scaler saved to {output_dir}")
    
    # Also sync into backend/app/ml/models/parkinsons/
    backend_model_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "backend", "app", "ml", "models", "parkinsons"
    )
    try:
        os.makedirs(backend_model_dir, exist_ok=True)
        shutil.copy2(model_path, os.path.join(backend_model_dir, "random_forest_model.pkl"))
        shutil.copy2(scaler_path, os.path.join(backend_model_dir, "scaler.pkl"))
        shutil.copy2(metadata_path, os.path.join(backend_model_dir, "metadata.json"))
        print(f"Synced model artifacts to backend: {backend_model_dir}")
    except Exception as e:
        print(f"Notice: Could not sync to backend directory: {e}")
        
    return model, scaler

if __name__ == "__main__":
    train_and_evaluate(model_type='random_forest')
