import os
import sys
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from src.feature_extractor import extract_vocal_biomarkers_22
from src.data_loader import VOCAL_BIOMARKER_COLUMNS

SAMPLE_RATE = 16000
DURATION = 5
OUTPUT_FILE = os.path.join(BASE_DIR, "recorded_voice_parkinsons.wav")
MODELS_DIR = os.path.join(BASE_DIR, "output", "models")
MODEL_PATH = os.path.join(MODELS_DIR, "random_forest_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

INSTRUCTIONS = """
======================================================================
     SWAASCARE - PARKINSON'S VOCAL BIOMARKER SCREENING CLI
======================================================================
Protocol:
Please hold a sustained "Ahhh" vowel sound into the microphone for
the duration of the recording (5 seconds).
"""

def main():
    print(INSTRUCTIONS)
    
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print(f"[ERROR] Trained model artifacts not found at {MODELS_DIR}.")
        print("Please run: python -m src.train")
        sys.exit(1)

    input("Press ENTER when you are ready to begin recording...")

    print(f"\nRecording for {DURATION} seconds... Speak now!")
    recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    
    for remaining in range(DURATION, 0, -1):
        print(f"  Time remaining: {remaining}s...", end="\r")
        time.sleep(1)
        
    sd.wait()
    print("\nRecording finished! Processing audio...")

    sf.write(OUTPUT_FILE, recording, SAMPLE_RATE)
    print(f"Saved audio sample to {OUTPUT_FILE}")

    # Extract features
    biomarkers = extract_vocal_biomarkers_22(OUTPUT_FILE)
    
    # Load model and scaler
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    features_scaled = scaler.transform(biomarkers.reshape(1, -1))
    prediction = int(model.predict(features_scaled)[0])
    probs = model.predict_proba(features_scaled)[0]

    healthy_conf = round(float(probs[0]) * 100, 1)
    parkinson_conf = round(float(probs[1]) * 100, 1)

    print("\n" + "="*50)
    print(" VOCAL BIOMARKER ASSESSMENT RESULTS")
    print("="*50)
    if prediction == 0:
        print("[STATUS] HEALTHY VOCAL PROFILE")
        print(f"  Healthy Confidence:     {healthy_conf}%")
        print(f"  Parkinson's Risk Index: {parkinson_conf}%")
        print("  Acoustic stability and neuromuscular cord control appear normal.")
    else:
        print("[STATUS] PARKINSON'S PROFILE DETECTED")
        print(f"  Parkinson's Risk Index: {parkinson_conf}%")
        print(f"  Confidence:             {parkinson_conf}%")
        print("  Noticeable micro-tremor or frequency/amplitude perturbation detected.")

    print("\nKey Measured Vocal Biomarkers:")
    print(f"  Fundamental Frequency (Fo): {biomarkers[0]:.2f} Hz")
    print(f"  Jitter (%):                 {biomarkers[3]:.4f} %")
    print(f"  Shimmer:                    {biomarkers[8]:.4f}")
    print(f"  Harmonics-to-Noise (HNR):   {biomarkers[15]:.2f} dB")
    print(f"  Pitch Period Entropy (PPE): {biomarkers[21]:.4f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
