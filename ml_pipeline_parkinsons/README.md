# SwaasCare — Parkinson's Disease Vocal Biomarker ML Pipeline

This is the **Parkinson's Disease ML module** for the SwaasCare Multi-Disease Voice Biomarker Disease Tracking Platform. It non-invasively screens voice recordings for neurological micro-tremors and vocal instability using acoustic biomarkers.

Based on the UCI/Oxford Parkinson's Disease dataset and the Kaggle kernel workflow (`nasrulhakim86/parkinson-s-disease-detection-from-vocal-biomarker`).

## Features Extracted
- **Fundamental Frequency ($F_0$) Variations:** Average (`MDVP:Fo`), maximum (`MDVP:Fhi`), and minimum (`MDVP:Flo`) pitch frequencies.
- **Vocal Frequency Jitter (Micro-tremors):** `MDVP:Jitter(%)`, `MDVP:Jitter(Abs)`, `MDVP:RAP`, `MDVP:PPQ`, `Jitter:DDP`.
- **Vocal Amplitude Shimmer (Intensity instability):** `MDVP:Shimmer`, `MDVP:Shimmer(dB)`, `Shimmer:APQ3`, `Shimmer:APQ5`, `MDVP:APQ`, `Shimmer:DDA`.
- **Harmonic-to-Noise Ratios:** `NHR` (Noise-to-Harmonics Ratio) and `HNR` (Harmonics-to-Noise Ratio).
- **Nonlinear Dynamics & Fractal Complexity:** `RPDE`, `DFA`, `spread1`, `spread2`, `D2`, and `PPE` (Pitch Period Entropy).

## Model Performance
- **Algorithm:** Random Forest Classifier (Optimized with Balanced Weights)
- **Accuracy:** 92.31%
- **Precision:** 93.33%
- **Recall:** 96.55%
- **F1 Score:** 94.92%
- **Evaluation Split:** 80/20 Stratified Split on 195 clinical voice recordings

## Project Structure
```
ml_pipeline_parkinsons/
├── requirements.txt                   # All dependencies
├── README.md                          # Documentation
├── api.py                             # FastAPI REST service
├── demo.py                            # Interactive Gradio live web demo
├── record_and_predict.py              # CLI microphone recording & screening
├── data/
│   └── parkinsons.csv                 # 22 vocal biomarkers + status dataset
├── notebooks/
│   ├── kernel-metadata.json           # Kaggle kernel metadata
│   └── parkinson-s-disease-detection-from-vocal-biomarker.ipynb # Full EDA & benchmark notebook
├── src/
│   ├── __init__.py
│   ├── data_loader.py                 # Dataset loader and validator
│   ├── feature_extractor.py           # Librosa YIN pitch & biomarker extractor
│   ├── model.py                       # Model factory (RandomForest, GradientBoosting)
│   ├── train.py                       # Training, evaluation & artifact synchronization
│   └── predict.py                     # Inference pipeline for audio & tabular input
└── output/
    └── models/
        ├── random_forest_model.pkl    # Trained classifier
        ├── scaler.pkl                 # Fitted feature standardizer
        └── metadata.json              # Training benchmark metrics
```

## Quick Start Pipeline

### 1. Install Dependencies
```bash
cd ml_pipeline_parkinsons
pip install -r requirements.txt
```

### 2. Train and Benchmark the Model
```bash
python -m src.train
```

### 3. Launch Live Web Demo (Gradio)
```bash
python demo.py
# Opens interactive UI at http://127.0.0.1:7862
```

### 4. Test With Microphone (Terminal CLI)
```bash
python record_and_predict.py
```

### 5. Start Dedicated REST API (FastAPI)
```bash
python api.py
# REST API documentation at http://localhost:8002/docs
```

## Dataset Source & Citations
- Dataset: **Oxford Parkinson's Disease Detection Dataset** (UCI Machine Learning Repository / Kaggle `parkinsons-disease-data-set`).
- Reference Kaggle Kernel: `nasrulhakim86/parkinson-s-disease-detection-from-vocal-biomarker`
- Max A. Little, Patrick E. McSharry, Stephen J. Roberts, Declan A.E. Costello, Irene M. Moroz. "Suitability of dysphonia measurements for telemonitoring of Parkinson's disease", IEEE Transactions on Biomedical Engineering (2009).
