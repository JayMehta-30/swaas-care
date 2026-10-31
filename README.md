# SwaasCare — Multi-Disease Voice Biomarker Platform

This repository houses the Machine Learning pipelines, backend, and frontend code for the SwaasCare voice biomarker health monitoring platform.

## ML Pipelines

We have developed multiple independent AI/ML pipelines for tracking different disease markers via voice:

- **[Depression Screening Module](./ml_pipeline_depression/)**: Classifies voice samples for depression markers using Random Forest with 97.5% accuracy.
- **[Asthma Screening Module](./ml_pipeline_asthma/)**: Classifies voice samples for asthma indicators using an optimized Random Forest model with `librosa.yin`.
- **[Parkinson's Disease Screening Module](./ml_pipeline_parkinsons/)**: Classifies vocal biomarkers for Parkinson's disease micro-tremors and acoustic instability using a trained Random Forest classifier (92.3% accuracy, 94.9% F1) integrated from the UCI/Oxford dataset and Kaggle kernel `nasrulhakim86/parkinson-s-disease-detection-from-vocal-biomarker`.

Each module is fully containerized with its own:
- Live Gradio Demo (`demo.py`)
- FastAPI Backend Service (`api.py`)
- Terminal Mic Recorder (`record_and_predict.py`)
- Trained Classifier & Scaler (`output/models/`)
- Benchmark & Analysis Notebooks (`notebooks/`)

### Setup Instructions

To run any module, navigate into its specific directory and install its dependencies:
```bash
cd ml_pipeline_parkinsons
pip install -r requirements.txt
```

Train or run prediction:
```bash
# 1. Train model
python -m src.train

# 2. Launch Gradio web UI
python demo.py

# 3. Or launch REST API
python api.py
```

### Full-Stack Backend Integration
The trained Parkinson's vocal biomarker model and scaler are integrated into `backend/app/ml/models/parkinsons/`, powering the multi-disease voice analysis endpoint (`/api/analyze`) consumed by the Next.js frontend.
