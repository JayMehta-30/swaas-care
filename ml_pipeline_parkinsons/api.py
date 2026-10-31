import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import tempfile
from pydantic import BaseModel
from typing import Optional, List, Dict
from src.predict import predict_voice
from src.data_loader import VOCAL_BIOMARKER_COLUMNS

app = FastAPI(
    title="SwaasCare Parkinson's Vocal Biomarker API",
    description="REST API for Parkinson's Disease screening using acoustic vocal biomarkers.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "models")
MODEL_PATH = os.path.join(MODELS_DIR, "random_forest_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

class BiomarkerInput(BaseModel):
    features: List[float]

@app.get("/")
def root():
    return {
        "status": "healthy",
        "service": "SwaasCare Parkinson's Disease Detection API",
        "model_loaded": os.path.exists(MODEL_PATH)
    }

@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    """
    Upload an audio file (.wav, .mp3, etc.) for vocal biomarker extraction and screening.
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise HTTPException(status_code=500, detail="Model artifacts not found. Please run src.train first.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = predict_voice(tmp_path, MODEL_PATH, SCALER_PATH)
        return {
            "status": "success",
            "filename": file.filename,
            **result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.post("/predict/biomarkers")
def predict_from_biomarkers(payload: BiomarkerInput):
    """
    Screening directly using the 22 vocal biomarker measurements.
    """
    if len(payload.features) != 22:
        raise HTTPException(status_code=400, detail=f"Expected 22 vocal biomarker features, got {len(payload.features)}")
    try:
        result = predict_voice(payload.features, MODEL_PATH, SCALER_PATH)
        return {
            "status": "success",
            **result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8002, reload=True)
