import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
import joblib
import numpy as np
import tempfile
import soundfile as sf
from src.feature_extractor import extract_vocal_biomarkers_22
from src.data_loader import VOCAL_BIOMARKER_COLUMNS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "output", "models")
MODEL_PATH  = os.path.join(MODELS_DIR, "random_forest_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

CUSTOM_CSS = """
.result-box      { border-radius: 12px; padding: 16px; margin-top: 8px; font-size: 15px; }
.normal-box      { background: #d1fae5; border: 2px solid #10b981; color: #065f46; }
.parkinsons-box  { background: #fee2e2; border: 2px solid #ef4444; color: #7f1d1d; }
footer { display: none !important; }
"""

INSTRUCTIONS = """### Voice Assessment Protocol:
For the most accurate vocal biomarker screening:
1. Hold a sustained **"Ahhh"** vowel sound for 3 to 5 seconds at a comfortable, steady volume.
2. Alternatively, read a sentence at your natural pace.
3. Keep the microphone 10-15 cm from your mouth in a quiet environment.
"""

def predict_from_audio(audio):
    if audio is None:
        return (
            "No audio detected. Please record or upload a voice sample.",
            "",
            gr.update(visible=False),
            gr.update(visible=False),
            ""
        )

    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return (
            "ERROR: Trained model artifacts not found. Please run `python -m src.train` first.",
            "",
            gr.update(visible=False),
            gr.update(visible=False),
            ""
        )

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    sr, data = audio

    if data.dtype != np.float32:
        max_val = np.iinfo(data.dtype).max if np.issubdtype(data.dtype, np.integer) else 1.0
        data = data.astype(np.float32) / max_val

    if data.ndim > 1:
        data = np.mean(data, axis=1)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        sf.write(tmp_path, data, sr)

    try:
        biomarkers = extract_vocal_biomarkers_22(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    features_scaled = scaler.transform(biomarkers.reshape(1, -1))
    prediction = int(model.predict(features_scaled)[0])
    probs = model.predict_proba(features_scaled)[0]

    healthy_conf = round(float(probs[0]) * 100, 1)
    parkinson_conf = round(float(probs[1]) * 100, 1)

    # Key biomarkers table
    biomarkers_md = f"""
| Vocal Biomarker | Measured Value | Standard Healthy Range |
| :--- | :--- | :--- |
| **Pitch Mean (Fo)** | {biomarkers[0]:.2f} Hz | 100 - 240 Hz |
| **Pitch Max (Fhi)** | {biomarkers[1]:.2f} Hz | 120 - 280 Hz |
| **Jitter (%)** | {biomarkers[3]:.4f} % | < 0.006 % |
| **Shimmer** | {biomarkers[8]:.4f} | < 0.035 |
| **HNR (Harmonics-to-Noise)** | {biomarkers[15]:.2f} dB | > 20 dB |
| **PPE (Pitch Period Entropy)** | {biomarkers[21]:.4f} | < 0.20 |
"""

    if prediction == 0:
        label = "Healthy Vocal Profile"
        summary = (
            f"### Result: Healthy Neurological Vocal Profile Detected\n\n"
            f"- **Confidence:** {healthy_conf}%\n"
            f"- **Parkinson's Risk Index:** {parkinson_conf}%\n"
            f"- Vocal stability and neuromuscular cords control appear normal."
        )
        vis_norm = gr.update(visible=True)
        vis_park = gr.update(visible=False)
    else:
        label = "Parkinson's Profile Detected"
        summary = (
            f"### Result: Potential Vocal Biomarkers for Parkinson's Detected\n\n"
            f"- **Confidence:** {parkinson_conf}%\n"
            f"- **Parkinson's Risk Index:** {parkinson_conf}%\n"
            f"- Noticeable vocal frequency perturbation (jitter) or amplitude instability (shimmer) detected."
        )
        vis_norm = gr.update(visible=False)
        vis_park = gr.update(visible=True)

    return label, summary, vis_norm, vis_park, biomarkers_md

with gr.Blocks(css=CUSTOM_CSS, title="SwaasCare — Parkinson's Disease Screening") as demo:
    gr.Markdown("# SwaasCare — Parkinson's Disease Vocal Biomarker Screening")
    gr.Markdown("Real-time non-invasive screening using acoustic vocal measurements from the UCI/Oxford Parkinson's dataset.")
    gr.Markdown(INSTRUCTIONS)

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Record or Upload Voice")
            analyze_btn = gr.Button("Analyze Voice Biomarkers", variant="primary")
            
        with gr.Column(scale=1):
            status_text = gr.Markdown("### Ready for Voice Input")
            
            with gr.Group(visible=False) as normal_group:
                gr.Markdown('<div class="result-box normal-box">Healthy Profile: Stable vocal neuromuscular control.</div>')
                
            with gr.Group(visible=False) as parkinson_group:
                gr.Markdown('<div class="result-box parkinsons-box">Parkinson\'s Profile: Elevated acoustic micro-tremor pattern detected.</div>')

            result_summary = gr.Markdown()
            biomarkers_display = gr.Markdown()

    analyze_btn.click(
        fn=predict_from_audio,
        inputs=[audio_input],
        outputs=[status_text, result_summary, normal_group, parkinson_group, biomarkers_display]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7862)
