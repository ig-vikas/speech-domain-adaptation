"""
app.py - FastAPI Web App for Domain-Adapted Speech Recognition
================================================================
Serves the fine-tuned wav2vec2 model via a web interface.
Users can upload audio files or record from their microphone.

Usage:
  python app.py
  Then open http://localhost:8000 in your browser
"""

import os
import io
import sys
import time
import warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import torch
import numpy as np
import soundfile as sf

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)
from utils import calculate_wer, calculate_cer

# ============================================================
# Global model variables (loaded once at startup)
# ============================================================
baseline_model = None
baseline_processor = None
adapted_model = None
adapted_processor = None
device = "cpu"

app = FastAPI(title="Speech Domain Adaptation Demo")

# Serve results folder for images
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
if os.path.exists(RESULTS_DIR):
    app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")
if os.path.exists(DATA_DIR):
    app.mount("/project-data", StaticFiles(directory=DATA_DIR), name="project-data")


# ============================================================
# Load models at startup
# ============================================================
@app.on_event("startup")
def load_models():
    global baseline_model, baseline_processor
    global adapted_model, adapted_processor, device

    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  [DEVICE] Using: {device}")

    # Load baseline model
    print("  [LOADING] Baseline model (wav2vec2-base-960h)...")
    baseline_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    baseline_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    baseline_model.to(device).eval()
    print("  [OK] Baseline model loaded")

    # Load adapted model
    adapted_dir = os.path.join(PROJECT_DIR, "adapted_model")
    if os.path.exists(adapted_dir):
        print(f"  [LOADING] Adapted model from {adapted_dir}...")
        adapted_processor = Wav2Vec2Processor.from_pretrained(adapted_dir)
        adapted_model = Wav2Vec2ForCTC.from_pretrained(adapted_dir)
        adapted_model.to(device).eval()
        print("  [OK] Adapted model loaded")
    else:
        print("  [WARNING] adapted_model/ not found. Only baseline will be available.")

    print("  [READY] Server is ready!\n")


# ============================================================
# Helper: transcribe audio array
# ============================================================
def transcribe_audio(model, processor, audio_array, sr=16000):
    """Run inference on an audio numpy array."""
    # Resample if needed
    if sr != 16000:
        import librosa
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)

    inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = logits.argmax(dim=-1)
    text = processor.batch_decode(predicted_ids)[0].strip()
    return text


def get_metadata_rows():
    """Load metadata rows from data/metadata.csv if available."""
    import csv

    metadata_path = os.path.join(DATA_DIR, "metadata.csv")
    rows = []
    if not os.path.exists(metadata_path):
        return rows

    with open(metadata_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def build_audio_url(file_path: str) -> str:
    """Convert an absolute local file path under data/ to a static URL."""
    try:
        abs_file = os.path.abspath(file_path)
        abs_data = os.path.abspath(DATA_DIR)
        if abs_file.startswith(abs_data + os.sep):
            rel = os.path.relpath(abs_file, abs_data).replace("\\", "/")
            return f"/project-data/{rel}"
    except Exception:
        pass
    return ""


# ============================================================
# API Routes
# ============================================================

@app.get("/", response_class=HTMLResponse)
def home():
    """Serve the main webpage."""
    html_path = os.path.join(PROJECT_DIR, "templates", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...), reference: str = ""):
    """
    Transcribe an uploaded audio file using both models.
    Returns predictions from baseline and adapted model with metrics.
    """
    try:
        start_time = time.time()

        # Read the uploaded audio
        contents = await audio.read()

        # Try soundfile first (handles WAV, FLAC, OGG)
        # Fall back to wave module for raw PCM WAV
        audio_array = None
        sr = 16000
        try:
            audio_array, sr = sf.read(io.BytesIO(contents), dtype="float32")
        except Exception:
            # Fallback: try the standard wave module
            import wave
            import struct
            try:
                wav_io = io.BytesIO(contents)
                with wave.open(wav_io, 'rb') as wf:
                    sr = wf.getframerate()
                    n_frames = wf.getnframes()
                    n_channels = wf.getnchannels()
                    sampwidth = wf.getsampwidth()
                    raw = wf.readframes(n_frames)
                    if sampwidth == 2:
                        fmt = f'<{n_frames * n_channels}h'
                        samples = struct.unpack(fmt, raw)
                        audio_array = np.array(samples, dtype=np.float32) / 32768.0
                    elif sampwidth == 4:
                        fmt = f'<{n_frames * n_channels}i'
                        samples = struct.unpack(fmt, raw)
                        audio_array = np.array(samples, dtype=np.float32) / 2147483648.0
                    else:
                        raise ValueError(f"Unsupported sample width: {sampwidth}")
                    if n_channels > 1:
                        audio_array = audio_array.reshape(-1, n_channels).mean(axis=1)
            except Exception as e2:
                raise ValueError(
                    f"Could not decode audio. Please upload a WAV file (16-bit PCM). "
                    f"soundfile error: format not recognised. wave error: {e2}"
                )

        # Handle stereo -> mono
        if audio_array is not None and len(audio_array.shape) > 1:
            audio_array = np.mean(audio_array, axis=1)

        duration = len(audio_array) / sr

        # Baseline transcription
        baseline_text = ""
        baseline_time = 0
        if baseline_model is not None:
            t0 = time.time()
            baseline_text = transcribe_audio(baseline_model, baseline_processor, audio_array, sr)
            baseline_time = time.time() - t0

        # Adapted transcription
        adapted_text = ""
        adapted_time = 0
        if adapted_model is not None:
            t0 = time.time()
            adapted_text = transcribe_audio(adapted_model, adapted_processor, audio_array, sr)
            adapted_time = time.time() - t0

        total_time = time.time() - start_time

        # Calculate WER/CER if reference provided
        result = {
            "baseline_text": baseline_text,
            "adapted_text": adapted_text,
            "duration": round(duration, 2),
            "baseline_time": round(baseline_time, 2),
            "adapted_time": round(adapted_time, 2),
            "total_time": round(total_time, 2),
        }

        if reference.strip():
            ref = reference.strip().upper()
            b_wer = calculate_wer(ref, baseline_text) if baseline_text else None
            a_wer = calculate_wer(ref, adapted_text) if adapted_text else None
            b_cer = calculate_cer(ref, baseline_text) if baseline_text else None
            a_cer = calculate_cer(ref, adapted_text) if adapted_text else None
            result["reference"] = ref
            result["baseline_wer"] = round(b_wer * 100, 2) if b_wer is not None else None
            result["adapted_wer"] = round(a_wer * 100, 2) if a_wer is not None else None
            result["baseline_cer"] = round(b_cer * 100, 2) if b_cer is not None else None
            result["adapted_cer"] = round(a_cer * 100, 2) if a_cer is not None else None

        return JSONResponse(result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sample-files")
def get_sample_files(
    domain: str = Query(default="all", description="all | source | target"),
    limit: int = Query(default=40, ge=1, le=200)
):
    """Return sample audio records for selection and quick testing."""
    samples = []
    rows = get_metadata_rows()
    domain = domain.strip().lower()

    if domain in {"source", "target"}:
        rows = [r for r in rows if r.get("domain", "").lower() == domain]

    # Keep deterministic ordering for UI pickers.
    rows = sorted(rows, key=lambda r: (r.get("domain", ""), r.get("file_id", "")))

    for row in rows[:limit]:
        file_path = row.get("file_path", "")
        samples.append({
            "file_id": row.get("file_id", ""),
            "domain": row.get("domain", ""),
            "transcript": row.get("transcript", ""),
            "duration": row.get("duration", ""),
            "file_path": file_path,
            "audio_url": build_audio_url(file_path),
        })

    return JSONResponse(samples)


@app.get("/transcribe-sample/{file_id}")
def transcribe_sample(file_id: str):
    """Transcribe a sample from the dataset by file_id."""
    rows = get_metadata_rows()
    if not rows:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Find the sample
    target_row = None
    for row in rows:
        if row.get("file_id") == file_id:
            target_row = row
            break

    if not target_row:
        raise HTTPException(status_code=404, detail=f"Sample {file_id} not found")

    if baseline_model is None or baseline_processor is None:
        raise HTTPException(status_code=503, detail="Baseline model not available")

    # Load and transcribe
    audio_array, sr = sf.read(target_row["file_path"], dtype="float32")
    reference = target_row["transcript"].upper().strip()

    baseline_text = transcribe_audio(baseline_model, baseline_processor, audio_array, sr)
    adapted_text = ""
    if adapted_model is not None and adapted_processor is not None:
        adapted_text = transcribe_audio(adapted_model, adapted_processor, audio_array, sr)

    b_wer = calculate_wer(reference, baseline_text)
    a_wer = calculate_wer(reference, adapted_text) if adapted_text else None

    return JSONResponse({
        "file_id": file_id,
        "domain": target_row["domain"],
        "reference": reference,
        "baseline_text": baseline_text,
        "adapted_text": adapted_text,
        "baseline_wer": round(b_wer * 100, 2),
        "adapted_wer": round(a_wer * 100, 2) if a_wer is not None else None,
        "duration": target_row["duration"],
        "audio_url": build_audio_url(target_row.get("file_path", "")),
    })


@app.get("/project-results")
def get_project_results():
    """Return the project results (baseline vs adapted)."""
    import json
    results = {}

    baseline_path = os.path.join(RESULTS_DIR, "baseline_results.json")
    adapted_path = os.path.join(RESULTS_DIR, "adapted_results.json")

    if os.path.exists(baseline_path):
        with open(baseline_path, "r") as f:
            results["baseline"] = json.load(f)
    if os.path.exists(adapted_path):
        with open(adapted_path, "r") as f:
            results["adapted"] = json.load(f)

    return JSONResponse(results)


@app.get("/project-overview")
def get_project_overview():
    """Return pipeline status, dataset summary, and artifact availability."""
    import json

    metadata_rows = get_metadata_rows()
    source_count = sum(1 for r in metadata_rows if r.get("domain") == "source")
    target_count = sum(1 for r in metadata_rows if r.get("domain") == "target")

    baseline_path = os.path.join(RESULTS_DIR, "baseline_results.json")
    adapted_path = os.path.join(RESULTS_DIR, "adapted_results.json")
    training_loss_path = os.path.join(RESULTS_DIR, "training_loss.json")
    wer_chart_path = os.path.join(RESULTS_DIR, "wer_comparison.png")
    loss_chart_path = os.path.join(RESULTS_DIR, "training_loss.png")
    spec_chart_path = os.path.join(RESULTS_DIR, "spectrogram_comparison.png")
    sample_preds_path = os.path.join(RESULTS_DIR, "sample_predictions.txt")
    adapted_model_dir = os.path.join(PROJECT_DIR, "adapted_model")

    baseline_metrics = None
    if os.path.exists(baseline_path):
        with open(baseline_path, "r", encoding="utf-8") as f:
            baseline_data = json.load(f)
        baseline_metrics = {
            "source_wer": baseline_data.get("source_domain", {}).get("avg_wer"),
            "target_wer": baseline_data.get("target_domain", {}).get("avg_wer"),
        }

    adapted_metrics = None
    if os.path.exists(adapted_path):
        with open(adapted_path, "r", encoding="utf-8") as f:
            adapted_data = json.load(f)
        adapted_metrics = {
            "source_wer": adapted_data.get("source_domain", {}).get("avg_wer"),
            "target_wer": adapted_data.get("target_domain", {}).get("avg_wer"),
            "target_improvement_pct": adapted_data.get("comparison", {}).get("target_wer_improvement_pct"),
        }

    pipeline_steps = [
        {
            "name": "1_data_preparation.py",
            "description": "Prepare source/target audio and metadata",
            "complete": source_count > 0 and target_count > 0,
        },
        {
            "name": "2_baseline_evaluation.py",
            "description": "Evaluate baseline model",
            "complete": os.path.exists(baseline_path),
        },
        {
            "name": "3_fine_tune.py",
            "description": "Fine-tune on target domain",
            "complete": os.path.exists(adapted_model_dir) and os.path.exists(training_loss_path),
        },
        {
            "name": "4_final_evaluation.py",
            "description": "Compare baseline vs adapted",
            "complete": os.path.exists(adapted_path) and os.path.exists(wer_chart_path),
        },
    ]

    artifacts = [
        {"name": "Metadata CSV", "path": "data/metadata.csv", "exists": os.path.exists(os.path.join(DATA_DIR, "metadata.csv")), "url": "/project-data/metadata.csv"},
        {"name": "Baseline Results", "path": "results/baseline_results.json", "exists": os.path.exists(baseline_path), "url": "/results/baseline_results.json"},
        {"name": "Adapted Results", "path": "results/adapted_results.json", "exists": os.path.exists(adapted_path), "url": "/results/adapted_results.json"},
        {"name": "Sample Predictions", "path": "results/sample_predictions.txt", "exists": os.path.exists(sample_preds_path), "url": "/results/sample_predictions.txt"},
        {"name": "WER Chart", "path": "results/wer_comparison.png", "exists": os.path.exists(wer_chart_path), "url": "/results/wer_comparison.png"},
        {"name": "Training Loss", "path": "results/training_loss.png", "exists": os.path.exists(loss_chart_path), "url": "/results/training_loss.png"},
        {"name": "Spectrogram", "path": "results/spectrogram_comparison.png", "exists": os.path.exists(spec_chart_path), "url": "/results/spectrogram_comparison.png"},
    ]

    return JSONResponse({
        "models": {
            "baseline_loaded": baseline_model is not None,
            "adapted_loaded": adapted_model is not None,
            "device": device,
        },
        "dataset": {
            "source_count": source_count,
            "target_count": target_count,
            "total_count": source_count + target_count,
        },
        "pipeline": pipeline_steps,
        "artifacts": artifacts,
        "metrics": {
            "baseline": baseline_metrics,
            "adapted": adapted_metrics,
        },
    })


# ============================================================
# Run the server
# ============================================================
if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 50)
    print("  Speech Domain Adaptation - Web Demo")
    print("  Open http://localhost:8000 in your browser")
    print("=" * 50 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
