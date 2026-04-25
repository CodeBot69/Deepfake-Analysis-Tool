"""
Deepfake Analysis Tool — Central Configuration
"""
import os
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
HEATMAPS_DIR = RESULTS_DIR / "heatmaps"
FRAMES_DIR = RESULTS_DIR / "frames"

# Create dirs on import
for d in [UPLOAD_DIR, RESULTS_DIR, HEATMAPS_DIR, FRAMES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── API Settings ─────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
MAX_UPLOAD_SIZE_MB = 500
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".jpg", ".jpeg", ".png", ".bmp"}

# ─── Model Settings ──────────────────────────────────────
DEVICE = os.getenv("DEVICE", "auto")  # "auto", "cpu", "cuda"
IMAGE_SIZE = 299  # XceptionNet input size
FACE_DETECTION_CONFIDENCE = 0.5

# ─── Ensemble Weights ────────────────────────────────────
# Weights for each detector in the ensemble (must sum to 1.0)
ENSEMBLE_WEIGHTS = {
    "spatial": 0.45,
    "frequency": 0.30,
    "lipsync": 0.25,
}

# ─── Video Processing ────────────────────────────────────
FRAME_SAMPLE_RATE = 5       # Extract 1 frame every N frames
MAX_FRAMES_TO_ANALYZE = 60  # Cap frames for performance
AUDIO_SAMPLE_RATE = 22050

# ─── Confidence Thresholds ────────────────────────────────
FAKE_THRESHOLD = 0.5        # Scores above this → "FAKE"
HIGH_CONFIDENCE = 0.8       # Scores above this → "High Confidence"
LOW_CONFIDENCE = 0.3        # Scores below this → "Low Confidence"
