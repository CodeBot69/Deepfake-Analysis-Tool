<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776ab?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Next.js-16-000000?style=for-the-badge&logo=next.js&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.4-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

<h1 align="center">🔬 Aegis-Vision — Deepfake Analysis Tool</h1>

<p align="center">
  <strong>A high-precision, multi-modal ensemble deepfake detection system combining spatial analysis, frequency forensics, and lip-sync verification — with an interactive Next.js dashboard and Grad-CAM heatmap visualization.</strong>
</p>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Detection Pipeline](#detection-pipeline)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [API Reference](#api-reference)
- [Frontend](#frontend)
- [Configuration](#configuration)
- [Testing](#testing)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

**Aegis-Vision** is an end-to-end deepfake detection system that analyzes images and videos for signs of AI-generated manipulation. It employs a **multi-model ensemble** approach — combining three independent detection strategies — to provide robust, high-confidence verdicts with explainable results.

The system is built as a decoupled architecture with a **FastAPI** backend for ML inference and a **Next.js** frontend for interactive analysis, making it suitable for both research and production deployment.

---

## Key Features

| Feature | Description |
|---|---|
| 🧠 **Multi-Model Ensemble** | Weighted combination of spatial, frequency, and lip-sync detectors for robust predictions |
| 🔍 **Spatial Artifact Detection** | XceptionNet CNN trained to detect blending boundaries, texture inconsistencies, and compression artifacts |
| 〰️ **Frequency Domain Analysis** | 2D FFT-based detection of GAN fingerprints and periodic spectral anomalies |
| 🗣️ **Lip-Sync Verification** | Audio-visual temporal correlation analysis using MediaPipe face landmarks and MFCC audio features |
| 🌡️ **Grad-CAM Heatmaps** | Visual explanations highlighting which facial regions triggered the detection |
| 🎞️ **Frame-by-Frame Playback** | Interactive video scrubber with per-frame scores and heatmap overlays |
| 📊 **Score Timeline** | Temporal visualization of detection scores across video frames |
| 📤 **Drag-and-Drop Upload** | Support for images (JPG, PNG, BMP) and videos (MP4, AVI, MOV, MKV, WebM) up to 500 MB |
| ⚡ **Async Processing** | Background job queue with real-time progress polling |
| 🎨 **Glassmorphism UI** | Premium dark-mode interface with smooth animations and responsive design |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Next.js Frontend                         │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌───────────────┐  │
│  │  Upload   │  │  Progress │  │  Frame   │  │   Results     │  │
│  │  Zone     │  │  Tracker  │  │  Player  │  │   Dashboard   │  │
│  └──────────┘  └───────────┘  └──────────┘  └───────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │ REST API (HTTP)
┌───────────────────────────▼─────────────────────────────────────┐
│                       FastAPI Backend                            │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  Analysis Pipeline                        │   │
│  │  ┌──────────┐  ┌──────────────┐  ┌───────────────────┐  │   │
│  │  │  Video   │  │    Face      │  │    Ensemble       │  │   │
│  │  │ Processor│──│  Detection   │──│   Aggregator      │  │   │
│  │  └──────────┘  └──────────────┘  └───────────────────┘  │   │
│  │       │                                    ▲              │   │
│  │       ▼                                    │              │   │
│  │  ┌──────────┐  ┌──────────────┐  ┌────────┴──────────┐  │   │
│  │  │  Audio   │  │   Spatial    │  │    Frequency      │  │   │
│  │  │ Extractor│  │  Detector    │  │    Detector       │  │   │
│  │  └──────────┘  │ (XceptionNet)│  │    (FFT)          │  │   │
│  │       │        └──────────────┘  └───────────────────┘  │   │
│  │       ▼                                                   │   │
│  │  ┌──────────────────┐  ┌──────────────────────────────┐  │   │
│  │  │  Lip-Sync        │  │   Heatmap Generator          │  │   │
│  │  │  Detector        │  │   (Grad-CAM Overlays)        │  │   │
│  │  └──────────────────┘  └──────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Detection Pipeline

### 1. Spatial Artifact Detection — `SpatialDetector`

Uses a pretrained **XceptionNet** (via `timm`) with a custom binary classification head to detect visual manipulation artifacts in individual frames.

- **Input**: Cropped face region (299×299)
- **Process**: Forward pass through Xception → softmax probabilities → Grad-CAM heatmap generation
- **Output**: Fake probability score (0–1) + pixel-level heatmap
- **Detects**: Blending boundaries, texture inconsistencies, compression artifacts, color mismatches

### 2. Frequency Domain Analysis — `FrequencyDetector`

Analyzes the **2D Fourier Transform** of face crops to detect GAN-specific spectral fingerprints.

- **Input**: Grayscale face crop (256×256)
- **Process**:
  1. Apply Hanning window → 2D FFT → log magnitude spectrum
  2. Compute azimuthal (radial) average of power spectrum
  3. Extract three spectral features:
     - **High-frequency energy ratio** — GANs produce unusual high-frequency content
     - **Periodicity score** — Transposed convolutions create periodic spectral peaks
     - **Spectral falloff rate** — Natural images follow 1/f^α; GANs have flatter falloff
- **Output**: Weighted combination score (0–1)

### 3. Lip-Sync Consistency — `LipSyncDetector`

Measures temporal correlation between **mouth movements** and **audio energy** to detect dubbed or face-swapped videos.

- **Input**: Video frames + extracted audio (WAV)
- **Process**:
  1. Extract lip landmarks (MediaPipe Face Mesh) → mouth openness signal
  2. Extract audio RMS energy envelope (librosa) → resample to frame count
  3. Compute normalized cross-correlation between the two signals
- **Output**: Fake score based on correlation (low correlation → likely manipulated)
- **Fallback**: Lip-only motion analysis when audio is unavailable (detects unnaturally smooth/jerky motion)

### 4. Ensemble Aggregation — `EnsembleDetector`

Combines all detector outputs using configurable weighted averaging:

| Detector | Default Weight |
|---|---|
| Spatial (XceptionNet) | **0.45** |
| Frequency (FFT) | **0.30** |
| Lip-Sync | **0.25** |

**Video-level aggregation** uses a robust formula: `60% median + 25% mean + 15% max` of per-frame scores, with temporal consistency analysis.

**Confidence scoring** is derived from inter-model agreement (lower standard deviation across detectors = higher confidence).

---

## Project Structure

```
Deepfake-Analysis-Tool/
├── app.py                        # FastAPI backend entry point
├── config.py                     # Global configuration and paths
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
├── TECHNOLOGIES.md               # Full technology stack documentation
├── README.md                     # This file
│
├── models/
│   ├── __init__.py
│   ├── spatial_detector.py       # XceptionNet + Grad-CAM spatial analysis
│   ├── frequency_detector.py     # FFT-based frequency domain detector
│   ├── lipsync_detector.py       # Audio-visual lip-sync consistency
│   └── ensemble.py               # Weighted score aggregation (frame & video)
│
├── utils/
│   ├── __init__.py
│   ├── video_processor.py        # Frame extraction, face detection, audio extraction
│   └── heatmap_generator.py      # Grad-CAM overlay rendering & visualization
│
├── tests/
│   ├── __init__.py
│   ├── test_detectors.py         # Unit tests for detection models
│   └── test_api.py               # API endpoint integration tests
│
└── frontend/                     # Next.js 16 frontend application
    ├── package.json
    ├── next.config.ts
    ├── tsconfig.json
    └── src/
        └── app/
            ├── layout.tsx        # Root layout with metadata & fonts
            ├── page.tsx          # Main analysis dashboard
            ├── page.module.css   # Component-level styles
            ├── globals.css       # Design system tokens & base styles
            └── favicon.ico
```

---

## Getting Started

### Prerequisites

- **Python** 3.10+
- **Node.js** 20+
- **ffmpeg** (for audio extraction from videos)
- **CUDA** (optional, for GPU acceleration)

### 1. Clone the Repository

```bash
git clone https://github.com/CodeBot69/Deepfake-Analysis-Tool.git
cd Deepfake-Analysis-Tool
```

### 2. Backend Setup

```bash
# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Frontend Setup

```bash
cd frontend
npm install
cd ..
```

### 4. Run the Application

**Start the backend** (from the project root):
```bash
python app.py
```
The API server will start at `http://localhost:8000`.

**Start the frontend** (in a separate terminal):
```bash
cd frontend
npm run dev
```
The frontend will be available at `http://localhost:3000`.

### 5. Open the Dashboard

Navigate to **http://localhost:3000** in your browser, drag-and-drop a video or image, and click **"Analyze for Deepfakes"**.

---

## API Reference

### Health Check

```http
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "device": "auto",
  "ensemble_weights": { "spatial": 0.45, "frequency": 0.30, "lipsync": 0.25 }
}
```

### Submit Media for Analysis

```http
POST /api/analyze
Content-Type: multipart/form-data
```

| Parameter | Type | Description |
|---|---|---|
| `file` | `File` | Image or video file (max 500 MB) |

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "queued",
  "message": "Analysis started"
}
```

### Poll for Results

```http
GET /api/results/{job_id}
```

**Response (completed):**
```json
{
  "job_id": "uuid-string",
  "status": "completed",
  "progress": 1.0,
  "result": {
    "type": "video",
    "overall": {
      "final_score": 0.73,
      "verdict": "FAKE",
      "confidence": 0.85,
      "confidence_label": "High Confidence",
      "per_detector": {
        "spatial": { "mean": 0.78, "max": 0.92, "std": 0.08 },
        "frequency": { "mean": 0.65, "max": 0.71, "std": 0.04 }
      },
      "temporal_consistency": 0.88,
      "frame_scores": [0.72, 0.74, 0.71, ...]
    },
    "lipsync": {
      "score": 0.68,
      "correlation": 0.12,
      "analysis_type": "full_lipsync"
    },
    "frames": [
      { "url": "/static/results/frames/abc_f0001.jpg", "heatmap_url": "/static/results/heatmaps/abc_f0001_heatmap.png", "index": 0 }
    ]
  }
}
```

### Analyze Single Frame (Synchronous)

```http
POST /api/analyze/frame
Content-Type: multipart/form-data
```

Returns immediate results without background processing. Useful for quick image-only analysis.

### List All Jobs

```http
GET /api/jobs
```

---

## Frontend

The frontend is built with **Next.js 16** (App Router) and features a premium glassmorphism dark-mode design.

### Key UI Components

| Component | Description |
|---|---|
| **Upload Zone** | Drag-and-drop area with format badges and file preview |
| **Progress Tracker** | Animated progress bar with status messages describing the current pipeline stage |
| **Verdict Card** | Circular SVG gauge displaying the final score with animated stroke |
| **Detector Breakdown** | Per-detector score bars (Spatial, Frequency, Lip-Sync) with weights |
| **Frame Player** | Video scrubber with play/pause, next/prev, and frame slider |
| **Heatmap Toggle** | Overlay Grad-CAM heatmaps on analyzed frames |
| **Score Timeline** | Clickable bar chart showing per-frame detection scores |
| **Media Info** | Video metadata (duration, FPS, resolution, frames analyzed) |

### Design System

- **Color Palette**: Deep navy (`#0a0a0f`) with indigo (`#6366f1`) and cyan (`#22d3ee`) accents
- **Typography**: Inter (sans-serif) + JetBrains Mono (monospace)
- **Effects**: Glassmorphism cards, backdrop blur, animated gradients, smooth micro-animations
- **Animations**: `fadeInUp`, `slideInRight`, `gaugeGrow`, `bgPulse`, `shimmer`

---

## Configuration

All settings are centralized in [`config.py`](config.py):

| Setting | Default | Description |
|---|---|---|
| `DEVICE` | `auto` | Inference device (`auto`, `cpu`, `cuda`) |
| `IMAGE_SIZE` | `299` | Input size for XceptionNet |
| `FACE_DETECTION_CONFIDENCE` | `0.5` | MediaPipe face detection threshold |
| `ENSEMBLE_WEIGHTS` | `{spatial: 0.45, frequency: 0.30, lipsync: 0.25}` | Detector weight distribution |
| `FRAME_SAMPLE_RATE` | `5` | Extract 1 frame every N frames |
| `MAX_FRAMES_TO_ANALYZE` | `60` | Maximum frames per video |
| `MAX_UPLOAD_SIZE_MB` | `500` | Maximum upload file size |
| `FAKE_THRESHOLD` | `0.5` | Score threshold for FAKE verdict |
| `HIGH_CONFIDENCE` | `0.8` | High confidence threshold |
| `LOW_CONFIDENCE` | `0.3` | Low confidence threshold |
| `AUDIO_SAMPLE_RATE` | `22050` | Audio sample rate for lip-sync analysis |

Override settings via environment variables:

```bash
DEVICE=cuda API_PORT=9000 python app.py
```

---

## Testing

Run the full test suite with pytest:

```bash
# Run all tests
pytest tests/ -v

# Run detector unit tests only
pytest tests/test_detectors.py -v

# Run API integration tests only
pytest tests/test_api.py -v
```

Tests cover:
- ✅ Frequency detector spectral analysis
- ✅ Ensemble score aggregation (frame & video level)
- ✅ Lip-sync detector (with and without audio)
- ✅ API endpoints (`/api/health`, `/api/analyze`, `/api/results`)
- ✅ File upload validation and error handling

---

## Technologies Used

| Category | Technologies |
|---|---|
| **Backend** | Python 3.10+, FastAPI 0.115, Uvicorn, Pydantic |
| **Deep Learning** | PyTorch 2.4, TorchVision, timm (XceptionNet) |
| **Computer Vision** | OpenCV, Pillow, MediaPipe |
| **Audio Processing** | librosa, soundfile |
| **Scientific Computing** | NumPy, SciPy (FFT, signal processing) |
| **Visualization** | Matplotlib (Grad-CAM heatmaps, colormaps) |
| **Frontend** | Next.js 16, React 19, TypeScript 5, CSS Modules |
| **Testing** | pytest, httpx |

For the complete technology stack with versions, see [`TECHNOLOGIES.md`](TECHNOLOGIES.md).

---

## Contributing

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Built with ❤️ by <a href="https://github.com/CodeBot69">CodeBot69</a>
</p>
