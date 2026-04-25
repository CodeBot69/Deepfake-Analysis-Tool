"""
Deepfake Analysis Tool — FastAPI Backend
High-performance API for multi-model ensemble deepfake detection.
"""
import uuid
import json
import time
import shutil
import traceback
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import numpy as np
import cv2

import config
from utils.video_processor import VideoProcessor
from utils.heatmap_generator import HeatmapGenerator

# ─── App Setup ────────────────────────────────────────────
app = FastAPI(
    title="Deepfake Analysis Tool",
    description="High-precision deepfake detection using multi-model ensemble analysis",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static heatmaps & results
app.mount("/static/results", StaticFiles(directory=str(config.RESULTS_DIR)), name="results")

# ─── Global State ─────────────────────────────────────────
# In-memory job store (use Redis/DB in production)
jobs: dict = {}
video_processor = VideoProcessor()
heatmap_generator = HeatmapGenerator()

# Lazy-loaded detectors (heavy models loaded on first use)
# _UNSET sentinel distinguishes "not yet tried" from "tried but unavailable"
_UNSET = object()
_spatial_detector = _UNSET
_frequency_detector = None
_lipsync_detector = None
_ensemble_detector = None


def get_spatial_detector():
    global _spatial_detector
    if _spatial_detector is _UNSET:
        try:
            from models.spatial_detector import SpatialDetector
            _spatial_detector = SpatialDetector()
        except (OSError, ImportError) as e:
            # torch/DLL unavailable in this environment — skip spatial analysis
            import logging
            logging.getLogger(__name__).warning(
                "SpatialDetector unavailable (torch not loaded): %s", e
            )
            _spatial_detector = None
    return _spatial_detector


def get_frequency_detector():
    global _frequency_detector
    if _frequency_detector is None:
        from models.frequency_detector import FrequencyDetector
        _frequency_detector = FrequencyDetector()
    return _frequency_detector


def get_lipsync_detector():
    global _lipsync_detector
    if _lipsync_detector is None:
        from models.lipsync_detector import LipSyncDetector
        _lipsync_detector = LipSyncDetector()
    return _lipsync_detector


def get_ensemble_detector():
    global _ensemble_detector
    if _ensemble_detector is None:
        from models.ensemble import EnsembleDetector
        _ensemble_detector = EnsembleDetector()
    return _ensemble_detector


# ─── Schemas ──────────────────────────────────────────────
class AnalysisStatus(BaseModel):
    job_id: str
    status: str  # "queued", "processing", "completed", "failed"
    progress: float = 0.0
    result: Optional[dict] = None
    error: Optional[str] = None


# ─── Endpoints ────────────────────────────────────────────
@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "device": config.DEVICE,
        "ensemble_weights": config.ENSEMBLE_WEIGHTS,
    }


@app.post("/api/analyze")
async def analyze_media(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """Upload a video or image for full ensemble deepfake analysis.

    Returns a job_id to poll for results at /api/results/{job_id}.
    """
    # Validate file extension
    ext = Path(file.filename).suffix.lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {config.ALLOWED_EXTENSIONS}"
        )

    # Save uploaded file
    job_id = str(uuid.uuid4())
    job_dir = config.UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    file_path = job_dir / f"input{ext}"
    with open(file_path, "wb") as f:
        content = await file.read()
        if len(content) > config.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
            shutil.rmtree(job_dir)
            raise HTTPException(status_code=413, detail="File too large")
        f.write(content)

    # Initialize job status
    jobs[job_id] = {
        "status": "queued",
        "progress": 0.0,
        "file_path": str(file_path),
        "file_name": file.filename,
        "result": None,
        "error": None,
        "created_at": time.time(),
    }

    # Run analysis in background
    background_tasks.add_task(run_analysis, job_id, str(file_path))

    return {"job_id": job_id, "status": "queued", "message": "Analysis started"}


@app.post("/api/analyze/frame")
async def analyze_single_frame(file: UploadFile = File(...)):
    """Analyze a single image frame synchronously.

    Returns immediate results without background processing.
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        raise HTTPException(status_code=400, detail=f"Image format required, got {ext}")

    content = await file.read()
    nparr = np.frombuffer(content, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Cannot decode image")

    try:
        result = analyze_frame_sync(frame)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/results/{job_id}")
async def get_results(job_id: str):
    """Retrieve analysis results for a job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    return AnalysisStatus(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        result=job["result"],
        error=job["error"],
    )


@app.get("/api/jobs")
async def list_jobs():
    """List all analysis jobs."""
    return {
        jid: {
            "status": j["status"],
            "progress": j["progress"],
            "file_name": j.get("file_name"),
            "created_at": j.get("created_at"),
        }
        for jid, j in jobs.items()
    }


# ─── Analysis Logic ──────────────────────────────────────
def analyze_frame_sync(frame: np.ndarray) -> dict:
    """Run all detectors on a single frame synchronously.

    SpatialDetector is skipped gracefully when torch is unavailable
    (e.g. missing fbgemm.dll), falling back to frequency-only ensemble.
    """
    spatial = get_spatial_detector()  # May be None if torch not available
    frequency = get_frequency_detector()
    ensemble = get_ensemble_detector()

    # Detect faces
    faces = video_processor.detect_and_crop_faces(frame)
    if not faces:
        # Analyze whole frame if no face detected
        faces = [(frame, (0, 0, frame.shape[1], frame.shape[0]))]

    face_results = []
    for face_crop, bbox in faces:
        frequency_result = frequency.analyze_frame(face_crop)

        spatial_score = None
        heatmap = None
        if spatial is not None:
            spatial_result = spatial.analyze_frame(face_crop)
            spatial_score = spatial_result["score"]
            heatmap = spatial_result.get("heatmap")

        # Build ensemble — skip spatial when unavailable
        ensemble_kwargs = {"frequency_score": frequency_result["score"]}
        if spatial_score is not None:
            ensemble_kwargs["spatial_score"] = spatial_score
        else:
            # Provide a neutral spatial score so ensemble still works
            ensemble_kwargs["spatial_score"] = frequency_result["score"]

        ensemble_result = ensemble.aggregate_frame(**ensemble_kwargs)

        # Generate and save heatmap (only when spatial model available)
        heatmap_url = None
        if heatmap is not None:
            heatmap_id = str(uuid.uuid4())[:8]
            heatmap_path = config.HEATMAPS_DIR / f"{heatmap_id}.png"
            heatmap_generator.save_heatmap(
                frame, heatmap, str(heatmap_path), face_bbox=bbox
            )
            heatmap_url = f"/static/results/heatmaps/{heatmap_id}.png"

        face_results.append({
            "bbox": list(bbox),
            "spatial_score": spatial_score,
            "frequency_score": frequency_result["score"],
            "spatial_available": spatial is not None,
            "ensemble": ensemble_result,
            "heatmap_url": heatmap_url,
        })

    return {
        "type": "image",
        "faces_detected": len(face_results),
        "faces": face_results,
        "overall": face_results[0]["ensemble"] if face_results else None,
    }


def run_analysis(job_id: str, file_path: str):
    """Background task for full video/image analysis."""
    try:
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["progress"] = 0.05

        is_image = video_processor.is_image(file_path)

        if is_image:
            frame = video_processor.load_image(file_path)
            jobs[job_id]["progress"] = 0.2
            result = analyze_frame_sync(frame)
            jobs[job_id]["progress"] = 0.9

            # Save original frame for frontend
            frame_id = str(uuid.uuid4())[:8]
            frame_path = config.FRAMES_DIR / f"{frame_id}.jpg"
            cv2.imwrite(str(frame_path), frame)
            result["frames"] = [{"url": f"/static/results/frames/{frame_id}.jpg", "index": 0}]

        else:
            # Video analysis
            result = analyze_video(job_id, file_path)

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 1.0
        jobs[job_id]["result"] = result

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["progress"] = 0.0
        print(f"Analysis error for {job_id}: {traceback.format_exc()}")


def analyze_video(job_id: str, file_path: str) -> dict:
    """Full video analysis pipeline."""
    spatial = get_spatial_detector()
    frequency = get_frequency_detector()
    lipsync = get_lipsync_detector()
    ensemble = get_ensemble_detector()

    # Extract frames
    jobs[job_id]["progress"] = 0.1
    frames, metadata = video_processor.extract_frames(file_path)

    if not frames:
        raise ValueError("No frames could be extracted from video")

    # Extract audio
    jobs[job_id]["progress"] = 0.15
    job_dir = Path(file_path).parent
    audio_path = video_processor.extract_audio(file_path, str(job_dir))

    # Analyze each frame
    frame_results = []
    saved_frames = []
    total = len(frames)

    for i, frame in enumerate(frames):
        progress = 0.2 + (i / total) * 0.6
        jobs[job_id]["progress"] = progress

        # Detect faces
        faces = video_processor.detect_and_crop_faces(frame)
        if not faces:
            faces = [(frame, (0, 0, frame.shape[1], frame.shape[0]))]

        # Use the primary (largest) face
        face_crop, bbox = max(faces, key=lambda f: f[0].shape[0] * f[0].shape[1])

        # Frequency analysis
        frequency_result = frequency.analyze_frame(face_crop)

        # Save frame
        frame_id = f"{job_id[:8]}_f{i:04d}"
        frame_path = config.FRAMES_DIR / f"{frame_id}.jpg"
        cv2.imwrite(str(frame_path), frame)

        # Spatial analysis (graceful fallback)
        heatmap_url = None
        spatial_score = frequency_result["score"]
        
        if spatial is not None:
            spatial_result = spatial.analyze_frame(face_crop)
            spatial_score = spatial_result["score"]
            if spatial_result.get("heatmap") is not None:
                heatmap_path = config.HEATMAPS_DIR / f"{frame_id}_heatmap.png"
                heatmap_generator.save_heatmap(
                    frame, spatial_result["heatmap"], str(heatmap_path), face_bbox=bbox
                )
                heatmap_url = f"/static/results/heatmaps/{frame_id}_heatmap.png"

        saved_frames.append({
            "url": f"/static/results/frames/{frame_id}.jpg",
            "heatmap_url": heatmap_url,
            "index": i,
        })

        frame_results.append({
            "frame_index": i,
            "spatial_score": spatial_score,
            "frequency_score": frequency_result["score"],
        })

    # Lip-sync analysis (whole video)
    jobs[job_id]["progress"] = 0.82
    lipsync_result = lipsync.analyze(frames, audio_path)
    lipsync_score = lipsync_result["score"]

    # Ensemble aggregation per frame
    jobs[job_id]["progress"] = 0.88
    ensemble_frame_results = []
    for fr in frame_results:
        ens = ensemble.aggregate_frame(
            spatial_score=fr["spatial_score"],
            frequency_score=fr["frequency_score"],
            lipsync_score=lipsync_score,
        )
        ensemble_frame_results.append(ens)

    # Video-level aggregation
    jobs[job_id]["progress"] = 0.93
    video_result = ensemble.aggregate_video(ensemble_frame_results)

    return {
        "type": "video",
        "metadata": metadata,
        "overall": video_result,
        "lipsync": {
            "score": lipsync_result["score"],
            "correlation": lipsync_result.get("correlation"),
            "analysis_type": lipsync_result.get("analysis_type"),
        },
        "frame_results": [
            {
                **fr,
                "ensemble": efr,
            }
            for fr, efr in zip(frame_results, ensemble_frame_results)
        ],
        "frames": saved_frames,
    }


# ─── Main ─────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True,
    )
