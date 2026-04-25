"""
Video Processor — Frame extraction, face cropping, and audio extraction.
"""
import cv2
import numpy as np
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional
import config


class VideoProcessor:
    """Processes video files for deepfake analysis.

    Handles:
    - Frame extraction at configurable sample rate
    - Face detection and cropping using MediaPipe
    - Audio track extraction to WAV using ffmpeg
    """

    def __init__(self):
        self._face_detection = None  # Lazy-loaded

    @property
    def face_detection(self):
        if self._face_detection is None:
            import mediapipe as mp
            self._face_detection = mp.solutions.face_detection.FaceDetection(
                model_selection=1,  # Full-range model
                min_detection_confidence=config.FACE_DETECTION_CONFIDENCE,
            )
        return self._face_detection

    def extract_frames(
        self,
        video_path: str,
        sample_rate: int = None,
        max_frames: int = None,
    ) -> Tuple[List[np.ndarray], dict]:
        """Extract frames from video at a specified sample rate.

        Args:
            video_path: Path to the video file
            sample_rate: Extract every Nth frame (default from config)
            max_frames: Maximum number of frames to extract

        Returns:
            Tuple of (list of BGR frames, video metadata dict)
        """
        sample_rate = sample_rate or config.FRAME_SAMPLE_RATE
        max_frames = max_frames or config.MAX_FRAMES_TO_ANALYZE

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        metadata = {
            "fps": fps,
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "duration": total_frames / fps if fps > 0 else 0,
        }

        frames = []
        frame_idx = 0

        while cap.isOpened() and len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_rate == 0:
                frames.append(frame)

            frame_idx += 1

        cap.release()

        metadata["frames_extracted"] = len(frames)
        metadata["sample_rate"] = sample_rate
        return frames, metadata

    def detect_and_crop_faces(
        self,
        frame: np.ndarray,
        padding: float = 0.3,
    ) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """Detect faces in a frame and return cropped face regions.

        Args:
            frame: BGR numpy array
            padding: Fraction of face size to add as padding

        Returns:
            List of (cropped_face, (x1, y1, x2, y2)) tuples
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb)

        faces = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box

                # Convert relative to absolute coordinates
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)

                # Add padding
                pad_x = int(bw * padding)
                pad_y = int(bh * padding)
                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_y)
                x2 = min(w, x1 + bw + 2 * pad_x)
                y2 = min(h, y1 + bh + 2 * pad_y)

                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    faces.append((crop, (x1, y1, x2, y2)))

        return faces

    def extract_audio(self, video_path: str, output_dir: str = None) -> Optional[str]:
        """Extract audio track from video to WAV file.

        Uses ffmpeg subprocess for reliable audio extraction.

        Args:
            video_path: Path to the video file
            output_dir: Directory to save the WAV file

        Returns:
            Path to the extracted WAV file, or None if extraction fails.
        """
        if output_dir:
            audio_path = str(Path(output_dir) / "audio.wav")
        else:
            audio_path = tempfile.mktemp(suffix=".wav")

        try:
            cmd = [
                "ffmpeg", "-i", video_path,
                "-vn",  # No video
                "-acodec", "pcm_s16le",
                "-ar", str(config.AUDIO_SAMPLE_RATE),
                "-ac", "1",  # Mono
                "-y",  # Overwrite
                audio_path,
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=60,
            )
            if result.returncode == 0 and Path(audio_path).exists():
                return audio_path
            return None
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None

    def is_image(self, file_path: str) -> bool:
        """Check if the file is an image (not video)."""
        ext = Path(file_path).suffix.lower()
        return ext in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    def is_video(self, file_path: str) -> bool:
        """Check if the file is a video."""
        ext = Path(file_path).suffix.lower()
        return ext in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}

    def load_image(self, image_path: str) -> np.ndarray:
        """Load an image file as BGR numpy array."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot load image: {image_path}")
        return img

    def __del__(self):
        if self._face_detection is not None:
            self._face_detection.close()
