"""
Lip-Sync Consistency Detector — Audio-Visual Temporal Analysis
Detects temporal mismatches between audio speech patterns and lip movements.
Dubbed or manipulated videos often have low correlation between mouth
movement and audio energy.
"""
import numpy as np
import cv2
from typing import Dict, List, Optional
import config


class _FaceMeshStub:
    """No-op stub for mediapipe FaceMesh when mp.solutions is unavailable (mediapipe >= 0.10)."""

    class _Result:
        multi_face_landmarks = None  # No faces detected

    def process(self, _frame):
        return self._Result()

    def close(self):
        pass


class LipSyncDetector:
    """Detects audio-visual lip-sync inconsistencies.

    Approach:
    1. Extract lip landmarks from video frames using MediaPipe Face Mesh
    2. Compute mouth openness signal over time
    3. Extract audio energy envelope using librosa MFCC
    4. Measure temporal correlation between the two signals
    5. Low correlation → likely manipulated content
    """

    def __init__(self):
        self._face_mesh = None  # Lazy-loaded
        self.audio_sr = config.AUDIO_SAMPLE_RATE

    @property
    def face_mesh(self):
        """Lazy-load MediaPipe Face Mesh to avoid import-time overhead.

        Handles both mediapipe <0.10 (mp.solutions) and >=0.10 (no solutions).
        Falls back to a no-op stub when neither API is available.
        """
        if self._face_mesh is None:
            import mediapipe as mp
            try:
                # mediapipe < 0.10 — legacy solutions API
                self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=config.FACE_DETECTION_CONFIDENCE,
                    min_tracking_confidence=0.5,
                )
            except AttributeError:
                # mediapipe >= 0.10 removed mp.solutions — use stub
                self._face_mesh = _FaceMeshStub()
        return self._face_mesh

    def analyze(self, frames: List[np.ndarray], audio_path: Optional[str] = None) -> Dict:
        """Analyze lip-sync consistency across video frames.

        Args:
            frames: List of BGR numpy arrays (video frames)
            audio_path: Path to extracted audio file (WAV)

        Returns:
            Dict with 'score' (0-1), lip movement data, and analysis details.
        """
        # Extract lip movement signal
        lip_signal = self._extract_lip_movement(frames)

        if audio_path is None or len(lip_signal) < 5:
            # Can't do full lip-sync without audio — use lip movement variability
            return self._analyze_lip_only(lip_signal)

        # Extract audio energy envelope
        audio_signal = self._extract_audio_energy(audio_path, len(lip_signal))

        if audio_signal is None or len(audio_signal) < 5:
            return self._analyze_lip_only(lip_signal)

        # Compute correlation between lip movement and audio energy
        correlation = self._compute_sync_score(lip_signal, audio_signal)

        # Low correlation = likely fake (mis-synced audio)
        # Natural speech: correlation > 0.3, Fake: correlation < 0.15
        fake_score = 1.0 - np.clip((correlation + 0.2) / 0.7, 0.0, 1.0)

        return {
            "score": float(fake_score),
            "correlation": float(correlation),
            "lip_movement_signal": lip_signal.tolist(),
            "audio_available": True,
            "analysis_type": "full_lipsync",
            "details": {
                "num_frames_analyzed": len(frames),
                "lip_movement_variance": float(np.var(lip_signal)),
                "audio_energy_variance": float(np.var(audio_signal)),
            }
        }

    def _extract_lip_movement(self, frames: List[np.ndarray]) -> np.ndarray:
        """Extract mouth openness signal from video frames using face landmarks.

        Uses MediaPipe's lip landmarks to compute the ratio of vertical to
        horizontal mouth opening for each frame.
        """
        openness_values = []

        # Upper and lower lip landmark indices in MediaPipe Face Mesh
        UPPER_LIP = 13  # Upper lip center
        LOWER_LIP = 14  # Lower lip center
        LEFT_MOUTH = 78   # Left corner of mouth
        RIGHT_MOUTH = 308  # Right corner of mouth

        for frame in frames:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                h, w = frame.shape[:2]

                # Vertical opening (upper to lower lip)
                upper = np.array([landmarks[UPPER_LIP].x * w, landmarks[UPPER_LIP].y * h])
                lower = np.array([landmarks[LOWER_LIP].x * w, landmarks[LOWER_LIP].y * h])
                vertical = np.linalg.norm(upper - lower)

                # Horizontal span (mouth width)
                left = np.array([landmarks[LEFT_MOUTH].x * w, landmarks[LEFT_MOUTH].y * h])
                right = np.array([landmarks[RIGHT_MOUTH].x * w, landmarks[RIGHT_MOUTH].y * h])
                horizontal = np.linalg.norm(left - right) + 1e-6

                # Openness ratio
                openness = vertical / horizontal
                openness_values.append(openness)
            else:
                # No face detected — use previous value or 0
                openness_values.append(openness_values[-1] if openness_values else 0.0)

        return np.array(openness_values, dtype=np.float64)

    def _extract_audio_energy(self, audio_path: str, num_frames: int) -> Optional[np.ndarray]:
        """Extract audio energy envelope aligned with video frames.

        Uses librosa to compute RMS energy and resample to match frame count.
        """
        try:
            import librosa
            y, sr = librosa.load(audio_path, sr=self.audio_sr)

            if len(y) == 0:
                return None

            # Compute RMS energy
            hop_length = 512
            rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

            # Resample to match number of video frames
            if len(rms) == 0:
                return None

            indices = np.linspace(0, len(rms) - 1, num_frames).astype(int)
            resampled = rms[indices]

            # Normalize
            if resampled.max() > 0:
                resampled = resampled / resampled.max()

            return resampled

        except Exception:
            return None

    def _compute_sync_score(self, lip_signal: np.ndarray, audio_signal: np.ndarray) -> float:
        """Compute normalized cross-correlation between lip and audio signals.

        Higher correlation = better sync = more likely real.
        """
        # Ensure same length
        min_len = min(len(lip_signal), len(audio_signal))
        lip = lip_signal[:min_len]
        audio = audio_signal[:min_len]

        # Zero-mean normalization
        lip_centered = lip - lip.mean()
        audio_centered = audio - audio.mean()

        # Normalized cross-correlation
        lip_std = lip_centered.std() + 1e-10
        audio_std = audio_centered.std() + 1e-10

        correlation = np.correlate(
            lip_centered / lip_std,
            audio_centered / audio_std,
            mode='full'
        )
        max_corr = correlation.max() / min_len

        return float(np.clip(max_corr, -1.0, 1.0))

    def _analyze_lip_only(self, lip_signal: np.ndarray) -> Dict:
        """Fallback analysis when audio is not available.

        Analyzes lip movement patterns for unnaturally smooth or jerky motion,
        which can indicate face replacement without proper temporal coherence.
        """
        if len(lip_signal) < 3:
            return {
                "score": 0.5,  # Uncertain
                "correlation": None,
                "lip_movement_signal": lip_signal.tolist(),
                "audio_available": False,
                "analysis_type": "lip_only",
                "details": {"reason": "insufficient_frames"},
            }

        # Compute motion smoothness
        diffs = np.diff(lip_signal)
        jerk = np.diff(diffs)  # Second derivative

        # Unnatural motion: either too smooth (synthetic) or too jerky
        smoothness = 1.0 / (1.0 + np.var(jerk) * 100)
        jerkiness = np.mean(np.abs(jerk))

        # Combined score: suspicious if too smooth OR too jerky
        if smoothness > 0.9:
            score = 0.6  # Suspiciously smooth
        elif jerkiness > 0.1:
            score = 0.55  # Suspiciously jerky
        else:
            score = 0.3  # Appears natural

        return {
            "score": float(score),
            "correlation": None,
            "lip_movement_signal": lip_signal.tolist(),
            "audio_available": False,
            "analysis_type": "lip_only",
            "details": {
                "smoothness": float(smoothness),
                "jerkiness": float(jerkiness),
                "num_frames_analyzed": len(lip_signal),
            }
        }

    def __del__(self):
        if self._face_mesh is not None:
            self._face_mesh.close()
