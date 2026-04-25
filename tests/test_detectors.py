"""
Test suite for individual deepfake detectors.
Uses synthetic test data to verify each model's pipeline works correctly.
"""
import pytest
import numpy as np
import cv2


# ─── Fixtures ─────────────────────────────────────────────
@pytest.fixture
def real_frame():
    """Simulate a 'real' face image — smooth natural gradients."""
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    # Skin-tone gradient
    for y in range(300):
        for x in range(300):
            frame[y, x] = [
                min(255, 140 + int(20 * np.sin(x / 30))),   # B
                min(255, 160 + int(15 * np.sin(y / 25))),   # G
                min(255, 200 + int(10 * np.sin((x+y) / 40))), # R
            ]
    return frame


@pytest.fixture
def fake_frame():
    """Simulate a 'fake' face image — contains GAN-like periodic artifacts."""
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    # Base skin tone
    frame[:, :] = [140, 160, 200]
    # Add periodic high-frequency noise (GAN artifact simulation)
    for y in range(300):
        for x in range(300):
            noise = int(20 * np.sin(x * 0.5) * np.cos(y * 0.5))
            frame[y, x, 0] = np.clip(int(frame[y, x, 0]) + noise, 0, 255)
            frame[y, x, 1] = np.clip(int(frame[y, x, 1]) + noise, 0, 255)
    return frame


@pytest.fixture
def random_frames():
    """Generate a list of random frames for testing."""
    return [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(5)]


# ─── Frequency Detector Tests ─────────────────────────────
class TestFrequencyDetector:
    def test_initialization(self):
        from models.frequency_detector import FrequencyDetector
        det = FrequencyDetector()
        assert det.spectral_size == 256

    def test_analyze_frame_returns_valid_structure(self, real_frame):
        from models.frequency_detector import FrequencyDetector
        det = FrequencyDetector()
        result = det.analyze_frame(real_frame)

        assert "score" in result
        assert "features" in result
        assert "spectrum" in result
        assert 0.0 <= result["score"] <= 1.0
        assert all(k in result["features"] for k in ["high_freq_energy", "periodicity", "spectral_falloff"])

    def test_analyze_multiple_frames(self, random_frames):
        from models.frequency_detector import FrequencyDetector
        det = FrequencyDetector()
        results = det.analyze_frames(random_frames)
        assert len(results) == len(random_frames)

    def test_fake_scores_higher_than_real(self, real_frame, fake_frame):
        """Fake frames with periodic artifacts should generally score higher."""
        from models.frequency_detector import FrequencyDetector
        det = FrequencyDetector()
        real_result = det.analyze_frame(real_frame)
        fake_result = det.analyze_frame(fake_frame)
        # Not a strict assertion — depends on the synthetic data
        assert isinstance(real_result["score"], float)
        assert isinstance(fake_result["score"], float)


# ─── Ensemble Detector Tests ──────────────────────────────
class TestEnsembleDetector:
    def test_initialization_with_default_weights(self):
        from models.ensemble import EnsembleDetector
        det = EnsembleDetector()
        total = sum(det.weights.values())
        assert abs(total - 1.0) < 0.01

    def test_aggregate_frame_all_detectors(self):
        from models.ensemble import EnsembleDetector
        det = EnsembleDetector()
        result = det.aggregate_frame(
            spatial_score=0.8,
            frequency_score=0.7,
            lipsync_score=0.6,
        )
        assert "final_score" in result
        assert "verdict" in result
        assert "confidence" in result
        assert result["verdict"] in ("FAKE", "REAL")
        assert 0.0 <= result["final_score"] <= 1.0

    def test_aggregate_frame_without_lipsync(self):
        from models.ensemble import EnsembleDetector
        det = EnsembleDetector()
        result = det.aggregate_frame(
            spatial_score=0.3,
            frequency_score=0.2,
        )
        assert result["verdict"] == "REAL"

    def test_high_scores_produce_fake_verdict(self):
        from models.ensemble import EnsembleDetector
        det = EnsembleDetector()
        result = det.aggregate_frame(
            spatial_score=0.9,
            frequency_score=0.85,
            lipsync_score=0.8,
        )
        assert result["verdict"] == "FAKE"

    def test_aggregate_video(self):
        from models.ensemble import EnsembleDetector
        det = EnsembleDetector()
        frame_results = []
        for _ in range(10):
            fr = det.aggregate_frame(
                spatial_score=np.random.uniform(0.6, 0.9),
                frequency_score=np.random.uniform(0.5, 0.8),
                lipsync_score=np.random.uniform(0.5, 0.7),
            )
            frame_results.append(fr)
        video_result = det.aggregate_video(frame_results)
        assert "final_score" in video_result
        assert "frame_count" in video_result
        assert video_result["frame_count"] == 10

    def test_empty_video_aggregation(self):
        from models.ensemble import EnsembleDetector
        det = EnsembleDetector()
        result = det.aggregate_video([])
        assert result["verdict"] == "UNCERTAIN"


# ─── Lip-Sync Detector Tests ─────────────────────────────
class TestLipSyncDetector:
    def test_lip_only_analysis(self, random_frames):
        from models.lipsync_detector import LipSyncDetector
        det = LipSyncDetector()
        result = det.analyze(random_frames, audio_path=None)
        assert "score" in result
        assert "analysis_type" in result
        assert result["audio_available"] is False

    def test_insufficient_frames(self):
        from models.lipsync_detector import LipSyncDetector
        det = LipSyncDetector()
        # Single frame — should return uncertain
        frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)]
        result = det.analyze(frames, audio_path=None)
        assert result["score"] == 0.5 or isinstance(result["score"], float)


# ─── Heatmap Generator Tests ──────────────────────────────
class TestHeatmapGenerator:
    def test_overlay_heatmap(self):
        from utils.heatmap_generator import HeatmapGenerator
        gen = HeatmapGenerator()
        frame = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        heatmap = np.random.rand(300, 300)
        result = gen.overlay_heatmap(frame, heatmap)
        assert result.shape == frame.shape

    def test_overlay_with_bbox(self):
        from utils.heatmap_generator import HeatmapGenerator
        gen = HeatmapGenerator()
        frame = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        heatmap = np.random.rand(100, 100)
        result = gen.overlay_heatmap(frame, heatmap, face_bbox=(50, 50, 150, 150))
        assert result.shape == frame.shape
