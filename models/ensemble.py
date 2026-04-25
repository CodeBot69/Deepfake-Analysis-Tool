"""
Ensemble Detector — Multi-Model Weighted Aggregator
Combines scores from spatial, frequency, and lip-sync detectors into a
final deepfake detection verdict with confidence scoring.
"""
import numpy as np
from typing import Dict, List, Optional
import config


class EnsembleDetector:
    """Ensemble aggregator that combines multiple detector outputs.

    Features:
    - Weighted average of detector scores
    - Confidence scoring based on inter-model agreement
    - Per-frame and video-level aggregation
    - Detailed per-detector breakdown
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or config.ENSEMBLE_WEIGHTS
        self._validate_weights()

    def _validate_weights(self):
        total = sum(self.weights.values())
        if not (0.99 < total < 1.01):
            # Normalize weights
            for key in self.weights:
                self.weights[key] /= total

    def aggregate_frame(
        self,
        spatial_score: float,
        frequency_score: float,
        lipsync_score: Optional[float] = None,
    ) -> Dict:
        """Aggregate detector scores for a single frame.

        Args:
            spatial_score: Score from spatial detector (0-1)
            frequency_score: Score from frequency detector (0-1)
            lipsync_score: Score from lip-sync detector (0-1), optional

        Returns:
            Dict with final score, per-detector breakdown, confidence, and verdict.
        """
        scores = {
            "spatial": spatial_score,
            "frequency": frequency_score,
        }

        if lipsync_score is not None:
            scores["lipsync"] = lipsync_score
            weights = self.weights
        else:
            # Redistribute lip-sync weight to other detectors
            weights = {
                "spatial": self.weights["spatial"] / (self.weights["spatial"] + self.weights["frequency"]),
                "frequency": self.weights["frequency"] / (self.weights["spatial"] + self.weights["frequency"]),
            }

        # Weighted average
        final_score = sum(scores[k] * weights[k] for k in scores)

        # Confidence based on inter-model agreement
        score_values = list(scores.values())
        agreement = 1.0 - np.std(score_values) * 2  # Lower std = higher agreement
        confidence = float(np.clip(agreement, 0.0, 1.0))

        # Verdict
        if final_score >= config.FAKE_THRESHOLD:
            verdict = "FAKE"
            if confidence >= config.HIGH_CONFIDENCE:
                confidence_label = "High Confidence"
            elif confidence >= config.LOW_CONFIDENCE:
                confidence_label = "Medium Confidence"
            else:
                confidence_label = "Low Confidence"
        else:
            verdict = "REAL"
            if confidence >= config.HIGH_CONFIDENCE:
                confidence_label = "High Confidence"
            elif confidence >= config.LOW_CONFIDENCE:
                confidence_label = "Medium Confidence"
            else:
                confidence_label = "Low Confidence"

        return {
            "final_score": float(final_score),
            "verdict": verdict,
            "confidence": confidence,
            "confidence_label": confidence_label,
            "per_detector": scores,
            "weights_used": {k: weights[k] for k in scores},
        }

    def aggregate_video(self, frame_results: List[Dict]) -> Dict:
        """Aggregate frame-level results into a video-level verdict.

        Uses temporal statistics across all frames for a robust decision.

        Args:
            frame_results: List of per-frame ensemble results

        Returns:
            Dict with video-level score, confidence, temporal analysis.
        """
        if not frame_results:
            return {
                "final_score": 0.5,
                "verdict": "UNCERTAIN",
                "confidence": 0.0,
                "confidence_label": "No Data",
                "frame_count": 0,
            }

        scores = [r["final_score"] for r in frame_results]
        scores_array = np.array(scores)

        # Video-level score: weighted combination of mean and max
        # (catches both consistent and intermittent fakes)
        mean_score = float(scores_array.mean())
        max_score = float(scores_array.max())
        median_score = float(np.median(scores_array))

        # 60% median + 25% mean + 15% max — robust to outliers
        video_score = 0.60 * median_score + 0.25 * mean_score + 0.15 * max_score

        # Temporal consistency: how stable are scores across frames?
        temporal_std = float(scores_array.std())
        temporal_consistency = 1.0 - min(temporal_std * 3, 1.0)

        # Overall confidence combines per-frame agreement and temporal consistency
        avg_frame_confidence = np.mean([r["confidence"] for r in frame_results])
        overall_confidence = float(0.6 * avg_frame_confidence + 0.4 * temporal_consistency)

        # Verdict
        if video_score >= config.FAKE_THRESHOLD:
            verdict = "FAKE"
        else:
            verdict = "REAL"

        if overall_confidence >= config.HIGH_CONFIDENCE:
            confidence_label = "High Confidence"
        elif overall_confidence >= config.LOW_CONFIDENCE:
            confidence_label = "Medium Confidence"
        else:
            confidence_label = "Low Confidence"

        # Per-detector video-level averages
        per_detector_avg = {}
        for key in frame_results[0]["per_detector"]:
            values = [r["per_detector"][key] for r in frame_results]
            per_detector_avg[key] = {
                "mean": float(np.mean(values)),
                "max": float(np.max(values)),
                "std": float(np.std(values)),
            }

        return {
            "final_score": float(video_score),
            "verdict": verdict,
            "confidence": overall_confidence,
            "confidence_label": confidence_label,
            "frame_count": len(frame_results),
            "temporal_consistency": temporal_consistency,
            "score_statistics": {
                "mean": mean_score,
                "median": median_score,
                "max": max_score,
                "std": temporal_std,
            },
            "per_detector": per_detector_avg,
            "frame_scores": scores,
        }
