"""
Deepfake Detection Models Package

Lazy imports to avoid loading heavy dependencies (torch, mediapipe) at package init.
"""


def __getattr__(name):
    if name == "SpatialDetector":
        from .spatial_detector import SpatialDetector
        return SpatialDetector
    elif name == "FrequencyDetector":
        from .frequency_detector import FrequencyDetector
        return FrequencyDetector
    elif name == "LipSyncDetector":
        from .lipsync_detector import LipSyncDetector
        return LipSyncDetector
    elif name == "EnsembleDetector":
        from .ensemble import EnsembleDetector
        return EnsembleDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SpatialDetector",
    "FrequencyDetector",
    "LipSyncDetector",
    "EnsembleDetector",
]
