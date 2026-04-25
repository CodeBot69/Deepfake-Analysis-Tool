"""
Frequency Domain Detector — FFT-based GAN Fingerprint Analysis
Detects GAN-specific periodic artifacts in the frequency spectrum of face images.
GANs tend to leave distinctive patterns in the 2D Fourier transform.
"""
import numpy as np
from scipy import fft as scipy_fft
from typing import Dict, Optional
import cv2


class FrequencyDetector:
    """Detects deepfakes by analyzing frequency domain artifacts.

    GAN-generated images exhibit characteristic periodic patterns in their
    2D Fourier transform (spectral peaks at specific frequencies) due to
    the transposed convolution operations used in generators.

    This detector:
    1. Computes 2D FFT of the face crop
    2. Extracts azimuthal average of the power spectrum
    3. Analyzes spectral features for GAN fingerprints
    4. Returns a fakeness score based on spectral anomalies
    """

    def __init__(self):
        self.spectral_size = 256  # Resize face crop for FFT analysis
        # Thresholds calibrated for common GAN architectures
        self.high_freq_weight = 0.4
        self.periodicity_weight = 0.35
        self.falloff_weight = 0.25

    def analyze_frame(self, frame: np.ndarray) -> Dict:
        """Analyze a single frame for frequency domain artifacts.

        Args:
            frame: BGR numpy array (H, W, 3)

        Returns:
            Dict with 'score', 'features', and 'spectrum' visualization.
        """
        # Convert to grayscale for spectral analysis
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Resize to standard size for consistent FFT
        gray_resized = cv2.resize(gray, (self.spectral_size, self.spectral_size))

        # Apply window function to reduce spectral leakage
        window = np.outer(
            np.hanning(self.spectral_size),
            np.hanning(self.spectral_size)
        )
        windowed = gray_resized.astype(np.float64) * window

        # 2D FFT
        f_transform = scipy_fft.fft2(windowed)
        f_shift = scipy_fft.fftshift(f_transform)
        magnitude_spectrum = np.log1p(np.abs(f_shift))

        # Extract features
        azimuthal_avg = self._azimuthal_average(magnitude_spectrum)
        high_freq_energy = self._high_frequency_energy(azimuthal_avg)
        periodicity_score = self._detect_periodicity(azimuthal_avg)
        spectral_falloff = self._spectral_falloff_rate(azimuthal_avg)

        # Combine features into fakeness score
        score = (
            self.high_freq_weight * high_freq_energy +
            self.periodicity_weight * periodicity_score +
            self.falloff_weight * (1.0 - spectral_falloff)  # Flatter falloff → more fake
        )
        score = float(np.clip(score, 0.0, 1.0))

        return {
            "score": score,
            "features": {
                "high_freq_energy": float(high_freq_energy),
                "periodicity": float(periodicity_score),
                "spectral_falloff": float(spectral_falloff),
            },
            "spectrum": magnitude_spectrum,
        }

    def _azimuthal_average(self, spectrum: np.ndarray) -> np.ndarray:
        """Compute azimuthal (radial) average of 2D power spectrum.

        This averages the spectrum over all angles at each radial distance
        from the center, producing a 1D profile of spectral energy vs frequency.
        """
        h, w = spectrum.shape
        center_y, center_x = h // 2, w // 2

        Y, X = np.ogrid[:h, :w]
        r = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2).astype(int)

        max_radius = min(center_x, center_y)
        radial_profile = np.zeros(max_radius)

        for radius in range(max_radius):
            mask = r == radius
            if mask.any():
                radial_profile[radius] = spectrum[mask].mean()

        return radial_profile

    def _high_frequency_energy(self, azimuthal: np.ndarray) -> float:
        """Measure energy concentration in high frequency bands.

        GAN-generated images often have unusual energy in high-frequency
        components compared to natural images.
        """
        n = len(azimuthal)
        if n == 0:
            return 0.0

        # Split into low and high frequency bands
        mid = n // 2
        low_energy = np.sum(azimuthal[:mid]) + 1e-10
        high_energy = np.sum(azimuthal[mid:])
        ratio = high_energy / low_energy

        # Normalize: natural images typically have ratio < 0.1
        # GAN images can have ratio > 0.2
        normalized = np.clip(ratio / 0.4, 0.0, 1.0)
        return float(normalized)

    def _detect_periodicity(self, azimuthal: np.ndarray) -> float:
        """Detect periodic peaks in the radial spectrum.

        GAN generators using strided transposed convolutions create
        periodic artifacts that manifest as peaks at regular intervals.
        """
        if len(azimuthal) < 10:
            return 0.0

        # Remove trend
        from scipy.signal import detrend
        detrended = detrend(azimuthal)

        # Autocorrelation to find periodicity
        autocorr = np.correlate(detrended, detrended, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]  # Keep positive lags

        if len(autocorr) < 2:
            return 0.0

        # Normalize
        autocorr = autocorr / (autocorr[0] + 1e-10)

        # Find peaks in autocorrelation (excluding lag 0)
        peaks = []
        for i in range(2, len(autocorr) - 1):
            if autocorr[i] > autocorr[i - 1] and autocorr[i] > autocorr[i + 1]:
                peaks.append(autocorr[i])

        if not peaks:
            return 0.0

        # Strong periodic peaks indicate GAN artifacts
        max_peak = max(peaks)
        return float(np.clip(max_peak * 2.0, 0.0, 1.0))

    def _spectral_falloff_rate(self, azimuthal: np.ndarray) -> float:
        """Measure how quickly spectral energy falls off with frequency.

        Natural images follow a 1/f^α power law. GAN images often have
        a flatter falloff (lower α), meaning more energy at high frequencies.
        """
        n = len(azimuthal)
        if n < 5:
            return 0.5

        # Fit log-log regression to estimate power law exponent
        freqs = np.arange(1, n)
        powers = azimuthal[1:]

        # Filter zeros
        valid = powers > 0
        if valid.sum() < 5:
            return 0.5

        log_freqs = np.log(freqs[valid])
        log_powers = np.log(powers[valid])

        # Linear regression in log-log space
        coeffs = np.polyfit(log_freqs, log_powers, 1)
        alpha = -coeffs[0]  # Negative slope = falloff rate

        # Natural images: α ≈ 1.5-2.5, GAN images: α ≈ 0.5-1.5
        # Normalize to [0, 1] where 1 = steep (natural), 0 = flat (GAN)
        normalized = np.clip((alpha - 0.5) / 2.0, 0.0, 1.0)
        return float(normalized)

    def analyze_frames(self, frames: list) -> list:
        """Analyze multiple frames."""
        return [self.analyze_frame(f) for f in frames]
