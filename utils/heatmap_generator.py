"""
Heatmap Generator — Grad-CAM overlay visualization.
Produces visually informative heatmap overlays on face images.
"""
import numpy as np
import cv2
from pathlib import Path
from typing import Optional
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class HeatmapGenerator:
    """Generates and overlays heatmap visualizations for deepfake detection.

    Creates Grad-CAM style heatmap overlays showing which facial regions
    the model flagged as potentially manipulated.
    """

    def __init__(self, colormap: str = "jet", alpha: float = 0.4):
        """
        Args:
            colormap: Matplotlib colormap name for heatmap coloring
            alpha: Transparency of heatmap overlay (0=transparent, 1=opaque)
        """
        self.colormap = colormap
        self.alpha = alpha
        self.cmap = matplotlib.colormaps[colormap]

    def overlay_heatmap(
        self,
        frame: np.ndarray,
        heatmap: np.ndarray,
        face_bbox: Optional[tuple] = None,
    ) -> np.ndarray:
        """Overlay heatmap on a video frame.

        Args:
            frame: Original BGR frame (H, W, 3)
            heatmap: Heatmap array (values 0-1)
            face_bbox: Optional (x1, y1, x2, y2) to place heatmap only on face

        Returns:
            BGR frame with heatmap overlay
        """
        output = frame.copy()

        if face_bbox:
            x1, y1, x2, y2 = face_bbox
            region = output[y1:y2, x1:x2]
            heatmap_resized = cv2.resize(heatmap, (region.shape[1], region.shape[0]))

            # Apply colormap
            colored = self._apply_colormap(heatmap_resized)
            blended = cv2.addWeighted(region, 1 - self.alpha, colored, self.alpha, 0)
            output[y1:y2, x1:x2] = blended
        else:
            heatmap_resized = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
            colored = self._apply_colormap(heatmap_resized)
            output = cv2.addWeighted(frame, 1 - self.alpha, colored, self.alpha, 0)

        return output

    def _apply_colormap(self, heatmap: np.ndarray) -> np.ndarray:
        """Apply matplotlib colormap to heatmap and convert to BGR.

        Args:
            heatmap: 2D array with values in [0, 1]

        Returns:
            BGR uint8 array
        """
        colored = self.cmap(heatmap)[:, :, :3]  # Drop alpha channel
        colored = (colored * 255).astype(np.uint8)
        colored = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)
        return colored

    def save_heatmap(
        self,
        frame: np.ndarray,
        heatmap: np.ndarray,
        output_path: str,
        face_bbox: Optional[tuple] = None,
        add_colorbar: bool = True,
    ) -> str:
        """Save heatmap overlay as an image file.

        Args:
            frame: Original BGR frame
            heatmap: Heatmap array (values 0-1)
            output_path: Where to save the image
            face_bbox: Optional face bounding box
            add_colorbar: Whether to add a colorbar legend

        Returns:
            Path to saved file
        """
        overlaid = self.overlay_heatmap(frame, heatmap, face_bbox)

        if add_colorbar:
            fig, (ax_img, ax_cb) = plt.subplots(
                1, 2, figsize=(12, 8),
                gridspec_kw={"width_ratios": [20, 1]}
            )

            rgb = cv2.cvtColor(overlaid, cv2.COLOR_BGR2RGB)
            ax_img.imshow(rgb)
            ax_img.set_title("Deepfake Detection Heatmap", fontsize=14, fontweight="bold")
            ax_img.axis("off")

            # Colorbar
            norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
            cb = matplotlib.colorbar.ColorbarBase(
                ax_cb, cmap=self.cmap, norm=norm, orientation="vertical"
            )
            cb.set_label("Manipulation Probability", fontsize=11)

            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
            plt.close(fig)
        else:
            cv2.imwrite(output_path, overlaid)

        return output_path

    def save_heatmap_transparent(
        self,
        heatmap: np.ndarray,
        output_path: str,
        size: tuple = None,
    ) -> str:
        """Save heatmap as a transparent PNG for frontend overlay.

        Args:
            heatmap: 2D array with values in [0, 1]
            output_path: Where to save the PNG
            size: Optional (width, height) to resize

        Returns:
            Path to saved file
        """
        if size:
            heatmap = cv2.resize(heatmap, size)

        # Create RGBA image
        colored = self.cmap(heatmap)  # [H, W, 4] with alpha
        # Set alpha based on heatmap intensity
        colored[:, :, 3] = heatmap * self.alpha

        rgba = (colored * 255).astype(np.uint8)
        rgba_bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)

        cv2.imwrite(output_path, rgba_bgra)
        return output_path

    def create_comparison(
        self,
        original: np.ndarray,
        overlaid: np.ndarray,
        output_path: str,
        score: float,
    ) -> str:
        """Create a side-by-side comparison image.

        Args:
            original: Original BGR frame
            overlaid: Frame with heatmap overlay
            output_path: Where to save
            score: Detection score to display

        Returns:
            Path to saved file
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original", fontsize=14)
        axes[0].axis("off")

        axes[1].imshow(cv2.cvtColor(overlaid, cv2.COLOR_BGR2RGB))
        verdict = "FAKE" if score >= 0.5 else "REAL"
        color = "#ff4444" if score >= 0.5 else "#44ff44"
        axes[1].set_title(
            f"Analysis: {verdict} (Score: {score:.2%})",
            fontsize=14, fontweight="bold", color=color
        )
        axes[1].axis("off")

        plt.suptitle("Deepfake Analysis Result", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return output_path
