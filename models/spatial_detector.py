"""
Spatial Artifact Detector — XceptionNet-based
Detects visual manipulation artifacts in individual frames using a pretrained
Xception network with Grad-CAM heatmap generation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
import timm
from typing import Dict, Tuple, Optional
import config


class GradCAM:
    """Grad-CAM implementation for generating heatmaps showing which regions
    the model focuses on for its deepfake detection decision."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, target_class: int = 1) -> np.ndarray:
        """Generate Grad-CAM heatmap for the given input.

        Args:
            input_tensor: Preprocessed image tensor [1, C, H, W]
            target_class: Class index to generate heatmap for (1=fake)

        Returns:
            Heatmap as numpy array [H, W] with values in [0, 1]
        """
        self.model.eval()
        output = self.model(input_tensor)

        self.model.zero_grad()
        target = output[0, target_class]
        target.backward(retain_graph=True)

        # Pool gradients across spatial dimensions
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # [1, C, 1, 1]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam


class SpatialDetector:
    """XceptionNet-based spatial deepfake detector.

    Uses transfer learning from ImageNet pretrained Xception to detect
    spatial artifacts such as blending boundaries, texture inconsistencies,
    and compression artifacts typical of face manipulation.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = self._resolve_device(device)
        self.model = self._build_model()
        self.grad_cam = self._setup_gradcam()
        self.transform = self._build_transform()

    def _resolve_device(self, device: Optional[str]) -> torch.device:
        if device and device != "auto":
            return torch.device(device)
        if config.DEVICE != "auto":
            return torch.device(config.DEVICE)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_model(self) -> nn.Module:
        """Build Xception model with binary classification head."""
        model = timm.create_model("xception", pretrained=True, num_classes=0)

        # Add binary classification head (real vs fake)
        num_features = model.num_features
        classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 2),  # 2 classes: real, fake
        )

        # Wrap into a full model
        full_model = XceptionWrapper(model, classifier)
        full_model.to(self.device)
        full_model.eval()
        return full_model

    def _setup_gradcam(self) -> GradCAM:
        """Set up Grad-CAM targeting the last convolutional block."""
        # Target the last block of Xception's feature extractor
        target_layer = self.model.backbone.act4
        return GradCAM(self.model, target_layer)

    def _build_transform(self) -> transforms.Compose:
        """Image preprocessing pipeline matching Xception's requirements."""
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            ),
        ])

    def analyze_frame(self, frame: np.ndarray) -> Dict:
        """Analyze a single frame for spatial manipulation artifacts.

        Args:
            frame: BGR numpy array from OpenCV (H, W, 3)

        Returns:
            Dict with 'score' (0-1, higher=more likely fake),
            'probabilities' (real, fake), and 'heatmap' (numpy array).
        """
        # Convert BGR to RGB PIL Image
        rgb_frame = frame[:, :, ::-1] if frame.shape[2] == 3 else frame
        pil_image = Image.fromarray(rgb_frame.astype(np.uint8))

        # Preprocess
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        input_tensor.requires_grad_(True)

        # Forward pass
        with torch.enable_grad():
            logits = self.model(input_tensor)
            probabilities = F.softmax(logits, dim=1)

            fake_prob = probabilities[0, 1].item()
            real_prob = probabilities[0, 0].item()

            # Generate heatmap
            heatmap = self.grad_cam.generate(input_tensor, target_class=1)

            # Resize heatmap to original frame size
            from PIL import Image as PILImage
            heatmap_resized = np.array(
                PILImage.fromarray((heatmap * 255).astype(np.uint8)).resize(
                    (frame.shape[1], frame.shape[0]),
                    PILImage.BILINEAR
                )
            ) / 255.0

        return {
            "score": fake_prob,
            "probabilities": {"real": real_prob, "fake": fake_prob},
            "heatmap": heatmap_resized,
        }

    def analyze_frames(self, frames: list) -> list:
        """Analyze multiple frames and return per-frame results."""
        results = []
        for frame in frames:
            result = self.analyze_frame(frame)
            results.append(result)
        return results


class XceptionWrapper(nn.Module):
    """Wrapper combining Xception backbone with classification head."""

    def __init__(self, backbone: nn.Module, classifier: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        # features is [B, C, H, W] from the backbone without pooling
        if features.dim() == 2:
            # Already pooled by backbone
            features = features.unsqueeze(-1).unsqueeze(-1)
        logits = self.classifier(features)
        return logits
