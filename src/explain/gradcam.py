"""
Grad-CAM (Selvaraju et al., 2017) for verifying the model attends to plant
morphology rather than background. Implemented with raw forward/backward
hooks (no extra dependency) so it works on both backbones the proposal
compares — they expose their final conv feature maps under different
attribute names, which is the only backbone-specific part.
"""
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


def _last_conv_layer(model: torch.nn.Module, model_name: str) -> torch.nn.Module:
    if model_name == "resnet50":
        return model.layer4[-1]
    if model_name == "efficientnet_v2_s":
        return model.features[-1]
    if model_name == "baseline":
        return model.features[-1]
    raise ValueError(
        f"Don't know the last conv layer for '{model_name}'. "
        f"Choose from: resnet50, efficientnet_v2_s, baseline."
    )


class GradCAM:
    def __init__(self, model: torch.nn.Module, model_name: str):
        self.model = model.eval()
        self.target_layer = _last_conv_layer(model, model_name)
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        self.target_layer.register_forward_hook(self._save_activations)
        self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, inp, out):
        self._activations = out.detach()

    def _save_gradients(self, module, grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def __call__(self, image_tensor: torch.Tensor, class_idx: Optional[int] = None) -> tuple[np.ndarray, int]:
        """
        image_tensor: shape (1, 3, H, W), already normalized as the model expects.
        Returns (heatmap as HxW float array in [0, 1], the class index used).
        """
        if image_tensor.ndim != 4 or image_tensor.shape[0] != 1:
            raise ValueError("GradCAM expects a single image with shape (1, 3, H, W).")

        self.model.zero_grad(set_to_none=True)
        logits = self.model(image_tensor)

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        score = logits[0, class_idx]
        score.backward()

        # Global-average-pool the gradients to get per-channel importance weights
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)        # (1, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)   # (1, 1, h, w)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()
        cam_max = cam.max()
        if cam_max > 0:
            cam = cam / cam_max
        return cam, class_idx


def overlay_heatmap(image_rgb_uint8: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend a [0,1] heatmap (simple red channel, no extra colormap
    dependency needed) onto an HxWx3 uint8 image for visual inspection."""
    heatmap_rgb = np.zeros_like(image_rgb_uint8, dtype=np.float32)
    heatmap_rgb[..., 0] = heatmap * 255  # red channel only
    blended = (1 - alpha) * image_rgb_uint8.astype(np.float32) + alpha * heatmap_rgb
    return np.clip(blended, 0, 255).astype(np.uint8)
