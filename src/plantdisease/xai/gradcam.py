from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..models.base import BaseClassifier


class GradCAM:
    """Grad-CAM (Selvaraju et al., 2017) for CNNs and Vision Transformers.

    The target layer and the token->feature-map reshape come from the model
    (:meth:`BaseClassifier.gradcam_target_layer` / ``gradcam_reshape_transform``), so the
    same class works for EfficientNet and ViT. Use as a context manager to remove hooks::

        with GradCAM(model) as cam:
            heatmaps = cam(images)          # (B, H, W) in [0, 1]
    """

    def __init__(
        self,
        model: BaseClassifier,
        target_layer: nn.Module | None = None,
        reshape_transform: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        self.model = model
        self.target_layer = target_layer or model.gradcam_target_layer()
        self.reshape = reshape_transform or model.gradcam_reshape_transform() or (lambda t: t)
        self._activations: Tensor | None = None
        self._gradients: Tensor | None = None
        self._handle = self.target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module: nn.Module, inputs, output: Tensor) -> None:
        self._activations = output.detach()
        if output.requires_grad:
            output.register_hook(self._store_gradient)

    def _store_gradient(self, grad: Tensor) -> None:
        self._gradients = grad.detach()

    def __call__(self, images: Tensor, target_classes: Tensor | None = None) -> Tensor:
        """Return one heatmap per image for ``target_classes`` (default: predicted class)."""
        was_training = self.model.training
        self.model.eval()
        with torch.enable_grad():
            images = images.clone().requires_grad_(True)
            logits = self.model(images)
            if target_classes is None:
                target_classes = logits.argmax(dim=1)
            self.model.zero_grad(set_to_none=True)
            logits.gather(1, target_classes.view(-1, 1)).sum().backward()
        self.model.train(was_training)

        activations = self.reshape(self._activations)  # (B, C, h, w)
        gradients = self.reshape(self._gradients)
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=images.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)
        flat = cam.flatten(1)
        lo, hi = flat.min(1)[0].view(-1, 1, 1), flat.max(1)[0].view(-1, 1, 1)
        return ((cam - lo) / (hi - lo + 1e-8)).cpu()

    def remove(self) -> None:
        self._handle.remove()

    def __enter__(self) -> GradCAM:
        return self

    def __exit__(self, *exc) -> None:
        self.remove()
