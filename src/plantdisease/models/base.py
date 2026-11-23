"""Common interface every classifier implements.

Trainer, evaluator, t-SNE and Grad-CAM only depend on this interface, so adding a new
architecture never requires touching them (Liskov substitution / dependency inversion).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from torch import Tensor, nn


class BaseClassifier(nn.Module, ABC):
    """Image classifier that also exposes embeddings and a Grad-CAM target layer."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes

    @abstractmethod
    def forward_features(self, x: Tensor) -> Tensor:
        """Return pooled embeddings of shape ``(B, D)``."""

    @abstractmethod
    def forward_head(self, features: Tensor) -> Tensor:
        """Map embeddings ``(B, D)`` to logits ``(B, num_classes)``."""

    def forward(self, x: Tensor, return_embedding: bool = False) -> Tensor:
        features = self.forward_features(x)
        return features if return_embedding else self.forward_head(features)

    def embed(self, x: Tensor) -> Tensor:
        return self.forward_features(x)

    @abstractmethod
    def gradcam_target_layer(self) -> nn.Module:
        """Layer whose activations/gradients Grad-CAM uses."""

    def gradcam_reshape_transform(self) -> Callable[[Tensor], Tensor] | None:
        """Converts the target layer's output to ``(B, C, H, W)``; ``None`` if already so."""
        return None
