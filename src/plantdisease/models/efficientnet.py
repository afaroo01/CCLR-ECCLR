from __future__ import annotations

import torch
import torchvision.models as tvm
from torch import Tensor, nn

from ..config.schema import ModelConfig
from .base import BaseClassifier
from .registry import MODELS


@MODELS.register("efficientnet")
class EfficientNetClassifier(BaseClassifier):
    """torchvision EfficientNet (b0..b7 via ``model.variant``) with a new classification head.

    The attribute is called ``efficientnet`` so ``best_model.pth`` files saved by the original
    notebooks load directly.
    """

    def __init__(self, cfg: ModelConfig, num_classes: int) -> None:
        super().__init__(num_classes)
        builder = getattr(tvm, cfg.variant, None)
        if builder is None or not cfg.variant.startswith("efficientnet"):
            raise ValueError(f"Unknown torchvision EfficientNet variant '{cfg.variant}'")
        self.efficientnet = builder(weights="DEFAULT" if cfg.pretrained else None)
        head = self.efficientnet.classifier
        if cfg.dropout is not None:
            head[0].p = cfg.dropout
        head[1] = nn.Linear(head[1].in_features, num_classes)

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.efficientnet.features(x)
        x = self.efficientnet.avgpool(x)
        return torch.flatten(x, 1)

    def forward_head(self, features: Tensor) -> Tensor:
        return self.efficientnet.classifier(features)

    def gradcam_target_layer(self) -> nn.Module:
        return self.efficientnet.features[-1]  # last conv block
