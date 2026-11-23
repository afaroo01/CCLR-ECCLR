from __future__ import annotations

import math
from collections.abc import Callable

import timm
from torch import Tensor, nn

from ..config.schema import ModelConfig
from .base import BaseClassifier
from .registry import MODELS


@MODELS.register("vit")
class ViTClassifier(BaseClassifier):
    """timm Vision Transformer (default ``vit_tiny_patch16_224``).

    The attribute is called ``model`` so ``best_vit_model.pth`` files saved by the original
    notebooks load directly. The embedding is the pre-logits [CLS] representation.
    """

    def __init__(self, cfg: ModelConfig, num_classes: int) -> None:
        super().__init__(num_classes)
        kwargs = {} if cfg.dropout is None else {"drop_rate": cfg.dropout}
        self.model = timm.create_model(
            cfg.variant, pretrained=cfg.pretrained, num_classes=num_classes, **kwargs
        )

    def forward_features(self, x: Tensor) -> Tensor:
        tokens = self.model.forward_features(x)
        return self.model.forward_head(tokens, pre_logits=True)

    def forward_head(self, features: Tensor) -> Tensor:
        return self.model.head(features)

    def gradcam_target_layer(self) -> nn.Module:
        return self.model.blocks[-1].norm1

    def gradcam_reshape_transform(self) -> Callable[[Tensor], Tensor] | None:
        n_prefix = getattr(self.model, "num_prefix_tokens", 1)

        def reshape(tokens: Tensor) -> Tensor:
            patches = tokens[:, n_prefix:, :]  # drop [CLS] (and register) tokens
            side = int(math.sqrt(patches.shape[1]))
            return patches.reshape(tokens.shape[0], side, side, -1).permute(0, 3, 1, 2)

        return reshape
