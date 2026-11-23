from __future__ import annotations

import torch
from torch import Tensor, nn

from .base import BaseClassifier


class PrototypicalNetwork(nn.Module):
    """Prototypical Networks (Snell et al., 2017) on top of any :class:`BaseClassifier`.

    Each class prototype is the mean support embedding; queries are scored by negative
    Euclidean distance to every prototype.
    """

    def __init__(self, backbone: BaseClassifier) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(
        self, support_images: Tensor, support_labels: Tensor, query_images: Tensor
    ) -> Tensor:
        z_support = self.backbone.embed(support_images)
        z_query = self.backbone.embed(query_images)
        n_way = int(support_labels.max().item()) + 1
        prototypes = torch.stack(
            [z_support[support_labels == c].mean(dim=0) for c in range(n_way)]
        )
        return -torch.cdist(z_query, prototypes)
