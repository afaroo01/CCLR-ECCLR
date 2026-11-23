"""Batch-level augmentations (CutMix / MixUp) implemented as picklable collate functions.

They return *soft* one-hot targets of shape ``(B, num_classes)``. ``nn.CrossEntropyLoss``
accepts these directly, so the trainer trains on the mixed labels (the notebooks took the
argmax, which throws away the CutMix label mixing).
"""

from __future__ import annotations

from collections.abc import Callable

from torch.utils.data import default_collate
from torchvision.transforms import v2

from ..registry import Registry

BATCH_MIXERS: Registry[Callable[..., Callable]] = Registry("batch_mixer")


class _MixCollate:
    def __init__(self, mixer: Callable) -> None:
        self.mixer = mixer

    def __call__(self, batch):
        return self.mixer(*default_collate(batch))


@BATCH_MIXERS.register("cutmix")
def cutmix(num_classes: int, alpha: float = 1.0) -> Callable:
    return _MixCollate(v2.CutMix(num_classes=num_classes, alpha=alpha))


@BATCH_MIXERS.register("mixup")
def mixup(num_classes: int, alpha: float = 1.0) -> Callable:
    return _MixCollate(v2.MixUp(num_classes=num_classes, alpha=alpha))


@BATCH_MIXERS.register("cutmix_or_mixup")
def cutmix_or_mixup(num_classes: int, alpha: float = 1.0) -> Callable:
    return _MixCollate(
        v2.RandomChoice(
            [v2.CutMix(num_classes=num_classes, alpha=alpha),
             v2.MixUp(num_classes=num_classes, alpha=alpha)]
        )
    )
