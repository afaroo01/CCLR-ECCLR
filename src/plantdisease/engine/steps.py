"""Batch processors: the *only* code that knows what a batch looks like.

The trainer and evaluator are written once against :class:`BatchProcessor`; swapping the
processor switches between ordinary mini-batches and few-shot episodes (strategy pattern).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class StepOutput:
    logits: Tensor  # (B, C) scores used for the loss
    targets: Tensor  # loss targets: class indices (B,) or soft labels (B, C)
    labels: Tensor  # hard *global* class labels (B,) for metrics
    predictions: Tensor  # hard *global* predicted labels (B,)

    @property
    def batch_size(self) -> int:
        return self.labels.shape[0]

    @property
    def num_correct(self) -> int:
        return int((self.predictions == self.labels).sum().item())


class BatchProcessor(ABC):
    @abstractmethod
    def __call__(self, model: nn.Module, batch, device: torch.device) -> StepOutput:
        """Move ``batch`` to ``device``, run ``model`` and package the outputs."""


class StandardBatchProcessor(BatchProcessor):
    """``(images, targets)`` batches; targets may be indices or CutMix/MixUp soft labels."""

    def __call__(self, model, batch, device) -> StepOutput:
        images, targets = batch
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        labels = targets.argmax(dim=1) if targets.ndim == 2 else targets
        return StepOutput(logits, targets, labels, logits.argmax(dim=1))


class EpisodicBatchProcessor(BatchProcessor):
    """Few-shot episodes from :class:`~plantdisease.data.episodic.TaskSampler`.

    Episode-local predictions are mapped back to dataset class ids, so confusion matrices
    and per-class reports are meaningful across episodes.
    """

    def __call__(self, model, batch, device) -> StepOutput:
        support, support_labels, query, query_labels, class_ids = batch
        logits = model(
            support.to(device, non_blocking=True),
            support_labels.to(device, non_blocking=True),
            query.to(device, non_blocking=True),
        )
        query_labels = query_labels.to(device)
        class_ids = class_ids.to(device)
        return StepOutput(
            logits=logits,
            targets=query_labels,
            labels=class_ids[query_labels],
            predictions=class_ids[logits.argmax(dim=1)],
        )
