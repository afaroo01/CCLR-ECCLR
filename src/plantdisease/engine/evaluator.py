from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..metrics import ClassificationMetrics, compute_classification_metrics
from ..metrics.classification import mean_confidence_interval
from ..utils.profiling import ResourceMonitor, ResourceUsage
from .steps import BatchProcessor


@dataclass
class EvaluationResult:
    loss: float
    y_true: np.ndarray
    y_pred: np.ndarray
    batch_accuracies: list[float] = field(default_factory=list)  # one per batch / episode
    usage: ResourceUsage | None = None

    @property
    def accuracy(self) -> float:
        return float((self.y_true == self.y_pred).mean()) if self.y_true.size else 0.0

    @property
    def batch_accuracy_ci95(self) -> float:
        """95% CI half-width over batches -- the usual few-shot ``acc ± ci`` over episodes."""
        return mean_confidence_interval(self.batch_accuracies)

    def metrics(self, class_names: Sequence[str]) -> ClassificationMetrics:
        return compute_classification_metrics(self.y_true, self.y_pred, class_names)


class Evaluator:
    """Runs a model over a loader without gradients and collects loss and predictions."""

    def __init__(
        self,
        processor: BatchProcessor,
        criterion: nn.Module,
        device: torch.device,
        amp: bool = False,
        progress: bool = True,
    ) -> None:
        self.processor = processor
        self.criterion = criterion
        self.device = device
        self.amp = amp and device.type == "cuda"
        self.progress = progress

    @torch.no_grad()
    def evaluate(
        self, model: nn.Module, loader: DataLoader, desc: str = "eval"
    ) -> EvaluationResult:
        model.eval()
        total_loss, total = 0.0, 0
        y_true: list[np.ndarray] = []
        y_pred: list[np.ndarray] = []
        batch_acc: list[float] = []

        with ResourceMonitor(desc, self.device) as monitor:
            for batch in tqdm(loader, desc=desc, leave=False, disable=not self.progress):
                with torch.autocast(self.device.type, enabled=self.amp):
                    out = self.processor(model, batch, self.device)
                    loss = self.criterion(out.logits, out.targets)
                total_loss += loss.item() * out.batch_size
                total += out.batch_size
                batch_acc.append(out.num_correct / out.batch_size)
                y_true.append(out.labels.cpu().numpy())
                y_pred.append(out.predictions.cpu().numpy())

        return EvaluationResult(
            loss=total_loss / max(total, 1),
            y_true=np.concatenate(y_true) if y_true else np.array([], dtype=int),
            y_pred=np.concatenate(y_pred) if y_pred else np.array([], dtype=int),
            batch_accuracies=batch_acc,
            usage=monitor.usage,
        )
