from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..utils.logging import get_logger
from ..utils.profiling import ResourceMonitor, ResourceUsage
from .callbacks import EarlyStopping, monitor_mode
from .evaluator import Evaluator
from .steps import BatchProcessor


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    lr: float
    epoch_time_s: float
    ram_peak_mb: float
    gpu_peak_mb: float | None = None


@dataclass
class TrainingHistory:
    records: list[EpochRecord] = field(default_factory=list)
    best_epoch: int = -1
    best_score: float = float("nan")
    stopped_early: bool = False
    usage: ResourceUsage | None = None

    def as_rows(self) -> list[dict]:
        return [asdict(r) for r in self.records]

    def column(self, name: str) -> list[float]:
        return [getattr(r, name) for r in self.records]

    def summary(self) -> dict[str, float]:
        out = {
            "train/epochs_run": len(self.records),
            "train/best_epoch": self.best_epoch,
            "train/best_score": self.best_score,
            "train/stopped_early": float(self.stopped_early),
        }
        if self.records:
            out["train/mean_epoch_time_s"] = sum(self.column("epoch_time_s")) / len(self.records)
        if self.usage:
            out.update(self.usage.as_dict("train/"))
        return out


class Trainer:
    """Generic epoch-based trainer with validation, early stopping and best-checkpointing.

    The batch format is delegated to ``processor``, so the same class trains supervised
    classifiers and few-shot Prototypical Networks.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        processor: BatchProcessor,
        evaluator: Evaluator,
        device: torch.device,
        epochs: int,
        scheduler: LRScheduler | None = None,
        early_stopping: EarlyStopping | None = None,
        monitor: str = "val_loss",
        checkpoint_path: Path | None = None,
        amp: bool = False,
        grad_clip_norm: float | None = None,
        progress: bool = True,
    ) -> None:
        if monitor not in ("val_loss", "val_acc"):
            raise ValueError(f"monitor must be 'val_loss' or 'val_acc', got {monitor!r}")
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.processor = processor
        self.evaluator = evaluator
        self.device = device
        self.epochs = epochs
        self.scheduler = scheduler
        self.monitor = monitor
        self.early_stopping = early_stopping or EarlyStopping(monitor_mode(monitor), patience=0)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.amp = amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler(device.type, enabled=self.amp)
        self.grad_clip_norm = grad_clip_norm
        self.progress = progress
        self.log = get_logger()

    def _train_one_epoch(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        bar = tqdm(loader, desc=f"epoch {epoch}", leave=False, disable=not self.progress)
        for batch in bar:
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(self.device.type, enabled=self.amp):
                out = self.processor(self.model, batch, self.device)
                loss = self.criterion(out.logits, out.targets)
            self.scaler.scale(loss).backward()
            if self.grad_clip_norm:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * out.batch_size
            correct += out.num_correct
            total += out.batch_size
            bar.set_postfix(loss=f"{total_loss / total:.4f}", acc=f"{correct / total:.4f}")
        return {"loss": total_loss / max(total, 1), "acc": correct / max(total, 1)}

    def _save_checkpoint(self) -> None:
        if self.checkpoint_path:
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), self.checkpoint_path)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> TrainingHistory:
        history = TrainingHistory()
        with ResourceMonitor("train", self.device) as monitor:
            for epoch in range(1, self.epochs + 1):
                with ResourceMonitor(f"epoch{epoch}", self.device) as epoch_monitor:
                    train = self._train_one_epoch(train_loader, epoch)
                    val = self.evaluator.evaluate(self.model, val_loader, desc="val")
                record = EpochRecord(
                    epoch=epoch,
                    train_loss=train["loss"],
                    train_acc=train["acc"],
                    val_loss=val.loss,
                    val_acc=val.accuracy,
                    lr=self.optimizer.param_groups[0]["lr"],
                    epoch_time_s=epoch_monitor.usage.wall_time_s,
                    ram_peak_mb=epoch_monitor.usage.ram_peak_mb,
                    gpu_peak_mb=epoch_monitor.usage.gpu_peak_allocated_mb,
                )
                history.records.append(record)

                if not (math.isfinite(record.train_loss) and math.isfinite(record.val_loss)):
                    raise FloatingPointError(
                        f"Training diverged at epoch {epoch} (train loss {record.train_loss}, "
                        f"val loss {record.val_loss}). No valid checkpoint for this seed."
                    )
                improved = self.early_stopping.step(getattr(record, self.monitor), epoch)
                if improved:
                    self._save_checkpoint()
                self.log.info(
                    "Epoch %d/%d | train loss %.4f acc %.4f | val loss %.4f acc %.4f | "
                    "lr %.2e | %.1fs%s",
                    epoch, self.epochs, record.train_loss, record.train_acc, record.val_loss,
                    record.val_acc, record.lr, record.epoch_time_s, "  *best*" if improved else "",
                )
                if self.scheduler is not None:
                    self.scheduler.step()
                if self.early_stopping.should_stop:
                    self.log.info("Early stopping at epoch %d", epoch)
                    history.stopped_early = True
                    break

        history.best_epoch = self.early_stopping.best_epoch
        history.best_score = self.early_stopping.best
        history.usage = monitor.usage
        self._restore_best()
        return history

    def _restore_best(self) -> None:
        if self.checkpoint_path and self.checkpoint_path.is_file():
            state = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
            self.log.info("Restored best weights from %s", self.checkpoint_path)
