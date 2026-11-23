from __future__ import annotations

import math


class EarlyStopping:
    """Tracks the best value of a monitored metric and signals when to stop.

    ``patience <= 0`` disables stopping but still tracks the best epoch (for checkpointing).
    """

    def __init__(self, mode: str = "min", patience: int = 5, min_delta: float = 0.0) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.best = math.inf if mode == "min" else -math.inf
        self.best_epoch = -1
        self.counter = 0

    def _is_improvement(self, value: float) -> bool:
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def step(self, value: float, epoch: int) -> bool:
        """Record ``value``; return ``True`` if it is a new best."""
        if self._is_improvement(value):
            self.best, self.best_epoch, self.counter = value, epoch, 0
            return True
        self.counter += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.patience > 0 and self.counter >= self.patience


def monitor_mode(monitor: str) -> str:
    """``val_loss`` is minimized, every other monitored metric (``val_acc``) maximized."""
    return "min" if monitor.endswith("loss") else "max"
