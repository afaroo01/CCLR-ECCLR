"""Experiment base class (template method): ``build -> train -> test -> explain``."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from torch import nn
from torch.utils.data import DataLoader

from ..config.loader import save_config
from ..config.schema import ExperimentConfig
from ..data.datamodule import DataModule
from ..engine import (
    BatchProcessor,
    EarlyStopping,
    Evaluator,
    Tester,
    Trainer,
    TrainingHistory,
    build_optimizer,
    build_scheduler,
)
from ..engine.callbacks import monitor_mode
from ..models.base import BaseClassifier
from ..registry import Registry
from ..utils.device import resolve_device
from ..utils.io import ensure_dir, save_csv, save_json
from ..utils.logging import add_file_handler, get_logger, remove_handler
from ..utils.profiling import ResourceMonitor, count_parameters, measure_inference_latency
from ..utils.seed import seed_everything
from ..xai.visualizer import XAIVisualizer

EXPERIMENTS: Registry[type] = Registry("experiment")


class Experiment(ABC):
    """One run of one experiment config with one seed.

    Subclasses implement :meth:`build`, :meth:`train` and :meth:`test`; :meth:`explain`
    (t-SNE / Grad-CAM) is shared. Everything is written to ``run_dir``.
    """

    def __init__(self, cfg: ExperimentConfig, seed: int, run_dir: Path) -> None:
        self.cfg = cfg
        self.seed = seed
        self.run_dir = ensure_dir(run_dir)
        self.device = resolve_device(cfg.device)
        self.log = get_logger()
        self.metrics: dict[str, float] = {"seed": seed}
        self.datamodule: DataModule | None = None
        self.model: nn.Module | None = None
        self.visualizer: XAIVisualizer | None = None

    # -- abstract steps ----------------------------------------------------------------------
    @abstractmethod
    def build(self, checkpoint: str | None = None) -> None:
        """Create ``self.datamodule``, ``self.model`` and ``self.visualizer``."""

    @abstractmethod
    def train(self) -> None:
        """Fit ``self.model`` (may be a no-op)."""

    @abstractmethod
    def test(self) -> None:
        """Evaluate ``self.model`` and add results to ``self.metrics``."""

    @property
    @abstractmethod
    def backbone(self) -> BaseClassifier:
        """The image classifier used for embeddings and Grad-CAM."""

    # -- template methods --------------------------------------------------------------------
    def run(self) -> dict[str, float]:
        """Full pipeline. Returns the flat metrics dict (also saved as ``metrics.json``)."""
        return self._execute(train=True)

    def evaluate(self, checkpoint: str | None = None) -> dict[str, float]:
        """Skip training: load weights, then test and explain."""
        return self._execute(train=False, checkpoint=checkpoint)

    def _execute(self, train: bool, checkpoint: str | None = None) -> dict[str, float]:
        handler = add_file_handler(self.run_dir / "run.log")
        try:
            seed_everything(self.seed, self.cfg.deterministic)
            save_config(self.cfg, self.run_dir / "config.yaml")
            self.log.info("=== %s | seed %d | device %s ===", self.cfg.name, self.seed, self.device)
            with ResourceMonitor("total", self.device) as total:
                self.build(checkpoint)
                self.metrics.update(count_parameters(self.backbone))
                if train:
                    self.train()
                self.test()
                self._profile_inference()
                self.explain()
            self.metrics.update(total.usage.as_dict())
            self.log.info("%s", total.usage)
            save_json(self.metrics, self.run_dir / "metrics.json")
            return self.metrics
        finally:
            remove_handler(handler)

    # -- shared steps ------------------------------------------------------------------------
    def _make_visualizer(self) -> XAIVisualizer:
        return XAIVisualizer(
            self.datamodule.classes, self.run_dir, self.device,
            show=self.cfg.xai.show, seed=self.seed,
        )

    def _save_data_summary(self) -> None:
        counts = self.datamodule.class_counts()
        save_json(counts, self.run_dir / "class_counts.json")
        for split, per_class in counts.items():
            self.log.info("%-6s %5d images | %s", split, sum(per_class.values()), per_class)
            self.metrics[f"data/{split}_size"] = sum(per_class.values())

    def _profile_inference(self) -> None:
        if not self.cfg.profile_inference:
            return
        size = self.cfg.data.image_size
        self.metrics.update(
            measure_inference_latency(self.backbone, (1, 3, size, size), self.device)
        )

    def explain(self) -> None:
        xai = self.cfg.xai
        if not xai.enabled:
            return
        dm, vis, model = self.datamodule, self.visualizer, self.backbone
        for split in (s for s in xai.splits if s in dm.available_splits):
            loader = dm.loader(split, shuffle=True)  # seeded shuffle -> varied but reproducible
            if xai.tsne:
                vis.plot_tsne(model, loader, f"tsne_{split}", xai.tsne_max_samples)
            if xai.gradcam:
                vis.plot_gradcam(model, loader, f"gradcam_{split}", xai.num_images)
        self.log.info("Figures written to %s", vis.figure_dir)

    def _fit(
        self,
        model: nn.Module,
        processor: BatchProcessor,
        evaluator: Evaluator,
        criterion: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> TrainingHistory:
        """Train with ``cfg.train`` settings; save history, curves and summary metrics."""
        tc = self.cfg.train
        es = tc.early_stopping
        optimizer = build_optimizer(model.parameters(), tc.optimizer)
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            processor=processor,
            evaluator=evaluator,
            device=self.device,
            epochs=tc.epochs,
            scheduler=build_scheduler(optimizer, tc.scheduler, tc.epochs),
            early_stopping=EarlyStopping(monitor_mode(es.monitor), es.patience, es.min_delta),
            monitor=es.monitor,
            checkpoint_path=self.run_dir / "checkpoints" / "best.pt",
            amp=tc.amp,
            grad_clip_norm=tc.grad_clip_norm,
        )
        history = trainer.fit(train_loader, val_loader)
        save_csv(history.as_rows(), self.run_dir / "history.csv")
        self.metrics.update(history.summary())
        self.log.info("%s", history.usage)
        if self.cfg.xai.enabled and self.cfg.xai.training_curves and history.records:
            self.visualizer.plot_training_curves(history)
        return history

    def _make_evaluator(self, processor: BatchProcessor, criterion: nn.Module) -> Evaluator:
        return Evaluator(processor, criterion, self.device, amp=self.cfg.train.amp)

    def _make_tester(self, evaluator: Evaluator, report_ci95: bool = False) -> Tester:
        xai = self.cfg.xai
        return Tester(
            evaluator,
            self.datamodule.classes,
            self.run_dir,
            visualizer=self.visualizer if xai.enabled and xai.confusion_matrix else None,
            report_ci95=report_ci95,
        )

    def _eval_splits(self, requested) -> list:
        available = self.datamodule.available_splits
        missing = [s for s in requested if s not in available]
        if missing:
            self.log.warning("Skipping unavailable splits %s (available: %s)", missing, available)
        return [s for s in requested if s in available]
