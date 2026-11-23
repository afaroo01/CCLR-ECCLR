from __future__ import annotations

from dataclasses import replace

from torch import nn

from ..data.datamodule import DataModule
from ..engine import StandardBatchProcessor
from ..models import BaseClassifier, build_model
from .base import EXPERIMENTS, Experiment


@EXPERIMENTS.register("supervised")
class SupervisedExperiment(Experiment):
    """Train a classifier on the source dataset; test on its held-out split and the target."""

    def build(self, checkpoint: str | None = None) -> None:
        cfg = self.cfg
        self.datamodule = DataModule(cfg, self.seed).setup()
        self._save_data_summary()
        self.visualizer = self._make_visualizer()

        model_cfg = replace(cfg.model, checkpoint=checkpoint) if checkpoint else cfg.model
        self.model = build_model(model_cfg, self.datamodule.num_classes, self.seed).to(self.device)

        self.processor = StandardBatchProcessor()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)
        self.evaluator = self._make_evaluator(self.processor, self.criterion)

        if cfg.xai.enabled and cfg.xai.sample_grid:
            self.visualizer.plot_samples(self.datamodule.loader("train"), "samples_train")

    @property
    def backbone(self) -> BaseClassifier:
        return self.model

    def train(self) -> None:
        dm = self.datamodule
        self._fit(
            self.model, self.processor, self.evaluator, self.criterion,
            dm.loader("train"), dm.loader("val"),
        )

    def test(self) -> None:
        tester = self._make_tester(self.evaluator)
        loaders = {
            split: self.datamodule.loader(split, shuffle=False)
            for split in self._eval_splits(self.cfg.eval_splits)
        }
        self.metrics.update(tester.test(self.model, loaders))
