from __future__ import annotations

from dataclasses import replace

from torch import nn
from torch.utils.data import DataLoader

from ..data.datamodule import SOURCE_SPLITS, TARGET_SPLIT, DataModule
from ..engine import EpisodicBatchProcessor
from ..models import BaseClassifier, PrototypicalNetwork, build_model
from .base import EXPERIMENTS, Experiment


@EXPERIMENTS.register("fewshot")
class FewShotExperiment(Experiment):
    """N-way K-shot Prototypical-Network evaluation on top of a (supervised) backbone.

    ``model.checkpoint`` points at the backbone weights -- typically the matching supervised
    run, e.g. ``outputs/exp7_vit_cclr/seed_{seed}/checkpoints/best.pt``. Optionally the
    backbone is meta-trained episodically first (``fewshot.n_train_tasks > 0``).
    """

    def build(self, checkpoint: str | None = None) -> None:
        cfg, fs = self.cfg, self.cfg.fewshot
        transforms = {split: fs.transform for split in (*SOURCE_SPLITS, TARGET_SPLIT)}
        self.datamodule = DataModule(cfg, self.seed, split_transforms=transforms).setup()
        self._save_data_summary()
        self.visualizer = self._make_visualizer()

        model_cfg = replace(cfg.model, checkpoint=checkpoint) if checkpoint else cfg.model
        self._backbone = build_model(model_cfg, self.datamodule.num_classes, self.seed)
        self.model = PrototypicalNetwork(self._backbone).to(self.device)

        self.processor = EpisodicBatchProcessor()
        self.criterion = nn.CrossEntropyLoss()
        self.evaluator = self._make_evaluator(self.processor, self.criterion)

        if cfg.xai.enabled and cfg.xai.sample_grid:
            split = fs.eval_splits[-1]
            support, _, query, _, _ = next(iter(self._episodes(split, n_tasks=1)))
            self.visualizer.plot_episode(
                support, query, fs.n_shot, fs.n_query, name=f"episode_{split}"
            )

    @property
    def backbone(self) -> BaseClassifier:
        return self._backbone

    def _episodes(self, split: str, n_tasks: int) -> DataLoader:
        fs = self.cfg.fewshot
        return self.datamodule.episodic_loader(split, n_tasks, fs.n_way, fs.n_shot, fs.n_query)

    def train(self) -> None:
        fs = self.cfg.fewshot
        if fs.n_train_tasks <= 0:
            self.log.info("fewshot.n_train_tasks=0 -> evaluating the pretrained backbone as-is")
            return
        self._fit(
            self.model, self.processor, self.evaluator, self.criterion,
            self._episodes("train", fs.n_train_tasks), self._episodes("val", fs.n_val_tasks),
        )

    def test(self) -> None:
        fs = self.cfg.fewshot
        tester = self._make_tester(self.evaluator, report_ci95=True)
        loaders = {
            split: self._episodes(split, fs.n_eval_tasks)
            for split in self._eval_splits(fs.eval_splits)
        }
        self.metrics.update(tester.test(self.model, loaders))
