from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ..config.schema import ExperimentConfig
from ..utils.io import load_json
from ..utils.logging import get_logger
from .base import EXPERIMENTS, Experiment
from .results import ResultAggregator, Summary, format_mean_std


class ExperimentRunner:
    """Runs one experiment config for several seeds and aggregates mean ± std.

    Output layout::

        <output_dir>/<name>/
            seed_<s>/            config.yaml, run.log, metrics.json, history.csv,
                                 checkpoints/best.pt, reports/, figures/
            summary.json|csv|md  mean / std / min / max over seeds
            per_seed.csv         one row per seed
    """

    def __init__(
        self,
        cfg: ExperimentConfig,
        seeds: Sequence[int] | None = None,
        resume: bool = False,
    ) -> None:
        self.cfg = cfg
        self.seeds = list(seeds) if seeds else list(cfg.seeds)
        self.resume = resume
        self.experiment_dir = Path(cfg.output_dir) / cfg.name
        self.log = get_logger()

    def run_dir(self, seed: int) -> Path:
        return self.experiment_dir / f"seed_{seed}"

    def build_experiment(self, seed: int) -> Experiment:
        return EXPERIMENTS.build(self.cfg.type, self.cfg, seed, self.run_dir(seed))

    def run(self) -> Summary:
        per_seed: dict[int, dict[str, float]] = {}
        for i, seed in enumerate(self.seeds, 1):
            metrics_file = self.run_dir(seed) / "metrics.json"
            if self.resume and metrics_file.is_file():
                self.log.info("[%d/%d] seed %d already done -> loading %s",
                              i, len(self.seeds), seed, metrics_file)
                per_seed[seed] = load_json(metrics_file)
                continue
            self.log.info(
                "[%d/%d] running %s with seed %d", i, len(self.seeds), self.cfg.name, seed
            )
            per_seed[seed] = self.build_experiment(seed).run()

        summary = ResultAggregator.aggregate(per_seed)
        ResultAggregator.save(per_seed, summary, self.experiment_dir, title=self.cfg.name)
        self._log_headline(summary)
        return summary

    def _log_headline(self, summary: Summary) -> None:
        headline = [k for k in summary if k.endswith(("/accuracy", "/macro_f1"))]
        lines = [f"{k:<28} {format_mean_std(summary[k])}" for k in headline]
        self.log.info(
            "Summary over %d seed(s) -> %s\n%s",
            len(self.seeds), self.experiment_dir / "summary.md", "\n".join(lines),
        )
