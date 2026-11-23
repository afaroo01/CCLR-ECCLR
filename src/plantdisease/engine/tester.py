from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from torch import nn
from torch.utils.data import DataLoader

from ..utils.io import ensure_dir, save_json
from ..utils.logging import get_logger
from .evaluator import Evaluator

if TYPE_CHECKING:
    from ..xai.visualizer import XAIVisualizer


class Tester:
    """Evaluates a trained model on several named splits and persists the results.

    For every split it writes ``reports/<split>_classification_report.txt`` and
    ``reports/<split>_metrics.json``, draws the confusion matrix (if a visualizer is given)
    and returns flat metrics such as ``{"target/accuracy": 0.71, "target/macro_f1": ...}``.
    """

    def __init__(
        self,
        evaluator: Evaluator,
        class_names: Sequence[str],
        output_dir: Path,
        visualizer: XAIVisualizer | None = None,
        report_ci95: bool = False,
    ) -> None:
        self.evaluator = evaluator
        self.class_names = list(class_names)
        self.report_dir = ensure_dir(Path(output_dir) / "reports")
        self.visualizer = visualizer
        self.report_ci95 = report_ci95
        self.log = get_logger()

    def test(self, model: nn.Module, loaders: dict[str, DataLoader]) -> dict[str, float]:
        results: dict[str, float] = {}
        for split, loader in loaders.items():
            result = self.evaluator.evaluate(model, loader, desc=f"test:{split}")
            metrics = result.metrics(self.class_names)

            scalars = {"loss": result.loss, **metrics.scalars()}
            if self.report_ci95:
                scalars["accuracy_ci95"] = result.batch_accuracy_ci95
            if result.usage:
                scalars.update(result.usage.as_dict(prefix="eval_"))
                scalars["eval_throughput_img_s"] = len(result.y_true) / max(
                    result.usage.wall_time_s, 1e-9
                )
            results.update({f"{split}/{k}": v for k, v in scalars.items()})

            (self.report_dir / f"{split}_classification_report.txt").write_text(metrics.report)
            save_json(
                {**scalars, "per_class": metrics.per_class,
                 "confusion_matrix": metrics.confusion_matrix},
                self.report_dir / f"{split}_metrics.json",
            )
            if self.visualizer is not None:
                self.visualizer.plot_confusion_matrix(
                    metrics.confusion_matrix, f"confusion_matrix_{split}", title=split
                )
            ci = f" ± {scalars['accuracy_ci95']:.4f}" if self.report_ci95 else ""
            self.log.info(
                "[%s] acc %.4f%s | macro-F1 %.4f | loss %.4f\n%s",
                split, metrics.accuracy, ci, metrics.macro_f1, result.loss, metrics.report,
            )
        return results
