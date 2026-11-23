from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from sklearn import metrics as skm


@dataclass
class ClassificationMetrics:
    accuracy: float
    balanced_accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    confusion_matrix: np.ndarray
    report: str
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)

    def scalars(self) -> dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
        }


def compute_classification_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], class_names: Sequence[str]
) -> ClassificationMetrics:
    """Standard multi-class metrics. Classes absent from ``y_true`` still get a CM row."""
    labels = list(range(len(class_names)))
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    p, r, f1, _ = skm.precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    report_dict = skm.classification_report(
        y_true, y_pred, labels=labels, target_names=list(class_names),
        zero_division=0, output_dict=True,
    )
    return ClassificationMetrics(
        accuracy=float(skm.accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(skm.balanced_accuracy_score(y_true, y_pred)),
        macro_precision=float(p),
        macro_recall=float(r),
        macro_f1=float(f1),
        weighted_f1=float(
            skm.f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        confusion_matrix=skm.confusion_matrix(y_true, y_pred, labels=labels),
        report=skm.classification_report(
            y_true, y_pred, labels=labels, target_names=list(class_names),
            zero_division=0, digits=4,
        ),
        per_class={c: report_dict[c] for c in class_names},
    )


def mean_confidence_interval(values: Sequence[float], z: float = 1.96) -> float:
    """Half-width of the normal-approximation CI of the mean (few-shot convention)."""
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return 0.0
    return float(z * values.std(ddof=1) / np.sqrt(values.size))
