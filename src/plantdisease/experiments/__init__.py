# Importing the modules registers the experiment types in EXPERIMENTS.
from . import fewshot, supervised  # noqa: F401
from .base import EXPERIMENTS, Experiment
from .fewshot import FewShotExperiment
from .results import ResultAggregator, compare_experiments
from .runner import ExperimentRunner
from .supervised import SupervisedExperiment

__all__ = [
    "EXPERIMENTS",
    "Experiment",
    "ExperimentRunner",
    "FewShotExperiment",
    "ResultAggregator",
    "SupervisedExperiment",
    "compare_experiments",
]
