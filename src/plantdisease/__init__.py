"""Plant-disease classification experiments: supervised (EfficientNet / ViT with AutoAugment,
CutMix, CCLR, ECCLR) and few-shot (Prototypical Networks), with multi-seed aggregation,
resource profiling and Grad-CAM / t-SNE explainability.

Typical notebook usage::

    from plantdisease import load_config, ExperimentRunner
    cfg = load_config("configs/experiments/supervised/exp1_efficientnet_autoaugment.yaml",
                      overrides=["train.epochs=5"])
    summary = ExperimentRunner(cfg, seeds=[0, 1, 2]).run()
"""

from .config import ExperimentConfig, load_config
from .experiments import ExperimentRunner, FewShotExperiment, SupervisedExperiment

__version__ = "0.1.0"

__all__ = [
    "ExperimentConfig",
    "ExperimentRunner",
    "FewShotExperiment",
    "SupervisedExperiment",
    "load_config",
]
