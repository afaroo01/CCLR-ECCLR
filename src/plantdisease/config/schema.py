"""Typed configuration schema.

Every YAML file under ``configs/`` is validated against :class:`ExperimentConfig`, so a
typo in a key or a wrong value type fails fast instead of silently being ignored.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DatasetSpec:
    """Where a dataset lives on disk and how its folder names map to canonical classes.

    ``roots`` is a list so that e.g. PlantDoc's ``train/`` and ``test/`` folders can be merged.
    ``class_map`` maps *folder name -> canonical class name*. ``None`` means the folders are
    already named after the canonical classes. Folders not in the map are ignored.
    """

    roots: list[str] = field(default_factory=list)
    class_map: dict[str, str] | None = None


@dataclass
class DataConfig:
    source: str = "plantvillage"  # key into ``datasets``: used for train/val/test
    target: str | None = "plantdoc"  # key into ``datasets``: cross-domain evaluation
    classes: list[str] = field(default_factory=list)
    image_size: int = 224
    preresize: bool = True  # LANCZOS-resize on load (what the notebooks' make_ds did)
    split_ratios: list[float] = field(default_factory=lambda: [0.7, 0.2, 0.1])  # train/val/test
    split_seed: int | None = 42  # null -> use the run seed (splits change per seed)
    stratify: bool = False
    batch_size: int = 32
    eval_batch_size: int | None = None  # null -> batch_size
    num_workers: int = 2
    pin_memory: bool = True
    train_transform: str = "autoaugment"
    eval_transform: str = "standard"
    rrc_scale: list[float] = field(default_factory=lambda: [0.75, 1.0])
    normalize_mean: list[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: list[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    batch_mixer: str | None = None  # e.g. "cutmix" / "mixup"
    mixer_alpha: float = 1.0


@dataclass
class ModelConfig:
    name: str = "efficientnet"  # key in MODELS registry
    variant: str = "efficientnet_b0"  # torchvision / timm architecture name
    pretrained: bool = True
    checkpoint: str | None = None  # may contain "{seed}" -> formatted per run
    dropout: float | None = None


@dataclass
class OptimizerConfig:
    name: str = "adam"
    lr: float = 1e-3
    weight_decay: float = 0.0
    momentum: float = 0.9  # SGD only


@dataclass
class SchedulerConfig:
    name: str | None = "step"  # step | cosine | null
    step_size: int = 5
    gamma: float = 0.1
    min_lr: float = 0.0


@dataclass
class EarlyStoppingConfig:
    monitor: str = "val_loss"  # val_loss (minimized) | val_acc (maximized)
    patience: int = 5  # <= 0 disables stopping (best checkpoint is still kept)
    min_delta: float = 0.0


@dataclass
class TrainConfig:
    epochs: int = 25
    label_smoothing: float = 0.0
    amp: bool = False
    grad_clip_norm: float | None = None
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)


@dataclass
class FewShotConfig:
    n_way: int | None = None  # null -> all classes
    n_shot: int = 5
    n_query: int = 10
    n_eval_tasks: int = 100
    eval_splits: list[str] = field(default_factory=lambda: ["test", "target"])
    transform: str = "fewshot_basic"
    # Optional episodic meta-training of the backbone (0 = evaluation only, as in the notebooks)
    n_train_tasks: int = 0
    n_val_tasks: int = 100


@dataclass
class XAIConfig:
    enabled: bool = True
    splits: list[str] = field(default_factory=lambda: ["test", "target"])
    gradcam: bool = True
    tsne: bool = True
    confusion_matrix: bool = True
    training_curves: bool = True
    sample_grid: bool = True
    num_images: int = 16
    tsne_max_samples: int = 2000
    show: bool = False  # plt.show() in notebooks; figures are always saved to disk


@dataclass
class ExperimentConfig:
    name: str = "experiment"
    type: str = "supervised"  # key in EXPERIMENTS registry: supervised | fewshot
    description: str = ""
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    output_dir: str = "outputs"
    device: str = "auto"  # auto | cpu | cuda | cuda:1 | mps
    deterministic: bool = False
    eval_splits: list[str] = field(default_factory=lambda: ["val", "test", "target"])
    profile_inference: bool = True
    paths: dict[str, str] = field(default_factory=lambda: {"data_root": "data"})
    datasets: dict[str, DatasetSpec] = field(default_factory=dict)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    fewshot: FewShotConfig = field(default_factory=FewShotConfig)
    xai: XAIConfig = field(default_factory=XAIConfig)
