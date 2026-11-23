"""``DataModule``: the single place that turns a config into datasets and DataLoaders."""

from __future__ import annotations

from torch.utils.data import DataLoader

from ..config.schema import ExperimentConfig
from ..utils.device import resolve_device
from ..utils.seed import make_generator, seed_worker
from .datasets import LeafImageDataset, TransformSubset
from .episodic import TaskSampler
from .mixing import BATCH_MIXERS
from .splits import DatasetSplitter
from .transforms import build_transform

SOURCE_SPLITS = ("train", "val", "test")
TARGET_SPLIT = "target"


class DataModule:
    """Builds the source train/val/test splits and the (optional) cross-domain target set.

    Args:
        cfg: experiment config (uses ``cfg.data``, ``cfg.datasets`` and ``cfg.model``).
        seed: run seed; drives shuffling, episode sampling and (if ``data.split_seed`` is
            null) the train/val/test split.
        split_transforms: optional ``{split: transform_name}`` overriding the defaults
            (``data.train_transform`` for train, ``data.eval_transform`` otherwise).
    """

    def __init__(
        self,
        cfg: ExperimentConfig,
        seed: int,
        split_transforms: dict[str, str] | None = None,
    ) -> None:
        self.cfg = cfg
        self.data_cfg = cfg.data
        self.seed = seed
        self._split_transforms = {
            "train": cfg.data.train_transform,
            "val": cfg.data.eval_transform,
            "test": cfg.data.eval_transform,
            TARGET_SPLIT: cfg.data.eval_transform,
            **(split_transforms or {}),
        }
        self._datasets: dict[str, TransformSubset] = {}

    # -- setup -------------------------------------------------------------------------------
    @property
    def classes(self) -> list[str]:
        return sorted(self.data_cfg.classes)

    @property
    def num_classes(self) -> int:
        return len(self.data_cfg.classes)

    @property
    def available_splits(self) -> list[str]:
        self.setup()
        return list(self._datasets)

    def _build_base(self, dataset_key: str) -> LeafImageDataset:
        if dataset_key not in self.cfg.datasets:
            raise KeyError(
                f"Dataset '{dataset_key}' is not defined under `datasets:`. "
                f"Known: {sorted(self.cfg.datasets)}"
            )
        spec = self.cfg.datasets[dataset_key]
        return LeafImageDataset(
            roots=spec.roots,
            classes=self.data_cfg.classes,
            class_map=spec.class_map,
            image_size=self.data_cfg.image_size if self.data_cfg.preresize else None,
        )

    def _transform(self, split: str):
        return build_transform(self._split_transforms[split], self.data_cfg, self.cfg.model)

    def setup(self) -> DataModule:
        if self._datasets:
            return self
        source = self._build_base(self.data_cfg.source)
        split_seed = self.seed if self.data_cfg.split_seed is None else self.data_cfg.split_seed
        splits = DatasetSplitter(
            self.data_cfg.split_ratios, split_seed, self.data_cfg.stratify
        ).split(source.targets)
        for name in SOURCE_SPLITS:
            self._datasets[name] = TransformSubset(
                source, getattr(splits, name), self._transform(name)
            )
        if self.data_cfg.target:
            target = self._build_base(self.data_cfg.target)
            self._datasets[TARGET_SPLIT] = TransformSubset(
                target, range(len(target)), self._transform(TARGET_SPLIT)
            )
        return self

    def dataset(self, split: str) -> TransformSubset:
        self.setup()
        if split not in self._datasets:
            raise KeyError(f"Unknown split '{split}'. Available: {list(self._datasets)}")
        return self._datasets[split]

    def class_counts(self) -> dict[str, dict[str, int]]:
        self.setup()
        return {split: ds.class_counts() for split, ds in self._datasets.items()}

    # -- loaders -----------------------------------------------------------------------------
    def _loader_kwargs(self, persistent: bool = False) -> dict:
        # Only the training loader keeps its workers alive between epochs. On Windows each
        # worker is a full Python process (~0.5-1 GB), so persistent workers on every
        # loader (train + val + test + target) can exhaust system RAM.
        workers = self.data_cfg.num_workers
        return dict(
            num_workers=workers,
            pin_memory=self.data_cfg.pin_memory and resolve_device(self.cfg.device).type == "cuda",
            worker_init_fn=seed_worker,
            persistent_workers=persistent and workers > 0,
        )

    def loader(self, split: str, shuffle: bool | None = None) -> DataLoader:
        """Standard mini-batch loader. Training batches get the configured batch mixer."""
        is_train = split == "train"
        collate_fn = None
        if is_train and self.data_cfg.batch_mixer:
            collate_fn = BATCH_MIXERS.build(
                self.data_cfg.batch_mixer, self.num_classes, self.data_cfg.mixer_alpha
            )
        batch_size = self.data_cfg.batch_size if is_train else (
            self.data_cfg.eval_batch_size or self.data_cfg.batch_size
        )
        return DataLoader(
            self.dataset(split),
            batch_size=batch_size,
            shuffle=is_train if shuffle is None else shuffle,
            drop_last=is_train and collate_fn is not None,
            collate_fn=collate_fn,
            generator=make_generator(self.seed),
            **self._loader_kwargs(persistent=is_train),
        )

    def episodic_loader(
        self, split: str, n_tasks: int, n_way: int | None, n_shot: int, n_query: int
    ) -> DataLoader:
        """Loader yielding one N-way K-shot episode per batch (see :class:`TaskSampler`)."""
        dataset = self.dataset(split)
        sampler = TaskSampler(
            dataset.targets,
            n_way=n_way or self.num_classes,
            n_shot=n_shot,
            n_query=n_query,
            n_tasks=n_tasks,
            seed=self.seed,
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=sampler.episodic_collate_fn,
            **self._loader_kwargs(),
        )
