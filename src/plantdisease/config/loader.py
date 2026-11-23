"""YAML config loading with ``_base_`` inheritance and command-line overrides.

A config file may list one or more parent files under ``_base_`` (paths relative to the
file). Parents are merged first, children override them, and finally ``overrides``
(``"train.epochs=10"`` style dot-list strings) are applied. The result is validated
against :class:`~plantdisease.config.schema.ExperimentConfig`.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from .schema import ExperimentConfig

BASE_KEY = "_base_"
PathLike = str | Path


def _load_raw(path: Path, _stack: tuple = ()) -> DictConfig:
    path = path.resolve()
    if path in _stack:
        raise ValueError(f"Circular _base_ inheritance: {' -> '.join(map(str, _stack + (path,)))}")
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    cfg = OmegaConf.load(path)
    bases = cfg.pop(BASE_KEY, None) or []
    if isinstance(bases, str):
        bases = [bases]

    merged = OmegaConf.create()
    for base in bases:
        merged = OmegaConf.merge(merged, _load_raw(path.parent / base, _stack + (path,)))
    return OmegaConf.merge(merged, cfg)


def load_config(
    path: PathLike, overrides: Iterable[str] | None = None
) -> ExperimentConfig:
    """Load, merge, override, validate and resolve an experiment config."""
    raw = _load_raw(Path(path))
    if overrides:
        raw = OmegaConf.merge(raw, OmegaConf.from_dotlist(list(overrides)))
    structured = OmegaConf.merge(OmegaConf.structured(ExperimentConfig), raw)
    OmegaConf.resolve(structured)  # expand ${paths.data_root} etc.
    return OmegaConf.to_object(structured)  # type: ignore[return-value]


def config_to_yaml(cfg: ExperimentConfig) -> str:
    return OmegaConf.to_yaml(OmegaConf.structured(cfg))


def save_config(cfg: ExperimentConfig, path: PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(config_to_yaml(cfg))
