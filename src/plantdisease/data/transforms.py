"""Named image transform pipelines (selected by ``data.train_transform`` etc. in configs).

Each builder has the signature ``(data_cfg, model_cfg) -> callable`` and receives a PIL
image. Register new ones with ``@TRANSFORMS.register("name")``.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torchvision.transforms import AutoAugmentPolicy, v2

from ..config.schema import DataConfig, ModelConfig
from ..registry import Registry

TransformBuilder = Callable[[DataConfig, ModelConfig], Callable]
TRANSFORMS: Registry[TransformBuilder] = Registry("transform")


def _to_float() -> list:
    return [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]


def _normalize(cfg: DataConfig) -> v2.Normalize:
    return v2.Normalize(mean=cfg.normalize_mean, std=cfg.normalize_std)


def _random_resized_crop(cfg: DataConfig) -> v2.RandomResizedCrop:
    return v2.RandomResizedCrop(cfg.image_size, scale=tuple(cfg.rrc_scale), ratio=(1.0, 1.0))


@TRANSFORMS.register("autoaugment")
def autoaugment(cfg: DataConfig, model_cfg: ModelConfig) -> Callable:
    """Supervised training pipeline from exp1/3/4/5/7/8 (ImageNet AutoAugment + RRC)."""
    return v2.Compose(
        [*_to_float(), v2.AutoAugment(AutoAugmentPolicy.IMAGENET), _normalize(cfg),
         _random_resized_crop(cfg)]
    )


@TRANSFORMS.register("standard")
def standard(cfg: DataConfig, model_cfg: ModelConfig) -> Callable:
    """Evaluation pipeline used by all supervised notebooks (normalize + resize)."""
    return v2.Compose([*_to_float(), _normalize(cfg), v2.Resize((cfg.image_size, cfg.image_size))])


@TRANSFORMS.register("fewshot_basic")
def fewshot_basic(cfg: DataConfig, model_cfg: ModelConfig) -> Callable:
    """Few-shot notebooks' pipeline: tensor in [0, 1] + random resized crop (no normalization)."""
    return v2.Compose([*_to_float(), _random_resized_crop(cfg)])


@TRANSFORMS.register("fewshot_standard")
def fewshot_standard(cfg: DataConfig, model_cfg: ModelConfig) -> Callable:
    """Deterministic alternative for few-shot evaluation, matching the backbone's training norm."""
    return standard(cfg, model_cfg)


def _timm_transform(model_cfg: ModelConfig, is_training: bool) -> Callable:
    import timm.data

    data_config = timm.data.resolve_data_config({}, model=model_cfg.variant)
    return timm.data.create_transform(**data_config, is_training=is_training)


@TRANSFORMS.register("timm_train")
def timm_train(cfg: DataConfig, model_cfg: ModelConfig) -> Callable:
    """timm's default training transform for ``model.variant`` (exp6: ViT + CutMix)."""
    return _timm_transform(model_cfg, is_training=True)


@TRANSFORMS.register("timm_eval")
def timm_eval(cfg: DataConfig, model_cfg: ModelConfig) -> Callable:
    return _timm_transform(model_cfg, is_training=False)


def build_transform(name: str, data_cfg: DataConfig, model_cfg: ModelConfig) -> Callable:
    return TRANSFORMS.build(name, data_cfg, model_cfg)
