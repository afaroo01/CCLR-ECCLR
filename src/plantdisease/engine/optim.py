"""Optimizer and LR-scheduler factories (selected by name in ``train.optimizer/scheduler``)."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from torch import nn, optim
from torch.optim.lr_scheduler import LRScheduler

from ..config.schema import OptimizerConfig, SchedulerConfig
from ..registry import Registry

OPTIMIZERS: Registry[Callable[..., optim.Optimizer]] = Registry("optimizer")
SCHEDULERS: Registry[Callable[..., LRScheduler]] = Registry("scheduler")


@OPTIMIZERS.register("adam")
def _adam(params, cfg: OptimizerConfig) -> optim.Optimizer:
    return optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)


@OPTIMIZERS.register("adamw")
def _adamw(params, cfg: OptimizerConfig) -> optim.Optimizer:
    return optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)


@OPTIMIZERS.register("sgd")
def _sgd(params, cfg: OptimizerConfig) -> optim.Optimizer:
    return optim.SGD(
        params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay, nesterov=True
    )


@SCHEDULERS.register("step")
def _step(optimizer, cfg: SchedulerConfig, epochs: int) -> LRScheduler:
    return optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)


@SCHEDULERS.register("cosine")
def _cosine(optimizer, cfg: SchedulerConfig, epochs: int) -> LRScheduler:
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=cfg.min_lr)


def build_optimizer(params: Iterable[nn.Parameter], cfg: OptimizerConfig) -> optim.Optimizer:
    return OPTIMIZERS.build(cfg.name, params, cfg)


def build_scheduler(
    optimizer: optim.Optimizer, cfg: SchedulerConfig, epochs: int
) -> LRScheduler | None:
    if not cfg.name:
        return None
    return SCHEDULERS.build(cfg.name, optimizer, cfg, epochs)
