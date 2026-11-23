from __future__ import annotations

from pathlib import Path

import torch

from ..config.schema import ModelConfig
from ..utils.logging import get_logger
from .base import BaseClassifier
from .registry import MODELS


def resolve_checkpoint(path: str | None, seed: int | None = None) -> Path | None:
    """Expand a ``{seed}`` placeholder and check the file exists."""
    if not path:
        return None
    resolved = Path(path.format(seed=seed) if seed is not None else path)
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {resolved}. Train the backbone first or set "
            f"model.checkpoint=/path/to/weights.pth"
        )
    return resolved


def load_weights(model: torch.nn.Module, path: str | Path, device="cpu") -> None:
    """Load a plain ``state_dict`` or a dict with a ``model_state`` / ``state_dict`` key."""
    state = torch.load(path, map_location=device, weights_only=True)
    if isinstance(state, dict):
        state = state.get("model_state", state.get("state_dict", state))
    prefix = "backbone."  # checkpoint of a meta-trained PrototypicalNetwork
    if all(k.startswith(prefix) for k in state) and not any(
        k.startswith(prefix) for k in model.state_dict()
    ):
        state = {k[len(prefix):]: v for k, v in state.items()}
    model.load_state_dict(state)
    get_logger().info("Loaded weights from %s", path)


def build_model(
    cfg: ModelConfig, num_classes: int, seed: int | None = None
) -> BaseClassifier:
    """Instantiate ``MODELS[cfg.name]`` and load ``cfg.checkpoint`` if provided."""
    model = MODELS.build(cfg.name, cfg, num_classes)
    checkpoint = resolve_checkpoint(cfg.checkpoint, seed)
    if checkpoint is not None:
        load_weights(model, checkpoint)
    return model
