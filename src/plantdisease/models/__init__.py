# Importing the architecture modules registers them in MODELS.
from . import efficientnet, vit  # noqa: F401
from .base import BaseClassifier
from .factory import build_model, load_weights, resolve_checkpoint
from .protonet import PrototypicalNetwork
from .registry import MODELS

__all__ = [
    "MODELS",
    "BaseClassifier",
    "PrototypicalNetwork",
    "build_model",
    "load_weights",
    "resolve_checkpoint",
]
