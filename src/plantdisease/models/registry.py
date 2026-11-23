from __future__ import annotations

from collections.abc import Callable

from ..registry import Registry

MODELS: Registry[Callable] = Registry("model")
