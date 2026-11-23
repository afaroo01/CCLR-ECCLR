"""A tiny generic registry used to plug components in by name from config files.

Adding a new model / transform / optimizer / experiment type means writing a class or
function and decorating it with ``@REGISTRY.register("name")`` -- no existing code has to
change (open/closed principle).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Maps string keys to builders (classes or factory functions)."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._items: dict[str, T] = {}

    def register(self, key: str) -> Callable[[T], T]:
        def decorator(obj: T) -> T:
            if key in self._items:
                raise KeyError(f"'{key}' is already registered in the {self._name} registry")
            self._items[key] = obj
            return obj

        return decorator

    def get(self, key: str) -> T:
        try:
            return self._items[key]
        except KeyError:
            raise KeyError(
                f"Unknown {self._name} '{key}'. Available: {', '.join(self.keys())}"
            ) from None

    def build(self, key: str, *args, **kwargs):
        return self.get(key)(*args, **kwargs)  # type: ignore[operator]

    @property
    def name(self) -> str:
        return self._name

    def keys(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, key: str) -> bool:
        return key in self._items

    def __repr__(self) -> str:
        return f"Registry(name={self._name!r}, items={self.keys()})"
