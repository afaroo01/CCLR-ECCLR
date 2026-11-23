from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

PathLike = str | Path


def ensure_dir(path: PathLike) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _to_serializable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_json(data: Any, path: PathLike) -> None:
    ensure_dir(Path(path).parent)
    Path(path).write_text(json.dumps(data, indent=2, default=_to_serializable))


def load_json(path: PathLike) -> Any:
    return json.loads(Path(path).read_text())


def save_csv(rows: Iterable[dict[str, Any]], path: PathLike) -> None:
    rows = list(rows)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:  # union of keys, first-seen order
        fieldnames.extend(k for k in row if k not in fieldnames)
    ensure_dir(Path(path).parent)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
