"""Shared fixtures: a tiny synthetic copy of the real folder layout (no real data needed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from plantdisease.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"

PLANTVILLAGE_FOLDERS = [
    "Apple___healthy", "Apple___Cedar_apple_rust", "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy", "Grape___Black_rot", "Grape___healthy", "Tomato___healthy",
    "Tomato___Septoria_leaf_spot", "Corn___healthy",  # last one is unmapped -> ignored
]
PLANTDOC_FOLDERS = [
    "Apple_leaf", "Apple_rust_leaf", "Bell_pepper_leaf", "Bell_pepper_leaf_spot", "grape_leaf",
    "grape_leaf_black_rot", "Tomato_leaf", "Tomato_Septoria_leaf_spot",
]


def _write_images(folder: Path, n: int, color: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(color)
    for i in range(n):
        arr = rng.integers(0, 60, (48, 48, 3), dtype=np.uint8)
        arr[..., color % 3] += np.uint8(30 * (color % 7) + 40)
        Image.fromarray(arr).save(folder / f"img_{i}.png")


@pytest.fixture(scope="session")
def data_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("data")
    for c, name in enumerate(PLANTVILLAGE_FOLDERS):
        _write_images(root / "plantvillage" / name, 20, c)
    for c, name in enumerate(PLANTDOC_FOLDERS):
        _write_images(root / "plantdoc" / "train" / name, 6, c)
        _write_images(root / "plantdoc" / "test" / name, 4, c)
    return root


def tiny_overrides(data_root: Path, out: Path, *extra: str):
    return [
        f"paths.data_root={data_root}", f"output_dir={out}", "device=cpu",
        "model.pretrained=false", "train.epochs=1", "data.image_size=64", "data.batch_size=8",
        "data.num_workers=0", "data.pin_memory=false", "xai.tsne_max_samples=40",
        "xai.num_images=4", "profile_inference=false", *extra,
    ]


@pytest.fixture
def load(data_root, tmp_path):
    def _load(rel: str, *extra: str):
        return load_config(CONFIGS / rel, tiny_overrides(data_root, tmp_path / "outputs", *extra))
    return _load
