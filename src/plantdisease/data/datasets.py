"""Dataset classes.

``LeafImageDataset`` replaces the notebooks' ``make_ds`` + ``ImageFolder`` combo: instead of
copying/renaming/resizing images into a working folder, it reads them in place and maps
each source folder to a canonical class name on the fly. That lets PlantVillage, the
CCLR/ECCLR variants and PlantDoc (all with different folder names) share one label space.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from ..utils.logging import get_logger

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

Transform = Callable | None


class LeafImageDataset(Dataset):
    """Image-folder dataset over one or more roots with folder -> class remapping.

    Args:
        roots: directories that each contain one sub-folder per (source) class.
        classes: canonical class names. Label ``i`` is ``sorted(classes)[i]``.
        class_map: source folder name -> canonical class. ``None`` = identity.
        transform: applied to the PIL image (usually ``None``; see :class:`TransformSubset`).
        image_size: if set, LANCZOS-resize to a square of this size on load.
    """

    def __init__(
        self,
        roots: Sequence[str],
        classes: Sequence[str],
        class_map: dict[str, str] | None = None,
        transform: Transform = None,
        image_size: int | None = None,
    ) -> None:
        if not roots:
            raise ValueError("LeafImageDataset needs at least one root directory")
        self.roots = [Path(r) for r in roots]
        self.classes: list[str] = sorted(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.class_map = class_map
        self.transform = transform
        self.image_size = image_size
        self.samples: list[tuple[str, int]] = self._scan()
        if not self.samples:
            raise RuntimeError(
                f"No images found for classes {self.classes} under {list(map(str, self.roots))}. "
                "Check `datasets.<name>.roots` and `class_map` in your config."
            )

    def _scan(self) -> list[tuple[str, int]]:
        log = get_logger()
        samples: list[tuple[str, int]] = []
        for root in self.roots:
            if not root.is_dir():
                raise FileNotFoundError(
                    f"Dataset root '{root}' does not exist. Set it in configs/base.yaml or "
                    f"override with --set paths.data_root=... / datasets.<name>.roots=[...]"
                )
            for folder in sorted(p for p in root.iterdir() if p.is_dir()):
                target = self.class_map.get(folder.name) if self.class_map else folder.name
                if target is None or target not in self.class_to_idx:
                    log.debug("Skipping folder %s (not mapped to a target class)", folder)
                    continue
                label = self.class_to_idx[target]
                files = sorted(
                    f for f in folder.iterdir()
                    if f.suffix.lower() in IMG_EXTENSIONS and not f.name.startswith("._")
                )
                samples.extend((str(f), label) for f in files)
        return samples

    @property
    def targets(self) -> list[int]:
        return [label for _, label in self.samples]

    def load_image(self, index: int) -> Image.Image:
        path, _ = self.samples[index]
        with Image.open(path) as img:
            img = img.convert("RGB")
            if self.image_size:
                img = img.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
            return img

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image = self.load_image(index)
        if self.transform is not None:
            image = self.transform(image)
        return image, self.samples[index][1]

    def class_counts(self, indices: Sequence[int] | None = None) -> dict[str, int]:
        labels = self.targets if indices is None else [self.samples[i][1] for i in indices]
        counts = Counter(labels)
        return {self.classes[k]: counts.get(k, 0) for k in range(len(self.classes))}


class TransformSubset(Dataset):
    """A subset of a base dataset with its *own* transform.

    The notebooks built three ``ImageFolder`` instances over the same folder just to give
    train/val/test different transforms; this wrapper does it with one scan.
    """

    def __init__(self, base: LeafImageDataset, indices: Sequence[int], transform: Transform):
        self.base = base
        self.indices = list(indices)
        self.transform = transform

    @property
    def classes(self) -> list[str]:
        return self.base.classes

    @property
    def targets(self) -> list[int]:
        return [self.base.samples[i][1] for i in self.indices]

    def class_counts(self) -> dict[str, int]:
        return self.base.class_counts(self.indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        base_index = self.indices[index]
        image = self.base.load_image(base_index)
        if self.transform is not None:
            image = self.transform(image)
        return image, self.base.samples[base_index][1]
