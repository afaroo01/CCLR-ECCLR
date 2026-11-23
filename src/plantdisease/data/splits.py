from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class SplitIndices:
    train: list[int]
    val: list[int]
    test: list[int]


class DatasetSplitter:
    """Train/val/test index splitter.

    Reproduces the notebooks exactly: ``train = int(r0*n)``, ``val = int(r1*n)``,
    ``test = rest``; first split off val+test, then split that into val and test, both
    with ``random_state=seed``. ``stratify=True`` keeps class proportions in every split.
    """

    def __init__(self, ratios: Sequence[float], seed: int, stratify: bool = False) -> None:
        if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
            raise ValueError(f"split_ratios must be 3 numbers summing to 1, got {list(ratios)}")
        self.ratios = list(ratios)
        self.seed = seed
        self.stratify = stratify

    def split(self, labels: Sequence[int], stratify: bool | None = None) -> SplitIndices:
        stratify = self.stratify if stratify is None else stratify
        n = len(labels)
        n_train = int(self.ratios[0] * n)
        n_val = int(self.ratios[1] * n)
        n_test = n - n_train - n_val
        if min(n_train, n_val, n_test) <= 0:
            raise ValueError(f"Dataset of {n} images is too small for ratios {self.ratios}")

        indices = list(range(n))
        train, rest = train_test_split(
            indices,
            test_size=n_val + n_test,
            random_state=self.seed,
            stratify=list(labels) if stratify else None,
        )
        val, test = train_test_split(
            rest,
            test_size=n_test,
            random_state=self.seed,
            stratify=[labels[i] for i in rest] if stratify else None,
        )
        return SplitIndices(train=list(train), val=list(val), test=list(test))
