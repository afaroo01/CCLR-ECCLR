"""N-way K-shot episode sampling (adapted from easyfsl's ``TaskSampler``)."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterator, Sequence

import torch
from torch import Tensor
from torch.utils.data import Sampler

from ..utils.logging import get_logger

Episode = tuple[Tensor, Tensor, Tensor, Tensor, Tensor]


class TaskSampler(Sampler[list[int]]):
    """Yields the dataset indices of one few-shot task per iteration.

    Each task samples ``n_way`` classes, then ``n_shot + n_query`` images per class.
    Use :meth:`episodic_collate_fn` as the DataLoader ``collate_fn``.

    Args:
        labels: label of every item in the dataset the loader iterates over.
        seed: makes the sequence of tasks reproducible.
    """

    def __init__(
        self,
        labels: Sequence[int],
        n_way: int,
        n_shot: int,
        n_query: int,
        n_tasks: int,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_tasks = n_tasks
        self._rng = random.Random(seed)

        self.items_per_label: dict[int, list[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            self.items_per_label[int(label)].append(idx)
        self._validate()

    def _validate(self) -> None:
        """Drop classes with < n_shot + n_query images; fail if fewer than n_way remain."""
        needed = self.n_shot + self.n_query
        too_small = {k: len(v) for k, v in self.items_per_label.items() if len(v) < needed}
        for label in too_small:
            del self.items_per_label[label]
        if self.n_way > len(self.items_per_label):
            raise ValueError(
                f"n_way={self.n_way} but only {len(self.items_per_label)} classes have at least "
                f"n_shot + n_query = {needed} images (too small: {too_small})"
            )
        if too_small:
            get_logger().warning(
                "TaskSampler: ignoring classes %s with < %d images", too_small, needed
            )

    def __len__(self) -> int:
        return self.n_tasks

    def __iter__(self) -> Iterator[list[int]]:
        labels = sorted(self.items_per_label)
        for _ in range(self.n_tasks):
            task: list[int] = []
            for label in self._rng.sample(labels, self.n_way):
                task.extend(
                    self._rng.sample(self.items_per_label[label], self.n_shot + self.n_query)
                )
            yield task

    def episodic_collate_fn(self, batch: list[tuple[Tensor, int]]) -> Episode:
        """Return ``(support_images, support_labels, query_images, query_labels, class_ids)``.

        Labels are episode-local (``0..n_way-1``); ``class_ids[i]`` is the dataset label of
        episode class ``i``, used to map predictions back to global classes for metrics.
        """
        labels = [int(label) for _, label in batch]
        class_ids = list(dict.fromkeys(labels))  # order-preserving unique
        images = torch.stack([img for img, _ in batch])
        local = torch.tensor([class_ids.index(label) for label in labels])

        k = self.n_shot + self.n_query
        images = images.view(self.n_way, k, *images.shape[1:])
        local = local.view(self.n_way, k)

        support_images = images[:, : self.n_shot].reshape(-1, *images.shape[2:])
        query_images = images[:, self.n_shot :].reshape(-1, *images.shape[2:])
        support_labels = local[:, : self.n_shot].flatten()
        query_labels = local[:, self.n_shot :].flatten()
        return support_images, support_labels, query_images, query_labels, torch.tensor(class_ids)
