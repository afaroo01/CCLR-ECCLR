from __future__ import annotations

import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from ..models.base import BaseClassifier


@torch.no_grad()
def extract_embeddings(
    model: BaseClassifier,
    loader: DataLoader,
    device: torch.device,
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(features (N, D), labels (N,))`` from :meth:`BaseClassifier.embed`."""
    model.eval()
    features, labels, seen = [], [], 0
    for images, targets in loader:
        features.append(model.embed(images.to(device)).cpu().numpy())
        labels.append((targets.argmax(1) if targets.ndim == 2 else targets).numpy())
        seen += len(images)
        if max_samples and seen >= max_samples:
            break
    feats, labs = np.concatenate(features), np.concatenate(labels)
    return (feats[:max_samples], labs[:max_samples]) if max_samples else (feats, labs)


def tsne_projection(features: np.ndarray, seed: int = 0, perplexity: float = 30.0) -> np.ndarray:
    """2-D t-SNE scaled to ``[0, 1]`` on each axis."""
    perplexity = min(perplexity, max(1.0, (len(features) - 1) / 3))
    tsne = TSNE(n_components=2, random_state=seed, perplexity=perplexity, init="pca")
    points = tsne.fit_transform(features)
    points -= points.min(0)
    return points / np.maximum(points.max(0), 1e-12)
