"""All plotting lives here. Figures are saved to ``<run_dir>/figures/`` and optionally shown."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from ..engine.trainer import TrainingHistory
from ..models.base import BaseClassifier
from ..utils.io import ensure_dir
from .embeddings import extract_embeddings, tsne_projection
from .gradcam import GradCAM


def to_display(image: Tensor) -> np.ndarray:
    """``(C, H, W)`` tensor (any normalization) -> ``(H, W, C)`` float array in ``[0, 1]``."""
    img = image.detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
    return (img - img.min()) / max(img.max() - img.min(), 1e-8)


def _hard_labels(labels: Tensor) -> Tensor:
    return labels.argmax(1) if labels.ndim == 2 else labels


class XAIVisualizer:
    """Explainability and diagnostic plots: samples, curves, confusion matrices, t-SNE, Grad-CAM."""

    def __init__(
        self,
        class_names: Sequence[str],
        output_dir: Path,
        device: torch.device,
        show: bool = False,
        seed: int = 0,
        cols: int = 8,
    ) -> None:
        self.class_names = list(class_names)
        self.figure_dir = ensure_dir(Path(output_dir) / "figures")
        self.device = device
        self.show = show
        self.seed = seed
        self.cols = cols
        palette = sns.color_palette("tab10" if len(class_names) <= 10 else "tab20")
        self.colors = {c: palette[i % len(palette)] for i, c in enumerate(self.class_names)}

    # -- helpers -----------------------------------------------------------------------------
    def _finish(self, fig: plt.Figure, name: str) -> Path:
        path = self.figure_dir / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        if self.show:
            plt.show()
        plt.close(fig)
        return path

    def _grid(self, n: int, cell: float = 2.2) -> tuple[plt.Figure, list[plt.Axes]]:
        cols = min(self.cols, n)
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * cell, rows * (cell + 0.3)),
                                 squeeze=False)
        flat = list(axes.flat)
        for ax in flat:
            ax.axis("off")
        return fig, flat

    def _short(self, label: int) -> str:
        return self.class_names[label].replace("___", "\n")

    # -- plots -------------------------------------------------------------------------------
    def plot_samples(self, loader: DataLoader, name: str, max_images: int = 32) -> Path:
        images, labels = next(iter(loader))
        labels = _hard_labels(labels)
        n = min(max_images, len(images))
        fig, axes = self._grid(n)
        for ax, img, lab in zip(axes, images[:n], labels[:n], strict=False):
            ax.imshow(to_display(img))
            ax.set_title(self._short(int(lab)), fontsize=7)
        fig.suptitle(name)
        return self._finish(fig, name)

    def plot_training_curves(self, history: TrainingHistory, name: str = "training_curves") -> Path:
        epochs = history.column("epoch")
        fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax, metric in ((ax_loss, "loss"), (ax_acc, "acc")):
            ax.plot(epochs, history.column(f"train_{metric}"), "o-", label=f"train {metric}")
            ax.plot(epochs, history.column(f"val_{metric}"), "s-", label=f"val {metric}")
            if history.best_epoch > 0:
                ax.axvline(history.best_epoch, color="grey", ls="--", lw=1, label="best epoch")
            ax.set_xlabel("epoch")
            ax.set_ylabel(metric)
            ax.grid(alpha=0.3)
            ax.legend()
        ax_loss.set_title("Loss")
        ax_acc.set_title("Accuracy")
        return self._finish(fig, name)

    def plot_confusion_matrix(
        self, cm: np.ndarray, name: str, title: str = "", normalize: bool = False
    ) -> Path:
        data = cm.astype(float) / np.maximum(cm.sum(1, keepdims=True), 1) if normalize else cm
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            data, annot=True, fmt=".2f" if normalize else "g", cmap="Blues", ax=ax,
            xticklabels=self.class_names, yticklabels=self.class_names,
        )
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(f"Confusion matrix {title}".strip())
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        return self._finish(fig, name)

    def plot_tsne(
        self, model: BaseClassifier, loader: DataLoader, name: str, max_samples: int = 2000
    ) -> Path:
        features, labels = extract_embeddings(model, loader, self.device, max_samples)
        points = tsne_projection(features, seed=self.seed)
        fig, ax = plt.subplots(figsize=(10, 6))
        for idx, cls in enumerate(self.class_names):
            mask = labels == idx
            if mask.any():
                ax.scatter(points[mask, 0], points[mask, 1], s=12, alpha=0.7,
                           color=self.colors[cls], label=cls, linewidths=0, rasterized=True)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9, frameon=True)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"t-SNE of embeddings ({name})")
        return self._finish(fig, name)

    def plot_gradcam(
        self, model: BaseClassifier, loader: DataLoader, name: str, num_images: int = 16,
        alpha: float = 0.5,
    ) -> Path:
        images, labels = next(iter(loader))
        images, labels = images[:num_images], _hard_labels(labels)[:num_images]
        with GradCAM(model) as cam:
            heatmaps = cam(images.to(self.device))
        with torch.no_grad():
            preds = model(images.to(self.device)).argmax(1).cpu()

        cmap = matplotlib.colormaps["jet"]
        fig, axes = self._grid(len(images), cell=2.4)
        for ax, img, heat, true, pred in zip(axes, images, heatmaps, labels, preds, strict=False):
            overlay = (1 - alpha) * to_display(img) + alpha * cmap(heat.numpy())[..., :3]
            ax.imshow(np.clip(overlay, 0, 1))
            ok = int(true) == int(pred)
            ax.set_title(
                f"T: {self._short(int(true))}\nP: {self._short(int(pred))}",
                fontsize=6, color="green" if ok else "red",
            )
        fig.suptitle(f"Grad-CAM ({name})")
        return self._finish(fig, name)

    def plot_episode(
        self, support: Tensor, query: Tensor, n_shot: int, n_query: int, name: str = "episode"
    ) -> Path:
        """Support and query images of one few-shot task, one row per class."""
        fig, (ax_s, ax_q) = plt.subplots(
            1, 2, figsize=(2 + 1.2 * (n_shot + n_query), 6),
            gridspec_kw={"width_ratios": [n_shot, n_query]},
        )
        for ax, imgs, per_row, title in (
            (ax_s, support, n_shot, "support"), (ax_q, query, n_query, "query")
        ):
            disp = torch.stack([torch.from_numpy(to_display(i)).permute(2, 0, 1) for i in imgs])
            ax.imshow(make_grid(disp, nrow=per_row, padding=2).permute(1, 2, 0).numpy())
            ax.set_title(title)
            ax.axis("off")
        return self._finish(fig, name)
