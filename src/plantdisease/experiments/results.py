"""Aggregation of per-seed metrics into mean ± std, and cross-experiment comparison tables."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from ..utils.io import load_json, save_csv, save_json

PerSeed = Mapping[int, Mapping[str, float]]
Summary = dict[str, dict[str, float]]

DEFAULT_REPORT_METRICS = (
    "test/accuracy",
    "test/macro_f1",
    "target/accuracy",
    "target/accuracy_ci95",
    "target/macro_f1",
    "train/epochs_run",
    "train/wall_time_s",
    "train/gpu_peak_allocated_mb",
    "train/ram_peak_mb",
    "inference/per_image_ms",
)


class ResultAggregator:
    """Turns ``{seed: {metric: value}}`` into ``{metric: {mean, std, min, max, n}}``.

    ``std`` is the *sample* standard deviation (``ddof=1``), the usual choice when
    reporting variability across a handful of random seeds; it is 0 for a single seed.
    """

    @staticmethod
    def aggregate(per_seed: PerSeed) -> Summary:
        keys: list[str] = []
        for metrics in per_seed.values():
            keys.extend(k for k in metrics if k not in keys and k != "seed")
        summary: Summary = {}
        for key in keys:
            values = np.array(
                [m[key] for m in per_seed.values() if isinstance(m.get(key), (int, float))],
                dtype=float,
            )
            if values.size == 0:
                continue
            summary[key] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "min": float(values.min()),
                "max": float(values.max()),
                "n": int(values.size),
            }
        return summary

    @staticmethod
    def save(per_seed: PerSeed, summary: Summary, experiment_dir: Path, title: str = "") -> None:
        experiment_dir = Path(experiment_dir)
        save_json({"per_seed": per_seed, "summary": summary}, experiment_dir / "summary.json")
        save_csv(
            [{"metric": k, **v} for k, v in summary.items()], experiment_dir / "summary.csv"
        )
        save_csv([{"seed": s, **m} for s, m in per_seed.items()], experiment_dir / "per_seed.csv")
        (experiment_dir / "summary.md").write_text(
            summary_markdown(summary, title=title, seeds=list(per_seed))
        )


def format_mean_std(stats: Mapping[str, float], scale: float = 1.0, digits: int = 4) -> str:
    return f"{stats['mean'] * scale:.{digits}f} ± {stats['std'] * scale:.{digits}f}"


def summary_markdown(
    summary: Summary, title: str = "", seeds: Sequence[int] | None = None
) -> str:
    lines = [f"# {title}".rstrip(), ""]
    if seeds is not None:
        lines += [f"Seeds: {list(seeds)} (n={len(seeds)})", ""]
    lines += ["| metric | mean ± std | min | max |", "|---|---|---|---|"]
    for key, s in summary.items():
        lines.append(f"| {key} | {format_mean_std(s)} | {s['min']:.4f} | {s['max']:.4f} |")
    return "\n".join(lines) + "\n"


def compare_experiments(
    experiment_dirs: Sequence[Path], metrics: Sequence[str] = DEFAULT_REPORT_METRICS
) -> list[dict[str, str]]:
    """One row per experiment with ``mean ± std`` for each requested metric."""
    rows = []
    for exp_dir in experiment_dirs:
        path = Path(exp_dir) / "summary.json"
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found -- run the experiment first")
        data = load_json(path)
        summary = data["summary"]
        row = {"experiment": Path(exp_dir).name, "seeds": len(data["per_seed"])}
        for metric in metrics:
            row[metric] = format_mean_std(summary[metric]) if metric in summary else "-"
        rows.append(row)
    return rows


def rows_to_markdown(rows: Sequence[Mapping[str, object]]) -> str:
    if not rows:
        return ""
    headers = list(rows[0])
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(r.get(h, "")) for h in headers) + " |" for r in rows]
    return "\n".join(out) + "\n"
