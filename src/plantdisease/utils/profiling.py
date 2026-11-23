"""Time and memory measurement utilities.

``ResourceMonitor`` is a context manager that records, for the enclosed block:

* wall-clock and process CPU time,
* host RAM (RSS) at start/end and the peak, sampled by a lightweight background thread,
* peak GPU memory allocated / reserved by PyTorch (CUDA), or driver memory on Apple MPS.

Example::

    with ResourceMonitor("train", device) as mon:
        trainer.fit(...)
    print(mon.usage.as_dict())
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass

import psutil
import torch
from torch import nn

_MB = 1024**2


@dataclass
class ResourceUsage:
    name: str
    wall_time_s: float = 0.0
    cpu_time_s: float = 0.0
    ram_start_mb: float = 0.0
    ram_end_mb: float = 0.0
    ram_peak_mb: float = 0.0
    gpu_peak_allocated_mb: float | None = None
    gpu_peak_reserved_mb: float | None = None

    def as_dict(self, prefix: str | None = None) -> dict[str, float]:
        """Flat numeric dict, keys prefixed (default: ``"<name>/"``). ``None`` values dropped."""
        prefix = f"{self.name}/" if prefix is None else prefix
        return {
            f"{prefix}{k}": v
            for k, v in asdict(self).items()
            if k != "name" and v is not None
        }

    def __str__(self) -> str:
        gpu = (
            f", GPU peak {self.gpu_peak_allocated_mb:.0f} MB"
            if self.gpu_peak_allocated_mb is not None
            else ""
        )
        return (
            f"[{self.name}] {format_seconds(self.wall_time_s)} wall, "
            f"RAM peak {self.ram_peak_mb:.0f} MB{gpu}"
        )


class ResourceMonitor:
    """Context manager measuring time, RAM and GPU memory of a code block.

    Monitors can be nested (e.g. a validation pass inside a training run): CUDA peak
    counters are reset on entry, so each child propagates its peaks to its parent on exit
    to keep the parent's numbers correct.
    """

    _active: list[ResourceMonitor] = []

    def __init__(
        self, name: str, device: torch.device | None = None, sample_interval_s: float = 0.2
    ) -> None:
        self.name = name
        self.device = device or torch.device("cpu")
        self.sample_interval_s = sample_interval_s
        self.usage = ResourceUsage(name=name)
        self._process = psutil.Process()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._mps_peak = 0
        self._child_gpu_alloc = 0.0
        self._child_gpu_reserved = 0.0

    # -- helpers -----------------------------------------------------------------------------
    def _rss_mb(self) -> float:
        return self._process.memory_info().rss / _MB

    def _is_mps(self) -> bool:
        return self.device.type == "mps" and hasattr(torch, "mps")

    def _sample(self) -> None:
        while not self._stop.wait(self.sample_interval_s):
            self.usage.ram_peak_mb = max(self.usage.ram_peak_mb, self._rss_mb())
            if self._is_mps():
                self._mps_peak = max(self._mps_peak, torch.mps.driver_allocated_memory())

    # -- context protocol --------------------------------------------------------------------
    def __enter__(self) -> ResourceMonitor:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            if ResourceMonitor._active:  # keep the parent's peak before we reset the counters
                parent = ResourceMonitor._active[-1]
                parent._child_gpu_alloc = max(
                    parent._child_gpu_alloc, torch.cuda.max_memory_allocated(self.device) / _MB
                )
                parent._child_gpu_reserved = max(
                    parent._child_gpu_reserved, torch.cuda.max_memory_reserved(self.device) / _MB
                )
            torch.cuda.reset_peak_memory_stats(self.device)
        self.usage.ram_start_mb = self.usage.ram_peak_mb = self._rss_mb()
        self._cpu_start = self._process.cpu_times()
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        ResourceMonitor._active.append(self)
        return self

    def __exit__(self, *exc) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.usage.wall_time_s = time.perf_counter() - self._t0
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        cpu = self._process.cpu_times()
        self.usage.cpu_time_s = (cpu.user - self._cpu_start.user) + (
            cpu.system - self._cpu_start.system
        )
        self.usage.ram_end_mb = self._rss_mb()
        self.usage.ram_peak_mb = max(self.usage.ram_peak_mb, self.usage.ram_end_mb)
        if self.device.type == "cuda":
            self.usage.gpu_peak_allocated_mb = max(
                torch.cuda.max_memory_allocated(self.device) / _MB, self._child_gpu_alloc
            )
            self.usage.gpu_peak_reserved_mb = max(
                torch.cuda.max_memory_reserved(self.device) / _MB, self._child_gpu_reserved
            )
        elif self._is_mps():
            peak = max(self._mps_peak, torch.mps.driver_allocated_memory()) / _MB
            self.usage.gpu_peak_allocated_mb = max(peak, self._child_gpu_alloc)
        self._propagate_to_parent()

    def _propagate_to_parent(self) -> None:
        if self in ResourceMonitor._active:
            ResourceMonitor._active.remove(self)
        if ResourceMonitor._active:
            parent = ResourceMonitor._active[-1]
            parent._child_gpu_alloc = max(
                parent._child_gpu_alloc, self.usage.gpu_peak_allocated_mb or 0.0
            )
            parent._child_gpu_reserved = max(
                parent._child_gpu_reserved, self.usage.gpu_peak_reserved_mb or 0.0
            )


@contextmanager
def timer() -> Iterator[dict[str, float]]:
    """Minimal wall-clock timer: ``with timer() as t: ...; t["seconds"]``."""
    result: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["seconds"] = time.perf_counter() - start


def count_parameters(model: nn.Module) -> dict[str, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = (
        sum(p.numel() * p.element_size() for p in model.parameters())
        + sum(b.numel() * b.element_size() for b in model.buffers())
    ) / _MB
    return {
        "model/params_total": total,
        "model/params_trainable": trainable,
        "model/size_mb": size_mb,
    }


@torch.no_grad()
def measure_inference_latency(
    model: nn.Module,
    input_shape: tuple[int, ...],
    device: torch.device,
    warmup: int = 5,
    iterations: int = 20,
) -> dict[str, float]:
    """Average forward-pass latency for a batch of shape ``input_shape``."""
    was_training = model.training
    model.eval()
    x = torch.randn(*input_shape, device=device)

    def _sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()

    for _ in range(warmup):
        model(x)
    _sync()
    start = time.perf_counter()
    for _ in range(iterations):
        model(x)
    _sync()
    elapsed = (time.perf_counter() - start) / iterations
    model.train(was_training)
    batch = input_shape[0]
    return {
        "inference/batch_latency_ms": elapsed * 1000,
        "inference/per_image_ms": elapsed * 1000 / batch,
        "inference/throughput_img_s": batch / elapsed,
    }


def format_seconds(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h)}h{int(m):02d}m{s:04.1f}s" if h else f"{int(m)}m{s:04.1f}s"
