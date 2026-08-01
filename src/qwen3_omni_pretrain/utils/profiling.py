"""Small profiling helpers with no import-time accelerator initialization."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import platform
import resource
import time
from typing import Callable, TypeVar

import torch


T = TypeVar("T")


@dataclass(frozen=True)
class TimedResult:
    result: object
    elapsed_seconds: float


@dataclass(frozen=True)
class PeakMemory:
    bytes: int
    device: str
    platform: str
    source: str


def synchronize_device(device: torch.device | str | None = None) -> None:
    """Synchronize only when the caller explicitly targets an available GPU."""

    resolved = torch.device(device) if device is not None else torch.device("cpu")
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA synchronization requested but CUDA is unavailable")
        torch.cuda.synchronize(resolved)


def synchronized_wall_clock(
    function: Callable[..., T],
    *args: object,
    device: torch.device | str | None = None,
    **kwargs: object,
) -> TimedResult:
    """Measure a callable with accelerator synchronization around the sample."""

    synchronize_device(device)
    start = time.perf_counter()
    result = function(*args, **kwargs)
    synchronize_device(device)
    elapsed = time.perf_counter() - start
    return TimedResult(result=result, elapsed_seconds=elapsed)


def percentile(samples: tuple[float, ...] | list[float], q: float) -> float:
    """Return a linearly interpolated percentile from finite raw samples."""

    if not isinstance(q, (int, float)) or isinstance(q, bool):
        raise TypeError("percentile q must be numeric")
    q_float = float(q)
    if not math.isfinite(q_float) or not 0.0 <= q_float <= 1.0:
        raise ValueError("percentile q must be finite and within [0, 1]")
    if not samples:
        raise ValueError("percentile requires at least one sample")
    converted = tuple(float(sample) for sample in samples)
    if any(not math.isfinite(sample) for sample in converted):
        raise ValueError("percentile samples must be finite")
    ordered = sorted(converted)
    position = (len(ordered) - 1) * q_float
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def reset_peak_memory(device: torch.device | str | None = None) -> None:
    """Reset CUDA peak statistics; CPU RSS has no reset operation."""

    resolved = torch.device(device) if device is not None else torch.device("cpu")
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA peak reset requested but CUDA is unavailable")
        torch.cuda.reset_peak_memory_stats(resolved)


def measure_peak_memory(device: torch.device | str | None = None) -> PeakMemory:
    """Read CUDA allocated peak or process CPU maximum RSS in bytes."""

    resolved = torch.device(device) if device is not None else torch.device("cpu")
    system = platform.system()
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA peak memory requested but CUDA is unavailable")
        return PeakMemory(
            bytes=int(torch.cuda.max_memory_allocated(resolved)),
            device=str(resolved),
            platform=system,
            source="torch.cuda.max_memory_allocated",
        )

    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB, whereas macOS reports bytes.
    rss_bytes = rss if system == "Darwin" else rss * 1024
    return PeakMemory(
        bytes=rss_bytes,
        device="cpu",
        platform=system,
        source="resource.getrusage.ru_maxrss",
    )


def cpu_count() -> int:
    """Expose the effective logical CPU count for report construction."""

    return int(os.cpu_count() or 1)


__all__ = [
    "PeakMemory",
    "TimedResult",
    "cpu_count",
    "measure_peak_memory",
    "percentile",
    "reset_peak_memory",
    "synchronize_device",
    "synchronized_wall_clock",
]
