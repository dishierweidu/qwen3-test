"""Shared construction helpers for correctness-gated benchmark CLIs."""

from __future__ import annotations

import platform
from pathlib import Path
import subprocess
import sys
from typing import Callable, Iterable

import torch
import yaml

from qwen3_omni_pretrain.evaluation.contracts import (
    CorrectnessGate,
    ExperimentReport,
    RuntimeEnvironment,
)
from qwen3_omni_pretrain.models.hybrid_swa_moe.configuration_hybrid_swa_moe import (
    HybridSwaMoeConfig,
)
from qwen3_omni_pretrain.utils.profiling import synchronized_wall_clock


_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def resolve_dtype(value: str) -> torch.dtype:
    if not isinstance(value, str):
        raise TypeError("dtype must be a string")
    try:
        return _DTYPES[value.removeprefix("torch.")]
    except KeyError as error:
        raise ValueError("dtype must be float32, float16, or bfloat16") from error


def load_hybrid_config(path_value: str | Path) -> HybridSwaMoeConfig:
    path = Path(path_value)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise TypeError("benchmark config must contain a mapping")
    return HybridSwaMoeConfig(**raw)


def positive_int(value: object, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def positive_int_values(
    values: Iterable[object], *, name: str
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be an iterable of integers")
    try:
        copied = tuple(values)
    except TypeError as error:
        raise TypeError(f"{name} must be an iterable of integers") from error
    if not copied:
        raise ValueError(f"{name} must not be empty")
    validated = tuple(
        positive_int(value, name=f"{name} item") for value in copied
    )
    if len(set(validated)) != len(validated):
        raise ValueError(f"{name} must not contain duplicates")
    return validated


def require_passing_gates(gates: tuple[CorrectnessGate, ...]) -> None:
    if not gates or not all(gate.passed for gate in gates):
        failed = [gate.name for gate in gates if not gate.passed]
        raise RuntimeError(
            "benchmark correctness gates failed before timing: "
            + ", ".join(failed or ["missing gates"])
        )


def timing_samples(
    function: Callable[[], object],
    *,
    repetitions: int,
    warmup: int,
    device: torch.device,
) -> tuple[float, ...]:
    positive_int(repetitions, name="repetitions")
    if type(warmup) is not int or warmup < 0:
        raise ValueError("warmup must be a non-negative integer")
    for _ in range(warmup):
        synchronized_wall_clock(function, device=device)
    return tuple(
        synchronized_wall_clock(function, device=device).elapsed_seconds
        for _ in range(repetitions)
    )


def runtime_environment(
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    kernels: tuple[str, ...],
    fallbacks: tuple[str, ...],
) -> RuntimeEnvironment:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if dirty.strip():
            commit += "-dirty"
    except (OSError, subprocess.SubprocessError):
        commit = "unknown-working-tree"
    hardware = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda" and torch.cuda.is_available()
        else platform.processor() or platform.machine() or "unknown-cpu"
    )
    return RuntimeEnvironment(
        git_commit=commit,
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        device=str(device),
        precision=str(dtype).removeprefix("torch."),
        hardware=hardware,
        platform=platform.platform(),
        kernels=kernels,
        fallbacks=fallbacks,
        seed=seed,
    )


def write_report(report: ExperimentReport, output: str | None) -> None:
    payload = report.to_json()
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload + "\n", encoding="utf-8")
    else:
        sys.stdout.write(payload + "\n")


__all__ = [
    "load_hybrid_config",
    "positive_int",
    "positive_int_values",
    "require_passing_gates",
    "resolve_dtype",
    "runtime_environment",
    "timing_samples",
    "write_report",
]
