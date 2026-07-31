"""Auditable contracts for architecture experiments and benchmarks.

The report types in this module deliberately keep correctness evidence next to
performance measurements.  A benchmark cannot be serialized as a successful
experiment unless every declared correctness gate passed and raw timing samples
are retained.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from types import MappingProxyType
from typing import Mapping

from qwen3_omni_pretrain.architecture.manifest import ProfileManifest
from qwen3_omni_pretrain.architecture.summary import ArchitectureSummary


_JSON_SCALAR = (str, int, float, bool)


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{field} must be finite")
    return converted


def _immutable_scalar_mapping(
    value: Mapping[str, str | int | float | bool],
    *,
    field: str,
) -> Mapping[str, str | int | float | bool]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    copied: dict[str, str | int | float | bool] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise TypeError(f"{field} keys must be non-empty strings")
        if item is None or not isinstance(item, _JSON_SCALAR):
            raise TypeError(f"{field} values must be JSON scalars")
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError(f"{field} values must be finite")
        copied[key] = item
    return MappingProxyType(copied)


def _require_exact_keys(
    raw: Mapping[str, object],
    expected: set[str],
    *,
    type_name: str,
) -> dict[str, object]:
    if not isinstance(raw, Mapping):
        raise TypeError(f"{type_name} must be a mapping")
    copied = dict(raw)
    missing = expected - set(copied)
    unknown = set(copied) - expected
    if missing or unknown:
        raise ValueError(
            f"invalid {type_name} keys: missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    return copied


@dataclass(frozen=True)
class RuntimeEnvironment:
    """Environment details required to reproduce a measurement."""

    git_commit: str
    python_version: str
    torch_version: str
    device: str
    precision: str
    hardware: str
    platform: str
    kernels: tuple[str, ...]
    fallbacks: tuple[str, ...]
    seed: int

    def __post_init__(self) -> None:
        for field in (
            "git_commit",
            "python_version",
            "torch_version",
            "device",
            "precision",
            "hardware",
            "platform",
        ):
            value = getattr(self, field)
            if not isinstance(value, str):
                raise TypeError(f"{field} must be a string")
            if not value.strip():
                raise ValueError(f"{field} must be non-empty")
        for field in ("kernels", "fallbacks"):
            value = getattr(self, field)
            if not isinstance(value, tuple) or any(
                not isinstance(item, str) or not item.strip() for item in value
            ):
                raise TypeError(f"{field} must be a tuple of non-empty strings")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")

    def to_dict(self) -> dict[str, object]:
        return {
            "git_commit": self.git_commit,
            "python_version": self.python_version,
            "torch_version": self.torch_version,
            "device": self.device,
            "precision": self.precision,
            "hardware": self.hardware,
            "platform": self.platform,
            "kernels": list(self.kernels),
            "fallbacks": list(self.fallbacks),
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "RuntimeEnvironment":
        values = _require_exact_keys(
            raw,
            {
                "git_commit",
                "python_version",
                "torch_version",
                "device",
                "precision",
                "hardware",
                "platform",
                "kernels",
                "fallbacks",
                "seed",
            },
            type_name="runtime environment",
        )
        for name in ("kernels", "fallbacks"):
            value = values[name]
            if not isinstance(value, list) or any(
                not isinstance(item, str) for item in value
            ):
                raise TypeError(f"{name} must be a list of strings")
            values[name] = tuple(value)
        return cls(**values)  # type: ignore[arg-type]


@dataclass(frozen=True)
class CorrectnessGate:
    name: str
    passed: bool
    tolerance: float | None
    observed: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("correctness gate name must be non-empty")
        if type(self.passed) is not bool:
            raise TypeError("correctness gate passed must be boolean")
        for field in ("tolerance", "observed"):
            value = getattr(self, field)
            if value is not None:
                converted = _finite_number(value, field=field)
                if field == "tolerance" and converted < 0:
                    raise ValueError("tolerance must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "passed": self.passed,
            "tolerance": self.tolerance,
            "observed": self.observed,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "CorrectnessGate":
        values = _require_exact_keys(
            raw,
            {"name", "passed", "tolerance", "observed"},
            type_name="correctness gate",
        )
        return cls(**values)  # type: ignore[arg-type]


def is_timing_or_throughput(measurement: "BenchmarkMeasurement") -> bool:
    name = measurement.name.lower()
    unit = measurement.unit.lower().replace(" ", "")
    return (
        any(token in name for token in ("latency", "elapsed", "duration", "throughput", "tokens_per_second", "samples_per_second"))
        or unit in {"s", "ms", "us", "ns", "seconds", "tokens/s", "samples/s", "items/s", "bytes/s"}
        or unit.endswith("/s")
    )


def _linear_percentile(samples: tuple[float, ...], percentile: float) -> float:
    if not samples:
        raise ValueError("percentile requires at least one sample")
    ordered = sorted(samples)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


@dataclass(frozen=True)
class BenchmarkMeasurement:
    name: str
    value: float
    unit: str
    dimensions: Mapping[str, int | float | str | bool]
    raw_samples: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("measurement name must be non-empty")
        if not isinstance(self.unit, str) or not self.unit.strip():
            raise ValueError("measurement unit must be non-empty")
        _finite_number(self.value, field="measurement value")
        if not isinstance(self.raw_samples, tuple):
            raise TypeError("raw_samples must be a tuple")
        samples = tuple(
            _finite_number(sample, field="raw sample")
            for sample in self.raw_samples
        )
        object.__setattr__(self, "raw_samples", samples)
        object.__setattr__(
            self,
            "dimensions",
            _immutable_scalar_mapping(self.dimensions, field="dimensions"),
        )
        if is_timing_or_throughput(self) and any(sample < 0 for sample in samples):
            raise ValueError("timing/throughput raw samples must be non-negative")

        lowered = self.name.lower()
        if samples and ("p50" in lowered or "median" in lowered):
            object.__setattr__(self, "value", _linear_percentile(samples, 0.50))
        elif samples and "p90" in lowered:
            object.__setattr__(self, "value", _linear_percentile(samples, 0.90))

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "dimensions": dict(sorted(self.dimensions.items())),
            "raw_samples": list(self.raw_samples),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "BenchmarkMeasurement":
        values = _require_exact_keys(
            raw,
            {"name", "value", "unit", "dimensions", "raw_samples"},
            type_name="benchmark measurement",
        )
        samples = values["raw_samples"]
        if not isinstance(samples, list):
            raise TypeError("raw_samples must be a list")
        values["raw_samples"] = tuple(samples)
        return cls(**values)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ExperimentReport:
    manifest: ProfileManifest
    architecture: ArchitectureSummary
    environment: RuntimeEnvironment
    router_aux_loss_weight: float
    mtp_loss_weight: float
    correctness_gates: tuple[CorrectnessGate, ...]
    measurements: tuple[BenchmarkMeasurement, ...]
    comparison_metadata: Mapping[str, int | float | str | bool]

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, ProfileManifest):
            raise TypeError("manifest must be a ProfileManifest")
        if not isinstance(self.architecture, ArchitectureSummary):
            raise TypeError("architecture must be an ArchitectureSummary")
        if not isinstance(self.environment, RuntimeEnvironment):
            raise TypeError("environment must be a RuntimeEnvironment")
        if not isinstance(self.correctness_gates, tuple) or any(
            not isinstance(gate, CorrectnessGate)
            for gate in self.correctness_gates
        ):
            raise TypeError("correctness_gates must contain CorrectnessGate values")
        if not isinstance(self.measurements, tuple) or any(
            not isinstance(item, BenchmarkMeasurement)
            for item in self.measurements
        ):
            raise TypeError("measurements must contain BenchmarkMeasurement values")
        object.__setattr__(
            self,
            "comparison_metadata",
            _immutable_scalar_mapping(
                self.comparison_metadata,
                field="comparison_metadata",
            ),
        )
        self.validate()

    def validate(self) -> None:
        self.manifest.validate()
        if self.architecture.profile != self.manifest.architecture_profile.value:
            raise ValueError("architecture profile contradicts manifest")
        if (
            self.architecture.compatibility_level
            != self.manifest.compatibility_level.value
        ):
            raise ValueError("architecture compatibility contradicts manifest")
        for field in ("router_aux_loss_weight", "mtp_loss_weight"):
            value = _finite_number(getattr(self, field), field=field)
            if value < 0:
                raise ValueError("loss weights must be finite and non-negative")
        if self.measurements and (
            not self.correctness_gates
            or not all(gate.passed for gate in self.correctness_gates)
        ):
            raise ValueError(
                "performance measurements require passing correctness gates"
            )
        for measurement in self.measurements:
            if is_timing_or_throughput(measurement) and not measurement.raw_samples:
                raise ValueError(
                    "timing/throughput measurements require raw samples"
                )

        architecture_count = self.comparison_metadata.get("architecture_count", 1)
        if (
            type(architecture_count) is not int
            or architecture_count < 1
        ):
            raise ValueError("architecture_count must be a positive integer")
        if architecture_count > 1:
            comparable = self.comparison_metadata.get("weight_comparable")
            if type(comparable) is not bool:
                raise ValueError(
                    "multi-architecture comparison requires weight_comparable"
                )
            if comparable is False:
                reason = self.comparison_metadata.get(
                    "reason",
                    self.comparison_metadata.get("comparison_reason"),
                )
                if not isinstance(reason, str) or not reason.strip():
                    raise ValueError(
                        "non-comparable architectures require a comparison reason"
                    )

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "manifest": self.manifest.to_dict(),
            "architecture": self.architecture.to_dict(),
            "environment": self.environment.to_dict(),
            "router_aux_loss_weight": self.router_aux_loss_weight,
            "mtp_loss_weight": self.mtp_loss_weight,
            "correctness_gates": [gate.to_dict() for gate in self.correctness_gates],
            "measurements": [item.to_dict() for item in self.measurements],
            "comparison_metadata": dict(sorted(self.comparison_metadata.items())),
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
            allow_nan=False,
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "ExperimentReport":
        values = _require_exact_keys(
            raw,
            {
                "manifest",
                "architecture",
                "environment",
                "router_aux_loss_weight",
                "mtp_loss_weight",
                "correctness_gates",
                "measurements",
                "comparison_metadata",
            },
            type_name="experiment report",
        )
        gates = values["correctness_gates"]
        measurements = values["measurements"]
        if not isinstance(gates, list) or not isinstance(measurements, list):
            raise TypeError("report gates and measurements must be lists")
        return cls(
            manifest=ProfileManifest.from_dict(values["manifest"]),  # type: ignore[arg-type]
            architecture=ArchitectureSummary.from_dict(values["architecture"]),  # type: ignore[arg-type]
            environment=RuntimeEnvironment.from_dict(values["environment"]),  # type: ignore[arg-type]
            router_aux_loss_weight=values["router_aux_loss_weight"],  # type: ignore[arg-type]
            mtp_loss_weight=values["mtp_loss_weight"],  # type: ignore[arg-type]
            correctness_gates=tuple(CorrectnessGate.from_dict(item) for item in gates),  # type: ignore[arg-type]
            measurements=tuple(BenchmarkMeasurement.from_dict(item) for item in measurements),  # type: ignore[arg-type]
            comparison_metadata=values["comparison_metadata"],  # type: ignore[arg-type]
        )

    @classmethod
    def from_json(cls, payload: str) -> "ExperimentReport":
        if not isinstance(payload, str):
            raise TypeError("report JSON must be a string")
        raw = json.loads(
            payload,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value!r}")
            ),
        )
        return cls.from_dict(raw)


__all__ = [
    "BenchmarkMeasurement",
    "CorrectnessGate",
    "ExperimentReport",
    "RuntimeEnvironment",
    "is_timing_or_throughput",
]
