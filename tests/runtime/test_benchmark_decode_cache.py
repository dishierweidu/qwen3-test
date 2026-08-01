from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/benchmark_decode_cache.py"


def load_benchmark_module():
    spec = importlib.util.spec_from_file_location(
        "benchmark_decode_cache",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_deterministic_tiny_benchmark_has_complete_truthful_schema():
    benchmark = load_benchmark_module()
    result = benchmark.run_benchmark(
        prompt_length=8,
        output_length=3,
        warmup=1,
        repetitions=2,
        seed=17,
        device="cpu",
        dtype=torch.float32,
    )
    assert result["schema_version"] == 1
    assert result["prompt_length"] == 8
    assert result["output_length"] == 3
    assert result["warmup"] == 1
    assert result["repetitions"] == 2
    assert result["dtype"] == "float32"
    assert result["device"] == "cpu"
    assert result["synchronization"] == "synchronous-cpu"
    assert len(result["latency_seconds"]["cached_raw"]) == 2
    assert len(result["latency_seconds"]["uncached_raw"]) == 2
    assert result["latency_seconds"]["cached_mean"] > 0
    assert result["tokens_per_second"]["cached"] > 0
    assert result["peak_memory_bytes"] == 0
    assert result["cache_bytes"]["logical_total"] > 0
    assert result["cache_bytes"]["unique_allocated_total"] > 0
    assert result["cache_bytes"]["logical_by_partition"][
        "full_attention_kv"
    ] > 0
    assert result["max_abs"] <= 1e-5
    assert result["token_exact"] is True
    assert result["fallback"] == {"used": False, "code": None}
    assert result["architecture_manifest"]["architecture_profile"] == (
        "legacy_prototype"
    )
    assert result["architecture_summary"]["capabilities"][
        "incremental_decode_state"
    ] is True
    assert isinstance(result["implementation_commit"], str)
    assert result["implementation_commit"]


def test_parity_failure_is_raised_before_measurement(monkeypatch):
    benchmark = load_benchmark_module()
    measurements = []
    real_cached = benchmark._cached_trace

    def divergent(*args, **kwargs):
        logits, tokens = real_cached(*args, **kwargs)
        return logits + 1, tokens

    monkeypatch.setattr(benchmark, "_cached_trace", divergent)
    monkeypatch.setattr(
        benchmark,
        "_measure",
        lambda *args, **kwargs: measurements.append("timed"),
    )
    with pytest.raises(AssertionError, match="parity"):
        benchmark.run_benchmark(
            prompt_length=4,
            output_length=2,
            warmup=0,
            repetitions=1,
            seed=19,
        )
    assert measurements == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("prompt_length", True),
        ("prompt_length", 0),
        ("output_length", -1),
        ("warmup", -1),
        ("repetitions", 0),
    ],
)
def test_benchmark_rejects_invalid_controls(field, value):
    benchmark = load_benchmark_module()
    kwargs = {
        "prompt_length": 4,
        "output_length": 2,
        "warmup": 0,
        "repetitions": 1,
    }
    kwargs[field] = value
    with pytest.raises((TypeError, ValueError)):
        benchmark.run_benchmark(**kwargs)
