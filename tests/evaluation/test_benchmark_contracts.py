from __future__ import annotations

from argparse import Namespace

from qwen3_omni_pretrain.evaluation.contracts import ExperimentReport
from scripts.benchmark_hybrid_attention import run_benchmark as run_attention
from scripts.benchmark_legacy_deltanet import run_benchmark as run_legacy
from scripts.benchmark_moe import run_benchmark as run_moe
from scripts.benchmark_mtp import run_benchmark as run_mtp


HYBRID_CONFIG = "configs/model/hybrid_swa_moe_tiny.yaml"


def _common(**updates):
    values = {
        "config": HYBRID_CONFIG,
        "device": "cpu",
        "dtype": "float32",
        "seed": 71,
        "repetitions": 1,
        "warmup": 0,
    }
    values.update(updates)
    return Namespace(**values)


def _assert_complete(report: ExperimentReport) -> None:
    report.validate()
    restored = ExperimentReport.from_json(report.to_json())
    assert restored.to_dict() == report.to_dict()
    assert report.manifest is not None
    assert report.architecture.total_parameters > 0
    assert report.architecture.active_parameters_per_token > 0
    assert report.environment.git_commit
    assert report.environment.python_version
    assert report.correctness_gates
    assert all(gate.passed for gate in report.correctness_gates)
    assert report.measurements
    timing = [
        item
        for item in report.measurements
        if "latency" in item.name or "tokens_per_second" in item.name
    ]
    assert timing and all(item.raw_samples for item in timing)


def test_attention_reports_separate_ordinary_and_tied_comparisons():
    ordinary = run_attention(
        _common(
            prompt_lengths=[4, 8],
            output_lengths=[1, 2],
            attention_modes=["full", "swa"],
            paired_kv_heads=None,
        )
    )
    paired = run_attention(
        _common(
            prompt_lengths=[8],
            output_lengths=[1],
            attention_modes=["full", "swa"],
            paired_kv_heads=4,
        )
    )
    _assert_complete(ordinary)
    _assert_complete(paired)

    assert ordinary.comparison_metadata["weight_comparable"] is False
    assert ordinary.comparison_metadata["reason"] == (
        "full and SWA KV projection shapes differ"
    )
    assert paired.comparison_metadata["weight_comparable"] is True
    assert paired.comparison_metadata["kv_heads"] == 4
    assert paired.comparison_metadata["copied_parameter_digest"]
    assert paired.comparison_metadata["copied_parameter_inventory"]
    cache_cases = {
        (
            item.name,
            item.dimensions["prompt_length"],
            item.dimensions["output_length"],
        )
        for item in ordinary.measurements
        if item.name.endswith("_cache_bytes")
    }
    assert len(cache_cases) == 2 * 2 * 2


def test_moe_report_preserves_router_observability_and_loss_weights():
    report = run_moe(
        _common(sequence_lengths=[3, 4], experts=8, top_k=2)
    )
    _assert_complete(report)
    assert report.router_aux_loss_weight == 0.01
    assert report.mtp_loss_weight == 0.1
    names = [measurement.name for measurement in report.measurements]
    assert "router_entropy" in names
    assert "max_mean_load" in names
    assert names.count("expert_token_count") == 16
    assert {
        item.dimensions["tokens"]
        for item in report.measurements
        if item.name == "expert_token_count"
    } == {3, 4}


def test_mtp_report_records_effective_weight_and_prompt_categories():
    report = run_mtp(
        _common(
            sequence_length=6,
            batch_sizes=[1, 2],
            draft_lengths=[1, 3],
            prompt_sources=["natural", "random"],
        )
    )
    _assert_complete(report)
    assert report.mtp_loss_weight == 0.1
    assert report.comparison_metadata["implemented_predictors"] == 1
    source_dimensions = {
        item.dimensions.get("prompt_source")
        for item in report.measurements
        if "prompt_source" in item.dimensions
    }
    assert source_dimensions == {"natural", "random"}
    verification_cases = {
        (
            item.dimensions["prompt_source"],
            item.dimensions["draft_length"],
            item.dimensions["batch_size"],
        )
        for item in report.measurements
        if item.name == "verification_latency_p50"
    }
    assert verification_cases == {
        (source, draft_length, batch_size)
        for source in ("natural", "random")
        for draft_length in (1, 3)
        for batch_size in (1, 2)
    }
    distribution_gate = next(
        gate
        for gate in report.correctness_gates
        if gate.name == "sampled_distribution_target_parity"
    )
    assert distribution_gate.passed


def test_legacy_deltanet_is_absolute_non_comparable_reference():
    report = run_legacy(
        _common(
            config="configs/model/legacy_deltanet_tiny_benchmark.yaml",
            sequence_lengths=[3, 4],
        )
    )
    _assert_complete(report)
    assert report.comparison_metadata["weight_comparable"] is False
    assert report.comparison_metadata["reason"] == (
        "different model class and independently initialized weights"
    )
    assert report.comparison_metadata["primary"] == "mimo_v25_experimental"
    assert report.comparison_metadata["subject"] == "legacy_deltanet"
    assert report.mtp_loss_weight == 0.0
    assert {
        item.dimensions["sequence_length"]
        for item in report.measurements
        if item.name == "legacy_deltanet_latency_p50"
    } == {3, 4}
