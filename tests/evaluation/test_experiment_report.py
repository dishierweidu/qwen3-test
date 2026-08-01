from __future__ import annotations

from dataclasses import FrozenInstanceError
import math

import pytest

from qwen3_omni_pretrain.architecture.manifest import (
    ProfileManifest,
    SourceRevision,
)
from qwen3_omni_pretrain.architecture.profiles import (
    ArchitectureProfile,
    CompatibilityLevel,
)
from qwen3_omni_pretrain.architecture.summary import (
    ArchitectureSummary,
    LayerArchitecture,
)
from qwen3_omni_pretrain.evaluation.contracts import (
    BenchmarkMeasurement,
    CorrectnessGate,
    ExperimentReport,
    RuntimeEnvironment,
)
from qwen3_omni_pretrain.utils.profiling import measure_peak_memory, percentile


def _manifest() -> ProfileManifest:
    return ProfileManifest(
        architecture_profile=ArchitectureProfile.MIMO_V25_EXPERIMENTAL,
        compatibility_level=CompatibilityLevel.MIMO_STYLE_EXPERIMENT,
        sources={
            "architecture": SourceRevision(
                name="MiMo-V2.5",
                revision="63651580ca774f8504f676040460aed3e1244ac1",
            )
        },
        assumptions=("generic tiny mechanism experiment",),
        exact_official_checkpoint_compatible=False,
    )


def _architecture() -> ArchitectureSummary:
    return ArchitectureSummary(
        profile="mimo_v25_experimental",
        compatibility_level="MiMo-style-experiment",
        model_type="hybrid_swa_moe_experimental",
        tokenizer_vocab_size=32,
        embedding_vocab_size=32,
        total_parameters=10,
        active_parameters_per_token=8,
        routed_parameters=4,
        shared_parameters=0,
        dense_parameters=6,
        capabilities={"swa": True},
        layers=(
            LayerArchitecture(
                index=0,
                attention_type="sliding-window-attention",
                cache_type="sliding-window-kv-cache",
                ffn_type="dense",
                routed_experts=0,
                experts_per_token=0,
            ),
        ),
        unsupported_capabilities=("official-checkpoint-loading",),
    )


def _environment() -> RuntimeEnvironment:
    return RuntimeEnvironment(
        git_commit="deadbeef",
        python_version="3.10.0",
        torch_version="2.10.0",
        device="cpu",
        precision="float32",
        hardware="test cpu",
        platform="Linux",
        kernels=("eager",),
        fallbacks=(),
        seed=7,
    )


def _valid_report(
    *,
    raw_samples: tuple[float, ...] = (0.1, 0.2, 0.3),
    router_aux_loss_weight: float = 0.02,
    mtp_loss_weight: float = 0.1,
    comparison_metadata: dict[str, object] | None = None,
) -> ExperimentReport:
    return ExperimentReport(
        manifest=_manifest(),
        architecture=_architecture(),
        environment=_environment(),
        router_aux_loss_weight=router_aux_loss_weight,
        mtp_loss_weight=mtp_loss_weight,
        correctness_gates=(
            CorrectnessGate(
                name="one-shot/chunked parity",
                passed=True,
                tolerance=1e-5,
                observed=1e-7,
            ),
        ),
        measurements=(
            BenchmarkMeasurement(
                name="decode_latency_p50",
                value=999.0,
                unit="s",
                dimensions={"context": 128, "output": 32},
                raw_samples=raw_samples,
            ),
        ),
        comparison_metadata=comparison_metadata or {"architecture_count": 1},
    )


def test_report_rejects_performance_without_correctness_gate():
    with pytest.raises(ValueError, match="correctness"):
        ExperimentReport(
            manifest=_manifest(),
            architecture=_architecture(),
            environment=_environment(),
            router_aux_loss_weight=0.01,
            mtp_loss_weight=0.0,
            correctness_gates=(),
            measurements=(
                BenchmarkMeasurement(
                    name="decode_tokens_per_second",
                    value=10.0,
                    unit="tokens/s",
                    dimensions={"context": 128, "output": 32},
                    raw_samples=(10.0,),
                ),
            ),
            comparison_metadata={"architecture_count": 1},
        )


def test_report_round_trips_raw_samples_and_loss_weights():
    report = _valid_report(raw_samples=(1.0, 2.0, 4.0))
    restored = ExperimentReport.from_json(report.to_json())

    assert restored.router_aux_loss_weight == 0.02
    assert restored.mtp_loss_weight == 0.1
    assert restored.measurements[0].raw_samples == (1.0, 2.0, 4.0)
    assert restored.measurements[0].value == 2.0
    assert restored.to_dict() == report.to_dict()


@pytest.mark.parametrize("field", ["router_aux_loss_weight", "mtp_loss_weight"])
@pytest.mark.parametrize("value", [-1.0, math.nan, math.inf])
def test_report_rejects_invalid_loss_weights(field, value):
    kwargs = {"router_aux_loss_weight": 0.0, "mtp_loss_weight": 0.0}
    kwargs[field] = value
    with pytest.raises(ValueError, match="finite|non-negative"):
        _valid_report(**kwargs)


def test_timing_measurements_require_raw_samples():
    with pytest.raises(ValueError, match="raw samples"):
        _valid_report(raw_samples=())


def test_failed_gate_blocks_measurements():
    with pytest.raises(ValueError, match="correctness"):
        ExperimentReport(
            manifest=_manifest(),
            architecture=_architecture(),
            environment=_environment(),
            router_aux_loss_weight=0.0,
            mtp_loss_weight=0.0,
            correctness_gates=(CorrectnessGate("parity", False, 1e-5, 1.0),),
            measurements=(
                BenchmarkMeasurement(
                    "elapsed", 1.0, "s", {}, (1.0,)
                ),
            ),
            comparison_metadata={"architecture_count": 1},
        )


def test_multi_architecture_report_requires_comparability_reason():
    with pytest.raises(ValueError, match="weight_comparable"):
        _valid_report(comparison_metadata={"architecture_count": 2})
    with pytest.raises(ValueError, match="comparison reason"):
        _valid_report(
            comparison_metadata={
                "architecture_count": 2,
                "weight_comparable": False,
            }
        )

    report = _valid_report(
        comparison_metadata={
            "architecture_count": 2,
            "weight_comparable": False,
            "comparison_reason": "different KV-head counts",
        }
    )
    assert report.comparison_metadata["weight_comparable"] is False


def test_report_types_are_immutable_and_json_is_sorted():
    report = _valid_report()
    with pytest.raises(FrozenInstanceError):
        report.mtp_loss_weight = 1.0
    with pytest.raises(TypeError):
        report.comparison_metadata["x"] = True
    payload = report.to_json(indent=None)
    assert payload.index('"architecture"') < payload.index('"environment"')


def test_non_finite_json_and_measurements_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        BenchmarkMeasurement("loss", math.nan, "scalar", {}, ())
    with pytest.raises(ValueError, match="non-finite JSON"):
        ExperimentReport.from_json('{"value": NaN}')


def test_percentile_and_cpu_peak_memory_are_explicit():
    assert percentile([1.0, 2.0, 4.0], 0.5) == 2.0
    assert percentile([1.0, 2.0, 4.0], 0.9) == pytest.approx(3.6)
    memory = measure_peak_memory("cpu")
    assert memory.bytes > 0
    assert memory.device == "cpu"
    assert memory.platform
    assert memory.source == "resource.getrusage.ru_maxrss"
