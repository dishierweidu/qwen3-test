from __future__ import annotations

from unittest.mock import patch

import pytest

from qwen3_omni_pretrain.architecture.profiles import (
    ArchitectureProfile,
    CompatibilityLevel,
)
from qwen3_omni_pretrain.models.hybrid_swa_moe import HybridSwaMoeForCausalLM
from qwen3_omni_pretrain.parallel.expert_parallel import ParallelTopology
from qwen3_omni_pretrain.profiles.mimo_v25_experimental import factory as factory_module
from qwen3_omni_pretrain.profiles.mimo_v25_experimental.factory import (
    MimoV25ExperimentalFactory,
)
from qwen3_omni_pretrain.profiles.registry import (
    ProfileBuildRequest,
    get_profile_factory,
)


CONFIG_PATH = "configs/model/hybrid_swa_moe_tiny.yaml"


def _request(**updates) -> ProfileBuildRequest:
    values = {
        "profile": ArchitectureProfile.MIMO_V25_EXPERIMENTAL,
        "config_or_checkpoint": CONFIG_PATH,
        "device": "cpu",
        "dtype": "float32",
    }
    values.update(updates)
    return ProfileBuildRequest(**values)


def test_registry_builds_complete_experimental_model_and_summary():
    factory = get_profile_factory(ArchitectureProfile.MIMO_V25_EXPERIMENTAL)
    validated = factory.validate(_request())
    result = factory.build(
        _request(
            requested_capabilities=(
                "sliding_window_attention",
                "mtp_enabled",
            )
        )
    )

    assert isinstance(result.artifact, HybridSwaMoeForCausalLM)
    assert result.manifest == validated
    assert result.manifest.compatibility_level is (
        CompatibilityLevel.MIMO_STYLE_EXPERIMENT
    )
    assert result.manifest.exact_official_checkpoint_compatible is False
    summary = result.architecture_summary
    assert summary.model_type == "hybrid_swa_moe_experimental"
    assert [layer.attention_type for layer in summary.layers] == [
        "sliding-window-attention",
        "sliding-window-attention",
        "sliding-window-attention",
        "sliding-window-attention",
        "sliding-window-attention",
        "full-attention",
    ]
    assert summary.layers[0].ffn_type == "dense"
    assert {layer.ffn_type for layer in summary.layers[1:]} == {
        "routed-moe-only"
    }
    assert summary.capabilities["mtp_enabled"] is True
    assert summary.routed_parameters > 0
    assert summary.active_parameters_per_token < summary.total_parameters
    assert summary.unsupported_capabilities == ()


def test_factory_reports_requested_unsupported_capability():
    result = get_profile_factory(
        ArchitectureProfile.MIMO_V25_EXPERIMENTAL
    ).build(_request(requested_capabilities=("official_checkpoint_loading",)))
    assert result.architecture_summary.unsupported_capabilities == (
        "official_checkpoint_loading",
    )


def test_invalid_ep_topology_fails_before_config_io_or_model_allocation():
    invalid = ParallelTopology(
        data_parallel_size=1,
        pipeline_parallel_size=1,
        expert_parallel_size=2,
        tensor_parallel_size=2,
    )
    factory = MimoV25ExperimentalFactory(topology=invalid)
    request = _request(config_or_checkpoint="does-not-exist.yaml")
    with patch.object(
        factory_module,
        "HybridSwaMoeForCausalLM",
    ) as constructor:
        with pytest.raises(ValueError, match="EP.*TP"):
            factory.validate(request)
    constructor.assert_not_called()


def test_ep_two_requires_explicit_initialized_context():
    topology = ParallelTopology(
        data_parallel_size=1,
        pipeline_parallel_size=1,
        expert_parallel_size=2,
        tensor_parallel_size=1,
    )
    factory = MimoV25ExperimentalFactory(topology=topology)
    with pytest.raises(ValueError, match="requires.*context"):
        factory.validate(_request())


def test_factory_rejects_official_tokenizer_override():
    with pytest.raises(ValueError, match="official tokenizer"):
        get_profile_factory(
            ArchitectureProfile.MIMO_V25_EXPERIMENTAL
        ).validate(_request(tokenizer="XiaomiMiMo/MiMo-V2.5-Pro"))
