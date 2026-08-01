from __future__ import annotations

import json

import pytest
import torch

pytest.importorskip("transformers", minversion="5.2.0")

from qwen3_omni_pretrain.architecture.profiles import ArchitectureProfile
from qwen3_omni_pretrain.profiles.qwen35_omni_inspired.runtime import (
    Qwen35InspiredRuntime,
)
from qwen3_omni_pretrain.profiles.registry import (
    ProfileBuildRequest,
    ProfileBuildResult,
    get_profile_factory,
)
from qwen3_omni_pretrain.runtime.protocols import ModelPrefillInputs
from qwen3_omni_pretrain.runtime.state import StateOwner

from .test_thinker import tiny_profile_config, tiny_public_config


def test_checked_in_tiny_config_builds_complete_runtime():
    result = get_profile_factory(
        ArchitectureProfile.QWEN35_OMNI_INSPIRED
    ).build(
        ProfileBuildRequest(
            profile=ArchitectureProfile.QWEN35_OMNI_INSPIRED,
            config_or_checkpoint=(
                "configs/model/qwen35_omni_inspired_tiny.yaml"
            ),
            device="cpu",
        )
    )
    assert isinstance(result.artifact, Qwen35InspiredRuntime)
    assert result.architecture_summary.total_parameters > 0
    assert result.architecture_summary.capabilities["speech_talker"] is True


def test_factory_returns_complete_paper_inspired_runtime(tmp_path):
    config = tiny_profile_config(tiny_public_config())
    config_path = tmp_path / "qwen35.json"
    config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    request = ProfileBuildRequest(
        profile=ArchitectureProfile.QWEN35_OMNI_INSPIRED,
        config_or_checkpoint=str(config_path),
        device="cpu",
        requested_capabilities=("streaming_audio_encoder",),
    )
    factory = get_profile_factory(request.profile)
    result = factory.build(request)
    assert isinstance(result, ProfileBuildResult)
    assert isinstance(result.artifact, Qwen35InspiredRuntime)
    assert result.manifest.compatibility_level.value == "paper-inspired"
    assert not result.manifest.exact_official_checkpoint_compatible
    assert result.manifest.sources["codec"].name == (
        "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    )
    assert result.architecture_summary.compatibility_level == "paper-inspired"
    assert result.architecture_summary.capabilities["streaming_audio_encoder"] is False
    assert result.architecture_summary.capabilities["paper_inspired_aria"] is True
    assert result.architecture_summary.capabilities["speech_talker"] is True
    assert result.architecture_summary.capabilities["predecessor_codec_proxy"] is True
    assert "streaming_audio_encoder" in (
        result.architecture_summary.unsupported_capabilities
    )
    assert result.architecture_summary.total_parameters > 0
    assert [layer.attention_type for layer in result.architecture_summary.layers] == [
        "gated-delta-net",
        "gated-delta-net",
        "gated-delta-net",
        "full-attention",
    ]


def test_factory_config_contract_is_allocation_free_and_complete(tmp_path):
    config = tiny_profile_config(tiny_public_config())
    config_path = tmp_path / "qwen35.json"
    config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    request = ProfileBuildRequest(
        profile=ArchitectureProfile.QWEN35_OMNI_INSPIRED,
        config_or_checkpoint=str(config_path),
        requested_capabilities=(
            "speech_talker",
            "streaming_audio_encoder",
        ),
    )

    result = get_profile_factory(request.profile).build_config_contract(request)

    assert result.architecture_summary.total_parameters == 0
    assert result.architecture_summary.model_type == "qwen35_omni_inspired"
    assert result.architecture_summary.capabilities["model_runtime"] is True
    assert result.architecture_summary.capabilities["speech_talker"] is True
    assert result.architecture_summary.unsupported_capabilities == (
        "streaming_audio_encoder",
    )
    assert [layer.cache_type for layer in result.architecture_summary.layers] == [
        "recurrent-matrix-state",
        "recurrent-matrix-state",
        "recurrent-matrix-state",
        "kv-cache",
    ]


def test_factory_runtime_delegates_typed_prefill(tmp_path):
    config = tiny_profile_config(tiny_public_config())
    config_path = tmp_path / "qwen35.json"
    config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    runtime = get_profile_factory(
        ArchitectureProfile.QWEN35_OMNI_INSPIRED
    ).build(
        ProfileBuildRequest(
            profile=ArchitectureProfile.QWEN35_OMNI_INSPIRED,
            config_or_checkpoint=str(config_path),
            device="cpu",
        )
    ).artifact
    assert isinstance(runtime, Qwen35InspiredRuntime)
    ids = torch.tensor([[3, 4, 5]])
    output = runtime.prefill(
        inputs=ModelPrefillInputs(
            input_ids=ids,
            inputs_embeds=None,
            key_valid_mask=torch.ones_like(ids, dtype=torch.bool),
            position_batch=None,
        ),
        owner=StateOwner.fresh("runtime"),
        use_cache=True,
    )
    assert output.logits.shape == (1, 3, 128)
    assert output.decoder_state is not None
    assert output.decoder_state.seen_tokens.tolist() == [3]
