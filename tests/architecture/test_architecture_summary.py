from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from qwen3_omni_pretrain.architecture.manifest import ProfileManifest
from qwen3_omni_pretrain.architecture.summary import (
    ArchitectureSummary,
    LayerArchitecture,
    summarize_model,
)
from qwen3_omni_pretrain.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeConfig,
)
from qwen3_omni_pretrain.models.qwen3_omni_moe.modeling_thinker_text import (
    Qwen3OmniMoeThinkerTextModel,
)
from qwen3_omni_pretrain.profiles.legacy_prototype.config_adapter import (
    adapt_legacy_config_dict,
)
from scripts.inspect_architecture import load_and_adapt_legacy_yaml


LEGACY_CONFIG_SNAPSHOTS = {
    "qwen3_omni_1_3b_moe.yaml": (24, (4, 8, 12, 16, 20), "sparse"),
    "qwen3_omni_7b_moe.yaml": (
        28,
        (4, 8, 12, 16, 20, 24),
        "dense_ensemble",
    ),
    "qwen3_omni_14b_moe.yaml": (
        40,
        (4, 8, 12, 16, 20, 24, 28, 32, 36),
        "sparse",
    ),
    "qwen3_omni_30b_moe.yaml": (
        48,
        (4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44),
        "sparse",
    ),
    "qwen3_omni_70b_moe.yaml": (
        64,
        (4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60),
        "sparse",
    ),
    "qwen3_omni_120b_moe.yaml": (
        80,
        (4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76),
        "sparse",
    ),
}


def _tiny_config() -> Qwen3OmniMoeConfig:
    return Qwen3OmniMoeConfig(
        vocab_size=32,
        thinker_config={
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "max_position_embeddings": 16,
            "use_moe": True,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "routing_kind": "sparse",
            "moe_layer_indices": "1",
        },
    )


def test_legacy_summary_reports_real_layer_and_routing_types():
    config = _tiny_config()
    model = Qwen3OmniMoeThinkerTextModel(config)

    summary = summarize_model(model, config.profile_manifest)

    assert [layer.ffn_type for layer in summary.layers] == [
        "dense",
        "shared-dense-plus-routed-moe",
    ]
    assert summary.total_parameters > summary.active_parameters_per_token
    assert summary.profile == "legacy_prototype"


def test_summary_round_trips_through_strict_plain_mapping():
    config = _tiny_config()
    summary = summarize_model(
        Qwen3OmniMoeThinkerTextModel(config),
        config.profile_manifest,
    )

    serialized = summary.to_dict()
    restored = ArchitectureSummary.from_dict(serialized)

    assert restored == summary
    assert restored.to_dict() == serialized
    assert "profile_manifest" not in serialized


@pytest.mark.parametrize(
    ("mutation", "exception", "message"),
    [
        (
            lambda raw: raw.update({"surprise": True}),
            ValueError,
            "invalid architecture summary keys",
        ),
        (
            lambda raw: raw.update({"total_parameters": True}),
            TypeError,
            "total_parameters must be an integer",
        ),
        (
            lambda raw: raw.update({"routed_parameters": -1}),
            ValueError,
            "routed_parameters must be non-negative",
        ),
        (
            lambda raw: raw["layers"][1].update({"index": 3}),
            ValueError,
            "layer indices must be contiguous",
        ),
        (
            lambda raw: raw.update({"capabilities": {"streaming": 1}}),
            TypeError,
            "capabilities must map strings to booleans",
        ),
        (
            lambda raw: raw.update({"unsupported_capabilities": [1]}),
            TypeError,
            "unsupported_capabilities must contain strings",
        ),
    ],
)
def test_summary_mapping_rejects_unknown_keys_and_invalid_types(
    mutation,
    exception,
    message,
):
    config = _tiny_config()
    raw = summarize_model(
        Qwen3OmniMoeThinkerTextModel(config),
        config.profile_manifest,
    ).to_dict()
    mutation(raw)

    with pytest.raises(exception, match=message):
        ArchitectureSummary.from_dict(raw)


def test_summary_dataclasses_reject_mutable_sequence_fields():
    layer = LayerArchitecture(
        index=0,
        attention_type="full-attention",
        cache_type="kv-cache",
        ffn_type="dense",
        routed_experts=0,
        experts_per_token=0,
    )

    with pytest.raises(TypeError, match="layers must be a tuple"):
        ArchitectureSummary(
            profile="legacy_prototype",
            compatibility_level="legacy-prototype",
            model_type="qwen3_omni_prototype",
            tokenizer_vocab_size=32,
            embedding_vocab_size=32,
            total_parameters=1,
            active_parameters_per_token=1,
            routed_parameters=0,
            shared_parameters=0,
            dense_parameters=1,
            capabilities={},
            layers=[layer],
            unsupported_capabilities=(),
        )


@pytest.mark.parametrize(
    ("name", "snapshot"),
    LEGACY_CONFIG_SNAPSHOTS.items(),
)
def test_checked_in_legacy_architecture_snapshot(name, snapshot):
    layer_count, moe_indices, routing_kind = snapshot
    path = Path("configs/model") / name
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = Qwen3OmniMoeConfig(**adapt_legacy_config_dict(raw))
    manifest = ProfileManifest.from_dict(config.profile_manifest.to_dict())

    with torch.device("meta"):
        model = Qwen3OmniMoeThinkerTextModel(config)
    summary = summarize_model(model, manifest)

    assert len(summary.layers) == layer_count
    assert tuple(
        layer.index
        for layer in summary.layers
        if layer.routed_experts
    ) == moe_indices
    expected_ffn = (
        "shared-dense-plus-dense-ensemble"
        if routing_kind == "dense_ensemble"
        else "shared-dense-plus-routed-moe"
    )
    assert {
        summary.layers[index].ffn_type for index in moe_indices
    } == {expected_ffn}
    assert summary.profile == manifest.architecture_profile.value
    assert summary.compatibility_level == manifest.compatibility_level.value
    assert summary.total_parameters not in {
        1_300_000_000,
        7_000_000_000,
        14_000_000_000,
        30_000_000_000,
        70_000_000_000,
        120_000_000_000,
    }


def test_30b_standard_and_tp_models_report_identical_layers(monkeypatch):
    from qwen3_omni_pretrain.models.qwen3_omni_moe import (
        modeling_thinker_text_tp as tp_module,
    )

    path = Path("configs/model/qwen3_omni_30b_moe.yaml")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = Qwen3OmniMoeConfig(**adapt_legacy_config_dict(raw))
    monkeypatch.setattr(tp_module, "is_model_parallel_initialized", lambda: True)

    with torch.device("meta"):
        standard = Qwen3OmniMoeThinkerTextModel(config)
        tensor_parallel = tp_module.Qwen3OmniMoeThinkerTextModelTP(config)

    standard_layers = summarize_model(
        standard, config.profile_manifest
    ).layers
    tp_layers = summarize_model(
        tensor_parallel, config.profile_manifest
    ).layers
    assert tp_layers == standard_layers


def test_architecture_loader_rejects_comment_only_model_config():
    with pytest.raises(ValueError, match="model configuration is empty"):
        load_and_adapt_legacy_yaml(Path("configs/model/thinker_text.yaml"))


def test_summary_validation_does_not_mutate_input():
    config = _tiny_config()
    raw = summarize_model(
        Qwen3OmniMoeThinkerTextModel(config),
        config.profile_manifest,
    ).to_dict()
    original = deepcopy(raw)

    ArchitectureSummary.from_dict(raw)

    assert raw == original
