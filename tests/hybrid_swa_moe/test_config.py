from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

import pytest
import yaml

from qwen3_omni_pretrain.architecture.manifest import ProfileManifest
from qwen3_omni_pretrain.models.hybrid_swa_moe import HybridSwaMoeConfig
from qwen3_omni_pretrain.profiles.mimo_v25_experimental.manifest import (
    MIMO_V25_PRO_REVISION,
    MIMO_V25_REVISION,
    mimo_experiment_manifest,
)


CONFIG_PATH = Path("configs/model/hybrid_swa_moe_tiny.yaml")


def tiny_hybrid_config_dict() -> dict[str, object]:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return raw


def load_tiny_hybrid_config() -> HybridSwaMoeConfig:
    return HybridSwaMoeConfig(**tiny_hybrid_config_dict())


def test_tiny_config_has_explicit_layer_types_and_final_mtp():
    config = load_tiny_hybrid_config()
    assert config.model_type == "hybrid_swa_moe_experimental"
    assert config.mtp_num_predictors == 1
    assert config.mtp_loss_weight == 0.1
    assert config.attention_layer_types == [
        "swa", "swa", "swa", "swa", "swa", "full"
    ]
    assert config.ffn_layer_types == [
        "dense",
        "routed_moe", "routed_moe", "routed_moe",
        "routed_moe", "routed_moe",
    ]


def test_config_round_trips_with_manifest():
    config = load_tiny_hybrid_config()
    serialized = config.to_dict()
    restored = HybridSwaMoeConfig(**serialized)
    assert restored.to_dict()["attention_layer_types"] == serialized[
        "attention_layer_types"
    ]
    assert restored.profile_manifest == config.profile_manifest


def test_config_rejects_non_sparse_router():
    raw = tiny_hybrid_config_dict()
    raw["num_experts_per_token"] = raw["num_experts"]
    with pytest.raises(ValueError, match="top_k < num_experts"):
        HybridSwaMoeConfig(**raw)


@pytest.mark.parametrize("value_scale", [0.0, -1.0, math.nan, math.inf])
def test_config_rejects_invalid_value_scale(value_scale):
    raw = tiny_hybrid_config_dict()
    raw["value_scale"] = value_scale
    with pytest.raises(ValueError, match="value_scale"):
        HybridSwaMoeConfig(**raw)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("attention_layer_types", ["swa"] * 5, "length"),
        (
            "attention_layer_types",
            ["deltanet", "swa", "swa", "swa", "swa", "full"],
            "unsupported",
        ),
        (
            "ffn_layer_types",
            ["routed_moe"] * 6,
            "layer 0",
        ),
        ("rotary_dim", 7, "rotary_dim"),
        ("swa_window_size", 0, "swa_window_size"),
        ("mtp_num_predictors", 4, "mtp_num_predictors"),
    ],
)
def test_config_rejects_invalid_structural_fields(field, value, message):
    raw = tiny_hybrid_config_dict()
    raw[field] = value
    with pytest.raises(ValueError, match=message):
        HybridSwaMoeConfig(**raw)


def test_config_rejects_deltanet_fields_and_missing_mtp_head():
    raw = tiny_hybrid_config_dict()
    raw["use_deltanet"] = False
    with pytest.raises(ValueError, match="DeltaNet"):
        HybridSwaMoeConfig(**raw)

    raw = tiny_hybrid_config_dict()
    raw["mtp_num_predictors"] = 0
    with pytest.raises(ValueError, match="mtp_loss_weight"):
        HybridSwaMoeConfig(**raw)


@pytest.mark.parametrize(
    "mutation",
    [
        {"model_type": "mimo_v2"},
        {"architecture_profile": "mimo_v2"},
        {"_name_or_path": "XiaomiMiMo/MiMo-V2.5-Pro"},
        {"architectures": ["MiMoV2ForCausalLM"]},
    ],
)
def test_config_rejects_official_mimo_identity(mutation):
    raw = tiny_hybrid_config_dict()
    raw.update(mutation)
    with pytest.raises(ValueError, match="MiMo|mimo_v25_experimental"):
        HybridSwaMoeConfig(**raw)


def test_manifest_pins_both_sources_and_never_claims_checkpoint_compatibility():
    manifest = mimo_experiment_manifest()
    restored = ProfileManifest.from_dict(manifest.to_dict())

    assert restored.compatibility_level.value == "MiMo-style-experiment"
    assert restored.exact_official_checkpoint_compatible is False
    revisions = {source.revision for source in restored.sources.values()}
    assert revisions == {MIMO_V25_REVISION, MIMO_V25_PRO_REVISION}
    assert any("not an official MiMo" in item for item in restored.assumptions)


def test_supplied_manifest_must_match_canonical_contract():
    raw = tiny_hybrid_config_dict()
    manifest = deepcopy(mimo_experiment_manifest().to_dict())
    manifest["validated_context_length"] = 64
    raw["profile_manifest"] = manifest
    with pytest.raises(ValueError, match="contradicts"):
        HybridSwaMoeConfig(**raw)
