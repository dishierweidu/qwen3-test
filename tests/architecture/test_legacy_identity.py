from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from qwen3_omni_pretrain.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeConfig,
)
from qwen3_omni_pretrain.profiles.legacy_prototype.config_adapter import (
    LEGACY_MODEL_TYPE,
    adapt_legacy_config_dict,
)

EXPECTED_LEGACY_CONFIG_PATHS = tuple(
    Path("configs/model") / name
    for name in (
        "qwen3_omni_1_3b_moe.yaml",
        "qwen3_omni_7b_moe.yaml",
        "qwen3_omni_14b_moe.yaml",
        "qwen3_omni_30b_moe.yaml",
        "qwen3_omni_70b_moe.yaml",
        "qwen3_omni_120b_moe.yaml",
    )
)


def test_old_colliding_model_type_is_migrated_with_warning():
    raw = {"model_type": "qwen3_omni_moe", "vocab_size": 32}

    with pytest.warns(DeprecationWarning, match="qwen3_omni_moe"):
        migrated = adapt_legacy_config_dict(raw)

    assert migrated["model_type"] == LEGACY_MODEL_TYPE
    assert migrated["architecture_profile"] == "legacy_prototype"
    assert raw["model_type"] == "qwen3_omni_moe"


def test_new_legacy_config_round_trip_never_saves_official_model_type():
    config = Qwen3OmniMoeConfig(vocab_size=32)

    saved = config.to_dict()

    assert config.model_type == "qwen3_omni_prototype"
    assert saved["architecture_profile"] == "legacy_prototype"
    assert saved["profile_manifest"]["compatibility_level"] == (
        "legacy-prototype"
    )


def test_thinker_structural_values_override_top_level_mirrors():
    thinker_values = {
        "hidden_size": 16,
        "intermediate_size": 24,
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 128,
        "use_moe": True,
        "num_experts": 4,
        "num_experts_per_tok": 2,
    }
    config = Qwen3OmniMoeConfig(
        hidden_size=1,
        intermediate_size=2,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        max_position_embeddings=8,
        use_moe=False,
        num_experts=1,
        num_experts_per_tok=1,
        thinker_config=thinker_values,
    )

    saved = config.to_dict()

    for field, expected in thinker_values.items():
        assert getattr(config, field) == expected
        assert saved[field] == expected
        assert saved["thinker_config"][field] == expected


def test_serialization_refreshes_structural_mirrors_from_thinker():
    config = Qwen3OmniMoeConfig()
    config.thinker_config.hidden_size = 64

    saved = config.to_dict()

    assert config.hidden_size == 64
    assert saved["hidden_size"] == 64
    assert saved["thinker_config"]["hidden_size"] == 64


@pytest.mark.parametrize(
    ("num_experts", "num_experts_per_tok", "indices", "routing_kind"),
    [
        (4, 2, "4,8,12,16,20,24,28,32,36", "sparse"),
        (16, 2, "4,8,12,16,20,24,28,32,36,40,44", "sparse"),
        (2, 2, "4,8,12,16,20,24", "dense_ensemble"),
    ],
)
def test_old_moe_layer_selection_preserves_standard_model_graph(
    num_experts,
    num_experts_per_tok,
    indices,
    routing_kind,
):
    raw = {
        "model_type": "qwen3_omni_moe",
        "thinker_config": {
            "use_moe": num_experts == 2,
            "num_experts": num_experts,
            "num_experts_per_tok": num_experts_per_tok,
            "moe_layer_indices": indices,
        },
    }
    original = deepcopy(raw)

    with pytest.warns(DeprecationWarning):
        migrated = adapt_legacy_config_dict(raw)

    assert migrated["thinker_config"]["use_moe"] is True
    assert migrated["thinker_config"]["routing_kind"] == routing_kind
    assert migrated["thinker_config"]["moe_layer_indices"] == indices
    assert raw == original


@pytest.mark.parametrize(
    ("raw", "exception", "message"),
    [
        (
            {"model_type": "qwen3_omni"},
            ValueError,
            "not a legacy prototype config",
        ),
        (
            {
                "model_type": LEGACY_MODEL_TYPE,
                "architecture_profile": "qwen3_omni_reference",
            },
            ValueError,
            "not a legacy prototype profile",
        ),
        (
            {
                "model_type": LEGACY_MODEL_TYPE,
                "thinker_config": {"use_moe": 1},
            },
            TypeError,
            "use_moe must be boolean",
        ),
    ],
)
def test_adapter_rejects_nonlegacy_identities_and_nonboolean_moe(
    raw,
    exception,
    message,
):
    with pytest.raises(exception, match=message):
        adapt_legacy_config_dict(raw)


@pytest.mark.parametrize(
    ("num_experts", "num_experts_per_tok", "message"),
    [
        (0, 1, "num_experts must be positive"),
        (-1, 1, "num_experts must be positive"),
        (2, 0, "num_experts_per_tok must be between"),
        (2, -1, "num_experts_per_tok must be between"),
        (2, 3, "num_experts_per_tok must be between"),
    ],
)
def test_adapter_rejects_invalid_effective_moe_expert_counts(
    num_experts,
    num_experts_per_tok,
    message,
):
    raw = {
        "model_type": LEGACY_MODEL_TYPE,
        "thinker_config": {
            "use_moe": True,
            "num_experts": num_experts,
            "num_experts_per_tok": num_experts_per_tok,
        },
    }

    with pytest.raises(ValueError, match=message):
        adapt_legacy_config_dict(raw)


def test_checked_in_legacy_config_set_is_exact():
    actual = set(Path("configs/model").glob("qwen3_omni_*_moe.yaml"))

    assert actual == set(EXPECTED_LEGACY_CONFIG_PATHS)


@pytest.mark.parametrize(
    "path",
    EXPECTED_LEGACY_CONFIG_PATHS,
)
def test_checked_in_legacy_configs_use_non_colliding_identity(path):
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    config = Qwen3OmniMoeConfig(**raw)
    saved = config.to_dict()

    assert raw["model_type"] == "qwen3_omni_prototype"
    assert raw["architecture_profile"] == "legacy_prototype"
    assert saved["model_type"] == "qwen3_omni_prototype"


def test_serialized_manifest_cannot_claim_another_profile():
    with pytest.raises(ValueError, match="profile_manifest"):
        Qwen3OmniMoeConfig(
            profile_manifest={
                "architecture_profile": "qwen3_omni_reference",
                "compatibility_level": "structure-aligned",
                "sources": {},
                "assumptions": [],
                "exact_official_checkpoint_compatible": False,
                "validated_context_length": 0,
            }
        )
