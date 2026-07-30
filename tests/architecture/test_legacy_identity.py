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
    "path",
    sorted(Path("configs/model").glob("qwen3_omni_*_moe.yaml")),
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
