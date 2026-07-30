from pathlib import Path

import pytest
import yaml

from qwen3_omni_pretrain.architecture.config_validation import (
    RoutingKind,
    parse_layer_indices,
    validate_legacy_thinker_config,
)
from qwen3_omni_pretrain.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeConfig,
    Qwen3OmniMoeThinkerConfig,
)


MOE_CONFIG_PATHS = tuple(
    sorted(Path("configs/model").glob("qwen3_omni_*_moe.yaml"))
)


def test_boolean_and_layer_indices_cannot_disagree():
    config = Qwen3OmniMoeThinkerConfig(
        num_hidden_layers=4,
        use_moe=False,
        moe_layer_indices="1,3",
    )

    with pytest.raises(ValueError, match="use_moe"):
        validate_legacy_thinker_config(config)


def test_sparse_route_requires_topk_smaller_than_expert_count():
    config = Qwen3OmniMoeThinkerConfig(
        use_moe=True,
        num_experts=2,
        num_experts_per_tok=2,
        routing_kind="sparse",
    )

    with pytest.raises(ValueError, match="top_k < num_experts"):
        validate_legacy_thinker_config(config)


def test_explicit_dense_ensemble_allows_topk_equal_to_experts():
    config = Qwen3OmniMoeThinkerConfig(
        use_moe=True,
        num_experts=2,
        num_experts_per_tok=2,
        routing_kind="dense_ensemble",
    )

    validate_legacy_thinker_config(config)

    assert config.routing_kind is RoutingKind.DENSE_ENSEMBLE


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("1,two", "comma-separated integer list"),
        ("1,1", "duplicate"),
        ("1,4", "out-of-range"),
    ],
)
def test_layer_indices_are_parsed_strictly(value, message):
    with pytest.raises(ValueError, match=message):
        parse_layer_indices(
            value,
            layer_count=4,
            field="moe_layer_indices",
        )


def test_deltanet_indices_require_deltanet_enablement():
    config = Qwen3OmniMoeThinkerConfig(
        num_hidden_layers=4,
        use_deltanet=False,
        deltanet_layer_indices="0,1",
    )

    with pytest.raises(ValueError, match="use_deltanet"):
        validate_legacy_thinker_config(config)


@pytest.mark.parametrize(
    ("thinker_config", "kwargs", "message"),
    [
        ({"num_hidden_layers": 0}, {}, "num_hidden_layers"),
        (
            {
                "hidden_size": 10,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
            },
            {},
            "hidden_size",
        ),
        (
            {"num_attention_heads": 6, "num_key_value_heads": 4},
            {},
            "num_attention_heads",
        ),
        ({}, {"rope_partial_factor": 0}, "rope_partial_factor"),
        ({}, {"rope_partial_factor": 1.1}, "rope_partial_factor"),
    ],
)
def test_invalid_structure_is_rejected_during_config_construction(
    thinker_config,
    kwargs,
    message,
):
    with pytest.raises(ValueError, match=message):
        Qwen3OmniMoeConfig(thinker_config=thinker_config, **kwargs)


@pytest.mark.parametrize("path", MOE_CONFIG_PATHS)
def test_checked_in_moe_configs_declare_constructible_routing(path):
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    thinker = raw["thinker_config"]

    assert thinker["routing_kind"] in {
        RoutingKind.SPARSE.value,
        RoutingKind.DENSE_ENSEMBLE.value,
    }

    config = Qwen3OmniMoeConfig(**raw)

    assert config.thinker_config.routing_kind is RoutingKind(
        thinker["routing_kind"]
    )
