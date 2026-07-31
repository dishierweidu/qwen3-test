from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from qwen3_omni_pretrain.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeConfig,
)
from qwen3_omni_pretrain.models.qwen3_omni_moe.modeling_thinker_text import (
    Qwen3OmniMoeThinkerTextModel,
)
from qwen3_omni_pretrain.runtime import (
    CacheCapabilityError,
    CacheErrorCode,
    CacheSupport,
    cache_support_for_legacy_layers,
    require_incremental_decode_support,
    scan_legacy_decoder_layers,
    validate_generation_operations,
)


class MultiHeadSelfAttention:
    pass


class TensorParallelMultiHeadSelfAttention:
    pass


class GatedDeltaNetAttention:
    pass


class StandardLayer:
    def __init__(self, block_type: str, attention: object) -> None:
        self.block_type = block_type
        self.self_attn = attention
        self.cache_type = "kv-cache"


class TensorParallelThinkerDecoderLayer(StandardLayer):
    pass


def full_attention_layers(count: int = 2):
    return tuple(
        StandardLayer("attn", MultiHeadSelfAttention())
        for _ in range(count)
    )


def tiny_constructed_model(*, deltanet_layer_indices):
    config = Qwen3OmniMoeConfig(
        vocab_size=16,
        thinker_config={
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "max_position_embeddings": 16,
            "use_moe": False,
            "use_flash_attention": False,
            "use_deltanet": True,
            "deltanet_layer_indices": deltanet_layer_indices,
            "deltanet_num_heads": 2,
        },
    )
    return Qwen3OmniMoeThinkerTextModel(config)


def test_error_codes_and_unimplemented_support_are_stable_and_frozen():
    assert {code.value for code in CacheErrorCode} == {
        "DELTA_NET_UNSUPPORTED",
        "BEAM_UNSUPPORTED",
        "TRUNCATE_UNSUPPORTED",
        "SPECULATIVE_UNSUPPORTED",
        "PROFILE_RUNTIME_UNSUPPORTED",
        "CONTEXT_OVERFLOW",
        "STATE_OWNER_MISMATCH",
    }
    support = CacheSupport.unimplemented()
    assert support.as_dict() == {
        "incremental_decode_state": False,
        "streaming_generation": False,
        "beam_search": False,
        "state_truncate": False,
        "speculative_decode": False,
    }
    with pytest.raises(FrozenInstanceError):
        support.incremental_decode_state = True
    with pytest.raises(TypeError, match="booleans"):
        CacheSupport(
            incremental_decode_state=1,
            streaming_generation=False,
            beam_search=False,
            state_truncate=False,
            speculative_decode=False,
        )


def test_capability_error_exposes_typed_code_reason_and_plain_mapping():
    error = CacheCapabilityError(
        CacheErrorCode.DELTA_NET_UNSUPPORTED,
        "recurrent state is not implemented",
    )
    assert error.code is CacheErrorCode.DELTA_NET_UNSUPPORTED
    assert error.reason == "recurrent state is not implemented"
    assert error.to_dict() == {
        "code": "DELTA_NET_UNSUPPORTED",
        "reason": "recurrent state is not implemented",
    }
    assert "DELTA_NET_UNSUPPORTED" in str(error)

    with pytest.raises(TypeError, match="CacheErrorCode"):
        CacheCapabilityError("DELTA_NET_UNSUPPORTED", "reason")
    with pytest.raises(ValueError, match="non-empty"):
        CacheCapabilityError(CacheErrorCode.BEAM_UNSUPPORTED, "  ")


@pytest.mark.parametrize("num_beams", [True, 0, -1, 2, 1.0, "1"])
def test_invalid_beam_requests_fail_before_downstream_calls(num_beams):
    calls = []

    with pytest.raises(CacheCapabilityError) as captured:
        validate_generation_operations(
            num_beams=num_beams,
            state_truncate=False,
            speculative_decode=False,
        )
        calls.append("tokenizer")
        calls.append("model")
        calls.append("media")

    assert captured.value.code is CacheErrorCode.BEAM_UNSUPPORTED
    assert calls == []


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("state_truncate", True, CacheErrorCode.TRUNCATE_UNSUPPORTED),
        ("state_truncate", 1, CacheErrorCode.TRUNCATE_UNSUPPORTED),
        (
            "speculative_decode",
            True,
            CacheErrorCode.SPECULATIVE_UNSUPPORTED,
        ),
        (
            "speculative_decode",
            1,
            CacheErrorCode.SPECULATIVE_UNSUPPORTED,
        ),
    ],
)
def test_truncate_and_speculation_fail_with_stable_codes(field, value, code):
    kwargs = {
        "num_beams": 1,
        "state_truncate": False,
        "speculative_decode": False,
    }
    kwargs[field] = value
    with pytest.raises(CacheCapabilityError) as captured:
        validate_generation_operations(**kwargs)
    assert captured.value.code is code


def test_valid_generation_operation_scope_is_a_noop():
    assert (
        validate_generation_operations(
            num_beams=1,
            state_truncate=False,
            speculative_decode=False,
        )
        is None
    )


def test_layer_scan_uses_actual_blocks_and_ignores_summary_cache_text():
    layers = (
        StandardLayer("attn", MultiHeadSelfAttention()),
        StandardLayer("deltanet", GatedDeltaNetAttention()),
        StandardLayer("attn", MultiHeadSelfAttention()),
    )
    scan = scan_legacy_decoder_layers(layers)
    assert scan.full_attention_layer_indices == (0, 2)
    assert scan.deltanet_layer_indices == (1,)
    assert scan.tensor_parallel is False
    assert scan.all_full_attention is False


def test_layer_scan_detects_implicit_three_to_one_constructed_topology():
    layers = tuple(
        StandardLayer(
            "deltanet" if index % 4 in (0, 1, 2) else "attn",
            (
                GatedDeltaNetAttention()
                if index % 4 in (0, 1, 2)
                else MultiHeadSelfAttention()
            ),
        )
        for index in range(8)
    )
    scan = scan_legacy_decoder_layers(layers)
    assert scan.deltanet_layer_indices == (0, 1, 2, 4, 5, 6)
    assert scan.full_attention_layer_indices == (3, 7)


@pytest.mark.parametrize(
    ("indices", "expected_delta"),
    [
        ("1,3", (1, 3)),
        (None, (0, 1, 2)),
    ],
)
def test_layer_scan_reads_real_explicit_and_implicit_constructed_models(
    indices,
    expected_delta,
):
    model = tiny_constructed_model(
        deltanet_layer_indices=indices,
    )
    scan = scan_legacy_decoder_layers(model.layers)
    assert scan.deltanet_layer_indices == expected_delta
    assert scan.full_attention_layer_indices == tuple(
        index for index in range(4) if index not in expected_delta
    )


def test_tensor_parallel_scan_requires_validated_local_shard_gate():
    layers = tuple(
        TensorParallelThinkerDecoderLayer(
            "attn",
            TensorParallelMultiHeadSelfAttention(),
        )
        for _ in range(2)
    )
    scan = scan_legacy_decoder_layers(layers)
    assert scan.tensor_parallel is True
    assert scan.all_full_attention is True

    unavailable = cache_support_for_legacy_layers(
        layers,
        protocol_implemented=True,
        tp_local_shard_validated=False,
    )
    assert unavailable.incremental_decode_state is False
    available = cache_support_for_legacy_layers(
        layers,
        protocol_implemented=True,
        tp_local_shard_validated=True,
    )
    assert available.incremental_decode_state is True


def test_support_stays_false_until_protocol_is_implemented():
    layers = full_attention_layers()
    assert (
        cache_support_for_legacy_layers(layers).incremental_decode_state
        is False
    )
    assert cache_support_for_legacy_layers(
        layers,
        protocol_implemented=True,
    ).incremental_decode_state is True


def test_explicit_and_implicit_deltanet_fail_before_compute():
    topologies = (
        (
            StandardLayer("attn", MultiHeadSelfAttention()),
            StandardLayer("deltanet", GatedDeltaNetAttention()),
        ),
        tuple(
            StandardLayer(
                "deltanet" if index % 4 in (0, 1, 2) else "attn",
                (
                    GatedDeltaNetAttention()
                    if index % 4 in (0, 1, 2)
                    else MultiHeadSelfAttention()
                ),
            )
            for index in range(4)
        ),
    )
    for layers in topologies:
        calls = {"embedding": 0, "attention": 0, "deltanet": 0}
        with pytest.raises(CacheCapabilityError) as captured:
            require_incremental_decode_support(
                layers,
                protocol_implemented=True,
            )
            for name in calls:
                calls[name] += 1
        assert captured.value.code is CacheErrorCode.DELTA_NET_UNSUPPORTED
        assert calls == {"embedding": 0, "attention": 0, "deltanet": 0}


def test_unimplemented_protocol_fails_before_compute_with_profile_code():
    with pytest.raises(CacheCapabilityError) as captured:
        require_incremental_decode_support(
            full_attention_layers(),
            protocol_implemented=False,
        )
    assert captured.value.code is CacheErrorCode.PROFILE_RUNTIME_UNSUPPORTED


@pytest.mark.parametrize(
    "layers",
    [
        (),
        (object(),),
        (StandardLayer("attn", GatedDeltaNetAttention()),),
        (StandardLayer("deltanet", MultiHeadSelfAttention()),),
    ],
)
def test_layer_scan_rejects_empty_malformed_or_inconsistent_layers(layers):
    with pytest.raises((TypeError, ValueError)):
        scan_legacy_decoder_layers(layers)
