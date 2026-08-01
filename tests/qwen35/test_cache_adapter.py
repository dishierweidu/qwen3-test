from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers", minversion="5.2.0")

from transformers import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeDynamicCache,
)

from qwen3_omni_pretrain.profiles.qwen35_omni_inspired.cache_adapter import (
    Qwen35CacheAdapter,
    Qwen35GDNState,
    expected_gdn_conv_shape,
    expected_gdn_recurrent_shape,
)


def tiny_public_qwen35_config() -> Qwen3_5MoeTextConfig:
    return Qwen3_5MoeTextConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
    )


def test_gdn_state_round_trip_preserves_conv_and_matrix_tensors():
    config = tiny_public_qwen35_config()
    native = Qwen3_5MoeDynamicCache(config)
    native.conv_states[0] = torch.randn(
        expected_gdn_conv_shape(config, batch_size=1)
    )
    native.recurrent_states[0] = torch.randn(
        expected_gdn_recurrent_shape(config, batch_size=1)
    )
    common = Qwen35CacheAdapter(config).from_native(
        native,
        request_id="r1",
        seen_tokens=torch.tensor([5], dtype=torch.long),
        key_valid_mask=torch.ones(1, 5, dtype=torch.bool),
        position_ids=torch.arange(5).view(1, 5),
    )
    assert isinstance(common.gdn_state, Qwen35GDNState)
    restored = Qwen35CacheAdapter(config).to_native(common, request_id="r1")
    torch.testing.assert_close(restored.conv_states[0], native.conv_states[0])
    torch.testing.assert_close(
        restored.recurrent_states[0], native.recurrent_states[0]
    )
    assert restored.conv_states[1] is None


def test_full_attention_state_round_trip_clones_mutable_native_tensors():
    config = tiny_public_qwen35_config()
    native = Qwen3_5MoeDynamicCache(config)
    native.key_cache[3] = torch.randn(1, 2, 3, 8)
    native.value_cache[3] = torch.randn(1, 2, 3, 8)
    original_key = native.key_cache[3].clone()
    common = Qwen35CacheAdapter(config).from_native(
        native,
        request_id="full",
        seen_tokens=torch.tensor([3]),
        key_valid_mask=torch.ones(1, 3, dtype=torch.bool),
        position_ids=torch.arange(3).view(1, 3),
    )
    native.key_cache[3].zero_()
    restored = Qwen35CacheAdapter(config).to_native(
        common,
        request_id="full",
    )
    torch.testing.assert_close(restored.key_cache[3], original_key)
    assert restored.key_cache[3].data_ptr() != common.full_attention_kv[3].key.data_ptr()


def test_cached_heterogeneous_or_padded_batch_fails_before_native_call():
    config = tiny_public_qwen35_config()
    native = Qwen3_5MoeDynamicCache(config)
    native.conv_states[0] = torch.randn(expected_gdn_conv_shape(config, 2))
    native.recurrent_states[0] = torch.randn(
        expected_gdn_recurrent_shape(config, 2)
    )
    state = Qwen35CacheAdapter(config).from_native(
        native,
        request_id="r1",
        seen_tokens=torch.tensor([4, 2]),
        key_valid_mask=torch.tensor(
            [[True, True, True, True], [True, True, False, False]]
        ),
        position_ids=torch.tensor([[0, 1, 2, 3], [0, 1, 0, 0]]),
    )
    with pytest.raises(ValueError, match="equal-length.*without padding"):
        Qwen35CacheAdapter(config).to_native(state, request_id="r1")


def test_wrong_rank_recurrent_state_is_rejected():
    config = tiny_public_qwen35_config()
    native = Qwen3_5MoeDynamicCache(config)
    native.conv_states[0] = torch.zeros(expected_gdn_conv_shape(config, 1))
    native.recurrent_states[0] = torch.zeros(1, 4, 4)
    with pytest.raises(ValueError, match="rank-4"):
        Qwen35CacheAdapter(config).from_native(
            native,
            request_id="r1",
            seen_tokens=torch.tensor([1]),
            key_valid_mask=torch.ones(1, 1, dtype=torch.bool),
            position_ids=torch.zeros(1, 1, dtype=torch.long),
        )


def test_request_ownership_is_checked():
    config = tiny_public_qwen35_config()
    state = Qwen35CacheAdapter(config).from_native(
        Qwen3_5MoeDynamicCache(config),
        request_id="one",
        seen_tokens=torch.tensor([1]),
        key_valid_mask=torch.ones(1, 1, dtype=torch.bool),
        position_ids=torch.zeros(1, 1, dtype=torch.long),
    )
    with pytest.raises(ValueError, match="different request_id"):
        Qwen35CacheAdapter(config).to_native(state, request_id="two")
