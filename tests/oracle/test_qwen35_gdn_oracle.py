from __future__ import annotations

import torch
import pytest

pytest.importorskip("transformers", minversion="5.2.0")

from transformers import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeDynamicCache,
    Qwen3_5MoeGatedDeltaNet,
)


def _config() -> Qwen3_5MoeTextConfig:
    return Qwen3_5MoeTextConfig(
        vocab_size=32,
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
    )


def test_public_gdn_contains_all_disclosed_projections_and_state_shapes():
    config = _config()
    module = Qwen3_5MoeGatedDeltaNet(config, layer_idx=2)
    for name in (
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_b",
        "in_proj_a",
        "conv1d",
        "dt_bias",
        "A_log",
    ):
        assert hasattr(module, name)
    cache = Qwen3_5MoeDynamicCache(config)
    module(
        torch.randn(2, 5, 32),
        cache_params=cache,
        cache_position=torch.arange(5),
        attention_mask=torch.ones(2, 5),
    )
    assert cache.conv_states[2].shape == (2, 48, 4)
    assert cache.recurrent_states[2].shape == (2, 2, 8, 8)
    assert cache.recurrent_states[2].ndim == 4


def test_public_gdn_full_sequence_matches_token_by_token_without_padding():
    torch.manual_seed(7)
    config = _config()
    full = Qwen3_5MoeGatedDeltaNet(config, layer_idx=2).eval()
    step = Qwen3_5MoeGatedDeltaNet(config, layer_idx=2).eval()
    step.load_state_dict(full.state_dict())
    hidden = torch.randn(2, 7, 32)
    expected = full(hidden, attention_mask=torch.ones(2, 7))
    cache = Qwen3_5MoeDynamicCache(config)
    pieces = []
    for index in range(hidden.shape[1]):
        pieces.append(
            step(
                hidden[:, index : index + 1],
                cache_params=cache,
                cache_position=torch.tensor([index]),
                attention_mask=torch.ones(2, 1),
            )
        )
    actual = torch.cat(pieces, dim=1)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    assert cache.recurrent_states[2].shape == (2, 2, 8, 8)


def test_public_gdn_forward_and_parameter_gradients_are_finite():
    torch.manual_seed(11)
    module = Qwen3_5MoeGatedDeltaNet(_config(), layer_idx=2)
    hidden = torch.randn(1, 4, 32, requires_grad=True)
    output = module(hidden, attention_mask=torch.ones(1, 4))
    output.square().mean().backward()
    assert torch.isfinite(output).all()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in module.parameters()
    )
