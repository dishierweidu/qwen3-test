from __future__ import annotations

import torch

from qwen3_omni_pretrain.models.hybrid_swa_moe import (
    HybridSelfAttention,
    HybridSwaMoeConfig,
)


def attention_config(
    *,
    window_size: int = 8,
    attention_sink: bool = False,
    value_scale: float | None = None,
) -> HybridSwaMoeConfig:
    return HybridSwaMoeConfig(
        vocab_size=32,
        hidden_size=32,
        num_hidden_layers=6,
        num_attention_heads=4,
        full_num_key_value_heads=2,
        swa_num_key_value_heads=2,
        qk_head_dim=8,
        v_head_dim=6,
        rotary_dim=4,
        attention_layer_types=["swa"] * 5 + ["full"],
        ffn_layer_types=["dense"] + ["routed_moe"] * 5,
        swa_window_size=window_size,
        full_rope_theta=10_000.0,
        swa_rope_theta=10_000.0,
        attention_sink=attention_sink,
        num_experts=4,
        num_experts_per_token=2,
        expert_intermediate_size=32,
        dense_intermediate_size=64,
        value_scale=value_scale,
        router_aux_loss_weight=0.01,
        mtp_num_predictors=1,
        mtp_loss_weight=0.1,
    )


def tied_attention_pair(
    *, window_size: int = 128, attention_sink: bool = False
) -> tuple[HybridSelfAttention, HybridSelfAttention]:
    config = attention_config(
        window_size=window_size,
        attention_sink=attention_sink,
    )
    full = HybridSelfAttention(
        config,
        layer_index=5,
        attention_type="full",
        num_key_value_heads=2,
        rope_theta=10_000.0,
        attention_sink=attention_sink,
    )
    swa = HybridSelfAttention(
        config,
        layer_index=0,
        attention_type="swa",
        num_key_value_heads=2,
        rope_theta=10_000.0,
        attention_sink=attention_sink,
    )
    swa.load_state_dict(full.state_dict())
    return full, swa


def test_swa_matches_full_when_sequence_fits_window_and_sink_is_off():
    torch.manual_seed(3)
    full, swa = tied_attention_pair(window_size=128, attention_sink=False)
    hidden = torch.randn(2, 32, 32)
    positions = torch.arange(32).expand(2, -1)

    expected, _ = full(hidden, position_ids=positions)
    actual, _ = swa(hidden, position_ids=positions)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_sink_matches_explicit_extra_softmax_logit_formula():
    torch.manual_seed(5)
    config = attention_config(window_size=8, attention_sink=True)
    module = HybridSelfAttention(config, layer_index=0)
    module.attention_sink_bias.data.copy_(torch.tensor([-1.0, 0.0, 0.5, 1.0]))
    hidden = torch.randn(1, 4, 32)
    positions = torch.arange(4).unsqueeze(0)

    actual, _ = module(hidden, position_ids=positions)
    query, key, value = module._project(hidden, positions)
    key = key.repeat_interleave(2, dim=1)
    value = value.repeat_interleave(2, dim=1)
    scores = torch.matmul(query, key.transpose(-1, -2)) * module.scaling
    causal = torch.ones((4, 4), dtype=torch.bool).tril()
    scores = scores.masked_fill(~causal.view(1, 1, 4, 4), float("-inf"))
    sink = module.attention_sink_bias.view(1, 4, 1, 1).expand(1, 4, 4, 1)
    probabilities = torch.softmax(
        torch.cat((scores, sink), dim=-1).float(), dim=-1
    )[..., :-1].to(query.dtype)
    expected = torch.matmul(probabilities, value)
    expected = module.o_proj(expected.transpose(1, 2).reshape(1, 4, -1))

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_token_outside_window_cannot_change_current_output():
    torch.manual_seed(7)
    config = attention_config(window_size=4)
    module = HybridSelfAttention(config, layer_index=0)
    hidden = torch.randn(1, 7, 32)
    changed = hidden.clone()
    changed[:, 0] += 100.0
    positions = torch.arange(7).unsqueeze(0)

    baseline, _ = module(hidden, position_ids=positions)
    perturbed, _ = module(changed, position_ids=positions)

    torch.testing.assert_close(
        baseline[:, -1], perturbed[:, -1], atol=1e-6, rtol=1e-6
    )
    assert not torch.allclose(baseline[:, 0], perturbed[:, 0])


def test_value_scale_is_applied_to_v_exactly_once():
    torch.manual_seed(11)
    base_config = attention_config(value_scale=None)
    scaled_config = attention_config(value_scale=2.5)
    base = HybridSelfAttention(base_config, layer_index=0)
    scaled = HybridSelfAttention(scaled_config, layer_index=0)
    scaled.load_state_dict(base.state_dict())
    hidden = torch.randn(2, 5, 32)
    positions = torch.arange(5).expand(2, -1)

    expected, _ = base(hidden, position_ids=positions)
    actual, _ = scaled(hidden, position_ids=positions)

    torch.testing.assert_close(actual, expected * 2.5, atol=2e-6, rtol=2e-6)


def test_distinct_rope_bases_change_only_selected_module_behavior():
    torch.manual_seed(13)
    config = attention_config(window_size=1_000)
    first = HybridSelfAttention(
        config, layer_index=0, attention_type="swa", rope_theta=1_000.0
    )
    second = HybridSelfAttention(
        config, layer_index=0, attention_type="swa", rope_theta=1_000_000.0
    )
    second.load_state_dict(first.state_dict())
    hidden = torch.randn(1, 6, 32)
    positions = (torch.arange(6) * 100).unsqueeze(0)

    first_output, _ = first(hidden, position_ids=positions)
    second_output, _ = second(hidden, position_ids=positions)

    assert first.rope_theta == 1_000.0
    assert second.rope_theta == 1_000_000.0
    assert not torch.allclose(first_output, second_output)


def test_bfloat16_output_preserves_input_dtype():
    config = attention_config()
    module = HybridSelfAttention(config, layer_index=0).to(torch.bfloat16)
    hidden = torch.randn(1, 3, 32, dtype=torch.bfloat16)
    output, _ = module(
        hidden,
        position_ids=torch.arange(3).unsqueeze(0),
    )
    assert output.dtype is torch.bfloat16
