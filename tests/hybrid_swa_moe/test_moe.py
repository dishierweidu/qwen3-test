from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from qwen3_omni_pretrain.models.hybrid_swa_moe.moe import (
    RoutedSwiGLUMoE,
    SwiGLU,
)


def tiny_routed_moe(
    *, experts: int = 8, top_k: int = 2
) -> RoutedSwiGLUMoE:
    return RoutedSwiGLUMoE(
        hidden_size=16,
        intermediate_size=24,
        num_experts=experts,
        num_experts_per_token=top_k,
    )


def test_swiglu_has_three_bias_free_projections():
    module = SwiGLU(8, 12)
    assert module(torch.randn(2, 3, 8)).shape == (2, 3, 8)
    assert all(parameter.ndim == 2 for parameter in module.parameters())


def test_selected_weights_sum_to_one_and_stats_count_every_route():
    moe = tiny_routed_moe(experts=8, top_k=2)
    output = moe(torch.randn(2, 5, 16), collect_stats=True)

    torch.testing.assert_close(
        output.selected_weights.sum(dim=-1),
        torch.ones((2, 5)),
    )
    assert output.stats is not None
    assert output.stats.expert_token_counts.sum().item() == 2 * 5 * 2
    assert output.stats.router_entropy.isfinite()
    assert output.stats.max_mean_load >= 1.0
    assert not output.stats.router_entropy.requires_grad


def test_forced_routes_give_every_expert_finite_gradients():
    moe = tiny_routed_moe(experts=8, top_k=2)
    hidden = torch.randn(1, 8, 16, requires_grad=True)
    primary = torch.arange(8)
    logits = torch.full((8, 8), -20.0)
    logits[torch.arange(8), primary] = 20.0
    logits[torch.arange(8), (primary + 1) % 8] = 10.0
    with patch.object(
        moe.router,
        "forward",
        return_value=logits.view(1, 8, 8),
    ):
        output = moe(hidden, collect_stats=True)
    output.hidden_states.square().mean().backward()

    assert output.stats is not None
    assert output.stats.expert_token_counts.tolist() == [2] * 8
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    for expert in moe.experts:
        assert all(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            and parameter.grad.abs().sum() > 0
            for parameter in expert.parameters()
        )


def test_padding_routes_do_not_contribute_to_counts_or_output():
    moe = tiny_routed_moe(experts=4, top_k=2)
    hidden = torch.randn(2, 3, 16)
    mask = torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.bool)
    output = moe(hidden, token_mask=mask, collect_stats=True)

    assert output.stats is not None
    assert output.stats.expert_token_counts.sum().item() == 4 * 2
    assert torch.count_nonzero(output.hidden_states[1, 1:]) == 0


def test_bfloat16_output_preserves_dtype():
    moe = tiny_routed_moe(experts=4, top_k=2).to(torch.bfloat16)
    output = moe(torch.randn(1, 4, 16, dtype=torch.bfloat16))
    assert output.hidden_states.dtype is torch.bfloat16
    assert output.aux_loss.dtype is torch.float32


def test_top_k_is_never_clamped():
    with pytest.raises(ValueError, match="top_k < num_experts"):
        tiny_routed_moe(experts=4, top_k=4)
