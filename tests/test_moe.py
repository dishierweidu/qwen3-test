import pytest
import torch

from qwen3_omni_pretrain.models.qwen3_omni_moe.modules.moe import Qwen3OmniMoeMLP
from qwen3_omni_pretrain.models.qwen3_omni_moe.modules import (
    moe as moe_module,
)


def test_topk_scores_are_renormalized_per_token():
    moe = Qwen3OmniMoeMLP(
        hidden_size=4,
        intermediate_size=8,
        num_experts=3,
        num_experts_per_tok=2,
        use_shared_expert=False,
        renormalize_topk=True,
    )
    gate_probs = torch.tensor(
        [
            [0.60, 0.30, 0.10],
            [0.20, 0.50, 0.30],
        ],
        dtype=torch.float32,
    )

    _, _, scores = moe._dispatch_tokens(gate_probs)

    assert torch.allclose(scores.view(2, 2).sum(dim=-1), torch.ones(2))


def test_internal_shared_expert_is_rejected():
    with pytest.raises(ValueError, match="decoder shared_mlp"):
        Qwen3OmniMoeMLP(
            hidden_size=4,
            intermediate_size=8,
            num_experts=2,
            num_experts_per_tok=1,
            use_shared_expert=True,
        )


def test_nonfinite_router_probabilities_propagate_to_global_output_check():
    moe = Qwen3OmniMoeMLP(
        hidden_size=4,
        intermediate_size=8,
        num_experts=2,
        num_experts_per_tok=1,
        use_shared_expert=False,
    )
    with torch.no_grad():
        moe.gate.weight.fill_(float("nan"))

    output, aux_loss = moe(torch.ones(1, 1, 4))

    assert not torch.isfinite(output).all()
    assert not torch.isfinite(aux_loss)
    assert bool(moe._nonfinite_diagnostic.item())


def test_config_rejects_legacy_internal_shared_expert():
    from qwen3_omni_pretrain.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeConfig

    with pytest.raises(ValueError, match="single shared dense path"):
        Qwen3OmniMoeConfig(thinker_config={"use_shared_expert": True})


def test_shared_topk_selector_respects_renormalization_flag():
    probabilities = torch.tensor([[0.6, 0.3, 0.1]], dtype=torch.float32)
    normalized, normalized_indices = moe_module.select_topk_routes(
        probabilities, k=2, renormalize=True
    )
    raw, raw_indices = moe_module.select_topk_routes(
        probabilities, k=2, renormalize=False
    )
    assert torch.equal(normalized_indices, raw_indices)
    assert normalized.sum(dim=-1).item() == pytest.approx(1.0)
    assert raw.sum(dim=-1).item() == pytest.approx(0.9)


def test_standard_moe_bfloat16_forward_preserves_output_dtype():
    model = Qwen3OmniMoeMLP(
        hidden_size=2,
        intermediate_size=4,
        num_experts=2,
        num_experts_per_tok=1,
        use_shared_expert=False,
    ).to(dtype=torch.bfloat16)
    output, _ = model(
        torch.ones(1, 1, 2, dtype=torch.bfloat16)
    )
    assert output.dtype is torch.bfloat16
