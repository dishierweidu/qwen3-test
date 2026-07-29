import pytest
import torch

from qwen3_omni_pretrain.models.qwen3_omni_moe.modules.moe import Qwen3OmniMoeMLP


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


def test_nonfinite_router_probabilities_raise():
    moe = Qwen3OmniMoeMLP(
        hidden_size=4,
        intermediate_size=8,
        num_experts=2,
        num_experts_per_tok=1,
        use_shared_expert=False,
    )
    with torch.no_grad():
        moe.gate.weight.fill_(float("nan"))

    with pytest.raises(FloatingPointError, match="router probabilities"):
        moe(torch.ones(1, 1, 4))


def test_config_rejects_legacy_internal_shared_expert():
    from qwen3_omni_pretrain.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeConfig

    with pytest.raises(ValueError, match="single shared dense path"):
        Qwen3OmniMoeConfig(thinker_config={"use_shared_expert": True})
