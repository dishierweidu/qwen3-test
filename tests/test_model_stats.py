import torch

from qwen3_omni_pretrain.models.qwen3_omni_moe.modules.moe import Qwen3OmniMoeMLP
from qwen3_omni_pretrain.utils.model_stats import collect_parameter_stats
from qwen3_omni_pretrain.models.hybrid_swa_moe.moe import RoutedSwiGLUMoE


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = torch.nn.Linear(4, 4, bias=False)
        self.moe = Qwen3OmniMoeMLP(
            hidden_size=4,
            intermediate_size=8,
            num_experts=4,
            num_experts_per_tok=2,
            use_shared_expert=False,
        )


def test_parameter_stats_count_unique_total_and_active_moe_parameters():
    model = ToyModel()
    stats = collect_parameter_stats(model)

    dense = 4 * 4
    gate = 4 * 4
    one_expert = (4 * 8) + (8 * 4)
    expected_total = dense + gate + (4 * one_expert)
    expected_active = dense + gate + (2 * one_expert)

    assert stats.total_parameters == expected_total
    assert stats.trainable_parameters == expected_total
    assert stats.estimated_active_parameters_per_token == expected_active
    assert stats.routed_parameters == 4 * one_expert
    assert stats.shared_parameters == 0
    assert stats.dense_parameters == dense + gate
    assert stats.routed_modules == 1
    assert stats.is_estimate is True


def test_parameter_stats_use_routed_parameter_protocol_for_hybrid_moe():
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 4, bias=False),
        RoutedSwiGLUMoE(
            hidden_size=4,
            intermediate_size=8,
            num_experts=4,
            num_experts_per_token=2,
        ),
    )
    stats = collect_parameter_stats(model)
    one_expert = (4 * 8) + (4 * 8) + (8 * 4)

    assert stats.routed_modules == 1
    assert stats.routed_parameters == 4 * one_expert
    assert stats.total_parameters > stats.estimated_active_parameters_per_token
    assert stats.dense_parameters == stats.total_parameters - 4 * one_expert
