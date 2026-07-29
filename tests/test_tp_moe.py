import torch

from qwen3_omni_pretrain.models.qwen3_omni_moe.modeling_thinker_text_tp import (
    TensorParallelMoeMLP,
)
from qwen3_omni_pretrain.models.qwen3_omni_moe.modules.moe import (
    Qwen3OmniMoeMLP,
)


def test_tp_world_size_one_matches_standard_routing_and_state_keys():
    standard = Qwen3OmniMoeMLP(
        hidden_size=1,
        intermediate_size=1,
        num_experts=3,
        num_experts_per_tok=2,
        use_shared_expert=False,
        router_normalize_init=False,
        renormalize_topk=True,
    )
    parallel = TensorParallelMoeMLP(
        hidden_size=1,
        intermediate_size=1,
        num_experts=3,
        num_experts_per_tok=2,
        use_shared_expert=False,
        router_normalize_init=False,
        renormalize_topk=True,
    )
    with torch.no_grad():
        gate = torch.log(torch.tensor([[0.6], [0.3], [0.1]]))
        standard.gate.weight.copy_(gate)
        parallel.gate.weight.copy_(gate)
        for standard_expert, parallel_expert in zip(
            standard.experts, parallel.experts
        ):
            standard_expert.fc1.weight.fill_(1)
            standard_expert.fc2.weight.fill_(1)
            parallel_expert.fc1.weight.fill_(1)
            parallel_expert.fc2.weight.fill_(1)

    standard_output, _ = standard(torch.ones(1, 1, 1))
    parallel_output, _ = parallel(torch.ones(1, 1, 1))

    assert torch.allclose(standard_output, parallel_output, atol=1e-6, rtol=1e-6)
    expected_keys = {
        "gate.weight",
        "experts.0.fc1.weight", "experts.0.fc2.weight",
        "experts.1.fc1.weight", "experts.1.fc2.weight",
        "experts.2.fc1.weight", "experts.2.fc2.weight",
    }
    assert set(standard.state_dict()) == expected_keys
    assert set(parallel.state_dict()) == expected_keys
