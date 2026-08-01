from __future__ import annotations

import inspect

import pytest
import torch

from qwen3_omni_pretrain.parallel import expert_parallel
from qwen3_omni_pretrain.parallel.expert_parallel import (
    ExpertParallelContext,
    combine_from_experts,
    dispatch_to_experts,
    validate_parallel_topology,
)


def test_expert_parallel_rejects_tp_combination():
    with pytest.raises(ValueError, match="EP.*TP"):
        validate_parallel_topology(
            world_size=4,
            data_parallel_size=1,
            pipeline_parallel_size=1,
            expert_parallel_size=2,
            tensor_parallel_size=2,
        )


def test_ep_one_preserves_existing_tp_pp_topology():
    topology = validate_parallel_topology(
        world_size=4,
        data_parallel_size=1,
        pipeline_parallel_size=2,
        expert_parallel_size=1,
        tensor_parallel_size=2,
    )
    assert topology.expert_parallel_size == 1
    assert topology.pipeline_parallel_size == 2
    assert topology.tensor_parallel_size == 2

    dp_topology = validate_parallel_topology(
        world_size=8,
        data_parallel_size=4,
        pipeline_parallel_size=1,
        expert_parallel_size=1,
        tensor_parallel_size=2,
    )
    assert dp_topology.world_size == 8


@pytest.mark.parametrize(
    ("world", "dp", "pp", "ep", "tp"),
    [
        (4, 2, 1, 2, 1),
        (4, 1, 2, 2, 1),
        (4, 1, 1, 2, 2),
        (3, 1, 1, 3, 1),
        (4, 1, 1, 2, 1),
    ],
)
def test_unsupported_ep_topologies_fail(world, dp, pp, ep, tp):
    with pytest.raises(ValueError, match="EP|world_size"):
        validate_parallel_topology(
            world_size=world,
            data_parallel_size=dp,
            pipeline_parallel_size=pp,
            expert_parallel_size=ep,
            tensor_parallel_size=tp,
        )


def test_ep_one_dispatch_combine_preserves_values_and_gradients():
    context = ExpertParallelContext(
        group=object(),  # EP=1 never enters a distributed collective.
        rank=0,
        world_size=1,
        local_expert_start=0,
        local_expert_end=4,
    )
    tokens = torch.randn(3, 5, requires_grad=True)
    expert_ids = torch.tensor([[0, 1], [2, 3], [1, 2]])
    weights = torch.tensor(
        [[0.25, 0.75], [0.4, 0.6], [0.1, 0.9]], requires_grad=True
    )
    dispatched = dispatch_to_experts(
        tokens, expert_ids, weights, context=context
    )
    combined = combine_from_experts(dispatched.token_values * 2, dispatched)

    torch.testing.assert_close(combined, tokens * 2)
    combined.sum().backward()
    torch.testing.assert_close(tokens.grad, torch.full_like(tokens, 2.0))
    assert weights.grad is not None and torch.isfinite(weights.grad).all()


def test_float_payload_helper_uses_autograd_all_to_all_only():
    source = inspect.getsource(expert_parallel._all_to_all_payload)
    assert "dist_nn.all_to_all" in source
    assert "all_to_all_single" not in source
