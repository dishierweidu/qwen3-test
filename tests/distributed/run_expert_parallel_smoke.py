"""Two-rank Gloo forward/backward parity smoke for Hybrid routed experts."""

from __future__ import annotations

import torch
import torch.distributed as dist

from qwen3_omni_pretrain.models.hybrid_swa_moe.moe import RoutedSwiGLUMoE
from qwen3_omni_pretrain.parallel.expert_parallel import (
    create_expert_parallel_context,
)
from qwen3_omni_pretrain.parallel.initialize import (
    destroy_model_parallel,
    get_expert_model_parallel_group,
    get_expert_model_parallel_rank,
    get_expert_model_parallel_world_size,
    initialize_model_parallel,
)


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, name: str) -> None:
    try:
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    except AssertionError as error:
        raise AssertionError(f"expert-parallel {name} parity failed") from error


def main() -> None:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 2:
        raise AssertionError("smoke test requires exactly two ranks")
    initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=2,
        backend="gloo",
    )
    if get_expert_model_parallel_world_size() != 2:
        raise AssertionError("initialized EP group must contain two ranks")
    if get_expert_model_parallel_rank() != rank:
        raise AssertionError("EP rank must match the two-rank world layout")
    context = create_expert_parallel_context(
        num_experts=4,
        group=get_expert_model_parallel_group(),
    )

    torch.manual_seed(101)
    reference = RoutedSwiGLUMoE(4, 7, 4, 2)
    parallel = RoutedSwiGLUMoE(
        4,
        7,
        4,
        2,
        expert_parallel_context=context,
    )
    with torch.no_grad():
        router_weight = torch.full((4, 4), -2.0)
        for token in range(4):
            router_weight[token, token] = 3.0
            router_weight[(token + 1) % 4, token] = 1.0
        reference.router.weight.copy_(router_weight)
        parallel.router.weight.copy_(router_weight)
        for local_index, local_expert in enumerate(parallel.experts):
            global_index = context.local_expert_start + local_index
            local_expert.load_state_dict(
                reference.experts[global_index].state_dict()
            )

    # Identical basis inputs make every expert receive two selected routes on
    # each source rank and make the expected cross-rank expert gradient exact.
    reference_input = torch.eye(4, requires_grad=True)
    parallel_input = torch.eye(4, requires_grad=True)
    reference_output = reference(
        reference_input.unsqueeze(0), collect_stats=True
    )
    parallel_output = parallel(
        parallel_input.unsqueeze(0), collect_stats=True
    )
    coefficients = torch.arange(1, 17, dtype=torch.float32).view(1, 4, 4)
    reference_loss = (
        reference_output.hidden_states * coefficients
    ).sum() + 0.01 * reference_output.aux_loss
    parallel_loss = (
        parallel_output.hidden_states * coefficients
    ).sum() + 0.01 * parallel_output.aux_loss
    reference_loss.backward()
    parallel_loss.backward()

    _assert_close(
        parallel_output.hidden_states,
        reference_output.hidden_states,
        "output",
    )
    _assert_close(parallel_loss.detach(), reference_loss.detach(), "loss")
    _assert_close(parallel_input.grad, reference_input.grad, "input gradient")
    _assert_close(
        parallel.router.weight.grad,
        reference.router.weight.grad,
        "router gradient",
    )
    for local_index, local_expert in enumerate(parallel.experts):
        global_index = context.local_expert_start + local_index
        reference_expert = reference.experts[global_index]
        for (name, parameter), (_, expected_parameter) in zip(
            local_expert.named_parameters(),
            reference_expert.named_parameters(),
        ):
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise AssertionError(f"rank {rank} expert {global_index} {name} grad missing")
            if parameter.grad.abs().sum() == 0:
                raise AssertionError(f"rank {rank} expert {global_index} {name} grad is zero")
            _assert_close(
                parameter.grad,
                expected_parameter.grad * world_size,
                f"expert {global_index} {name} gradient",
            )
    if parallel_input.grad is None or parallel_input.grad.abs().sum() == 0:
        raise AssertionError(f"rank {rank} input gradient is vacuous")
    if parallel_output.stats is None:
        raise AssertionError("router stats missing")
    if parallel_output.stats.expert_token_counts.tolist() != [2, 2, 2, 2]:
        raise AssertionError("deterministic routing did not cover every expert")

    dist.barrier()
    if rank == 0:
        print("expert-parallel smoke: PASS")
    destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
