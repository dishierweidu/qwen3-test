"""Two-rank expert-parallel routing with autograd-preserving payload exchange."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed.nn import functional as dist_nn


@dataclass(frozen=True)
class ParallelTopology:
    """Logical distributed layout in ``[DP, PP, EP, TP]`` order."""

    data_parallel_size: int
    pipeline_parallel_size: int
    expert_parallel_size: int
    tensor_parallel_size: int

    def __post_init__(self) -> None:
        for field in (
            "data_parallel_size",
            "pipeline_parallel_size",
            "expert_parallel_size",
            "tensor_parallel_size",
        ):
            value = getattr(self, field)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field} must be a positive integer")

    @property
    def world_size(self) -> int:
        return (
            self.data_parallel_size
            * self.pipeline_parallel_size
            * self.expert_parallel_size
            * self.tensor_parallel_size
        )


def validate_parallel_topology(
    *,
    world_size: int,
    data_parallel_size: int,
    pipeline_parallel_size: int,
    expert_parallel_size: int,
    tensor_parallel_size: int,
) -> ParallelTopology:
    if type(world_size) is not int or world_size <= 0:
        raise ValueError("world_size must be a positive integer")
    topology = ParallelTopology(
        data_parallel_size=data_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        expert_parallel_size=expert_parallel_size,
        tensor_parallel_size=tensor_parallel_size,
    )
    if topology.world_size != world_size:
        raise ValueError(
            "world_size must equal DP * PP * EP * TP for [DP,PP,EP,TP]"
        )
    if topology.expert_parallel_size > 1 and (
        topology.data_parallel_size != 1
        or topology.pipeline_parallel_size != 1
        or topology.expert_parallel_size != 2
        or topology.tensor_parallel_size != 1
    ):
        raise ValueError(
            "EP>1 currently requires exactly DP=1, PP=1, EP=2, TP=1; "
            "EP cannot be combined with DP, PP, or TP"
        )
    return topology


@dataclass(frozen=True)
class ExpertParallelContext:
    group: dist.ProcessGroup
    rank: int
    world_size: int
    local_expert_start: int
    local_expert_end: int

    def __post_init__(self) -> None:
        if self.group is None:
            raise TypeError("expert parallel group must not be None")
        for field in (
            "rank",
            "world_size",
            "local_expert_start",
            "local_expert_end",
        ):
            if type(getattr(self, field)) is not int:
                raise TypeError(f"{field} must be an integer")
        if self.world_size not in {1, 2}:
            raise ValueError("expert parallel world size must be one or two")
        if not 0 <= self.rank < self.world_size:
            raise ValueError("expert parallel rank is outside the group")
        if not 0 <= self.local_expert_start < self.local_expert_end:
            raise ValueError("local expert range must be non-empty and ordered")
        local_count = self.local_expert_end - self.local_expert_start
        if self.local_expert_start != self.rank * local_count:
            raise ValueError("expert ranges must be contiguous equal partitions")

    @property
    def local_expert_count(self) -> int:
        return self.local_expert_end - self.local_expert_start

    @property
    def num_experts(self) -> int:
        return self.local_expert_count * self.world_size


def create_expert_parallel_context(
    *,
    num_experts: int,
    group: dist.ProcessGroup | None = None,
) -> ExpertParallelContext:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized for EP")
    resolved_group = dist.group.WORLD if group is None else group
    world_size = dist.get_world_size(resolved_group)
    rank = dist.get_rank(resolved_group)
    if world_size not in {1, 2}:
        raise ValueError("the first EP implementation supports EP=1 or EP=2")
    if type(num_experts) is not int or num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if num_experts % world_size:
        raise ValueError("num_experts must be divisible by expert parallel size")
    local_count = num_experts // world_size
    return ExpertParallelContext(
        group=resolved_group,
        rank=rank,
        world_size=world_size,
        local_expert_start=rank * local_count,
        local_expert_end=(rank + 1) * local_count,
    )


@dataclass(frozen=True)
class DispatchedTokens:
    token_values: torch.Tensor
    local_expert_ids: torch.Tensor
    send_counts: tuple[int, ...]
    receive_counts: tuple[int, ...]
    send_order: torch.Tensor
    source_token_indices: torch.Tensor
    source_routing_weights: torch.Tensor
    num_source_tokens: int
    context: ExpertParallelContext

    def __post_init__(self) -> None:
        if self.token_values.ndim != 2:
            raise ValueError("dispatched token_values must have shape [R, H]")
        if (
            self.local_expert_ids.dtype is not torch.long
            or self.local_expert_ids.shape != (self.token_values.shape[0],)
        ):
            raise ValueError("local_expert_ids must have shape [R]")
        if len(self.send_counts) != self.context.world_size or len(
            self.receive_counts
        ) != self.context.world_size:
            raise ValueError("dispatch split counts must match EP world size")
        if sum(self.receive_counts) != self.token_values.shape[0]:
            raise ValueError("receive counts do not match dispatched routes")
        route_count = self.source_token_indices.numel()
        if (
            self.source_token_indices.dtype is not torch.long
            or self.source_token_indices.ndim != 1
            or self.source_routing_weights.shape != (route_count,)
            or self.send_order.shape != (route_count,)
            or sum(self.send_counts) != route_count
        ):
            raise ValueError("source route metadata has inconsistent shapes")
        if type(self.num_source_tokens) is not int or self.num_source_tokens <= 0:
            raise ValueError("num_source_tokens must be positive")


def _exchange_counts(
    send_counts: torch.Tensor,
    *,
    context: ExpertParallelContext,
) -> torch.Tensor:
    if context.world_size == 1:
        return send_counts.clone()
    receive_counts = torch.empty_like(send_counts)
    with torch.no_grad():
        dist.all_to_all_single(
            receive_counts,
            send_counts,
            group=context.group,
        )
    return receive_counts


def _all_to_all_payload(
    payload: torch.Tensor,
    *,
    send_counts: tuple[int, ...],
    receive_counts: tuple[int, ...],
    context: ExpertParallelContext,
) -> torch.Tensor:
    if context.world_size == 1:
        return payload
    input_chunks = list(payload.split(send_counts, dim=0))
    output_buffers = [
        torch.empty(
            (count, *payload.shape[1:]),
            dtype=payload.dtype,
            device=payload.device,
        )
        for count in receive_counts
    ]
    received_chunks = dist_nn.all_to_all(
        output_tensor_list=output_buffers,
        input_tensor_list=input_chunks,
        group=context.group,
    )
    return torch.cat(tuple(received_chunks), dim=0)


def dispatch_to_experts(
    token_values: torch.Tensor,
    selected_expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    *,
    context: ExpertParallelContext,
) -> DispatchedTokens:
    """Dispatch every selected route to its contiguous expert owner."""

    if not isinstance(context, ExpertParallelContext):
        raise TypeError("context must be ExpertParallelContext")
    if not isinstance(token_values, torch.Tensor) or token_values.ndim != 2:
        raise ValueError("token_values must have shape [N, H]")
    if not token_values.is_floating_point() or token_values.shape[0] == 0:
        raise ValueError("token_values must be non-empty and floating")
    if (
        not isinstance(selected_expert_ids, torch.Tensor)
        or selected_expert_ids.dtype is not torch.long
        or selected_expert_ids.ndim != 2
        or selected_expert_ids.shape[0] != token_values.shape[0]
    ):
        raise ValueError("selected_expert_ids must have long shape [N, K]")
    if (
        not isinstance(routing_weights, torch.Tensor)
        or routing_weights.shape != selected_expert_ids.shape
        or not routing_weights.is_floating_point()
    ):
        raise ValueError("routing_weights must have floating shape [N, K]")
    if (
        selected_expert_ids.device != token_values.device
        or routing_weights.device != token_values.device
    ):
        raise ValueError("route metadata and tokens must share a device")
    if bool(
        (
            (selected_expert_ids < 0)
            | (selected_expert_ids >= context.num_experts)
        ).any().item()
    ):
        raise ValueError("selected expert ID is outside the global range")

    num_tokens, top_k = selected_expert_ids.shape
    source_token_indices = (
        torch.arange(num_tokens, device=token_values.device)
        .unsqueeze(1)
        .expand(num_tokens, top_k)
        .reshape(-1)
    )
    flat_experts = selected_expert_ids.reshape(-1)
    flat_weights = routing_weights.reshape(-1)
    destinations = torch.div(
        flat_experts,
        context.local_expert_count,
        rounding_mode="floor",
    )
    send_order = torch.argsort(destinations, stable=True)
    sorted_destinations = destinations.index_select(0, send_order)
    send_counts_tensor = torch.bincount(
        sorted_destinations, minlength=context.world_size
    ).to(dtype=torch.long)
    receive_counts_tensor = _exchange_counts(
        send_counts_tensor, context=context
    )
    send_counts = tuple(int(item) for item in send_counts_tensor.tolist())
    receive_counts = tuple(
        int(item) for item in receive_counts_tensor.tolist()
    )

    sorted_tokens = token_values.index_select(
        0, source_token_indices.index_select(0, send_order)
    )
    received_tokens = _all_to_all_payload(
        sorted_tokens,
        send_counts=send_counts,
        receive_counts=receive_counts,
        context=context,
    )

    local_ids_to_send = (
        flat_experts.index_select(0, send_order)
        % context.local_expert_count
    ).contiguous()
    if context.world_size == 1:
        received_local_ids = local_ids_to_send
    else:
        received_local_ids = torch.empty(
            (sum(receive_counts),),
            dtype=torch.long,
            device=token_values.device,
        )
        with torch.no_grad():
            dist.all_to_all_single(
                received_local_ids,
                local_ids_to_send,
                output_split_sizes=list(receive_counts),
                input_split_sizes=list(send_counts),
                group=context.group,
            )
    return DispatchedTokens(
        token_values=received_tokens,
        local_expert_ids=received_local_ids,
        send_counts=send_counts,
        receive_counts=receive_counts,
        send_order=send_order,
        source_token_indices=source_token_indices,
        source_routing_weights=flat_weights,
        num_source_tokens=num_tokens,
        context=context,
    )


def combine_from_experts(
    expert_outputs: torch.Tensor,
    dispatched: DispatchedTokens,
) -> torch.Tensor:
    """Return expert results to source ranks, weight routes, and scatter-add."""

    if not isinstance(dispatched, DispatchedTokens):
        raise TypeError("dispatched must be DispatchedTokens")
    if (
        not isinstance(expert_outputs, torch.Tensor)
        or expert_outputs.ndim != 2
        or expert_outputs.shape != dispatched.token_values.shape
        or expert_outputs.device != dispatched.token_values.device
    ):
        raise ValueError("expert_outputs must match dispatched token_values")
    context = dispatched.context
    returned_sorted = _all_to_all_payload(
        expert_outputs,
        send_counts=dispatched.receive_counts,
        receive_counts=dispatched.send_counts,
        context=context,
    )
    sorted_weights = dispatched.source_routing_weights.index_select(
        0, dispatched.send_order
    ).to(returned_sorted.dtype)
    sorted_token_indices = dispatched.source_token_indices.index_select(
        0, dispatched.send_order
    )
    output = torch.zeros(
        (dispatched.num_source_tokens, expert_outputs.shape[-1]),
        dtype=expert_outputs.dtype,
        device=expert_outputs.device,
    )
    output.index_add_(
        0,
        sorted_token_indices,
        returned_sorted * sorted_weights.unsqueeze(-1),
    )
    return output


__all__ = [
    "DispatchedTokens",
    "ExpertParallelContext",
    "ParallelTopology",
    "combine_from_experts",
    "create_expert_parallel_context",
    "dispatch_to_experts",
    "validate_parallel_topology",
]
