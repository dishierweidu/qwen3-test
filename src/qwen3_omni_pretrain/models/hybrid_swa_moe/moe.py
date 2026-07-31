"""Dense SwiGLU and routed-only SwiGLU experts."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from qwen3_omni_pretrain.parallel.expert_parallel import (
    ExpertParallelContext,
    combine_from_experts,
    dispatch_to_experts,
)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        if hidden_size <= 0 or intermediate_size <= 0:
            raise ValueError("SwiGLU dimensions must be positive")
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


@dataclass(frozen=True)
class RouterStats:
    expert_token_counts: torch.Tensor
    router_entropy: torch.Tensor
    max_mean_load: torch.Tensor

    def __post_init__(self) -> None:
        for name in (
            "expert_token_counts",
            "router_entropy",
            "max_mean_load",
        ):
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a tensor")
            if value.requires_grad:
                raise ValueError(f"{name} must be detached")
        if self.expert_token_counts.ndim != 1:
            raise ValueError("expert_token_counts must have shape [E]")
        if self.router_entropy.ndim != 0 or self.max_mean_load.ndim != 0:
            raise ValueError("router entropy and load ratio must be scalars")


@dataclass(frozen=True)
class RoutedMoeOutput:
    hidden_states: torch.Tensor
    aux_loss: torch.Tensor
    stats: RouterStats | None
    selected_expert_ids: torch.Tensor
    selected_weights: torch.Tensor


class RoutedSwiGLUMoE(nn.Module):
    """Trainable top-k router plus experts, with no shared/dense path."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        num_experts_per_token: int,
        *,
        expert_parallel_context: ExpertParallelContext | None = None,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or intermediate_size <= 0:
            raise ValueError("MoE dimensions must be positive")
        if type(num_experts) is not int or num_experts <= 1:
            raise ValueError("num_experts must be greater than one")
        if (
            type(num_experts_per_token) is not int
            or num_experts_per_token <= 0
            or num_experts_per_token >= num_experts
        ):
            raise ValueError("routed MoE requires top_k < num_experts")
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.num_experts_per_tok = num_experts_per_token
        if expert_parallel_context is not None:
            if not isinstance(expert_parallel_context, ExpertParallelContext):
                raise TypeError(
                    "expert_parallel_context must be ExpertParallelContext"
                )
            if expert_parallel_context.num_experts != num_experts:
                raise ValueError(
                    "expert parallel context range does not cover num_experts"
                )
        self.expert_parallel_context = expert_parallel_context
        self.local_expert_start = (
            0
            if expert_parallel_context is None
            else expert_parallel_context.local_expert_start
        )
        self.local_expert_end = (
            num_experts
            if expert_parallel_context is None
            else expert_parallel_context.local_expert_end
        )
        self.local_num_experts = self.local_expert_end - self.local_expert_start
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList(
            SwiGLU(hidden_size, intermediate_size)
            for _ in range(self.local_num_experts)
        )

    @property
    def gate(self) -> nn.Linear:
        """Compatibility alias for generic model-inspection code."""

        return self.router

    def expert_parameter_groups(
        self,
    ) -> tuple[tuple[nn.Parameter, ...], ...]:
        return tuple(tuple(expert.parameters()) for expert in self.experts)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        token_mask: torch.Tensor | None = None,
        collect_stats: bool = False,
    ) -> RoutedMoeOutput:
        if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [B, T, H]")
        batch_size, sequence_length, hidden_size = hidden_states.shape
        if hidden_size != self.hidden_size:
            raise ValueError("hidden size does not match routed MoE")
        if token_mask is None:
            valid_mask = torch.ones(
                (batch_size, sequence_length),
                dtype=torch.bool,
                device=hidden_states.device,
            )
        else:
            if (
                not isinstance(token_mask, torch.Tensor)
                or token_mask.dtype is not torch.bool
                or token_mask.shape != (batch_size, sequence_length)
            ):
                raise ValueError("token_mask must have boolean shape [B, T]")
            valid_mask = token_mask

        flattened = hidden_states.reshape(-1, hidden_size)
        logits = self.router(hidden_states)
        if logits.shape != (batch_size, sequence_length, self.num_experts):
            raise ValueError("router must return logits with shape [B, T, E]")
        probabilities = torch.softmax(logits.float(), dim=-1).reshape(
            -1, self.num_experts
        )
        selected_weights, selected_ids = probabilities.topk(
            self.num_experts_per_token, dim=-1
        )
        selected_weights = selected_weights / selected_weights.sum(
            dim=-1, keepdim=True
        )
        flat_valid = valid_mask.reshape(-1)
        if self.expert_parallel_context is None:
            output = torch.zeros_like(flattened)
            for expert_id, expert in enumerate(self.experts):
                routes = (selected_ids == expert_id) & flat_valid.unsqueeze(-1)
                token_indices, route_indices = torch.nonzero(
                    routes, as_tuple=True
                )
                if token_indices.numel() == 0:
                    continue
                expert_output = expert(flattened.index_select(0, token_indices))
                route_weights = selected_weights[
                    token_indices, route_indices
                ].to(expert_output.dtype)
                output.index_add_(
                    0,
                    token_indices,
                    expert_output * route_weights.unsqueeze(-1),
                )
        else:
            effective_weights = selected_weights * flat_valid.unsqueeze(-1)
            dispatched = dispatch_to_experts(
                flattened,
                selected_ids,
                effective_weights,
                context=self.expert_parallel_context,
            )
            routed_output = torch.zeros_like(dispatched.token_values)
            for local_expert_id, expert in enumerate(self.experts):
                selected = dispatched.local_expert_ids == local_expert_id
                route_indices = torch.nonzero(
                    selected, as_tuple=False
                ).flatten()
                if route_indices.numel() == 0:
                    continue
                expert_output = expert(
                    dispatched.token_values.index_select(0, route_indices)
                )
                routed_output.index_copy_(
                    0, route_indices, expert_output
                )
            output = combine_from_experts(routed_output, dispatched)

        valid_probabilities = probabilities[flat_valid]
        valid_selected = selected_ids[flat_valid]
        if valid_probabilities.shape[0] == 0:
            aux_loss = probabilities.sum() * 0.0
            counts = torch.zeros(
                self.num_experts,
                dtype=torch.long,
                device=hidden_states.device,
            )
            entropy = probabilities.sum().detach() * 0.0
            max_mean_load = entropy.clone()
        else:
            importance = valid_probabilities.mean(dim=0)
            counts = torch.bincount(
                valid_selected.reshape(-1), minlength=self.num_experts
            )
            load = counts.to(probabilities.dtype) / valid_selected.numel()
            aux_loss = self.num_experts * torch.sum(importance * load)
            entropy = (
                -(valid_probabilities * valid_probabilities.clamp_min(1e-12).log())
                .sum(dim=-1)
                .mean()
                .detach()
            )
            mean_count = counts.to(torch.float32).mean().clamp_min(1.0)
            max_mean_load = (
                counts.to(torch.float32).max() / mean_count
            ).detach()
        stats = (
            RouterStats(
                expert_token_counts=counts.detach().clone(),
                router_entropy=entropy.detach().clone(),
                max_mean_load=max_mean_load.detach().clone(),
            )
            if collect_stats
            else None
        )
        return RoutedMoeOutput(
            hidden_states=output.view(
                batch_size, sequence_length, hidden_size
            ),
            aux_loss=aux_loss,
            stats=stats,
            selected_expert_ids=selected_ids.view(
                batch_size, sequence_length, self.num_experts_per_token
            ),
            selected_weights=selected_weights.view(
                batch_size, sequence_length, self.num_experts_per_token
            ),
        )


__all__ = [
    "RoutedMoeOutput",
    "RoutedSwiGLUMoE",
    "RouterStats",
    "SwiGLU",
]
