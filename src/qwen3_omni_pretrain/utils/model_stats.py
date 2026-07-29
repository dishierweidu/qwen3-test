from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Dict, Iterable, Set

import torch

from qwen3_omni_pretrain.models.qwen3_omni_moe.modules.moe import (
    Qwen3OmniMoeMLP,
)


@dataclass(frozen=True)
class ParameterStats:
    total_parameters: int
    trainable_parameters: int
    estimated_active_parameters_per_token: int
    routed_modules: int
    is_estimate: bool = True

    def to_dict(self) -> Dict[str, int | bool]:
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def _unique_parameters(parameters: Iterable[torch.nn.Parameter]):
    seen: Set[int] = set()
    for parameter in parameters:
        identity = id(parameter)
        if identity in seen:
            continue
        seen.add(identity)
        yield parameter


def collect_parameter_stats(model: torch.nn.Module) -> ParameterStats:
    """
    Count unique parameters and estimate parameters active for one token.

    The active estimate treats each ``Qwen3OmniMoeMLP`` router as active and
    replaces all routed-expert parameters with the average size of
    ``num_experts_per_tok`` experts. It does not model capacity drops, expert
    parallel padding, sequence-dependent routing, or activation memory.
    """
    all_parameters = list(_unique_parameters(model.parameters()))
    total_parameters = sum(
        int(parameter.numel()) for parameter in all_parameters
    )
    trainable_parameters = sum(
        int(parameter.numel())
        for parameter in all_parameters
        if parameter.requires_grad
    )

    active_parameters = total_parameters
    routed_modules = 0
    accounted_expert_parameter_ids: Set[int] = set()

    for module in model.modules():
        if not isinstance(module, Qwen3OmniMoeMLP):
            continue
        routed_modules += 1
        expert_parameters = list(
            _unique_parameters(
                parameter
                for expert in module.experts
                for parameter in expert.parameters()
            )
        )
        new_expert_parameters = [
            parameter
            for parameter in expert_parameters
            if id(parameter) not in accounted_expert_parameter_ids
        ]
        accounted_expert_parameter_ids.update(
            id(parameter) for parameter in new_expert_parameters
        )
        all_expert_parameters = sum(
            int(parameter.numel()) for parameter in new_expert_parameters
        )
        if module.num_experts <= 0:
            raise ValueError("MoE module has no experts")
        average_expert_parameters = (
            all_expert_parameters / module.num_experts
        )
        selected_expert_parameters = round(
            average_expert_parameters * module.num_experts_per_tok
        )
        active_parameters -= all_expert_parameters
        active_parameters += selected_expert_parameters

    return ParameterStats(
        total_parameters=total_parameters,
        trainable_parameters=trainable_parameters,
        estimated_active_parameters_per_token=int(active_parameters),
        routed_modules=routed_modules,
        is_estimate=True,
    )


def _format_count(value: int) -> str:
    for suffix, scale in (
        ("T", 10**12),
        ("B", 10**9),
        ("M", 10**6),
        ("K", 10**3),
    ):
        if value >= scale:
            return f"{value / scale:.3f}{suffix}"
    return str(value)


def format_parameter_stats(stats: ParameterStats) -> str:
    return "\n".join(
        [
            f"Total parameters: {_format_count(stats.total_parameters)}",
            f"Trainable parameters: {_format_count(stats.trainable_parameters)}",
            "Estimated active parameters/token: "
            f"{_format_count(stats.estimated_active_parameters_per_token)}",
            f"Routed MoE modules: {stats.routed_modules}",
            "Note: active parameters/token is a routing estimate, not measured FLOPs.",
        ]
    )
