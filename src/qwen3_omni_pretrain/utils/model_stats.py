from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from typing import Dict, Iterable, Protocol, Sequence, Set

import torch



class RoutedParameterInfo(Protocol):
    """Structural interface used to classify routed expert parameters."""

    num_experts: int
    num_experts_per_token: int

    def expert_parameter_groups(
        self,
    ) -> Sequence[Sequence[torch.nn.Parameter]]:
        ...


@dataclass(frozen=True)
class ParameterStats:
    total_parameters: int
    trainable_parameters: int
    estimated_active_parameters_per_token: int
    routed_modules: int
    is_estimate: bool = True
    routed_parameters: int = 0
    shared_parameters: int = 0
    dense_parameters: int = 0

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


def _parameter_numel(parameter: torch.nn.Parameter) -> int:
    """Return the logical size of ordinary and ZeRO-3 placeholder parameters."""
    deepspeed_numel = getattr(parameter, "ds_numel", None)
    if type(deepspeed_numel) is int and deepspeed_numel >= 0:
        return deepspeed_numel

    local_numel = int(parameter.numel())
    if local_numel:
        return local_numel

    deepspeed_shape = getattr(parameter, "ds_shape", None)
    if deepspeed_shape is not None:
        try:
            dimensions = tuple(int(size) for size in deepspeed_shape)
        except (TypeError, ValueError):
            dimensions = ()
        if dimensions and all(size >= 0 for size in dimensions):
            return math.prod(dimensions)

    return local_numel


def collect_parameter_stats(model: torch.nn.Module) -> ParameterStats:
    """
    Count unique parameters and estimate parameters active for one token.

    The active estimate treats every module implementing
    ``RoutedParameterInfo`` as routed and replaces all expert parameters with
    the average size of
    ``num_experts_per_tok`` experts. It does not model capacity drops, expert
    parallel padding, sequence-dependent routing, or activation memory.
    """
    all_parameters = list(_unique_parameters(model.parameters()))
    total_parameters = sum(
        _parameter_numel(parameter) for parameter in all_parameters
    )
    trainable_parameters = sum(
        _parameter_numel(parameter)
        for parameter in all_parameters
        if parameter.requires_grad
    )

    active_parameters = total_parameters
    routed_modules = 0
    accounted_expert_parameter_ids: Set[int] = set()
    shared_parameter_ids: Set[int] = set()

    for name, module in model.named_modules():
        if name.endswith("shared_mlp"):
            shared_parameter_ids.update(
                id(parameter)
                for parameter in _unique_parameters(module.parameters())
            )

        expert_groups_method = getattr(
            module, "expert_parameter_groups", None
        )
        if not callable(expert_groups_method):
            continue
        num_experts = getattr(module, "num_experts", None)
        num_experts_per_token = getattr(
            module,
            "num_experts_per_token",
            getattr(module, "num_experts_per_tok", None),
        )
        if type(num_experts) is not int or num_experts <= 0:
            raise ValueError("routed module has no valid num_experts")
        if (
            type(num_experts_per_token) is not int
            or not 0 < num_experts_per_token <= num_experts
        ):
            raise ValueError("routed module has invalid experts-per-token")
        expert_groups = tuple(tuple(group) for group in expert_groups_method())
        local_num_experts = getattr(
            module, "local_num_experts", num_experts
        )
        if type(local_num_experts) is not int or not (
            0 < local_num_experts <= num_experts
        ):
            raise ValueError("routed module has invalid local expert count")
        if len(expert_groups) != local_num_experts:
            raise ValueError(
                "expert_parameter_groups must return one group per local expert"
            )
        if any(
            any(not isinstance(parameter, torch.nn.Parameter) for parameter in group)
            for group in expert_groups
        ):
            raise TypeError("expert parameter groups must contain Parameters")
        routed_modules += 1
        expert_parameters = list(
            _unique_parameters(
                parameter
                for group in expert_groups
                for parameter in group
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
            _parameter_numel(parameter)
            for parameter in new_expert_parameters
        )
        average_expert_parameters = (
            all_expert_parameters / local_num_experts
        )
        selected_local_experts = min(
            local_num_experts,
            max(
                1,
                round(
                    num_experts_per_token
                    * local_num_experts
                    / num_experts
                ),
            ),
        )
        selected_expert_parameters = round(
            average_expert_parameters * selected_local_experts
        )
        active_parameters -= all_expert_parameters
        active_parameters += selected_expert_parameters

    routed_parameters = sum(
        _parameter_numel(parameter)
        for parameter in all_parameters
        if id(parameter) in accounted_expert_parameter_ids
    )
    shared_parameters = sum(
        _parameter_numel(parameter)
        for parameter in all_parameters
        if id(parameter) in shared_parameter_ids
        and id(parameter) not in accounted_expert_parameter_ids
    )
    dense_parameters = (
        total_parameters - routed_parameters - shared_parameters
    )

    return ParameterStats(
        total_parameters=total_parameters,
        trainable_parameters=trainable_parameters,
        estimated_active_parameters_per_token=int(active_parameters),
        routed_modules=routed_modules,
        routed_parameters=routed_parameters,
        shared_parameters=shared_parameters,
        dense_parameters=dense_parameters,
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


__all__ = [
    "ParameterStats",
    "RoutedParameterInfo",
    "collect_parameter_stats",
    "format_parameter_stats",
]
