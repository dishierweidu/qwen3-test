from __future__ import annotations

from enum import Enum
from typing import Any


class RoutingKind(str, Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    DENSE_ENSEMBLE = "dense_ensemble"


def parse_layer_indices(
    value: str | None,
    *,
    layer_count: int,
    field: str,
) -> tuple[int, ...]:
    if value is None or not value.strip():
        return ()
    parts = value.split(",")
    if any(not part.strip().isdigit() for part in parts):
        raise ValueError(f"{field} must be a comma-separated integer list")
    result = tuple(int(part.strip()) for part in parts)
    if len(set(result)) != len(result):
        raise ValueError(f"{field} contains duplicate layer indices")
    if any(index < 0 or index >= layer_count for index in result):
        raise ValueError(f"{field} contains an out-of-range layer index")
    return result


def validate_legacy_thinker_config(config: Any) -> None:
    if config.num_hidden_layers <= 0:
        raise ValueError("num_hidden_layers must be positive")
    if config.num_attention_heads <= 0:
        raise ValueError("num_attention_heads must be positive")
    if config.num_key_value_heads <= 0:
        raise ValueError("num_key_value_heads must be positive")
    if config.hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if config.hidden_size % config.num_attention_heads != 0:
        raise ValueError(
            "hidden_size must be divisible by num_attention_heads"
        )
    if config.num_attention_heads % config.num_key_value_heads != 0:
        raise ValueError(
            "num_attention_heads must be divisible by num_key_value_heads"
        )

    moe_layers = parse_layer_indices(
        config.moe_layer_indices,
        layer_count=config.num_hidden_layers,
        field="moe_layer_indices",
    )
    if moe_layers and not config.use_moe:
        raise ValueError("moe_layer_indices requires use_moe=true")

    deltanet_layers = parse_layer_indices(
        config.deltanet_layer_indices,
        layer_count=config.num_hidden_layers,
        field="deltanet_layer_indices",
    )
    if deltanet_layers and not config.use_deltanet:
        raise ValueError(
            "deltanet_layer_indices requires use_deltanet=true"
        )

    try:
        routing_kind = RoutingKind(config.routing_kind)
    except ValueError as error:
        allowed = ", ".join(kind.value for kind in RoutingKind)
        raise ValueError(
            f"routing_kind must be one of: {allowed}"
        ) from error
    config.routing_kind = routing_kind

    if routing_kind is RoutingKind.DENSE:
        if config.use_moe:
            raise ValueError("routing_kind=dense requires use_moe=false")
        return

    if not config.use_moe:
        raise ValueError(
            f"routing_kind={routing_kind.value} requires use_moe=true"
        )
    if config.num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if config.num_experts_per_tok <= 0:
        raise ValueError("num_experts_per_tok must be positive")
    if routing_kind is RoutingKind.SPARSE:
        if config.num_experts_per_tok >= config.num_experts:
            raise ValueError(
                "sparse routing requires top_k < num_experts"
            )
    elif config.num_experts_per_tok != config.num_experts:
        raise ValueError(
            "dense_ensemble routing requires top_k == num_experts"
        )
