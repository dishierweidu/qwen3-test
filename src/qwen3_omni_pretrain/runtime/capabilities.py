from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum


class CacheErrorCode(str, Enum):
    DELTA_NET_UNSUPPORTED = "DELTA_NET_UNSUPPORTED"
    BEAM_UNSUPPORTED = "BEAM_UNSUPPORTED"
    TRUNCATE_UNSUPPORTED = "TRUNCATE_UNSUPPORTED"
    SPECULATIVE_UNSUPPORTED = "SPECULATIVE_UNSUPPORTED"
    PROFILE_RUNTIME_UNSUPPORTED = "PROFILE_RUNTIME_UNSUPPORTED"
    CONTEXT_OVERFLOW = "CONTEXT_OVERFLOW"
    STATE_OWNER_MISMATCH = "STATE_OWNER_MISMATCH"


class CacheCapabilityError(RuntimeError):
    def __init__(
        self,
        code: CacheErrorCode,
        reason: str,
    ) -> None:
        if not isinstance(code, CacheErrorCode):
            raise TypeError("code must be CacheErrorCode")
        if type(reason) is not str:
            raise TypeError("reason must be a string")
        if not reason.strip():
            raise ValueError("reason must be non-empty")
        self.code = code
        self.reason = reason
        super().__init__(f"{code.value}: {reason}")

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code.value,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class CacheSupport:
    incremental_decode_state: bool
    streaming_generation: bool
    beam_search: bool
    state_truncate: bool
    speculative_decode: bool

    def __post_init__(self) -> None:
        values = (
            self.incremental_decode_state,
            self.streaming_generation,
            self.beam_search,
            self.state_truncate,
            self.speculative_decode,
        )
        if any(type(value) is not bool for value in values):
            raise TypeError("cache support fields must be booleans")

    @classmethod
    def unimplemented(cls) -> CacheSupport:
        return cls(
            incremental_decode_state=False,
            streaming_generation=False,
            beam_search=False,
            state_truncate=False,
            speculative_decode=False,
        )

    def as_dict(self) -> dict[str, bool]:
        return {
            "incremental_decode_state": self.incremental_decode_state,
            "streaming_generation": self.streaming_generation,
            "beam_search": self.beam_search,
            "state_truncate": self.state_truncate,
            "speculative_decode": self.speculative_decode,
        }


@dataclass(frozen=True)
class LegacyLayerScan:
    layer_count: int
    full_attention_layer_indices: tuple[int, ...]
    deltanet_layer_indices: tuple[int, ...]
    tensor_parallel: bool

    def __post_init__(self) -> None:
        if type(self.layer_count) is not int:
            raise TypeError("layer_count must be an integer")
        if self.layer_count <= 0:
            raise ValueError("layer_count must be positive")
        for name in (
            "full_attention_layer_indices",
            "deltanet_layer_indices",
        ):
            indices = getattr(self, name)
            if not isinstance(indices, tuple):
                raise TypeError(f"{name} must be a tuple")
            if any(type(index) is not int for index in indices):
                raise TypeError(f"{name} must contain integers")
            if tuple(sorted(set(indices))) != indices:
                raise ValueError(
                    f"{name} must be sorted and contain unique values"
                )
        if type(self.tensor_parallel) is not bool:
            raise TypeError("tensor_parallel must be a boolean")
        covered = (
            set(self.full_attention_layer_indices)
            | set(self.deltanet_layer_indices)
        )
        if (
            set(self.full_attention_layer_indices)
            & set(self.deltanet_layer_indices)
        ):
            raise ValueError("layer classifications must be disjoint")
        if covered != set(range(self.layer_count)):
            raise ValueError(
                "layer classifications must cover contiguous model layers"
            )

    @property
    def all_full_attention(self) -> bool:
        return not self.deltanet_layer_indices

    @property
    def cacheable_layer_indices(self) -> tuple[int, ...]:
        return self.full_attention_layer_indices


_FULL_ATTENTION_NAMES = frozenset(
    {
        "MultiHeadSelfAttention",
        "TensorParallelMultiHeadSelfAttention",
    }
)
_TENSOR_PARALLEL_ATTENTION_NAME = (
    "TensorParallelMultiHeadSelfAttention"
)
_DELTA_NET_ATTENTION_NAME = "GatedDeltaNetAttention"
_TENSOR_PARALLEL_LAYER_NAME = "TensorParallelThinkerDecoderLayer"


def _snapshot_layers(layers: object) -> tuple[object, ...]:
    if isinstance(layers, (str, bytes, Mapping)) or not isinstance(
        layers,
        Iterable,
    ):
        raise TypeError("layers must be a non-string iterable")
    snapshot = tuple(layers)
    if not snapshot:
        raise ValueError("layers must not be empty")
    return snapshot


def scan_legacy_decoder_layers(layers: object) -> LegacyLayerScan:
    snapshot = _snapshot_layers(layers)
    full_attention: list[int] = []
    deltanet: list[int] = []
    parallel_modes: set[bool] = set()

    for index, layer in enumerate(snapshot):
        block_type = getattr(layer, "block_type", None)
        if type(block_type) is not str:
            raise TypeError("every decoder layer must expose block_type")
        if block_type not in {"attn", "deltanet"}:
            raise ValueError(f"unsupported decoder block_type: {block_type!r}")
        attention = getattr(layer, "self_attn", None)
        if attention is None:
            raise TypeError("every decoder layer must expose self_attn")
        attention_name = type(attention).__name__
        layer_name = type(layer).__name__
        layer_is_parallel = (
            layer_name == _TENSOR_PARALLEL_LAYER_NAME
            or attention_name == _TENSOR_PARALLEL_ATTENTION_NAME
        )

        if attention_name == _DELTA_NET_ATTENTION_NAME:
            actual_block_type = "deltanet"
            deltanet.append(index)
        elif attention_name in _FULL_ATTENTION_NAMES:
            actual_block_type = "attn"
            full_attention.append(index)
            attention_is_parallel = (
                attention_name == _TENSOR_PARALLEL_ATTENTION_NAME
            )
            layer_declares_parallel = (
                layer_name == _TENSOR_PARALLEL_LAYER_NAME
            )
            if attention_is_parallel != layer_declares_parallel:
                raise ValueError(
                    "decoder layer and attention TP topology disagree"
                )
        else:
            raise TypeError(
                "unsupported decoder attention implementation: "
                f"{attention_name}"
            )

        if block_type != actual_block_type:
            raise ValueError(
                "decoder block_type disagrees with its attention "
                "implementation"
            )
        parallel_modes.add(layer_is_parallel)

    if len(parallel_modes) != 1:
        raise ValueError(
            "standard and tensor-parallel decoder layers cannot be mixed"
        )
    return LegacyLayerScan(
        layer_count=len(snapshot),
        full_attention_layer_indices=tuple(full_attention),
        deltanet_layer_indices=tuple(deltanet),
        tensor_parallel=next(iter(parallel_modes)),
    )


def _exact_boolean(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a boolean")
    return value


def cache_support_for_legacy_layers(
    layers: object,
    *,
    protocol_implemented: bool = False,
    tp_local_shard_validated: bool = False,
) -> CacheSupport:
    protocol_ready = _exact_boolean(
        protocol_implemented,
        "protocol_implemented",
    )
    tp_ready = _exact_boolean(
        tp_local_shard_validated,
        "tp_local_shard_validated",
    )
    scan = scan_legacy_decoder_layers(layers)
    incremental = (
        protocol_ready
        and scan.all_full_attention
        and (not scan.tensor_parallel or tp_ready)
    )
    return CacheSupport(
        incremental_decode_state=incremental,
        streaming_generation=False,
        beam_search=False,
        state_truncate=False,
        speculative_decode=False,
    )


def require_incremental_decode_support(
    layers: object,
    *,
    protocol_implemented: bool,
    tp_local_shard_validated: bool = False,
) -> LegacyLayerScan:
    protocol_ready = _exact_boolean(
        protocol_implemented,
        "protocol_implemented",
    )
    tp_ready = _exact_boolean(
        tp_local_shard_validated,
        "tp_local_shard_validated",
    )
    scan = scan_legacy_decoder_layers(layers)
    if scan.deltanet_layer_indices:
        indices = ",".join(
            str(index) for index in scan.deltanet_layer_indices
        )
        raise CacheCapabilityError(
            CacheErrorCode.DELTA_NET_UNSUPPORTED,
            "legacy DeltaNet has no validated recurrent cache "
            f"(layers: {indices})",
        )
    if not protocol_ready:
        raise CacheCapabilityError(
            CacheErrorCode.PROFILE_RUNTIME_UNSUPPORTED,
            "the concrete model does not implement incremental decode state",
        )
    if scan.tensor_parallel and not tp_ready:
        raise CacheCapabilityError(
            CacheErrorCode.PROFILE_RUNTIME_UNSUPPORTED,
            "tensor-parallel local-shard KV requires validated two-rank parity",
        )
    return scan


def validate_generation_operations(
    *,
    num_beams: object,
    state_truncate: object = False,
    speculative_decode: object = False,
) -> None:
    if type(num_beams) is not int or num_beams <= 0 or num_beams != 1:
        raise CacheCapabilityError(
            CacheErrorCode.BEAM_UNSUPPORTED,
            "incremental decode supports exactly num_beams=1",
        )
    if type(state_truncate) is not bool or state_truncate:
        raise CacheCapabilityError(
            CacheErrorCode.TRUNCATE_UNSUPPORTED,
            "generic decoder-state truncate is not implemented",
        )
    if type(speculative_decode) is not bool or speculative_decode:
        raise CacheCapabilityError(
            CacheErrorCode.SPECULATIVE_UNSUPPORTED,
            "speculative decode and rejection are not implemented",
        )


__all__ = [
    "CacheCapabilityError",
    "CacheErrorCode",
    "CacheSupport",
    "LegacyLayerScan",
    "cache_support_for_legacy_layers",
    "require_incremental_decode_support",
    "scan_legacy_decoder_layers",
    "validate_generation_operations",
]
