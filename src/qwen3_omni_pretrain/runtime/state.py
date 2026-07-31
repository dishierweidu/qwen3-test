from __future__ import annotations

import secrets
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import torch

from qwen3_omni_pretrain.multimodal.types import PositionBatch
from qwen3_omni_pretrain.runtime.capabilities import (
    CacheCapabilityError,
    CacheErrorCode,
)


_QWEN3_AXES = ("temporal", "height", "width")


def _require_tensor(value: object, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    return value


def _clone_detached(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().clone(memory_format=torch.contiguous_format)


def _require_bool_mask(
    value: object,
    name: str,
    *,
    shape: tuple[int, ...] | None = None,
) -> torch.Tensor:
    tensor = _require_tensor(value, name)
    if tensor.dtype is not torch.bool:
        raise TypeError(f"{name} must have dtype torch.bool")
    if shape is not None and tensor.shape != shape:
        raise ValueError(f"{name} must have shape {list(shape)}")
    return tensor


def _require_long_vector(
    value: object,
    name: str,
    *,
    batch_size: int | None = None,
) -> torch.Tensor:
    tensor = _require_tensor(value, name)
    if tensor.dtype is not torch.long:
        raise TypeError(f"{name} must have dtype torch.long")
    expected = "[B]" if batch_size is None else f"[{batch_size}]"
    if tensor.ndim != 1 or tensor.numel() == 0:
        raise ValueError(f"{name} must have non-empty shape {expected}")
    if batch_size is not None and tensor.shape != (batch_size,):
        raise ValueError(f"{name} must have shape {expected}")
    if bool((tensor < 0).any().item()):
        raise ValueError(f"{name} must be non-negative")
    return tensor


def _require_numeric(tensor: torch.Tensor, name: str) -> None:
    if tensor.is_complex() or not (
        tensor.is_floating_point()
        or tensor.dtype
        in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }
    ):
        raise TypeError(f"{name} must have a numeric non-complex dtype")
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} must contain finite values")


def _clone_position_batch(
    position: PositionBatch,
    key_valid_mask: torch.Tensor,
) -> PositionBatch:
    if not isinstance(position, PositionBatch):
        raise TypeError("position must be PositionBatch")
    cloned = PositionBatch(
        position_ids=_clone_detached(
            _require_tensor(position.position_ids, "position_ids")
        ),
        rope_deltas=_clone_detached(
            _require_tensor(position.rope_deltas, "rope_deltas")
        ),
        axis_names=position.axis_names,
    )
    cloned.validate(key_valid_mask)
    return cloned


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


@dataclass(frozen=True)
class StateOwner:
    display_request_id: str
    nonce: bytes

    def __post_init__(self) -> None:
        if type(self.display_request_id) is not str:
            raise TypeError("display_request_id must be a string")
        if not self.display_request_id.strip():
            raise ValueError("display_request_id must be nonblank")
        if type(self.nonce) is not bytes:
            raise TypeError("nonce must be bytes")
        if len(self.nonce) < 16:
            raise ValueError("nonce must contain at least 128 bits")

    @classmethod
    def fresh(cls, display_request_id: str) -> StateOwner:
        return cls(display_request_id, secrets.token_bytes(16))


@dataclass(frozen=True)
class LegacyPositionCursor:
    next_storage_position: torch.Tensor

    def __post_init__(self) -> None:
        position = _require_long_vector(
            self.next_storage_position,
            "next_storage_position",
        )
        object.__setattr__(
            self,
            "next_storage_position",
            _clone_detached(position),
        )

    @property
    def batch_size(self) -> int:
        return self.next_storage_position.shape[0]

    def clone_detached(self) -> LegacyPositionCursor:
        return LegacyPositionCursor(self.next_storage_position)

    def advance(self, storage_delta: torch.Tensor) -> LegacyPositionCursor:
        delta = _require_long_vector(
            storage_delta,
            "storage_delta",
            batch_size=self.batch_size,
        )
        if delta.device != self.next_storage_position.device:
            raise ValueError(
                "storage_delta and next_storage_position must share a device"
            )
        return LegacyPositionCursor(self.next_storage_position + delta)


@dataclass(frozen=True)
class Qwen3DisjointPositionCursor:
    next_text_position: torch.Tensor
    rope_deltas: torch.Tensor
    axis_names: tuple[str, ...]

    def __post_init__(self) -> None:
        next_position = _require_tensor(
            self.next_text_position,
            "next_text_position",
        )
        rope_deltas = _require_tensor(self.rope_deltas, "rope_deltas")
        if next_position.ndim != 1 or next_position.numel() == 0:
            raise ValueError("next_text_position must have non-empty shape [B]")
        _require_numeric(next_position, "next_text_position")
        if bool((next_position < 0).any().item()):
            raise ValueError("next_text_position must be non-negative")
        if rope_deltas.shape != (next_position.shape[0], 1):
            raise ValueError("rope_deltas must have shape [B, 1]")
        _require_numeric(rope_deltas, "rope_deltas")
        if (
            rope_deltas.dtype != next_position.dtype
            or rope_deltas.device != next_position.device
        ):
            raise ValueError(
                "next_text_position and rope_deltas must share dtype/device"
            )
        if self.axis_names != _QWEN3_AXES:
            raise ValueError(
                "Qwen3 disjoint cursor axes must be temporal/height/width"
            )
        object.__setattr__(
            self,
            "next_text_position",
            _clone_detached(next_position),
        )
        object.__setattr__(self, "rope_deltas", _clone_detached(rope_deltas))

    @property
    def batch_size(self) -> int:
        return self.next_text_position.shape[0]

    def clone_detached(self) -> Qwen3DisjointPositionCursor:
        return Qwen3DisjointPositionCursor(
            self.next_text_position,
            self.rope_deltas,
            self.axis_names,
        )


PositionCursor = LegacyPositionCursor | Qwen3DisjointPositionCursor


@dataclass(frozen=True)
class DecoderPositionState:
    cached: PositionBatch
    key_valid_mask: torch.Tensor
    continuation: PositionCursor

    def __post_init__(self) -> None:
        mask = _require_bool_mask(self.key_valid_mask, "key_valid_mask")
        if mask.ndim != 2 or mask.shape[0] == 0:
            raise ValueError("key_valid_mask must have non-empty shape [B, S]")
        mask = _clone_detached(mask)
        cached = _clone_position_batch(self.cached, mask)
        continuation = self.continuation
        if not isinstance(
            continuation,
            (LegacyPositionCursor, Qwen3DisjointPositionCursor),
        ):
            raise TypeError("continuation must be a supported position cursor")
        continuation = continuation.clone_detached()
        if continuation.batch_size != mask.shape[0]:
            raise ValueError("position cursor batch size does not match history")
        if isinstance(continuation, LegacyPositionCursor):
            if continuation.next_storage_position.device != mask.device:
                raise ValueError("legacy cursor and position history differ in device")
        else:
            if continuation.axis_names != cached.axis_names:
                raise ValueError("Qwen3 cursor axes do not match position history")
            if (
                continuation.next_text_position.device != mask.device
                or continuation.next_text_position.dtype
                != cached.position_ids.dtype
                or continuation.rope_deltas.device != mask.device
                or not torch.equal(
                    continuation.rope_deltas,
                    cached.rope_deltas,
                )
            ):
                raise ValueError(
                    "Qwen3 cursor dtype/device/delta must match position history"
                )
        object.__setattr__(self, "key_valid_mask", mask)
        object.__setattr__(self, "cached", cached)
        object.__setattr__(self, "continuation", continuation)

    @property
    def batch_size(self) -> int:
        return self.key_valid_mask.shape[0]

    @property
    def sequence_length(self) -> int:
        return self.key_valid_mask.shape[1]

    @property
    def device(self) -> torch.device:
        return self.key_valid_mask.device

    def clone_detached(self) -> DecoderPositionState:
        return DecoderPositionState(
            self.cached,
            self.key_valid_mask,
            self.continuation,
        )

    def append(
        self,
        *,
        current: PositionBatch,
        current_key_valid_mask: torch.Tensor,
        continuation: PositionCursor,
    ) -> DecoderPositionState:
        mask = _require_bool_mask(
            current_key_valid_mask,
            "current_key_valid_mask",
        )
        if mask.ndim != 2 or mask.shape[0] != self.batch_size:
            raise ValueError(
                "current_key_valid_mask must have shape [B, Q] with matching B"
            )
        mask = _clone_detached(mask)
        current = _clone_position_batch(current, mask)
        if (
            current.axis_names != self.cached.axis_names
            or current.position_ids.dtype != self.cached.position_ids.dtype
            or current.position_ids.device != self.cached.position_ids.device
        ):
            raise ValueError(
                "current and cached positions must share axes, dtype and device"
            )
        combined = PositionBatch(
            position_ids=torch.cat(
                (self.cached.position_ids, current.position_ids),
                dim=2,
            ),
            rope_deltas=current.rope_deltas,
            axis_names=current.axis_names,
        )
        return DecoderPositionState(
            combined,
            torch.cat((self.key_valid_mask, mask), dim=1),
            continuation,
        )

    def logical_tensor_bytes(self) -> int:
        cursor_tensors = (
            (self.continuation.next_storage_position,)
            if isinstance(self.continuation, LegacyPositionCursor)
            else (
                self.continuation.next_text_position,
                self.continuation.rope_deltas,
            )
        )
        return sum(
            _tensor_bytes(tensor)
            for tensor in (
                self.cached.position_ids,
                self.cached.rope_deltas,
                self.key_valid_mask,
                *cursor_tensors,
            )
        )


@dataclass(frozen=True)
class AttentionKV:
    key: torch.Tensor
    value: torch.Tensor
    key_valid_mask: torch.Tensor

    def __post_init__(self) -> None:
        key = _require_tensor(self.key, "key")
        value = _require_tensor(self.value, "value")
        if key.ndim != 4 or value.ndim != 4:
            raise ValueError("key and value must have shape [B, Hkv, S, D]")
        if any(size <= 0 for size in (key.shape[0], key.shape[1], key.shape[3])):
            raise ValueError("key batch, head and head-dimension must be positive")
        if any(size <= 0 for size in (value.shape[0], value.shape[1], value.shape[3])):
            raise ValueError("value batch, head and head-dimension must be positive")
        if key.shape[:3] != value.shape[:3]:
            raise ValueError("key and value must share B/Hkv/S dimensions")
        if not key.is_floating_point() or not value.is_floating_point():
            raise TypeError("key and value must have floating dtypes")
        if key.dtype != value.dtype or key.device != value.device:
            raise ValueError("key and value must share dtype and device")
        if not bool(torch.isfinite(key).all().item()) or not bool(
            torch.isfinite(value).all().item()
        ):
            raise ValueError("key and value must contain finite values")
        mask = _require_bool_mask(
            self.key_valid_mask,
            "key_valid_mask",
            shape=(key.shape[0], key.shape[2]),
        )
        if mask.device != key.device:
            raise ValueError("key, value and key_valid_mask must share a device")
        object.__setattr__(self, "key", _clone_detached(key))
        object.__setattr__(self, "value", _clone_detached(value))
        object.__setattr__(self, "key_valid_mask", _clone_detached(mask))

    @property
    def batch_size(self) -> int:
        return self.key.shape[0]

    @property
    def sequence_length(self) -> int:
        return self.key.shape[2]

    def clone_detached(self) -> AttentionKV:
        return AttentionKV(self.key, self.value, self.key_valid_mask)

    def append(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_valid_mask: torch.Tensor,
    ) -> AttentionKV:
        current = AttentionKV(key, value, key_valid_mask)
        if (
            current.batch_size != self.batch_size
            or current.key.shape[1] != self.key.shape[1]
            or current.key.shape[3] != self.key.shape[3]
            or current.value.shape[3] != self.value.shape[3]
            or current.key.dtype != self.key.dtype
            or current.key.device != self.key.device
        ):
            raise ValueError("current key/value topology does not match history")
        return AttentionKV(
            torch.cat((self.key, current.key), dim=2),
            torch.cat((self.value, current.value), dim=2),
            torch.cat(
                (self.key_valid_mask, current.key_valid_mask),
                dim=1,
            ),
        )

    def logical_tensor_bytes(self) -> int:
        return sum(
            _tensor_bytes(tensor)
            for tensor in (self.key, self.value, self.key_valid_mask)
        )


@dataclass(frozen=True)
class SlidingWindowKV:
    key: torch.Tensor
    value: torch.Tensor
    key_valid_mask: torch.Tensor
    position: PositionBatch
    window_size: int

    def __post_init__(self) -> None:
        if type(self.window_size) is not int:
            raise TypeError("window_size must be an integer")
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        cache = AttentionKV(self.key, self.value, self.key_valid_mask)
        if cache.sequence_length > self.window_size:
            raise ValueError("sliding-window storage cannot exceed window_size")
        position = _clone_position_batch(
            self.position,
            cache.key_valid_mask,
        )
        if position.position_ids.device != cache.key.device:
            raise ValueError("position and sliding-window KV must share a device")
        if bool(
            (cache.key_valid_mask.sum(dim=1) > self.window_size).any().item()
        ):
            raise ValueError("valid sliding-window history exceeds window_size")
        object.__setattr__(self, "key", cache.key)
        object.__setattr__(self, "value", cache.value)
        object.__setattr__(self, "key_valid_mask", cache.key_valid_mask)
        object.__setattr__(self, "position", position)

    @property
    def batch_size(self) -> int:
        return self.key.shape[0]

    @property
    def sequence_length(self) -> int:
        return self.key.shape[2]

    def clone_detached(self) -> SlidingWindowKV:
        return SlidingWindowKV(
            self.key,
            self.value,
            self.key_valid_mask,
            self.position,
            self.window_size,
        )

    def append(
        self,
        *,
        key: torch.Tensor,
        value: torch.Tensor,
        key_valid_mask: torch.Tensor,
        position: PositionBatch,
    ) -> SlidingWindowKV:
        current = AttentionKV(key, value, key_valid_mask)
        current_position = _clone_position_batch(
            position,
            current.key_valid_mask,
        )
        if (
            current.batch_size != self.batch_size
            or current.key.shape[1] != self.key.shape[1]
            or current.key.shape[3] != self.key.shape[3]
            or current.value.shape[3] != self.value.shape[3]
            or current.key.dtype != self.key.dtype
            or current.key.device != self.key.device
        ):
            raise ValueError("current key/value topology does not match history")
        if (
            current_position.axis_names != self.position.axis_names
            or current_position.position_ids.dtype
            != self.position.position_ids.dtype
            or current_position.position_ids.device
            != self.position.position_ids.device
        ):
            raise ValueError(
                "current and cached window positions must share axes/dtype/device"
            )

        all_key = torch.cat((self.key, current.key), dim=2)
        all_value = torch.cat((self.value, current.value), dim=2)
        all_mask = torch.cat(
            (self.key_valid_mask, current.key_valid_mask),
            dim=1,
        )
        all_positions = torch.cat(
            (
                self.position.position_ids,
                current_position.position_ids,
            ),
            dim=2,
        )
        retained: list[torch.Tensor] = []
        for row in range(self.batch_size):
            indices = torch.nonzero(all_mask[row], as_tuple=False).flatten()
            retained.append(indices[-self.window_size :])
        output_length = max((indices.numel() for indices in retained), default=0)
        output_key = torch.zeros(
            (
                self.batch_size,
                self.key.shape[1],
                output_length,
                self.key.shape[3],
            ),
            dtype=self.key.dtype,
            device=self.key.device,
        )
        output_value = torch.zeros(
            (
                self.batch_size,
                self.value.shape[1],
                output_length,
                self.value.shape[3],
            ),
            dtype=self.value.dtype,
            device=self.value.device,
        )
        output_mask = torch.zeros(
            (self.batch_size, output_length),
            dtype=torch.bool,
            device=self.key.device,
        )
        output_positions = torch.zeros(
            (
                all_positions.shape[0],
                self.batch_size,
                output_length,
            ),
            dtype=all_positions.dtype,
            device=all_positions.device,
        )
        for row, indices in enumerate(retained):
            count = indices.numel()
            if count == 0:
                continue
            output_key[row, :, :count] = all_key[row].index_select(1, indices)
            output_value[row, :, :count] = all_value[row].index_select(
                1,
                indices,
            )
            output_positions[:, row, :count] = all_positions[:, row].index_select(
                1,
                indices,
            )
            output_mask[row, :count] = True
        return SlidingWindowKV(
            key=output_key,
            value=output_value,
            key_valid_mask=output_mask,
            position=PositionBatch(
                position_ids=output_positions,
                rope_deltas=current_position.rope_deltas,
                axis_names=current_position.axis_names,
            ),
            window_size=self.window_size,
        )

    def logical_tensor_bytes(self) -> int:
        return sum(
            _tensor_bytes(tensor)
            for tensor in (
                self.key,
                self.value,
                self.key_valid_mask,
                self.position.position_ids,
                self.position.rope_deltas,
            )
        )


@runtime_checkable
class StatePartition(Protocol):
    @property
    def batch_size(self) -> int: ...

    def clone_detached(self) -> StatePartition: ...

    def logical_tensor_bytes(self) -> int: ...


@dataclass(frozen=True)
class LegacyProcessedPrefix:
    has_image: tuple[bool, ...]
    has_audio: tuple[bool, ...]
    prefix_storage_length: int

    def __post_init__(self) -> None:
        for name in ("has_image", "has_audio"):
            value = getattr(self, name)
            if not isinstance(value, tuple):
                raise TypeError(f"{name} must be a tuple")
            if not value:
                raise ValueError(f"{name} must not be empty")
            if any(type(flag) is not bool for flag in value):
                raise TypeError(f"{name} must contain exact booleans")
        if len(self.has_image) != len(self.has_audio):
            raise ValueError("media prefix flags must have equal batch length")
        if type(self.prefix_storage_length) is not int:
            raise TypeError("prefix_storage_length must be an integer")
        if self.prefix_storage_length != 2:
            raise ValueError("legacy processed prefix must occupy exactly 2 slots")

    @property
    def batch_size(self) -> int:
        return len(self.has_image)

    def clone_detached(self) -> LegacyProcessedPrefix:
        return LegacyProcessedPrefix(
            tuple(self.has_image),
            tuple(self.has_audio),
            self.prefix_storage_length,
        )


_GENERIC_PARTITIONS = (
    "gdn_state",
    "talker_state",
    "mtp_state",
    "codec_state",
)


@dataclass(frozen=True)
class DecoderState:
    owner: StateOwner
    seen_tokens: torch.Tensor
    position: DecoderPositionState | None
    full_attention_kv: Mapping[int, AttentionKV]
    swa_kv: Mapping[int, SlidingWindowKV]
    processed_media: LegacyProcessedPrefix | None
    gdn_state: StatePartition | None
    talker_state: StatePartition | None
    mtp_state: StatePartition | None
    codec_state: StatePartition | None

    def __post_init__(self) -> None:
        if not isinstance(self.owner, StateOwner):
            raise TypeError("owner must be StateOwner")
        seen = _require_long_vector(self.seen_tokens, "seen_tokens")
        seen = _clone_detached(seen)
        batch_size = seen.shape[0]
        position = self.position
        if position is not None:
            if not isinstance(position, DecoderPositionState):
                raise TypeError("position must be DecoderPositionState or None")
            position = position.clone_detached()
            if position.batch_size != batch_size:
                raise ValueError("position and seen_tokens batch sizes differ")
            if position.device != seen.device:
                raise ValueError("position and seen_tokens devices differ")

        full = self._clone_cache_mapping(
            self.full_attention_kv,
            AttentionKV,
            "full_attention_kv",
        )
        swa = self._clone_cache_mapping(
            self.swa_kv,
            SlidingWindowKV,
            "swa_kv",
        )
        self._validate_cache_mapping(full, batch_size, seen.device, "full")
        self._validate_cache_mapping(swa, batch_size, seen.device, "swa")
        if full:
            reference = next(iter(full.values()))
            for cache in tuple(full.values())[1:]:
                if cache.sequence_length != reference.sequence_length:
                    raise ValueError("full-attention cache lengths must match")
                if cache.key.dtype != reference.key.dtype:
                    raise ValueError("full-attention cache dtypes must match")
                if not torch.equal(
                    cache.key_valid_mask,
                    reference.key_valid_mask,
                ):
                    raise ValueError("full-attention cache masks must match")
            if position is None:
                raise ValueError("full-attention cache requires position history")
            if (
                position.sequence_length != reference.sequence_length
                or not torch.equal(
                    position.key_valid_mask,
                    reference.key_valid_mask,
                )
            ):
                raise ValueError(
                    "position history and full-attention cache mask/length differ"
                )

        processed = self.processed_media
        if processed is not None:
            if not isinstance(processed, LegacyProcessedPrefix):
                raise TypeError(
                    "processed_media must be LegacyProcessedPrefix or None"
                )
            processed = processed.clone_detached()
            if processed.batch_size != batch_size:
                raise ValueError("processed media and state batch sizes differ")

        partitions: dict[str, StatePartition | None] = {}
        for name in _GENERIC_PARTITIONS:
            partition = getattr(self, name)
            if partition is None:
                partitions[name] = None
                continue
            if not isinstance(partition, StatePartition):
                raise TypeError(f"{name} must implement StatePartition")
            if type(partition.batch_size) is not int:
                raise TypeError(f"{name}.batch_size must be an integer")
            if partition.batch_size != batch_size:
                raise ValueError(f"{name} and state batch sizes differ")
            cloned = partition.clone_detached()
            if not isinstance(cloned, StatePartition):
                raise TypeError(
                    f"{name}.clone_detached() must return StatePartition"
                )
            if cloned.batch_size != batch_size:
                raise ValueError(f"cloned {name} batch size changed")
            byte_count = cloned.logical_tensor_bytes()
            if type(byte_count) is not int or byte_count < 0:
                raise TypeError(
                    f"{name}.logical_tensor_bytes() must return non-negative int"
                )
            partitions[name] = cloned

        object.__setattr__(self, "seen_tokens", seen)
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "full_attention_kv", full)
        object.__setattr__(self, "swa_kv", swa)
        object.__setattr__(self, "processed_media", processed)
        for name, partition in partitions.items():
            object.__setattr__(self, name, partition)

    @staticmethod
    def _clone_cache_mapping(
        value: object,
        expected_type: type[AttentionKV] | type[SlidingWindowKV],
        name: str,
    ) -> Mapping[int, AttentionKV] | Mapping[int, SlidingWindowKV]:
        if not isinstance(value, Mapping):
            raise TypeError(f"{name} must be a mapping")
        copied: dict[int, AttentionKV | SlidingWindowKV] = {}
        for layer_index, cache in value.items():
            if type(layer_index) is not int:
                raise TypeError(f"{name} layer indices must be integers")
            if layer_index < 0:
                raise ValueError(f"{name} layer indices must be non-negative")
            if not isinstance(cache, expected_type):
                raise TypeError(
                    f"{name}[{layer_index}] must be {expected_type.__name__}"
                )
            copied[layer_index] = cache.clone_detached()
        return MappingProxyType(dict(sorted(copied.items())))

    @staticmethod
    def _validate_cache_mapping(
        caches: Mapping[int, AttentionKV] | Mapping[int, SlidingWindowKV],
        batch_size: int,
        device: torch.device,
        label: str,
    ) -> None:
        for cache in caches.values():
            if cache.batch_size != batch_size:
                raise ValueError(f"{label} cache and state batch sizes differ")
            if cache.key.device != device:
                raise ValueError(f"{label} cache and state devices differ")

    @classmethod
    def empty(
        cls,
        owner: StateOwner,
        *,
        batch_size: int,
        device: torch.device | str | None = None,
    ) -> DecoderState:
        if type(batch_size) is not int:
            raise TypeError("batch_size must be an integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        resolved_device = torch.device("cpu" if device is None else device)
        return cls(
            owner=owner,
            seen_tokens=torch.zeros(
                batch_size,
                dtype=torch.long,
                device=resolved_device,
            ),
            position=None,
            full_attention_kv={},
            swa_kv={},
            processed_media=None,
            gdn_state=None,
            talker_state=None,
            mtp_state=None,
            codec_state=None,
        )

    @property
    def batch_size(self) -> int:
        return self.seen_tokens.shape[0]

    @property
    def device(self) -> torch.device:
        return self.seen_tokens.device

    def assert_owner(self, owner: StateOwner) -> None:
        if not isinstance(owner, StateOwner):
            raise TypeError("owner must be StateOwner")
        if not secrets.compare_digest(self.owner.nonce, owner.nonce):
            raise CacheCapabilityError(
                CacheErrorCode.STATE_OWNER_MISMATCH,
                "decoder state belongs to a different request owner",
            )

    def clone_detached(self) -> DecoderState:
        return self._derive()

    def _derive(self, **changes: object) -> DecoderState:
        values = {
            "owner": self.owner,
            "seen_tokens": self.seen_tokens,
            "position": self.position,
            "full_attention_kv": self.full_attention_kv,
            "swa_kv": self.swa_kv,
            "processed_media": self.processed_media,
            "gdn_state": self.gdn_state,
            "talker_state": self.talker_state,
            "mtp_state": self.mtp_state,
            "codec_state": self.codec_state,
        }
        unknown = set(changes) - set(values)
        if unknown:
            raise TypeError(f"unknown DecoderState fields: {sorted(unknown)}")
        values.update(changes)
        return DecoderState(**values)

    def advance_seen_tokens(self, valid_count_delta: torch.Tensor) -> DecoderState:
        delta = _require_long_vector(
            valid_count_delta,
            "valid_count_delta",
            batch_size=self.batch_size,
        )
        if delta.device != self.device:
            raise ValueError("valid_count_delta and state must share a device")
        return self._derive(seen_tokens=self.seen_tokens + delta)

    def with_full_attention(
        self,
        caches: Mapping[int, AttentionKV],
        *,
        position: DecoderPositionState | None = None,
    ) -> DecoderState:
        return self._derive(
            full_attention_kv=caches,
            position=self.position if position is None else position,
        )

    def with_processed_media(
        self,
        processed_media: LegacyProcessedPrefix,
    ) -> DecoderState:
        return self._derive(processed_media=processed_media)

    def validate_full_attention_layers(
        self,
        expected_layer_indices: tuple[int, ...],
    ) -> None:
        if not isinstance(expected_layer_indices, tuple):
            raise TypeError("expected_layer_indices must be a tuple")
        if any(type(index) is not int for index in expected_layer_indices):
            raise TypeError("expected layer indices must be integers")
        if tuple(sorted(set(expected_layer_indices))) != expected_layer_indices:
            raise ValueError("expected layer indices must be sorted and unique")
        if tuple(self.full_attention_kv) != expected_layer_indices:
            raise ValueError(
                "full-attention cache layer set/order does not match the model"
            )

    @staticmethod
    def _partition_names() -> tuple[str, ...]:
        return (
            "seen_tokens",
            "position",
            "full_attention_kv",
            "swa_kv",
            "processed_media",
            *_GENERIC_PARTITIONS,
        )

    def _selected_partition_names(self, partition: str | None) -> tuple[str, ...]:
        if partition is None:
            return self._partition_names()
        if type(partition) is not str:
            raise TypeError("partition must be a string or None")
        if partition not in self._partition_names():
            raise ValueError(f"unknown state partition: {partition}")
        return (partition,)

    def _known_partition_tensors(self, name: str) -> tuple[torch.Tensor, ...]:
        if name == "seen_tokens":
            return (self.seen_tokens,)
        if name == "position":
            if self.position is None:
                return ()
            cursor = self.position.continuation
            cursor_tensors = (
                (cursor.next_storage_position,)
                if isinstance(cursor, LegacyPositionCursor)
                else (cursor.next_text_position, cursor.rope_deltas)
            )
            return (
                self.position.cached.position_ids,
                self.position.cached.rope_deltas,
                self.position.key_valid_mask,
                *cursor_tensors,
            )
        if name == "full_attention_kv":
            return tuple(
                tensor
                for cache in self.full_attention_kv.values()
                for tensor in (cache.key, cache.value, cache.key_valid_mask)
            )
        if name == "swa_kv":
            return tuple(
                tensor
                for cache in self.swa_kv.values()
                for tensor in (
                    cache.key,
                    cache.value,
                    cache.key_valid_mask,
                    cache.position.position_ids,
                    cache.position.rope_deltas,
                )
            )
        return ()

    def logical_tensor_bytes(self, partition: str | None = None) -> int:
        total = 0
        for name in self._selected_partition_names(partition):
            total += sum(
                _tensor_bytes(tensor)
                for tensor in self._known_partition_tensors(name)
            )
            if name in _GENERIC_PARTITIONS:
                value = getattr(self, name)
                if value is not None:
                    total += value.logical_tensor_bytes()
        return total

    def unique_allocated_bytes(self, partition: str | None = None) -> int:
        total = 0
        seen_storage: set[tuple[str, int | None, int]] = set()
        for name in self._selected_partition_names(partition):
            for tensor in self._known_partition_tensors(name):
                storage = tensor.untyped_storage()
                key = (
                    tensor.device.type,
                    tensor.device.index,
                    storage.data_ptr(),
                )
                if key not in seen_storage:
                    seen_storage.add(key)
                    total += _tensor_bytes(tensor)
            if name in _GENERIC_PARTITIONS:
                value = getattr(self, name)
                if value is not None:
                    total += value.logical_tensor_bytes()
        return total


__all__ = [
    "AttentionKV",
    "DecoderPositionState",
    "DecoderState",
    "LegacyPositionCursor",
    "LegacyProcessedPrefix",
    "Qwen3DisjointPositionCursor",
    "SlidingWindowKV",
    "StateOwner",
    "StatePartition",
]
