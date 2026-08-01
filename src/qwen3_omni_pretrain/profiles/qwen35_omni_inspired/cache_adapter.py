from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import torch

from qwen3_omni_pretrain.multimodal.types import PositionBatch
from qwen3_omni_pretrain.runtime.state import (
    AttentionKV,
    DecoderPositionState,
    DecoderState,
    LegacyPositionCursor,
    Qwen3DisjointPositionCursor,
    StateOwner,
)


def _clone(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().clone(memory_format=torch.contiguous_format)


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _text_config(config: object) -> object:
    if hasattr(config, "backbone_text_config"):
        raw = getattr(config, "backbone_text_config")
        from transformers import Qwen3_5MoeTextConfig

        return Qwen3_5MoeTextConfig(**dict(raw))
    nested = getattr(config, "text_config", None)
    return nested if nested is not None else config


def _layer_types(config: object) -> tuple[str, ...]:
    raw = getattr(config, "layer_types", None)
    if not isinstance(raw, (list, tuple)) or not raw:
        raise TypeError("Qwen3.5 public config must expose layer_types")
    result = tuple(raw)
    if any(value not in {"linear_attention", "full_attention"} for value in result):
        raise ValueError("Qwen3.5 config contains an unsupported layer type")
    return result


def expected_gdn_conv_shape(config: object, batch_size: int) -> tuple[int, int, int]:
    config = _text_config(config)
    key_dim = int(getattr(config, "linear_num_key_heads")) * int(
        getattr(config, "linear_key_head_dim")
    )
    value_dim = int(getattr(config, "linear_num_value_heads")) * int(
        getattr(config, "linear_value_head_dim")
    )
    return (
        batch_size,
        key_dim * 2 + value_dim,
        int(getattr(config, "linear_conv_kernel_dim")),
    )


def expected_gdn_recurrent_shape(
    config: object,
    batch_size: int,
) -> tuple[int, int, int, int]:
    config = _text_config(config)
    return (
        batch_size,
        int(getattr(config, "linear_num_value_heads")),
        int(getattr(config, "linear_key_head_dim")),
        int(getattr(config, "linear_value_head_dim")),
    )


def _clone_tensor_mapping(
    value: Mapping[int, torch.Tensor],
    name: str,
) -> Mapping[int, torch.Tensor]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    result: dict[int, torch.Tensor] = {}
    for index, tensor in value.items():
        if type(index) is not int or index < 0:
            raise ValueError(f"{name} layer indices must be non-negative integers")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name}[{index}] must be a tensor")
        if not tensor.is_floating_point():
            raise TypeError(f"{name}[{index}] must have a floating dtype")
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"{name}[{index}] must contain finite values")
        result[index] = _clone(tensor)
    return MappingProxyType(dict(sorted(result.items())))


@dataclass(frozen=True)
class Qwen35GDNState:
    gdn_convolution_state: Mapping[int, torch.Tensor]
    gdn_recurrent_matrix_state: Mapping[int, torch.Tensor]

    def __post_init__(self) -> None:
        convolution = _clone_tensor_mapping(
            self.gdn_convolution_state,
            "gdn_convolution_state",
        )
        recurrent = _clone_tensor_mapping(
            self.gdn_recurrent_matrix_state,
            "gdn_recurrent_matrix_state",
        )
        if set(convolution) != set(recurrent):
            raise ValueError("GDN convolution and recurrent layer sets must match")
        if not convolution:
            raise ValueError("Qwen35GDNState must contain at least one layer")
        batches = {tensor.shape[0] for tensor in (*convolution.values(), *recurrent.values())}
        if len(batches) != 1 or next(iter(batches)) <= 0:
            raise ValueError("all GDN states must share a positive batch size")
        devices = {tensor.device for tensor in (*convolution.values(), *recurrent.values())}
        if len(devices) != 1:
            raise ValueError("all GDN states must share a device")
        object.__setattr__(self, "gdn_convolution_state", convolution)
        object.__setattr__(self, "gdn_recurrent_matrix_state", recurrent)

    @property
    def batch_size(self) -> int:
        return next(iter(self.gdn_convolution_state.values())).shape[0]

    @property
    def device(self) -> torch.device:
        return next(iter(self.gdn_convolution_state.values())).device

    def clone_detached(self) -> Qwen35GDNState:
        return Qwen35GDNState(
            self.gdn_convolution_state,
            self.gdn_recurrent_matrix_state,
        )

    def logical_tensor_bytes(self) -> int:
        return sum(
            _tensor_bytes(tensor)
            for tensor in (
                *self.gdn_convolution_state.values(),
                *self.gdn_recurrent_matrix_state.values(),
            )
        )


def _validate_request_id(request_id: object) -> str:
    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError("request_id must be a non-empty string")
    return request_id


def _validate_history(
    *,
    seen_tokens: torch.Tensor,
    key_valid_mask: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(seen_tokens, torch.Tensor) or seen_tokens.dtype is not torch.long:
        raise TypeError("seen_tokens must have dtype torch.long")
    if seen_tokens.ndim != 1 or seen_tokens.numel() == 0:
        raise ValueError("seen_tokens must have non-empty shape [B]")
    if bool((seen_tokens < 0).any().item()):
        raise ValueError("seen_tokens must be non-negative")
    if not isinstance(key_valid_mask, torch.Tensor) or key_valid_mask.dtype is not torch.bool:
        raise TypeError("key_valid_mask must have dtype torch.bool")
    if key_valid_mask.ndim != 2 or key_valid_mask.shape[0] != seen_tokens.shape[0]:
        raise ValueError("key_valid_mask must have shape [B, S]")
    if not torch.equal(key_valid_mask.sum(dim=1).to(torch.long), seen_tokens):
        raise ValueError("key_valid_mask valid counts must equal seen_tokens")
    if not isinstance(position_ids, torch.Tensor):
        raise TypeError("position_ids must be a tensor")
    if position_ids.ndim == 2:
        position_ids = position_ids.unsqueeze(0)
    if position_ids.ndim != 3 or position_ids.shape[1:] != key_valid_mask.shape:
        raise ValueError("position_ids must have shape [B,S] or [A,B,S]")
    if position_ids.shape[0] not in {1, 3}:
        raise ValueError("position_ids must contain one or three axes")
    if position_ids.is_complex() or not (
        position_ids.is_floating_point()
        or position_ids.dtype in {torch.int32, torch.int64}
    ):
        raise TypeError("position_ids must have a real numeric dtype")
    if seen_tokens.device != key_valid_mask.device or position_ids.device != seen_tokens.device:
        raise ValueError("history tensors must share a device")
    invalid = ~key_valid_mask.unsqueeze(0).expand_as(position_ids)
    if bool((position_ids.masked_select(invalid) != 0).any().item()):
        raise ValueError("position_ids must be zero at masked positions")
    return _clone(seen_tokens), _clone(key_valid_mask), _clone(position_ids)


def _position_state(
    position_ids: torch.Tensor,
    key_valid_mask: torch.Tensor,
) -> DecoderPositionState:
    batch_size = key_valid_mask.shape[0]
    rope_deltas = torch.zeros(
        batch_size,
        1,
        dtype=position_ids.dtype,
        device=position_ids.device,
    )
    if position_ids.shape[0] == 1:
        axes = ("sequence",)
        next_positions = torch.zeros(
            batch_size,
            dtype=torch.long,
            device=position_ids.device,
        )
        for row in range(batch_size):
            valid = position_ids[0, row].masked_select(key_valid_mask[row])
            if valid.numel():
                next_positions[row] = int(valid.max().item()) + 1
        continuation = LegacyPositionCursor(next_positions)
    else:
        axes = ("temporal", "height", "width")
        next_positions = torch.zeros(
            batch_size,
            dtype=position_ids.dtype,
            device=position_ids.device,
        )
        for row in range(batch_size):
            valid_mask = key_valid_mask[row].view(1, -1).expand(3, -1)
            valid = position_ids[:, row].masked_select(valid_mask)
            if valid.numel():
                next_positions[row] = valid.max() + 1
        continuation = Qwen3DisjointPositionCursor(
            next_text_position=next_positions,
            rope_deltas=rope_deltas,
            axis_names=axes,
        )
    return DecoderPositionState(
        cached=PositionBatch(position_ids, rope_deltas, axes),
        key_valid_mask=key_valid_mask,
        continuation=continuation,
    )


class Qwen35CacheAdapter:
    def __init__(self, config: object) -> None:
        self.config = _text_config(config)
        self.layer_types = _layer_types(self.config)

    def _validate_gdn_tensor(
        self,
        tensor: torch.Tensor,
        *,
        layer_index: int,
        batch_size: int,
        recurrent: bool,
    ) -> None:
        expected = (
            expected_gdn_recurrent_shape(self.config, batch_size)
            if recurrent
            else expected_gdn_conv_shape(self.config, batch_size)
        )
        label = "recurrent matrix" if recurrent else "convolution"
        if recurrent and tensor.ndim != 4:
            raise ValueError(f"GDN {label} state must be rank-4 [B,H,K,V]")
        if tuple(tensor.shape) != expected:
            raise ValueError(
                f"GDN {label} state for layer {layer_index} must have shape {list(expected)}"
            )
        if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"GDN {label} state must be finite floating point")

    def from_native(
        self,
        native: object,
        *,
        request_id: str,
        seen_tokens: torch.Tensor,
        key_valid_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> DecoderState:
        request_id = _validate_request_id(request_id)
        seen, mask, positions = _validate_history(
            seen_tokens=seen_tokens,
            key_valid_mask=key_valid_mask,
            position_ids=position_ids,
        )
        batch_size, sequence_length = mask.shape
        for name in ("key_cache", "value_cache", "conv_states", "recurrent_states"):
            values = getattr(native, name, None)
            if not isinstance(values, list) or len(values) != len(self.layer_types):
                raise TypeError(f"native cache must expose {name} for every layer")

        full: dict[int, AttentionKV] = {}
        convolution: dict[int, torch.Tensor] = {}
        recurrent: dict[int, torch.Tensor] = {}
        for index, layer_type in enumerate(self.layer_types):
            key = native.key_cache[index]
            value = native.value_cache[index]
            conv = native.conv_states[index]
            matrix = native.recurrent_states[index]
            if layer_type == "full_attention":
                if conv is not None or matrix is not None:
                    raise ValueError("full-attention layer cannot contain GDN state")
                if (key is None) != (value is None):
                    raise ValueError("native full-attention key/value presence differs")
                if key is not None:
                    if key.shape[0] != batch_size or key.shape[2] != sequence_length:
                        raise ValueError("native full-attention cache shape contradicts history")
                    full[index] = AttentionKV(key, value, mask)
                continue
            if key is not None or value is not None:
                raise ValueError("GDN layer cannot contain full-attention KV")
            if (conv is None) != (matrix is None):
                raise ValueError("native GDN convolution/recurrent presence differs")
            if conv is not None:
                self._validate_gdn_tensor(
                    conv,
                    layer_index=index,
                    batch_size=batch_size,
                    recurrent=False,
                )
                self._validate_gdn_tensor(
                    matrix,
                    layer_index=index,
                    batch_size=batch_size,
                    recurrent=True,
                )
                convolution[index] = _clone(conv)
                recurrent[index] = _clone(matrix)

        gdn_state = (
            Qwen35GDNState(convolution, recurrent) if convolution else None
        )
        state = DecoderState(
            owner=StateOwner.fresh(request_id),
            seen_tokens=seen,
            position=_position_state(positions, mask),
            full_attention_kv=full,
            swa_kv={},
            processed_media=None,
            gdn_state=gdn_state,
            talker_state=None,
            mtp_state=None,
            codec_state=None,
        )
        if gdn_state is not None and gdn_state.device != state.device:
            raise ValueError("GDN and decoder history devices differ")
        return state

    def to_native(self, state: DecoderState, *, request_id: str) -> object:
        request_id = _validate_request_id(request_id)
        if not isinstance(state, DecoderState):
            raise TypeError("state must be DecoderState")
        if state.owner.display_request_id != request_id:
            raise ValueError("decoder state belongs to a different request_id")
        if state.position is None:
            raise ValueError("Qwen3.5 cached state requires position history")
        mask = state.position.key_valid_mask
        if not bool(mask.all().item()) or not bool(
            (state.seen_tokens == state.seen_tokens[0]).all().item()
        ):
            raise ValueError(
                "Qwen3.5 native cache requires an equal-length batch without padding"
            )
        if not torch.equal(
            mask.sum(dim=1).to(torch.long),
            state.seen_tokens,
        ):
            raise ValueError("position mask valid counts must equal seen_tokens")
        if state.swa_kv:
            raise ValueError("Qwen3.5 native cache cannot contain SWA state")

        try:
            from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
                Qwen3_5MoeDynamicCache,
            )
        except ImportError as exc:  # pragma: no cover - dependency guard.
            raise RuntimeError(
                "Qwen3.5 cache adapter requires transformers==5.2.0"
            ) from exc
        native = Qwen3_5MoeDynamicCache(self.config)
        gdn_state = state.gdn_state
        if gdn_state is not None and not isinstance(gdn_state, Qwen35GDNState):
            raise TypeError("state.gdn_state must be Qwen35GDNState")
        for index, layer_type in enumerate(self.layer_types):
            if layer_type == "full_attention":
                cache = state.full_attention_kv.get(index)
                if cache is not None:
                    if not torch.equal(cache.key_valid_mask, mask):
                        raise ValueError("full-attention cache masks must match history")
                    native.key_cache[index] = _clone(cache.key)
                    native.value_cache[index] = _clone(cache.value)
                continue
            if index in state.full_attention_kv:
                raise ValueError("GDN layer cannot contain full-attention KV")
            if gdn_state is None:
                continue
            conv = gdn_state.gdn_convolution_state.get(index)
            matrix = gdn_state.gdn_recurrent_matrix_state.get(index)
            if (conv is None) != (matrix is None):
                raise ValueError("GDN convolution/recurrent presence differs")
            if conv is not None:
                self._validate_gdn_tensor(
                    conv,
                    layer_index=index,
                    batch_size=state.batch_size,
                    recurrent=False,
                )
                self._validate_gdn_tensor(
                    matrix,
                    layer_index=index,
                    batch_size=state.batch_size,
                    recurrent=True,
                )
                native.conv_states[index] = _clone(conv)
                native.recurrent_states[index] = _clone(matrix)
        unexpected_gdn = (
            set()
            if gdn_state is None
            else set(gdn_state.gdn_convolution_state)
            - {i for i, kind in enumerate(self.layer_types) if kind == "linear_attention"}
        )
        if unexpected_gdn:
            raise ValueError(f"GDN state has unexpected layers: {sorted(unexpected_gdn)}")
        unexpected_full = set(state.full_attention_kv) - {
            i for i, kind in enumerate(self.layer_types) if kind == "full_attention"
        }
        if unexpected_full:
            raise ValueError(
                f"full-attention state has unexpected layers: {sorted(unexpected_full)}"
            )
        return native


__all__ = [
    "Qwen35CacheAdapter",
    "Qwen35GDNState",
    "expected_gdn_conv_shape",
    "expected_gdn_recurrent_shape",
]
