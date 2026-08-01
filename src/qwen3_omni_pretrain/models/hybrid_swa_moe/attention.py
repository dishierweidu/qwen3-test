"""Eager full and sliding-window attention for mechanism experiments."""

from __future__ import annotations

import math

import torch
from torch import nn

from qwen3_omni_pretrain.models.hybrid_swa_moe.configuration_hybrid_swa_moe import (
    HybridSwaMoeConfig,
)
from qwen3_omni_pretrain.multimodal.types import PositionBatch
from qwen3_omni_pretrain.runtime.state import AttentionKV, SlidingWindowKV


def _text_position_batch(position_ids: torch.Tensor) -> PositionBatch:
    return PositionBatch(
        position_ids=position_ids.unsqueeze(0),
        rope_deltas=torch.zeros(
            (position_ids.shape[0], 1),
            dtype=position_ids.dtype,
            device=position_ids.device,
        ),
        axis_names=("text",),
    )


def _apply_partial_rope(
    tensor: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    rotary_dim: int,
    theta: float,
) -> torch.Tensor:
    """Rotate an interleaved prefix of ``[B, H, T, D]`` Q/K tensors."""

    rotated = tensor[..., :rotary_dim]
    passthrough = tensor[..., rotary_dim:]
    frequencies = torch.arange(
        0,
        rotary_dim,
        2,
        dtype=torch.float32,
        device=tensor.device,
    )
    inverse = theta ** (-frequencies / rotary_dim)
    angles = position_ids.to(torch.float32).unsqueeze(-1) * inverse
    cos = angles.cos().unsqueeze(1).to(dtype=tensor.dtype)
    sin = angles.sin().unsqueeze(1).to(dtype=tensor.dtype)
    even = rotated[..., 0::2]
    odd = rotated[..., 1::2]
    rope = torch.stack(
        (even * cos - odd * sin, even * sin + odd * cos),
        dim=-1,
    ).flatten(-2)
    return torch.cat((rope, passthrough), dim=-1)


def _as_bool_mask(
    value: torch.Tensor,
    *,
    name: str,
    shape: tuple[int, int],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {list(shape)}")
    if value.dtype is torch.bool:
        return value
    if value.is_floating_point() or value.dtype in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        if not bool(((value == 0) | (value == 1)).all().item()):
            raise ValueError(f"{name} values must be 0 or 1")
        return value.to(torch.bool)
    raise TypeError(f"{name} must have boolean or numeric mask dtype")


def _empty_sliding_cache(
    *,
    key: torch.Tensor,
    value: torch.Tensor,
    position_ids: torch.Tensor,
    window_size: int,
) -> SlidingWindowKV:
    batch_size, kv_heads, _, qk_dim = key.shape
    v_dim = value.shape[-1]
    empty_mask = torch.zeros(
        (batch_size, 0), dtype=torch.bool, device=key.device
    )
    return SlidingWindowKV(
        key=torch.empty(
            (batch_size, kv_heads, 0, qk_dim),
            dtype=key.dtype,
            device=key.device,
        ),
        value=torch.empty(
            (batch_size, kv_heads, 0, v_dim),
            dtype=value.dtype,
            device=value.device,
        ),
        key_valid_mask=empty_mask,
        position=PositionBatch(
            position_ids=torch.empty(
                (1, batch_size, 0),
                dtype=position_ids.dtype,
                device=position_ids.device,
            ),
            rope_deltas=torch.zeros(
                (batch_size, 1),
                dtype=position_ids.dtype,
                device=position_ids.device,
            ),
            axis_names=("text",),
        ),
        window_size=window_size,
    )


class HybridSelfAttention(nn.Module):
    """Fused-QKV eager attention with strict persistent SWA state."""

    def __init__(
        self,
        config: HybridSwaMoeConfig,
        *,
        layer_index: int,
        attention_type: str | None = None,
        num_key_value_heads: int | None = None,
        rope_theta: float | None = None,
        attention_sink: bool | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(config, HybridSwaMoeConfig):
            raise TypeError("config must be HybridSwaMoeConfig")
        if type(layer_index) is not int or not 0 <= layer_index < config.num_hidden_layers:
            raise ValueError("layer_index is outside the configured layer range")
        selected_type = (
            config.attention_layer_types[layer_index]
            if attention_type is None
            else attention_type
        )
        if selected_type not in {"full", "swa"}:
            raise ValueError("attention_type must be full or swa")
        default_kv_heads = (
            config.full_num_key_value_heads
            if selected_type == "full"
            else config.swa_num_key_value_heads
        )
        kv_heads = default_kv_heads if num_key_value_heads is None else num_key_value_heads
        if type(kv_heads) is not int or kv_heads <= 0:
            raise ValueError("num_key_value_heads must be positive")
        if config.num_attention_heads % kv_heads:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        default_theta = (
            config.full_rope_theta if selected_type == "full" else config.swa_rope_theta
        )
        selected_theta = default_theta if rope_theta is None else float(rope_theta)
        if not math.isfinite(selected_theta) or selected_theta <= 0:
            raise ValueError("rope_theta must be finite and positive")
        sink_enabled = config.attention_sink if attention_sink is None else attention_sink
        if type(sink_enabled) is not bool:
            raise TypeError("attention_sink must be boolean")

        self.layer_index = layer_index
        self.attention_type = selected_type
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = kv_heads
        self.qk_head_dim = config.qk_head_dim
        self.v_head_dim = config.v_head_dim
        self.rotary_dim = config.rotary_dim
        self.rope_theta = selected_theta
        self.window_size = config.swa_window_size
        self.value_scale = config.value_scale
        self.scaling = self.qk_head_dim ** -0.5

        q_size = self.num_attention_heads * self.qk_head_dim
        k_size = self.num_key_value_heads * self.qk_head_dim
        v_size = self.num_key_value_heads * self.v_head_dim
        self.qkv_proj = nn.Linear(
            self.hidden_size,
            q_size + k_size + v_size,
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
        )
        if sink_enabled:
            self.attention_sink_bias = nn.Parameter(
                torch.zeros(self.num_attention_heads)
            )
        else:
            self.register_parameter("attention_sink_bias", None)

    @property
    def cache_type(self) -> str:
        return "sliding-window-kv-cache" if self.attention_type == "swa" else "kv-cache"

    def _project(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, _ = hidden_states.shape
        packed = self.qkv_proj(hidden_states)
        q_size = self.num_attention_heads * self.qk_head_dim
        k_size = self.num_key_value_heads * self.qk_head_dim
        query, key, value = packed.split(
            (q_size, k_size, self.num_key_value_heads * self.v_head_dim),
            dim=-1,
        )
        query = query.view(
            batch_size,
            sequence_length,
            self.num_attention_heads,
            self.qk_head_dim,
        ).transpose(1, 2)
        key = key.view(
            batch_size,
            sequence_length,
            self.num_key_value_heads,
            self.qk_head_dim,
        ).transpose(1, 2)
        value = value.view(
            batch_size,
            sequence_length,
            self.num_key_value_heads,
            self.v_head_dim,
        ).transpose(1, 2)
        query = _apply_partial_rope(
            query,
            position_ids,
            rotary_dim=self.rotary_dim,
            theta=self.rope_theta,
        )
        key = _apply_partial_rope(
            key,
            position_ids,
            rotary_dim=self.rotary_dim,
            theta=self.rope_theta,
        )
        if self.value_scale is not None:
            value = value * self.value_scale
        return query, key, value

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor,
        current_key_valid_mask: torch.BoolTensor | None = None,
        cache: AttentionKV | SlidingWindowKV | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, AttentionKV | SlidingWindowKV | None]:
        if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [B, Q, H]")
        batch_size, query_length, hidden_size = hidden_states.shape
        if hidden_size != self.hidden_size or query_length <= 0:
            raise ValueError("hidden_states shape contradicts attention configuration")
        if not hidden_states.is_floating_point():
            raise TypeError("hidden_states must have a floating dtype")
        if (
            not isinstance(position_ids, torch.Tensor)
            or position_ids.dtype is not torch.long
            or position_ids.shape != (batch_size, query_length)
        ):
            raise ValueError("position_ids must have dtype long and shape [B, Q]")
        if position_ids.device != hidden_states.device:
            raise ValueError("position_ids and hidden_states must share a device")

        expected_cache = SlidingWindowKV if self.attention_type == "swa" else AttentionKV
        if cache is not None and not isinstance(cache, expected_cache):
            raise TypeError(
                f"{self.attention_type} attention requires {expected_cache.__name__}"
            )
        if cache is not None:
            if cache.batch_size != batch_size or cache.key.device != hidden_states.device:
                raise ValueError("cache batch/device does not match hidden_states")
            if cache.key.dtype != hidden_states.dtype:
                raise ValueError("cache dtype does not match hidden_states")
            if cache.key.shape[1] != self.num_key_value_heads:
                raise ValueError("cache KV-head count does not match attention")

        prefix_length = 0 if cache is None else cache.sequence_length
        total_length = prefix_length + query_length
        if current_key_valid_mask is None:
            if attention_mask is not None and attention_mask.shape == (
                batch_size,
                total_length,
            ):
                current_mask = _as_bool_mask(
                    attention_mask[:, -query_length:],
                    name="current attention mask",
                    shape=(batch_size, query_length),
                )
            elif attention_mask is not None and attention_mask.shape == (
                batch_size,
                query_length,
            ):
                current_mask = _as_bool_mask(
                    attention_mask,
                    name="attention_mask",
                    shape=(batch_size, query_length),
                )
            else:
                current_mask = torch.ones(
                    (batch_size, query_length),
                    dtype=torch.bool,
                    device=hidden_states.device,
                )
        else:
            current_mask = _as_bool_mask(
                current_key_valid_mask,
                name="current_key_valid_mask",
                shape=(batch_size, query_length),
            )
            if attention_mask is not None and attention_mask.shape == (
                batch_size,
                query_length,
            ):
                current_mask = current_mask & _as_bool_mask(
                    attention_mask,
                    name="attention_mask",
                    shape=(batch_size, query_length),
                )
        if current_mask.device != hidden_states.device:
            raise ValueError("attention masks and hidden_states must share a device")

        query, current_key, current_value = self._project(
            hidden_states, position_ids
        )
        if cache is None:
            all_key = current_key
            all_value = current_value
            all_key_mask = current_mask
            all_key_positions = position_ids
        else:
            all_key = torch.cat((cache.key, current_key), dim=2)
            all_value = torch.cat((cache.value, current_value), dim=2)
            all_key_mask = torch.cat((cache.key_valid_mask, current_mask), dim=1)
            if isinstance(cache, SlidingWindowKV):
                cached_positions = cache.position.position_ids[0]
            else:
                # Full attention stores every slot.  Standard text positions are
                # reconstructed from valid-token counts; padded slots remain masked.
                cached_positions = (
                    cache.key_valid_mask.to(torch.long).cumsum(dim=1) - 1
                ).clamp_min(0)
            all_key_positions = torch.cat(
                (cached_positions, position_ids), dim=1
            )

        if attention_mask is not None and attention_mask.shape == (
            batch_size,
            total_length,
        ):
            all_key_mask = all_key_mask & _as_bool_mask(
                attention_mask,
                name="attention_mask",
                shape=(batch_size, total_length),
            )
        elif attention_mask is not None and attention_mask.shape not in {
            (batch_size, query_length),
            (batch_size, total_length),
        }:
            raise ValueError("attention_mask must describe current or total keys")

        repeat_factor = self.num_attention_heads // self.num_key_value_heads
        repeated_key = all_key.repeat_interleave(repeat_factor, dim=1)
        repeated_value = all_value.repeat_interleave(repeat_factor, dim=1)
        scores = torch.matmul(query, repeated_key.transpose(-1, -2)) * self.scaling

        causal = all_key_positions[:, None, :] <= position_ids[:, :, None]
        if self.attention_type == "swa":
            causal = causal & (
                all_key_positions[:, None, :]
                > position_ids[:, :, None] - self.window_size
            )
        allowed = causal & all_key_mask[:, None, :]
        # Invalid query slots must not create NaNs even when every key is masked.
        safe_allowed = allowed | (~current_mask[:, :, None])
        scores = scores.masked_fill(~safe_allowed[:, None, :, :], float("-inf"))

        if self.attention_sink_bias is not None:
            sink = self.attention_sink_bias.view(1, -1, 1, 1).expand(
                scores.shape[0], -1, scores.shape[2], 1
            )
            scores = torch.cat((scores, sink.to(scores.dtype)), dim=-1)
        probabilities = torch.softmax(scores.float(), dim=-1).to(query.dtype)
        if self.attention_sink_bias is not None:
            probabilities = probabilities[..., :-1]
        attended = torch.matmul(probabilities, repeated_value)
        attended = attended * current_mask[:, None, :, None].to(attended.dtype)
        attended = attended.transpose(1, 2).reshape(
            batch_size,
            query_length,
            self.num_attention_heads * self.v_head_dim,
        )
        output = self.o_proj(attended)

        next_cache: AttentionKV | SlidingWindowKV | None = None
        if use_cache:
            if self.attention_type == "full":
                next_cache = (
                    AttentionKV(current_key, current_value, current_mask)
                    if cache is None
                    else cache.append(current_key, current_value, current_mask)
                )
            else:
                sliding = (
                    _empty_sliding_cache(
                        key=current_key,
                        value=current_value,
                        position_ids=position_ids,
                        window_size=self.window_size,
                    )
                    if cache is None
                    else cache
                )
                assert isinstance(sliding, SlidingWindowKV)
                next_cache = sliding.append(
                    key=current_key,
                    value=current_value,
                    key_valid_mask=current_mask,
                    position=_text_position_batch(position_ids),
                )
        return output, next_cache


__all__ = ["HybridSelfAttention"]
