from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

from qwen3_omni_pretrain.multimodal.types import PositionBatch
from qwen3_omni_pretrain.runtime.state import DecoderState, StateOwner


def _require_tensor(value: object, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    return value


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


def _validate_floating(tensor: torch.Tensor, name: str) -> None:
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must have a floating dtype")
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} must contain finite values")


@dataclass(frozen=True)
class LegacyMediaPrefillInputs:
    pixel_values: torch.Tensor | None
    audio_values: torch.Tensor | None
    has_image: torch.Tensor
    has_audio: torch.Tensor

    def __post_init__(self) -> None:
        has_image = _require_bool_mask(self.has_image, "has_image")
        has_audio = _require_bool_mask(self.has_audio, "has_audio")
        if has_image.ndim != 1 or has_image.numel() == 0:
            raise ValueError("has_image must have non-empty shape [B]")
        if has_audio.shape != has_image.shape:
            raise ValueError("has_audio must have the same [B] shape as has_image")
        if has_audio.device != has_image.device:
            raise ValueError("media flags must share a device")
        self._validate_payload(
            self.pixel_values,
            has_image,
            "pixel_values",
        )
        self._validate_payload(
            self.audio_values,
            has_audio,
            "audio_values",
        )

    @staticmethod
    def _validate_payload(
        value: torch.Tensor | None,
        present: torch.Tensor,
        name: str,
    ) -> None:
        any_present = bool(present.any().item())
        if not any_present:
            if value is not None:
                raise ValueError(f"{name} must be None when every flag is false")
            return
        if value is None:
            raise ValueError(f"{name} is required when any flag is true")
        tensor = _require_tensor(value, name)
        if tensor.ndim < 2 or tensor.shape[0] != present.shape[0]:
            raise ValueError(f"{name} must have a matching leading batch axis")
        if tensor.device != present.device:
            raise ValueError(f"{name} and its presence flag must share a device")
        _validate_floating(tensor, name)

    @property
    def batch_size(self) -> int:
        return self.has_image.shape[0]

    @property
    def device(self) -> torch.device:
        return self.has_image.device


@dataclass(frozen=True)
class ModelPrefillInputs:
    input_ids: torch.Tensor | None
    inputs_embeds: torch.Tensor | None
    key_valid_mask: torch.Tensor
    position_batch: PositionBatch | None
    media: LegacyMediaPrefillInputs | None = None

    def __post_init__(self) -> None:
        if (self.input_ids is None) == (self.inputs_embeds is None):
            raise ValueError("exactly one of input_ids or inputs_embeds is required")
        mask = _require_bool_mask(self.key_valid_mask, "key_valid_mask")
        if mask.ndim != 2 or mask.shape[0] == 0 or mask.shape[1] == 0:
            raise ValueError("key_valid_mask must have non-empty shape [B, Q]")
        batch_size, query_length = mask.shape
        if self.input_ids is not None:
            input_ids = _require_tensor(self.input_ids, "input_ids")
            if input_ids.dtype is not torch.long:
                raise TypeError("input_ids must have dtype torch.long")
            if input_ids.shape != (batch_size, query_length):
                raise ValueError("input_ids must have shape [B, Q]")
            if input_ids.device != mask.device:
                raise ValueError("input_ids and key_valid_mask must share a device")
            if bool((input_ids < 0).any().item()):
                raise ValueError("input_ids must be non-negative")
        else:
            embeds = _require_tensor(self.inputs_embeds, "inputs_embeds")
            if embeds.ndim != 3 or embeds.shape[:2] != (
                batch_size,
                query_length,
            ):
                raise ValueError("inputs_embeds must have shape [B, Q, H]")
            if embeds.shape[2] == 0:
                raise ValueError("inputs_embeds hidden dimension must be positive")
            if embeds.device != mask.device:
                raise ValueError(
                    "inputs_embeds and key_valid_mask must share a device"
                )
            _validate_floating(embeds, "inputs_embeds")
        if self.position_batch is not None:
            if not isinstance(self.position_batch, PositionBatch):
                raise TypeError("position_batch must be PositionBatch or None")
            self.position_batch.validate(mask)
        if self.media is not None:
            if not isinstance(self.media, LegacyMediaPrefillInputs):
                raise TypeError("media must be LegacyMediaPrefillInputs or None")
            if self.media.batch_size != batch_size:
                raise ValueError("media and text batch sizes differ")
            if self.media.device != mask.device:
                raise ValueError("media and text inputs must share a device")

    @property
    def batch_size(self) -> int:
        return self.key_valid_mask.shape[0]

    @property
    def query_length(self) -> int:
        return self.key_valid_mask.shape[1]

    @property
    def device(self) -> torch.device:
        return self.key_valid_mask.device


@dataclass(frozen=True)
class ModelDecodeInputs:
    token_ids: torch.Tensor
    current_key_valid_mask: torch.Tensor
    position_batch: PositionBatch
    decoder_state: DecoderState

    def __post_init__(self) -> None:
        token_ids = _require_tensor(self.token_ids, "token_ids")
        if token_ids.dtype is not torch.long:
            raise TypeError("token_ids must have dtype torch.long")
        if token_ids.ndim != 2 or token_ids.shape[0] == 0 or token_ids.shape[1] == 0:
            raise ValueError("token_ids must have non-empty shape [B, Q]")
        if bool((token_ids < 0).any().item()):
            raise ValueError("token_ids must be non-negative")
        mask = _require_bool_mask(
            self.current_key_valid_mask,
            "current_key_valid_mask",
            shape=token_ids.shape,
        )
        if mask.device != token_ids.device:
            raise ValueError("token_ids and current mask must share a device")
        if not isinstance(self.position_batch, PositionBatch):
            raise TypeError("position_batch must be PositionBatch")
        self.position_batch.validate(mask)
        if not isinstance(self.decoder_state, DecoderState):
            raise TypeError("decoder_state must be DecoderState")
        if self.decoder_state.batch_size != token_ids.shape[0]:
            raise ValueError("decoder state and token batch sizes differ")
        if self.decoder_state.device != token_ids.device:
            raise ValueError("decoder state and token IDs must share a device")

    @property
    def batch_size(self) -> int:
        return self.token_ids.shape[0]

    @property
    def query_length(self) -> int:
        return self.token_ids.shape[1]

    @property
    def device(self) -> torch.device:
        return self.token_ids.device


def _validate_optional_scalar(value: torch.Tensor | None, name: str) -> None:
    if value is None:
        return
    tensor = _require_tensor(value, name)
    if tensor.ndim != 0:
        raise ValueError(f"{name} must be a scalar tensor")
    _validate_floating(tensor, name)


@dataclass(frozen=True)
class CausalLMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None
    ce_loss: torch.Tensor | None
    aux_loss: torch.Tensor | None
    decoder_state: DecoderState | None
    hidden_states: tuple[torch.Tensor, ...] | None

    def __post_init__(self) -> None:
        logits = _require_tensor(self.logits, "logits")
        if logits.ndim != 3 or any(size <= 0 for size in logits.shape):
            raise ValueError("logits must have non-empty shape [B, Q, V]")
        _validate_floating(logits, "logits")
        for name in ("loss", "ce_loss", "aux_loss"):
            value = getattr(self, name)
            _validate_optional_scalar(value, name)
            if value is not None and value.device != logits.device:
                raise ValueError(f"{name} and logits must share a device")
        if self.decoder_state is not None:
            if not isinstance(self.decoder_state, DecoderState):
                raise TypeError("decoder_state must be DecoderState or None")
            if self.decoder_state.batch_size != logits.shape[0]:
                raise ValueError("decoder_state and logits batch sizes differ")
            if self.decoder_state.device != logits.device:
                raise ValueError("decoder_state and logits must share a device")
        if self.hidden_states is not None:
            if not isinstance(self.hidden_states, tuple):
                raise TypeError("hidden_states must be a tuple or None")
            for hidden in self.hidden_states:
                hidden = _require_tensor(hidden, "hidden_states entry")
                if hidden.ndim != 3 or hidden.shape[:2] != logits.shape[:2]:
                    raise ValueError(
                        "hidden_states entries must share logits B/Q axes"
                    )
                if hidden.device != logits.device:
                    raise ValueError("hidden_states and logits must share a device")
                _validate_floating(hidden, "hidden_states entry")


@runtime_checkable
class CacheCapableModel(Protocol):
    def prefill(
        self,
        *,
        inputs: ModelPrefillInputs,
        owner: StateOwner,
        use_cache: bool,
    ) -> CausalLMOutput: ...

    def decode(
        self,
        *,
        inputs: ModelDecodeInputs,
        owner: StateOwner,
    ) -> CausalLMOutput: ...


def legacy_position_ids(
    position_batch: PositionBatch,
    key_valid_mask: torch.Tensor,
    *,
    max_position_embeddings: int,
) -> torch.Tensor:
    if type(max_position_embeddings) is not int:
        raise TypeError("max_position_embeddings must be an integer")
    if max_position_embeddings <= 0:
        raise ValueError("max_position_embeddings must be positive")
    mask = _require_bool_mask(key_valid_mask, "key_valid_mask")
    if not isinstance(position_batch, PositionBatch):
        raise TypeError("position_batch must be PositionBatch")
    position_batch.validate(mask)
    if position_batch.axis_names != ("sequence",):
        raise ValueError("legacy position adapter accepts exactly one sequence axis")
    values = position_batch.position_ids[0]
    valid_values = values.masked_select(mask)
    if values.is_floating_point() and not torch.equal(
        valid_values,
        torch.round(valid_values),
    ):
        raise ValueError("legacy positions must be integral")
    if valid_values.numel() and int(valid_values.max().item()) >= max_position_embeddings:
        raise ValueError("legacy position exceeds max_position_embeddings")
    return values.to(dtype=torch.long).detach().clone()


__all__ = [
    "CacheCapableModel",
    "CausalLMOutput",
    "LegacyMediaPrefillInputs",
    "ModelDecodeInputs",
    "ModelPrefillInputs",
    "legacy_position_ids",
]
