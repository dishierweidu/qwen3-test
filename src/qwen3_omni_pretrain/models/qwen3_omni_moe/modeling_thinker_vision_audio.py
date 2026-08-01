from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel

from qwen3_omni_pretrain.multimodal.types import PositionBatch
from qwen3_omni_pretrain.runtime.protocols import (
    CausalLMOutput,
    LegacyMediaPrefillInputs,
    ModelDecodeInputs,
    ModelPrefillInputs,
    legacy_position_ids,
)
from qwen3_omni_pretrain.runtime.state import (
    LegacyProcessedPrefix,
    StateOwner,
)

from .configuration_qwen3_omni_moe import Qwen3OmniMoeConfig
from .modeling_thinker_text import Qwen3OmniMoeThinkerTextModel


def _build_multimodal_attention_mask(
    text_attention_mask: torch.Tensor,
    has_image: torch.Tensor,
    has_audio: torch.Tensor,
) -> torch.Tensor:
    if text_attention_mask.dim() != 2:
        raise ValueError("text_attention_mask must have shape [B, T]")
    batch_size = text_attention_mask.size(0)
    if has_image.numel() != batch_size or has_audio.numel() != batch_size:
        raise ValueError("modality presence masks must match the text batch size")
    device = text_attention_mask.device
    dtype = text_attention_mask.dtype
    prefix = torch.stack(
        [
            has_image.to(device=device, dtype=dtype).reshape(batch_size),
            has_audio.to(device=device, dtype=dtype).reshape(batch_size),
        ],
        dim=1,
    )
    return torch.cat([prefix, text_attention_mask], dim=1)


class SimpleVisionEncoder(nn.Module):
    def __init__(self, hidden_size: int, image_size: int = 224):
        super().__init__()
        self.image_size = image_size
        self.proj = nn.Linear(3 * image_size * image_size, hidden_size)
        self.act = nn.Tanh()
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            pixel_values,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        x = x.reshape(x.size(0), -1)
        return self.norm(self.act(self.proj(x))).unsqueeze(1)


class SimpleAudioEncoder(nn.Module):
    def __init__(self, hidden_size: int, max_audio_len: int = 32000):
        super().__init__()
        self.max_audio_len = max_audio_len
        self.proj = nn.Linear(max_audio_len, hidden_size)
        self.act = nn.Tanh()
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, audio_values: torch.Tensor) -> torch.Tensor:
        if audio_values.dim() == 3:
            audio_values = audio_values.squeeze(1)
        return self.norm(self.act(self.proj(audio_values))).unsqueeze(1)


class Qwen3OmniMoeThinkerVisionAudioModel(PreTrainedModel):
    """Stage-2 wrapper that prepends one vision and one audio token."""

    config_class = Qwen3OmniMoeConfig
    _tied_weights_keys = ["thinker.lm_head.weight"]
    generation_prefix_storage_length = 2

    def __init__(self, config: Qwen3OmniMoeConfig):
        super().__init__(config)
        self.thinker = Qwen3OmniMoeThinkerTextModel(config)
        hidden_size = config.thinker_config.hidden_size
        self.vision_encoder = SimpleVisionEncoder(hidden_size, image_size=224)
        self.audio_encoder = SimpleAudioEncoder(hidden_size, max_audio_len=32000)
        self.post_init()

    @property
    def cache_support(self):
        return self.thinker.cache_support

    def _validate_typed_prefill_before_compute(
        self,
        inputs: ModelPrefillInputs,
        owner: StateOwner,
        use_cache: bool,
    ) -> tuple[torch.Tensor, LegacyMediaPrefillInputs]:
        if not isinstance(inputs, ModelPrefillInputs):
            raise TypeError("inputs must be ModelPrefillInputs")
        if not isinstance(owner, StateOwner):
            raise TypeError("owner must be StateOwner")
        if type(use_cache) is not bool:
            raise TypeError("use_cache must be a boolean")
        if use_cache:
            # Capability/training rejection must precede every media encoder.
            self.thinker._require_cache_runtime()
        device = self.thinker.embed_tokens.weight.device
        dtype = self.thinker.embed_tokens.weight.dtype
        payload = (
            inputs.input_ids
            if inputs.input_ids is not None
            else inputs.inputs_embeds
        )
        assert payload is not None
        if payload.device != device:
            raise ValueError("model input and parameters must share a device")
        if inputs.inputs_embeds is not None and inputs.inputs_embeds.dtype != dtype:
            raise ValueError("inputs_embeds dtype must match model parameters")
        if bool((~inputs.key_valid_mask.any(dim=1)).any().item()):
            raise ValueError("every text row must contain a valid token")

        media = inputs.media
        if media is None:
            media = LegacyMediaPrefillInputs(
                pixel_values=None,
                audio_values=None,
                has_image=torch.zeros(
                    inputs.batch_size,
                    dtype=torch.bool,
                    device=device,
                ),
                has_audio=torch.zeros(
                    inputs.batch_size,
                    dtype=torch.bool,
                    device=device,
                ),
            )
        if media.batch_size != inputs.batch_size or media.device != device:
            raise ValueError("media and text inputs must share batch/device")
        if media.pixel_values is not None:
            pixels = media.pixel_values
            if pixels.ndim != 4 or pixels.shape[1] != 3:
                raise ValueError("pixel_values must have shape [B, 3, H, W]")
            if pixels.shape[2] <= 0 or pixels.shape[3] <= 0:
                raise ValueError("pixel_values spatial dimensions must be positive")
            if pixels.dtype != dtype:
                raise ValueError("pixel_values dtype must match model parameters")
        if media.audio_values is not None:
            audio = media.audio_values
            valid_audio_shape = audio.ndim == 2 or (
                audio.ndim == 3 and audio.shape[1] == 1
            )
            if not valid_audio_shape:
                raise ValueError("audio_values must have shape [B, L] or [B, 1, L]")
            if audio.shape[-1] <= 0:
                raise ValueError("audio_values length must be positive")
            expected_length = getattr(self.audio_encoder, "max_audio_len", None)
            if expected_length is not None and audio.shape[-1] != expected_length:
                raise ValueError(
                    f"audio_values length must equal {expected_length}"
                )
            if audio.dtype != dtype:
                raise ValueError("audio_values dtype must match model parameters")

        max_positions = self.thinker.thinker_cfg.max_position_embeddings
        if inputs.query_length + 2 > max_positions:
            from qwen3_omni_pretrain.runtime.capabilities import (
                CacheCapabilityError,
                CacheErrorCode,
            )

            raise CacheCapabilityError(
                CacheErrorCode.CONTEXT_OVERFLOW,
                "multimodal prefix plus text exceeds max_position_embeddings",
            )
        if inputs.position_batch is None:
            if use_cache:
                raise ValueError("cache prefill requires explicit text positions")
            text_positions = torch.arange(
                inputs.query_length,
                dtype=torch.long,
                device=device,
            ).unsqueeze(0).expand(inputs.batch_size, -1)
        else:
            text_positions = legacy_position_ids(
                inputs.position_batch,
                inputs.key_valid_mask,
                max_position_embeddings=max_positions - 2,
            )
        return text_positions, media

    @staticmethod
    def _compose_prefill_position(
        *,
        text_positions: torch.Tensor,
        text_mask: torch.Tensor,
        media: LegacyMediaPrefillInputs,
        rope_deltas: torch.Tensor,
    ) -> PositionBatch:
        batch_size = text_mask.shape[0]
        prefix_positions = torch.zeros(
            (1, batch_size, 2),
            dtype=text_positions.dtype,
            device=text_positions.device,
        )
        prefix_positions[0, :, 1] = media.has_audio.to(
            dtype=text_positions.dtype
        )
        shifted_text = (text_positions + 2).masked_fill(~text_mask, 0)
        return PositionBatch(
            position_ids=torch.cat(
                (prefix_positions, shifted_text.unsqueeze(0)),
                dim=2,
            ),
            rope_deltas=rope_deltas,
            axis_names=("sequence",),
        )

    def prefill(
        self,
        *,
        inputs: ModelPrefillInputs,
        owner: StateOwner,
        use_cache: bool,
    ) -> CausalLMOutput:
        text_positions, media = self._validate_typed_prefill_before_compute(
            inputs,
            owner,
            use_cache,
        )
        batch_size = inputs.batch_size
        hidden_size = self.thinker.thinker_cfg.hidden_size
        dtype = self.thinker.embed_tokens.weight.dtype
        device = inputs.device
        with torch.inference_mode():
            text_embeds = (
                self.thinker.embed_tokens(inputs.input_ids)
                if inputs.input_ids is not None
                else inputs.inputs_embeds
            )
            assert text_embeds is not None
            vision = torch.zeros(
                (batch_size, 1, hidden_size),
                dtype=dtype,
                device=device,
            )
            audio = torch.zeros_like(vision)
            if bool(media.has_image.any().item()):
                assert media.pixel_values is not None
                indices = torch.nonzero(
                    media.has_image,
                    as_tuple=False,
                ).flatten()
                encoded = self.vision_encoder(
                    media.pixel_values.index_select(0, indices)
                )
                if encoded.shape != (indices.numel(), 1, hidden_size):
                    raise ValueError(
                        "vision encoder output must have shape [present, 1, H]"
                    )
                vision.index_copy_(0, indices, encoded.to(dtype=dtype))
            if bool(media.has_audio.any().item()):
                assert media.audio_values is not None
                indices = torch.nonzero(
                    media.has_audio,
                    as_tuple=False,
                ).flatten()
                encoded = self.audio_encoder(
                    media.audio_values.index_select(0, indices)
                )
                if encoded.shape != (indices.numel(), 1, hidden_size):
                    raise ValueError(
                        "audio encoder output must have shape [present, 1, H]"
                    )
                audio.index_copy_(0, indices, encoded.to(dtype=dtype))

            full_mask = torch.cat(
                (
                    media.has_image.unsqueeze(1),
                    media.has_audio.unsqueeze(1),
                    inputs.key_valid_mask,
                ),
                dim=1,
            )
            rope_deltas = (
                torch.zeros(
                    (batch_size, 1),
                    dtype=torch.long,
                    device=device,
                )
                if inputs.position_batch is None
                else inputs.position_batch.rope_deltas
            )
            full_positions = self._compose_prefill_position(
                text_positions=text_positions,
                text_mask=inputs.key_valid_mask,
                media=media,
                rope_deltas=rope_deltas,
            )
            result = self.thinker.prefill(
                inputs=ModelPrefillInputs(
                    input_ids=None,
                    inputs_embeds=torch.cat(
                        (vision, audio, text_embeds),
                        dim=1,
                    ),
                    key_valid_mask=full_mask,
                    position_batch=full_positions,
                    media=None,
                ),
                owner=owner,
                use_cache=use_cache,
            )
            if result.decoder_state is None:
                return result
            processed = LegacyProcessedPrefix(
                has_image=tuple(bool(value) for value in media.has_image.tolist()),
                has_audio=tuple(bool(value) for value in media.has_audio.tolist()),
                prefix_storage_length=2,
            )
            state = result.decoder_state.with_processed_media(processed)
            return CausalLMOutput(
                logits=result.logits,
                loss=result.loss,
                ce_loss=result.ce_loss,
                aux_loss=result.aux_loss,
                decoder_state=state,
                hidden_states=result.hidden_states,
            )

    def decode(
        self,
        *,
        inputs: ModelDecodeInputs,
        owner: StateOwner,
    ) -> CausalLMOutput:
        if not isinstance(inputs, ModelDecodeInputs):
            raise TypeError("inputs must be ModelDecodeInputs")
        if not isinstance(owner, StateOwner):
            raise TypeError("owner must be StateOwner")
        inputs.decoder_state.assert_owner(owner)
        processed = inputs.decoder_state.processed_media
        if processed is None or processed.prefix_storage_length != 2:
            raise ValueError("decode requires a complete processed media prefix")
        return self.thinker.decode(inputs=inputs, owner=owner)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor],
        pixel_values: torch.Tensor,
        audio_values: torch.Tensor,
        has_image: torch.Tensor,
        has_audio: torch.Tensor,
        output_hidden_states: bool = False,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        device = input_ids.device
        batch_size = input_ids.size(0)
        text_embeds = self.thinker.embed_tokens(input_ids)
        # Do not raise inside the forward graph: one distributed rank could
        # exit before peers reach the same collective. Non-finite values are
        # allowed to propagate to the synchronized output check in the trainer.
        vis_token = self.vision_encoder(pixel_values.to(device))
        aud_token = self.audio_encoder(audio_values.to(device))

        vis_token = vis_token * has_image.to(device).view(batch_size, 1, 1).float()
        aud_token = aud_token * has_audio.to(device).view(batch_size, 1, 1).float()
        inputs_embeds = torch.cat([vis_token, aud_token, text_embeds], dim=1)
        attn_full = _build_multimodal_attention_mask(
            attention_mask, has_image, has_audio
        )
        labels_full: Optional[torch.Tensor] = None
        if labels is not None:
            labels_full = torch.full(
                (batch_size, inputs_embeds.size(1)),
                fill_value=-100,
                dtype=labels.dtype,
                device=device,
            )
            labels_full[:, 2:] = labels.to(device)
        return self.thinker(
            input_ids=None,
            attention_mask=attn_full,
            labels=labels_full,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
        )
