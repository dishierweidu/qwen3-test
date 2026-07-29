from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel

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

    def __init__(self, config: Qwen3OmniMoeConfig):
        super().__init__(config)
        self.thinker = Qwen3OmniMoeThinkerTextModel(config)
        hidden_size = config.thinker_config.hidden_size
        self.vision_encoder = SimpleVisionEncoder(hidden_size, image_size=224)
        self.audio_encoder = SimpleAudioEncoder(hidden_size, max_audio_len=32000)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
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
        labels_full = torch.full(
            (batch_size, inputs_embeds.size(1)),
            fill_value=-100,
            dtype=labels.dtype,
            device=device,
        )
        labels_full[:, 2:] = labels
        return self.thinker(
            input_ids=None,
            attention_mask=attn_full,
            labels=labels_full,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
        )
