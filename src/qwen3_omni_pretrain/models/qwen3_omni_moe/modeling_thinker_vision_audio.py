# src/qwen3_omni_pretrain/models/qwen3_omni_moe/modeling_thinker_vision_audio.py

from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel

from .configuration_qwen3_omni_moe import Qwen3OmniMoeConfig
from .modeling_thinker_text import Qwen3OmniMoeThinkerTextModel


class SimpleVisionEncoder(nn.Module):
    """
    非常简化的 Vision encoder：
    - 输入: pixel_values [B, 3, H, W]
    - 输出: 一个 [B, 1, Hdim] 的“图像 token”
    后续可以替换成 ViT/CLIP。
    """

    def __init__(self, hidden_size: int, image_size: int = 224):
        super().__init__()
        self.image_size = image_size
        self.proj = nn.Linear(3 * image_size * image_size, hidden_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # 简单 resize 到固定大小
        x = F.interpolate(
            pixel_values,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )  # [B,3,H,W]
        x = x.view(x.size(0), -1)  # [B, 3*H*W]
        x = self.proj(x)           # [B, H]
        return x.unsqueeze(1)      # [B, 1, H]


class SimpleAudioEncoder(nn.Module):
    """
    非常简化的 Audio encoder：
    - 输入: audio_values [B, max_audio_len]
    - 输出: 一个 [B, 1, Hdim] 的“音频 token”
    后续可以替换成 Wav2Vec/Whisper encoder。
    """

    def __init__(self, hidden_size: int, max_audio_len: int = 32000):
        super().__init__()
        self.max_audio_len = max_audio_len
        self.proj = nn.Linear(max_audio_len, hidden_size)

    def forward(self, audio_values: torch.Tensor) -> torch.Tensor:
        if audio_values.dim() == 3:
            audio_values = audio_values.squeeze(1)
        x = self.proj(audio_values)  # [B, H]
        return x.unsqueeze(1)        # [B, 1, H]


class Qwen3OmniMoeThinkerVisionAudioModel(PreTrainedModel):
    """
    Omni Thinker (Stage2 多模态版)：
    - Vision / Audio -> 各 1 个 token
    - 拼接: [vis_token] [aud_token] [text_tokens...] -> 喂进文本 Thinker。
    """
    config_class = Qwen3OmniMoeConfig

    def __init__(self, config: Qwen3OmniMoeConfig):
        super().__init__(config)

        self.thinker = Qwen3OmniMoeThinkerTextModel(config)
        hidden_size = config.thinker_config.hidden_size

        self.vision_encoder = SimpleVisionEncoder(
            hidden_size=hidden_size,
            image_size=224,
        )
        self.audio_encoder = SimpleAudioEncoder(
            hidden_size=hidden_size,
            max_audio_len=32000,
        )

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
        """
        input_ids: [B, T_text]
        pixel_values: [B,3,H,W]
        audio_values: [B,max_audio_len]
        has_image/has_audio: [B]
        """

        device = input_ids.device
        B, T_text = input_ids.size()

        # 文本 embedding
        text_embeds = self.thinker.embed_tokens(input_ids)  # [B,T,H]

        # vision/audio token
        vis_token = self.vision_encoder(pixel_values.to(device))    # [B,1,H]
        aud_token = self.audio_encoder(audio_values.to(device))     # [B,1,H]

        # 对于没有图像/音频的样本，把对应 token 置 0，并 mask 掉
        has_image_f = has_image.to(device).view(B, 1, 1).float()
        has_audio_f = has_audio.to(device).view(B, 1, 1).float()

        vis_token = vis_token * has_image_f          # [B,1,H]
        aud_token = aud_token * has_audio_f          # [B,1,H]

        # 拼接序列：[vis][aud][text...]
        inputs_embeds = torch.cat([vis_token, aud_token, text_embeds], dim=1)  # [B,T+2,H]
        T_total = inputs_embeds.size(1)

        # attention_mask：没有的模态对应 token mask=0
        attn_full = torch.ones(B, T_total, device=device, dtype=attention_mask.dtype)
        # 没有 image 的，把 position 0 mask 掉
        attn_full[:, 0] = has_image.to(device)
        # 没有 audio 的，把 position 1 mask 掉
        attn_full[:, 1] = has_audio.to(device)

        # labels 对齐：前两个 multimodal token 不参与 loss，设为 -100
        labels_full = torch.full(
            (B, T_total), fill_value=-100, dtype=labels.dtype, device=device
        )
        labels_full[:, 2:] = labels  # 文本部分照抄

        outputs = self.thinker(
            input_ids=None,
            attention_mask=attn_full,
            labels=labels_full,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
        )

        return outputs
