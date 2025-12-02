# src/qwen3_omni_pretrain/models/qwen3_omni_moe/configuration_qwen3_omni_moe.py

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from transformers import PretrainedConfig


@dataclass
class Qwen3OmniMoeThinkerConfig:
    hidden_size: int = 3072
    intermediate_size: int = 8192
    num_hidden_layers: int = 28
    num_attention_heads: int = 24
    num_key_value_heads: int = 8
    max_position_embeddings: int = 4096

    use_moe: bool = False
    num_experts: int = 8
    num_experts_per_tok: int = 2
    
    # 新增：哪些层使用 MoE，字符串形式，例如 "0,2,4,6"
    moe_layer_indices: Optional[str] = None

    # 多模态 backbone 占位
    vision_hidden_size: int = 1152
    audio_hidden_size: int = 1024


@dataclass
class Qwen3OmniMoeTalkerConfig:
    hidden_size: int = 1536
    intermediate_size: int = 4096
    num_hidden_layers: int = 12
    num_attention_heads: int = 24

    num_code_groups: int = 8
    codebook_size: int = 1024

    accept_hidden_layer: int = -1  # 使用 Thinker 的哪一层 hidden state


@dataclass
class Qwen3OmniMoeCode2WavConfig:
    hidden_size: int = 1024
    num_layers: int = 12
    kernel_size: int = 5
    stride: int = 2


class Qwen3OmniMoeConfig(PretrainedConfig):
    model_type = "qwen3_omni_moe"

    def __init__(
        self,
        vocab_size: int = 152064,
        hidden_size: int = 3072,
        intermediate_size: int = 8192,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 24,
        num_key_value_heads: int = 8,
        max_position_embeddings: int = 4096,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[Dict[str, Any]] = None,
        use_moe: bool = False,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
        # special tokens
        bos_token_id: int = 151643,
        eos_token_id: int = 151645,
        pad_token_id: int = 151643,
        image_token_id: Optional[int] = None,
        audio_token_id: Optional[int] = None,
        video_token_id: Optional[int] = None,
        audio_start_token_id: Optional[int] = None,
        audio_end_token_id: Optional[int] = None,
        # 子配置（可以直接传 dict）
        thinker_config: Optional[Dict[str, Any]] = None,
        talker_config: Optional[Dict[str, Any]] = None,
        code2wav_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )

        # 顶层参数，主要给 text Thinker 用
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings

        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

        self.use_moe = use_moe
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

        # 多模态 token id
        self.image_token_id = image_token_id
        self.audio_token_id = audio_token_id
        self.video_token_id = video_token_id
        self.audio_start_token_id = audio_start_token_id
        self.audio_end_token_id = audio_end_token_id

        # 子配置：我们用 dataclass 来保存，但序列化时会转回 dict
        self.thinker_config = (
            Qwen3OmniMoeThinkerConfig(**thinker_config)
            if isinstance(thinker_config, dict)
            else (thinker_config or Qwen3OmniMoeThinkerConfig())
        )
        self.talker_config = (
            Qwen3OmniMoeTalkerConfig(**talker_config)
            if isinstance(talker_config, dict)
            else (talker_config or Qwen3OmniMoeTalkerConfig())
        )
        self.code2wav_config = (
            Qwen3OmniMoeCode2WavConfig(**code2wav_config)
            if isinstance(code2wav_config, dict)
            else (code2wav_config or Qwen3OmniMoeCode2WavConfig())
        )

    def to_dict(self):
        # 确保保存到磁盘时，子配置能转成可 JSON 序列化的 dict
        output = super().to_dict()
        output["thinker_config"] = self.thinker_config.__dict__
        output["talker_config"] = self.talker_config.__dict__
        output["code2wav_config"] = self.code2wav_config.__dict__
        return output
