from typing import Optional, Dict, Any
from dataclasses import dataclass
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

    # DecoderLayer already has shared_mlp. Keep the MoE module routed-only to
    # avoid executing two shared dense FFNs on every token.
    moe_shared_expert: bool = False
    moe_shared_intermediate_size: Optional[int] = None
    moe_router_init_std: float = 1e-3
    moe_router_normalize_init: bool = True
    moe_renormalize_topk: bool = True

    moe_layer_indices: Optional[str] = None
    moe_aux_loss_coef: float = 0.01
    use_flash_attention: bool = True

    vision_hidden_size: int = 1152
    audio_hidden_size: int = 1024

    headwise_attn_output_gate: bool = False
    elementwise_attn_output_gate: bool = False

    use_deltanet: bool = False
    deltanet_layer_indices: Optional[str] = None
    block_type: str = "attn"
    deltanet_kernel_size: int = 3
    deltanet_conv_kernel_size: Optional[int] = None
    deltanet_num_heads: int = 24
    deltanet_chunk_size: int = 0
    gradient_checkpointing: bool = False

    use_tensor_parallel: bool = False
    tensor_parallel_size: Optional[int] = None


@dataclass
class Qwen3OmniMoeTalkerConfig:
    hidden_size: int = 1536
    intermediate_size: int = 4096
    num_hidden_layers: int = 12
    num_attention_heads: int = 24
    num_code_groups: int = 8
    codebook_size: int = 1024
    accept_hidden_layer: int = -1


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
        rope_partial_factor: float = 1.0,
        use_moe: bool = False,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
        bos_token_id: int = 151643,
        eos_token_id: int = 151645,
        pad_token_id: int = 151643,
        image_token_id: Optional[int] = None,
        audio_token_id: Optional[int] = None,
        video_token_id: Optional[int] = None,
        audio_start_token_id: Optional[int] = None,
        audio_end_token_id: Optional[int] = None,
        headwise_attn_output_gate: Optional[bool] = None,
        elementwise_attn_output_gate: Optional[bool] = None,
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

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings

        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.rope_partial_factor = rope_partial_factor

        self.use_moe = use_moe
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

        self.image_token_id = image_token_id
        self.audio_token_id = audio_token_id
        self.video_token_id = video_token_id
        self.audio_start_token_id = audio_start_token_id
        self.audio_end_token_id = audio_end_token_id

        thinker_cfg_dict = (
            thinker_config
            if isinstance(thinker_config, dict)
            else (thinker_config.__dict__ if thinker_config else {})
        )
        if isinstance(thinker_cfg_dict, dict):
            mapping = {
                "use_shared_expert": "moe_shared_expert",
                "shared_intermediate_size": "moe_shared_intermediate_size",
                "moe_router_init_std": "moe_router_init_std",
                "moe_router_normalize_init": "moe_router_normalize_init",
                "moe_renormalize_topk": "moe_renormalize_topk",
            }
            thinker_cfg_dict = {
                mapping.get(key, key): value
                for key, value in thinker_cfg_dict.items()
            }

        self.thinker_config = (
            Qwen3OmniMoeThinkerConfig(**thinker_cfg_dict)
            if isinstance(thinker_cfg_dict, dict)
            else (thinker_cfg_dict or Qwen3OmniMoeThinkerConfig())
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

        if headwise_attn_output_gate is not None:
            self.thinker_config.headwise_attn_output_gate = bool(
                headwise_attn_output_gate
            )
        if elementwise_attn_output_gate is not None:
            self.thinker_config.elementwise_attn_output_gate = bool(
                elementwise_attn_output_gate
            )

        if getattr(self.thinker_config, "deltanet_kernel_size", None) is None:
            legacy_kernel = getattr(
                self.thinker_config, "deltanet_conv_kernel_size", None
            )
            if legacy_kernel is not None:
                self.thinker_config.deltanet_kernel_size = int(legacy_kernel)
        if getattr(self.thinker_config, "deltanet_kernel_size", None) is None:
            self.thinker_config.deltanet_kernel_size = 3

        if self.thinker_config.moe_shared_expert:
            raise ValueError(
                "thinker_config.use_shared_expert=true is no longer valid: "
                "ThinkerDecoderLayer.shared_mlp is the single shared dense path."
            )
        if self.thinker_config.moe_shared_intermediate_size is not None:
            raise ValueError(
                "shared_intermediate_size must be removed when the MoE module "
                "contains routed experts only."
            )

        self.headwise_attn_output_gate = (
            self.thinker_config.headwise_attn_output_gate
        )
        self.elementwise_attn_output_gate = (
            self.thinker_config.elementwise_attn_output_gate
        )

    def to_dict(self):
        output = super().to_dict()
        output["rope_partial_factor"] = getattr(
            self, "rope_partial_factor", 1.0
        )
        output["headwise_attn_output_gate"] = self.headwise_attn_output_gate
        output["elementwise_attn_output_gate"] = (
            self.elementwise_attn_output_gate
        )
        output["thinker_config"] = self.thinker_config.__dict__
        output["talker_config"] = self.talker_config.__dict__
        output["code2wav_config"] = self.code2wav_config.__dict__
        return output
