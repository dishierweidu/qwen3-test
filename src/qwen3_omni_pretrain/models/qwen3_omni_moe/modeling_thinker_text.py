# src/qwen3_omni_pretrain/models/qwen3_omni_moe/modeling_thinker_text.py

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from .configuration_qwen3_omni_moe import Qwen3OmniMoeConfig, Qwen3OmniMoeThinkerConfig
from .modules.moe import Qwen3OmniMoeMLP



class RMSNorm(nn.Module):
    """简化版 RMSNorm"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x


class RotaryEmbedding(nn.Module):
    """
    一维 RoPE，只针对 head_dim 维度。
    支持输入形状:
      - [B, T, D]
      - [B, num_heads, T, D]
    """

    def __init__(self, dim: int, max_position_embeddings: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # [T, dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)       # [T, dim]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)  # [T, dim]
        self.register_buffer("sin_cached", emb.sin(), persistent=False)  # [T, dim]

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D] 或 [B, num_heads, T, D]
        position_ids: [B, T]
        """
        if x.dim() == 3:
            # [B, T, D] -> 假装 num_heads=1
            bsz, seq_len, dim = x.size()
            assert dim == self.dim
            cos = self.cos_cached[position_ids]        # [B, T, D]
            sin = self.sin_cached[position_ids]        # [B, T, D]
            x1, x2 = x[..., ::2], x[..., 1::2]
            x_rot = torch.stack([-x2, x1], dim=-1).reshape_as(x)
            return x * cos + x_rot * sin

        elif x.dim() == 4:
            # [B, nh, T, D]
            bsz, nh, seq_len, dim = x.size()
            assert dim == self.dim
            # cos/sin: [B, T, D] -> [B, 1, T, D]
            cos = self.cos_cached[position_ids]        # [B, T, D]
            sin = self.sin_cached[position_ids]        # [B, T, D]
            cos = cos.unsqueeze(1)                     # [B, 1, T, D]
            sin = sin.unsqueeze(1)                     # [B, 1, T, D]
            x1, x2 = x[..., ::2], x[..., 1::2]
            x_rot = torch.stack([-x2, x1], dim=-1).reshape_as(x)
            return x * cos + x_rot * sin

        else:
            raise ValueError(f"Unsupported x.dim() for RoPE: {x.dim()}")


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        rotary_emb: Optional[RotaryEmbedding] = None,
    ) -> torch.Tensor:
        B, T, H = hidden_states.size()

        q = self.q_proj(hidden_states)  # [B, T, H]
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, nh, T, hd]
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if rotary_emb is not None and position_ids is not None:
            # position_ids: [B, T]
            q = rotary_emb(q, position_ids)  # [B, nh, T, hd]
            k = rotary_emb(k, position_ids)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)  # [B, nh, T, T]

        if attention_mask is not None:
            # attention_mask: [B, 1, 1, T] 或 [B, 1, T, T]
            scores = scores + attention_mask

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # [B, nh, T, hd]
        out = out.transpose(1, 2).contiguous().view(B, T, H)
        out = self.o_proj(out)
        return out


class MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class ThinkerDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3OmniMoeThinkerConfig, rotary_emb: RotaryEmbedding):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.use_moe = getattr(config, "use_moe", False)
        self.num_experts = getattr(config, "num_experts", 0)
        self.num_experts_per_tok = getattr(config, "num_experts_per_tok", 1)

        head_dim = config.hidden_size // config.num_attention_heads
        self.self_attn = MultiHeadSelfAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=head_dim,
        )
        self.attn_norm = RMSNorm(config.hidden_size)
        
        # Shared Dense FFN（所有层都有）
        self.shared_mlp = MLP(config.hidden_size, config.intermediate_size)

        # 可选的 MoE FFN：根据 use_moe 选择 Dense MLP 或 MoE MLP
        if self.use_moe and self.num_experts > 0 and self.num_experts_per_tok > 0:
            self.moe_mlp = Qwen3OmniMoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_experts=self.num_experts,
                num_experts_per_tok=self.num_experts_per_tok,
            )
        else:
            self.moe_mlp = None

        self.mlp_norm = RMSNorm(config.hidden_size)

        self.rotary_emb = rotary_emb

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Self-Attention
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        attn_out = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            rotary_emb=self.rotary_emb,
        )
        hidden_states = residual + attn_out

        # FFN: Shared Dense + Optional MoE
        residual = hidden_states
        hidden_states = self.mlp_norm(hidden_states)

        # Shared FFN
        shared_out = self.shared_mlp(hidden_states)

        aux_loss = None
        if self.moe_mlp is not None:
            moe_out, aux_loss = self.moe_mlp(hidden_states)  # aux_loss: scalar tensor
            mlp_out = shared_out + moe_out
        else:
            mlp_out = shared_out

        hidden_states = residual + mlp_out

        return hidden_states, aux_loss



class Qwen3OmniMoeThinkerTextModel(PreTrainedModel):
    config_class = Qwen3OmniMoeConfig

    def __init__(self, config: Qwen3OmniMoeConfig):
        super().__init__(config)

        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.max_position_embeddings = config.max_position_embeddings

        thinker_cfg = config.thinker_config
        assert isinstance(thinker_cfg, Qwen3OmniMoeThinkerConfig)
        self.thinker_cfg = thinker_cfg  # 保存一份，forward 里要用 moe_aux_loss_coef

        self.embed_tokens = nn.Embedding(
            config.vocab_size, thinker_cfg.hidden_size
        )

        # head_dim 用于 RoPE
        self.head_dim = thinker_cfg.hidden_size // thinker_cfg.num_attention_heads
        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=thinker_cfg.max_position_embeddings,
            base=config.rope_theta,
        )

        # 解析 moe_layer_indices: 例如 "0,2,4" -> {0,2,4}
        moe_layer_set = None
        if getattr(thinker_cfg, "moe_layer_indices", None) is not None:
            indices_str = thinker_cfg.moe_layer_indices
            if isinstance(indices_str, str) and indices_str.strip():
                moe_layer_set = set(
                    int(x) for x in indices_str.split(",") if x.strip().isdigit()
                )

        layers = []
        for layer_idx in range(thinker_cfg.num_hidden_layers):
            # 默认按 config.use_moe
            use_moe_layer = thinker_cfg.use_moe
            # 如果指定了 moe_layer_indices，则以它为准
            if moe_layer_set is not None:
                use_moe_layer = layer_idx in moe_layer_set

            # 为当前层构造一个“局部Config”拷贝，覆盖 use_moe
            local_cfg = Qwen3OmniMoeThinkerConfig(**thinker_cfg.__dict__)
            local_cfg.use_moe = use_moe_layer

            layers.append(ThinkerDecoderLayer(local_cfg, self.rotary_emb))

        self.layers = nn.ModuleList(layers)

        self.norm = RMSNorm(thinker_cfg.hidden_size)

        # LM head
        self.lm_head = nn.Linear(thinker_cfg.hidden_size, config.vocab_size, bias=False)
        
        # transformers 要绑词嵌入
        self.config.tie_word_embeddings = True
        # 让 lm_head.weight 和 embed_tokens.weight 指向同一个 Tensor
        self.lm_head.weight = self.embed_tokens.weight

        self.post_init()

    def _prepare_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        input_shape: Tuple[int, int],
        device: torch.device,
    ) -> torch.Tensor:
        bsz, tgt_len = input_shape
        if attention_mask is None:
            attention_mask = torch.ones((bsz, tgt_len), device=device)
        # causal mask [1, 1, T, T]
        causal_mask = torch.full(
            (tgt_len, tgt_len),
            fill_value=-float("inf"),
            device=device,
        )
        causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1,1,T,T]

        # padding mask [B, 1, 1, T]
        padding_mask = (1.0 - attention_mask[:, None, None, :]) * -1e4

        return causal_mask + padding_mask

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        inputs_embeds: Optional[torch.Tensor] = None,
    ):
        """
        input_ids: [B, T]
        labels: [B, T] 或 None
        - Stage1: 纯文本 → 传 input_ids（inputs_embeds=None）
        - Stage2: 多模态 → 传 inputs_embeds（input_ids 可以为 None，只用于 labels）
        """

        if inputs_embeds is not None:
            hidden_states = inputs_embeds
            device = hidden_states.device
            bsz, seq_len, _ = hidden_states.size()
        else:
            assert input_ids is not None, "input_ids or inputs_embeds must be provided"
            hidden_states = self.embed_tokens(input_ids)  # [B, T, H]
            device = input_ids.device
            bsz, seq_len = input_ids.size()

        if position_ids is None:
            position_ids = torch.arange(
                seq_len, dtype=torch.long, device=device
            ).unsqueeze(0).expand(bsz, -1)  # [B, T]

        attention_mask_full = self._prepare_attention_mask(
            attention_mask, (bsz, seq_len), device
        )

        all_hidden_states = [] if output_hidden_states else None
        total_aux_loss = None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            hidden_states, layer_aux = layer(
                hidden_states, attention_mask_full, position_ids
            )
            if layer_aux is not None:
                if total_aux_loss is None:
                    total_aux_loss = layer_aux
                else:
                    total_aux_loss = total_aux_loss + layer_aux

        hidden_states = self.norm(hidden_states)


        logits = self.lm_head(hidden_states)

        ce_loss = None
        aux_loss = total_aux_loss
        loss = None

        if labels is not None:
            # 计算标准自回归交叉熵
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            ce_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            if aux_loss is not None:
                loss = ce_loss + self.thinker_cfg.moe_aux_loss_coef * aux_loss
            else:
                loss = ce_loss

        output = {
            "logits": logits,
            "loss": loss,
        }
        if ce_loss is not None:
            output["ce_loss"] = ce_loss
        if aux_loss is not None:
            output["aux_loss"] = aux_loss

        if output_hidden_states:
            all_hidden_states.append(hidden_states)
            output["hidden_states"] = all_hidden_states

        return output

    
    # ---- 让 transformers 知道怎么 tie 权重 ----
    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

