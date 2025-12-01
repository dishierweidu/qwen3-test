# src/qwen3_omni_pretrain/models/qwen3_omni_moe/modeling_thinker_text.py

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from .configuration_qwen3_omni_moe import Qwen3OmniMoeConfig, Qwen3OmniMoeThinkerConfig


class RMSNorm(nn.Module):
    """简化版 RMSNorm，方便先跑起来"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x


class RotaryEmbedding(nn.Module):
    """简化的一维 RoPE，占位用。后续可以换成真正的 TMRoPE 三维实现。"""

    def __init__(self, dim: int, max_position_embeddings: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        # x: [B, T, H]
        cos = self.cos_cached[position_ids]  # [B, T, H]
        sin = self.sin_cached[position_ids]
        x1, x2 = x[..., ::2], x[..., 1::2]
        x_rotated = torch.stack([-x2, x1], dim=-1).reshape_as(x)
        return x * cos + x_rotated * sin


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads

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
            # 展开为 [B, 1, T, 1] 再广播
            pos = position_ids.unsqueeze(1)
            q = rotary_emb(q.transpose(1, 2), pos).transpose(1, 2)
            k = rotary_emb(k.transpose(1, 2), pos).transpose(1, 2)

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

        self.self_attn = MultiHeadSelfAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
        )
        self.attn_norm = RMSNorm(config.hidden_size)

        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.mlp_norm = RMSNorm(config.hidden_size)

        self.rotary_emb = rotary_emb

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
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

        # MLP
        residual = hidden_states
        hidden_states = self.mlp_norm(hidden_states)
        mlp_out = self.mlp(hidden_states)
        hidden_states = residual + mlp_out

        return hidden_states


class Qwen3OmniMoeThinkerTextModel(PreTrainedModel):
    config_class = Qwen3OmniMoeConfig

    def __init__(self, config: Qwen3OmniMoeConfig):
        super().__init__(config)

        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.max_position_embeddings = config.max_position_embeddings

        thinker_cfg = config.thinker_config
        assert isinstance(thinker_cfg, Qwen3OmniMoeThinkerConfig)

        self.embed_tokens = nn.Embedding(
            config.vocab_size, thinker_cfg.hidden_size
        )
        self.rotary_emb = RotaryEmbedding(
            dim=thinker_cfg.hidden_size,
            max_position_embeddings=thinker_cfg.max_position_embeddings,
            base=config.rope_theta,
        )

        self.layers = nn.ModuleList(
            [
                ThinkerDecoderLayer(thinker_cfg, self.rotary_emb)
                for _ in range(thinker_cfg.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(thinker_cfg.hidden_size)

        # LM head
        self.lm_head = nn.Linear(thinker_cfg.hidden_size, config.vocab_size, bias=False)

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
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ):
        """
        input_ids: [B, T]
        labels: [B, T] 或 None
        """

        device = input_ids.device
        bsz, seq_len = input_ids.size()

        if position_ids is None:
            position_ids = torch.arange(
                seq_len, dtype=torch.long, device=device
            ).unsqueeze(0).expand(bsz, -1)  # [B, T]

        attention_mask_full = self._prepare_attention_mask(
            attention_mask, (bsz, seq_len), device
        )

        hidden_states = self.embed_tokens(input_ids)  # [B, T, H]

        all_hidden_states = [] if output_hidden_states else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            hidden_states = layer(hidden_states, attention_mask_full, position_ids)

        hidden_states = self.norm(hidden_states)

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # 计算标准自回归交叉熵
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        output = {
            "logits": logits,
            "loss": loss,
        }
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
            output["hidden_states"] = all_hidden_states

        return output
