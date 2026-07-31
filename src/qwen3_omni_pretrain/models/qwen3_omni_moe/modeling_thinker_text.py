# src/qwen3_omni_pretrain/models/qwen3_omni_moe/modeling_thinker_text.py

import hashlib
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from qwen3_omni_pretrain.architecture.config_validation import (
    parse_layer_indices,
)
from qwen3_omni_pretrain.runtime.capabilities import (
    CacheCapabilityError,
    CacheErrorCode,
    cache_support_for_legacy_layers,
    require_incremental_decode_support,
)
from qwen3_omni_pretrain.runtime.protocols import (
    CausalLMOutput,
    ModelDecodeInputs,
    ModelPrefillInputs,
    legacy_position_ids,
)
from qwen3_omni_pretrain.runtime.state import (
    AttentionKV,
    DecoderPositionState,
    DecoderState,
    LegacyPositionCursor,
    StateOwner,
)

from .configuration_qwen3_omni_moe import Qwen3OmniMoeConfig, Qwen3OmniMoeThinkerConfig
from .modules.moe import Qwen3OmniMoeMLP


def _require_boolean_2d_mask(
    value: torch.Tensor,
    name: str,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype is not torch.bool:
        raise TypeError(f"{name} must have dtype torch.bool")
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] == 0:
        raise ValueError(f"{name} must have non-empty shape [B, S]")
    return value


def _build_rectangular_causal_bias(
    *,
    past_key_valid_mask: torch.Tensor | None,
    current_key_valid_mask: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build one final additive [B,1,Q,P+Q] causal/key-valid bias."""

    current = _require_boolean_2d_mask(
        current_key_valid_mask,
        "current_key_valid_mask",
    )
    if not dtype.is_floating_point:
        raise TypeError("attention bias dtype must be floating")
    batch_size, query_length = current.shape
    if past_key_valid_mask is None:
        past = torch.empty(
            (batch_size, 0),
            dtype=torch.bool,
            device=current.device,
        )
    else:
        past = _require_boolean_2d_mask(
            past_key_valid_mask,
            "past_key_valid_mask",
        )
        if past.shape[0] != batch_size:
            raise ValueError("past and current masks must share batch size")
        if past.device != current.device:
            raise ValueError("past and current masks must share a device")

    past_length = past.shape[1]
    key_mask = torch.cat((past, current), dim=1)
    key_indices = torch.arange(
        key_mask.shape[1],
        device=current.device,
    )
    query_limits = past_length + torch.arange(
        query_length,
        device=current.device,
    )
    causal = key_indices.unsqueeze(0) <= query_limits.unsqueeze(1)
    allowed = key_mask.unsqueeze(1) & causal.unsqueeze(0)
    missing = current & ~allowed.any(dim=-1)
    if bool(missing.any().item()):
        raise ValueError("a valid query must have at least one valid key")

    # Invalid queries never enter state as valid keys and are zeroed after the
    # output projection. Giving them a finite attention row avoids all-masked
    # softmax NaNs without changing their observable output.
    allowed = torch.where(
        current.unsqueeze(-1),
        allowed,
        torch.ones_like(allowed),
    )
    bias = torch.zeros(
        (batch_size, 1, query_length, key_mask.shape[1]),
        dtype=dtype,
        device=current.device,
    )
    return bias.masked_fill(~allowed.unsqueeze(1), torch.finfo(dtype).min)


class RMSNorm(nn.Module):
    r"""Zero-centered RMSNorm used in Qwen3-Next / Omni style blocks.

    y = x / rms(x) * (1 + weight)
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        # Zero-centered: 初始化为 0，而不是 1
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 计算 RMS，末维上归一化
        # 为了数值稳定，可以在 FP32 里做归一化，再 cast 回去
        orig_dtype = x.dtype
        if x.dtype in (torch.float16, torch.bfloat16):
            x_float = x.to(torch.float32)
            norm = x_float.pow(2).mean(-1, keepdim=True)
            x_float = x_float * torch.rsqrt(norm + self.eps)
            x = x_float.to(orig_dtype)
        else:
            norm = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(norm + self.eps)

        # 核心：乘以 (1 + weight)，而不是直接乘 weight
        scale = (1.0 + self.weight).to(x.dtype)
        return x * scale


class RotaryEmbedding(nn.Module):
    """
    一维 RoPE，支持部分维度旋转。
    支持输入形状:
      - [B, T, D]
      - [B, num_heads, T, D]

    参数:
      dim:        总的 head_dim（例如 128、192 等）
      rope_dim:   真正用于 RoPE 的前缀维度，必须为偶数且 <= dim
                  rope_dim = 0 表示完全不加 RoPE
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int,
        base: float = 10000.0,
        rope_dim: int = None,
    ):
        super().__init__()

        # 总维度（head_dim）
        self.total_dim = dim

        # 用于 RoPE 的前缀维度
        if rope_dim is None:
            rope_dim = dim
        # 保证非负且不超过 total_dim
        rope_dim = max(0, min(rope_dim, dim))
        # RoPE 维度必须为偶数，如果是奇数就减一
        if rope_dim % 2 == 1:
            rope_dim -= 1
        self.rope_dim = rope_dim

        if self.rope_dim > 0:
            inv_freq = 1.0 / (base ** (torch.arange(0, self.rope_dim, 2).float() / self.rope_dim))
            t = torch.arange(max_position_embeddings, dtype=torch.float)
            freqs = torch.einsum("i,j->ij", t, inv_freq)  # [T, rope_dim/2]
            emb = torch.cat((freqs, freqs), dim=-1)       # [T, rope_dim]
            self.register_buffer("cos_cached", emb.cos(), persistent=False)  # [T, rope_dim]
            self.register_buffer("sin_cached", emb.sin(), persistent=False)  # [T, rope_dim]
        else:
            # 不使用 RoPE 的情况
            self.register_buffer("cos_cached", None, persistent=False)
            self.register_buffer("sin_cached", None, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D] 或 [B, num_heads, T, D]
        position_ids: [B, T]
        """
        if self.rope_dim == 0:
            # 不做 RoPE，直接返回
            return x

        if x.dim() == 3:
            # [B, T, D]
            bsz, seq_len, dim = x.size()
            assert dim == self.total_dim

            cos = self.cos_cached[position_ids].to(dtype=x.dtype)
            sin = self.sin_cached[position_ids].to(dtype=x.dtype)

            # 拆分前 rope_dim 和剩余部分
            x_rope = x[..., :self.rope_dim]      # [B, T, rope_dim]
            x_pass = x[..., self.rope_dim:]      # [B, T, D - rope_dim]

            x1, x2 = x_rope[..., ::2], x_rope[..., 1::2]
            x_rot = torch.stack([-x2, x1], dim=-1).reshape_as(x_rope)
            x_rope = x_rope * cos + x_rot * sin

            if self.rope_dim == self.total_dim:
                return x_rope
            else:
                return torch.cat([x_rope, x_pass], dim=-1)

        elif x.dim() == 4:
            # [B, nh, T, D]
            bsz, nh, seq_len, dim = x.size()
            assert dim == self.total_dim

            cos = self.cos_cached[position_ids].to(dtype=x.dtype)
            sin = self.sin_cached[position_ids].to(dtype=x.dtype)
            cos = cos.unsqueeze(1)               # [B, 1, T, rope_dim]
            sin = sin.unsqueeze(1)               # [B, 1, T, rope_dim]

            x_rope = x[..., :self.rope_dim]      # [B, nh, T, rope_dim]
            x_pass = x[..., self.rope_dim:]      # [B, nh, T, D - rope_dim]

            x1, x2 = x_rope[..., ::2], x_rope[..., 1::2]
            x_rot = torch.stack([-x2, x1], dim=-1).reshape_as(x_rope)
            x_rope = x_rope * cos + x_rot * sin

            if self.rope_dim == self.total_dim:
                return x_rope
            else:
                return torch.cat([x_rope, x_pass], dim=-1)

        else:
            raise ValueError(f"Unsupported x.dim() for RoPE: {x.dim()}")


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, 
                 head_dim: int, 
                 use_flash_attention: bool = False,
                 headwise_attn_output_gate: bool = False,
                 elementwise_attn_output_gate: bool = False):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.use_flash_attention = use_flash_attention

        # 检查维度合法性
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(f"hidden_size must be divisible by num_heads")
        if self.num_kv_heads <= 0 or (self.num_heads % self.num_kv_heads) != 0:
            raise ValueError(f"num_heads must be a multiple of num_kv_heads for GQA")

        # Q 全头；K/V 只用 num_kv_heads 头 → 真正利用 GQA
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)

        # 输出还是 full hidden_size
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # --- NEW: SDPA output gate G1 ---
        self.headwise_attn_output_gate = headwise_attn_output_gate
        self.elementwise_attn_output_gate = elementwise_attn_output_gate
        self.use_output_gate = headwise_attn_output_gate or elementwise_attn_output_gate

        if self.headwise_attn_output_gate:
            # 基于输入 token → 每个 head 一个标量 gate
            self.gate_proj = nn.Linear(hidden_size, num_heads, bias=True)
        elif self.elementwise_attn_output_gate:
            # 基于输入 token → 每个 head、每个 channel 一个 gate
            self.gate_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        else:
            self.gate_proj = None
            
    @staticmethod
    def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        将 [B, num_kv_heads, T, head_dim] 复制成 [B, num_heads, T, head_dim]
        参考 Qwen2 / LLaMA 的 repeat_kv 实现。:contentReference[oaicite:2]{index=2}
        """
        batch, num_kv_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        # [B, num_kv_heads, 1, T, D] → expand → [B, num_kv_heads, n_rep, T, D]
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_kv_heads, n_rep, slen, head_dim
        )
        # 合并 kv_head 和 group 维度
        return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        rotary_emb: Optional[RotaryEmbedding] = None,
        past_key_value: AttentionKV | None = None,
        current_key_valid_mask: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, AttentionKV | None]:
        B, T, _ = hidden_states.size()
        if type(use_cache) is not bool:
            raise TypeError("use_cache must be a boolean")
        if current_key_valid_mask is None:
            current_key_valid_mask = torch.ones(
                (B, T),
                dtype=torch.bool,
                device=hidden_states.device,
            )
        current_key_valid_mask = _require_boolean_2d_mask(
            current_key_valid_mask,
            "current_key_valid_mask",
        )
        if current_key_valid_mask.shape != (B, T):
            raise ValueError("current_key_valid_mask must have shape [B, Q]")
        if current_key_valid_mask.device != hidden_states.device:
            raise ValueError("current mask and hidden states must share a device")
        if past_key_value is not None:
            if not isinstance(past_key_value, AttentionKV):
                raise TypeError("past_key_value must be AttentionKV or None")
            if (
                past_key_value.batch_size != B
                or past_key_value.key.shape[1] != self.num_kv_heads
                or past_key_value.key.shape[3] != self.head_dim
                or past_key_value.value.shape[3] != self.head_dim
                or past_key_value.key.dtype != hidden_states.dtype
                or past_key_value.key.device != hidden_states.device
            ):
                raise ValueError("past_key_value topology does not match attention")
        
        # --- NEW: compute query-dependent gate from pre-norm hidden_states ---
        gate = None
        if self.use_output_gate:
            if self.headwise_attn_output_gate:
                # [B, T, num_heads] -> [B, num_heads, T, 1]
                gate_logits = self.gate_proj(hidden_states)
                gate = torch.sigmoid(gate_logits).view(B, T, self.num_heads, 1)
                gate = gate.transpose(1, 2)  # [B, nh, T, 1]
            else:
                # elementwise: [B, T, H] -> [B, nh, T, hd]
                gate_logits = self.gate_proj(hidden_states)
                gate = torch.sigmoid(
                    gate_logits.view(B, T, self.num_heads, self.head_dim)
                )
                gate = gate.transpose(1, 2)  # [B, nh, T, hd]

        # 1) 线性映射得到 Q/K/V
        # Q: [B, T, nh * hd] K/V: [B, T, n_kv * hd]
        q = self.q_proj(hidden_states)  # [B, T, H]
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # 2) reshape 成多头形式
        # Q: [B, nh, T, hd] K/V: [B, n_kv, T, hd]
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 3) RoPE 位置编码（先在 n_kv heads 上加，再 repeat）
        if rotary_emb is not None and position_ids is not None:
            # position_ids: [B, T]
            q = rotary_emb(q, position_ids)  # [B, nh, T, hd]
            k = rotary_emb(k, position_ids)

        current_k = k
        current_v = v
        if past_key_value is None:
            combined_k = current_k
            combined_v = current_v
            combined_mask = current_key_valid_mask
        else:
            combined_k = torch.cat((past_key_value.key, current_k), dim=2)
            combined_v = torch.cat((past_key_value.value, current_v), dim=2)
            combined_mask = torch.cat(
                (
                    past_key_value.key_valid_mask,
                    current_key_valid_mask,
                ),
                dim=1,
            )
        expected_bias_shape = (B, 1, T, combined_k.shape[2])
        if attention_bias is None or attention_bias.shape != expected_bias_shape:
            raise ValueError(
                "attention_bias must have shape [B, 1, Q, P+Q]"
            )
        if attention_bias.device != hidden_states.device:
            raise ValueError("attention_bias and hidden states must share a device")
        if not attention_bias.is_floating_point():
            raise TypeError("attention_bias must have a floating dtype")

        # 4) 如果 num_kv_heads < num_heads，则复制 KV 以实现 GQA/MQA
        if self.num_kv_heads != self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = self._repeat_kv(combined_k, n_rep)
            v = self._repeat_kv(combined_v, n_rep)
        else:
            k = combined_k
            v = combined_v
            
        # 5) SDPA/FlashAttention
        if self.use_flash_attention:
            attn_mask = attention_bias
            
            # [FIX 1] Dtype 修正：必须让 bias (mask) 的类型匹配 query 的类型 (bf16/fp16)
            if attn_mask is not None and attn_mask.dtype != q.dtype:
                attn_mask = attn_mask.to(q.dtype)

            # [FIX 2] API 修正：移除旧版 sdpa_kernel(enable_flash=True)
            # PyTorch 2.4+ 默认会自动尝试 FlashAttention，无需强制 context
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False,
            )
        else:
            q_float = q.float()
            k_float = k.float()
            value_float = v.float()
            scores = torch.matmul(
                q_float,
                k_float.transpose(-1, -2),
            ) / math.sqrt(self.head_dim)
            if attention_bias is not None:
                # [FIX 1] 这里也加上 dtype 转换更安全
                if attention_bias.dtype != scores.dtype:
                    attention_bias = attention_bias.to(scores.dtype)
                scores = scores + attention_bias

            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn, value_float).to(dtype=v.dtype)
            
        # 6) Gate & Output
        if gate is not None:
            out = out * gate  # 形状兼容自动广播

        # 6) 合并 heads + 输出投影
        out = out.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        out = self.o_proj(out) # [B, T, hidden_size]
        out = out.masked_fill(~current_key_valid_mask.unsqueeze(-1), 0)

        present = None
        if use_cache:
            present = AttentionKV(
                key=combined_k,
                value=combined_v,
                key_valid_mask=combined_mask,
            )
        return out, present


class CausalConv1d(nn.Module):
    """
    一维因果卷积：
      - 输入: [B, T, H]
      - 输出: [B, T, H]
    只看当前及之前的 token（左填充）。
    """

    def __init__(self, hidden_size: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        assert kernel_size > 0 and kernel_size % 2 == 1, "kernel_size 必须为正奇数"
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.dilation = dilation

        # Conv1d 输入 [B, C, T]，这里 C=hidden_size
        self.conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            dilation=dilation,
            bias=True,
        )

        # 左填充长度 = (kernel_size - 1) * dilation
        self.pad = (kernel_size - 1) * dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, H]
        """
        bsz, seq_len, hidden = x.size()
        x = x.transpose(1, 2)
        x = F.pad(x, (self.pad, 0))
        x = self.conv(x)  # [B, H, T]
        x = x.transpose(1, 2)  # [B, T, H]
        return x
    
class GatedDeltaNetAttention(nn.Module):
    """
    简化版 Gated DeltaNet：
      - 前端 CausalConv1d 捕捉 N-gram
      - 线性投影到 Q/V（这里简单不用 K）
      - 使用 per-head 时间尺度门控 (A_log, dt_bias)
      - 通过逐步更新“状态向量” state_t 得到输出

    注意：
      - 这里的实现是结构/接口对齐版，便于后续替换为 fused kernel。
      - 当前版本时间复杂度 O(B * H * T * D)，适合先在中小模型上验证。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        conv_kernel_size: int = 3,
        num_heads_for_dt: Optional[int] = None,
        chunk_size: int = 0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.chunk_size = chunk_size

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(f"hidden_size must be divisible by num_heads")

        # 1) 前端因果卷积
        self.causal_conv = CausalConv1d(
            hidden_size=hidden_size,
            kernel_size=conv_kernel_size,
        )

        # 2) Q / V 投影（这里用 num_heads；后续可以扩展为 QK/V 不同 head 数）
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)

        # 3) 时间尺度门控：dt 投影到每个 head
        if num_heads_for_dt is None:
            num_heads_for_dt = num_heads
        self.num_heads_for_dt = num_heads_for_dt

        # 从 hidden_size 投影到 per-head dt，形状 [B, T, num_heads_for_dt]
        self.dt_proj = nn.Linear(hidden_size, num_heads_for_dt, bias=True)

        # A_log & dt_bias: 参考 Mamba/DeltaNet 风格的时间尺度参数化
        self.A_log = nn.Parameter(torch.zeros(num_heads_for_dt))
        self.dt_bias = nn.Parameter(torch.zeros(num_heads_for_dt))

        # 输出投影
        self.out_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        rotary_emb: Optional[RotaryEmbedding] = None,
    ) -> torch.Tensor:
        """
        hidden_states: [B, T, H]
        attention_mask: [B, 1, 1, T]（目前未使用，可用于将来做 padding/segment）
        position_ids: [B, T]（可选，用于对 Q 做 RoPE）
        """

        B, T, H = hidden_states.size()
        device = hidden_states.device
        dtype = hidden_states.dtype

        # 1) Causal Conv 提取局部模式
        x = self.causal_conv(hidden_states)  # [B, T, H]

        # 2) 线性投影到 Q / V
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        # q, v: [B, num_heads, T, head_dim]

        # 3) 可选：对 Q 做 RoPE（和 Attention 保持一致的接口）
        if rotary_emb is not None and position_ids is not None:
            q = rotary_emb(q, position_ids)

        dt_raw = self.dt_proj(hidden_states).transpose(1, 2)
        dt = F.softplus(dt_raw + self.dt_bias.view(1, -1, 1))
        A = -self.A_log.exp().view(1, -1, 1)
        alpha = torch.exp(A * dt)
        beta = dt
        
        # 5) 为了简单，将 dt 的 head 维对齐到 num_heads（若数量不同则 broadcast）
        if self.num_heads_for_dt != self.num_heads:
            # [B, H_dt, T] -> [B, num_heads, T] (repeat / crop)
            if self.num_heads_for_dt > self.num_heads:
                alpha = alpha[:, : self.num_heads, :]
                beta = beta[:, : self.num_heads, :]
            else:
                repeat_factor = (self.num_heads + self.num_heads_for_dt - 1) // self.num_heads_for_dt
                alpha = alpha.repeat_interleave(repeat_factor, dim=1)[:, : self.num_heads, :]
                beta = beta.repeat_interleave(repeat_factor, dim=1)[:, : self.num_heads, :]
        
        eps = 1e-6
        alpha = alpha.to(dtype)
        beta = beta.to(dtype)

        p = torch.cumprod(alpha, dim=2)
        contrib = beta.unsqueeze(-1) * v
        inv_p = 1.0 / (p.unsqueeze(-1) + eps)
        u = contrib * inv_p
        s = torch.cumsum(u, dim=2)
        state = p.unsqueeze(-1) * s
        out = q * state

        out = out.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        out = self.out_proj(out)  # [B, T, H]
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
        
        # 当前层的 block 类型：'attn' or 'deltanet'
        self.block_type = getattr(config, "block_type", "attn")

        head_dim = config.hidden_size // config.num_attention_heads
        if self.block_type == "deltanet":
            # DeltaNet 占位实现：Causal Conv1d + gated temporal dynamics
            kernel_size = getattr(config, "deltanet_kernel_size", 3)
            num_heads_for_dt = getattr(
                config, "deltanet_num_heads", config.num_attention_heads
            )
            chunk_size = getattr(config, "deltanet_chunk_size", 0)

            self.self_attn = GatedDeltaNetAttention(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                head_dim=head_dim,
                conv_kernel_size=kernel_size,
                num_heads_for_dt=num_heads_for_dt,
                chunk_size=chunk_size,
            )
        else:
            # 标准 MultiHeadSelfAttention（GatedAttention）
            self.self_attn = MultiHeadSelfAttention(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                head_dim=head_dim,
                use_flash_attention=getattr(config, "use_flash_attention", False),
                headwise_attn_output_gate=getattr(
                    config, "headwise_attn_output_gate", False
                ),
                elementwise_attn_output_gate=getattr(
                    config, "elementwise_attn_output_gate", False
                ),
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
                use_shared_expert=getattr(config, "moe_shared_expert", True),
                shared_intermediate_size=getattr(config, "moe_shared_intermediate_size", None),
                router_init_std=getattr(config, "moe_router_init_std", 1e-3),
                router_normalize_init=getattr(config, "moe_router_normalize_init", True),
                renormalize_topk=getattr(config, "moe_renormalize_topk", True),
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
        *,
        past_key_value: AttentionKV | None = None,
        current_key_valid_mask: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], AttentionKV | None]:
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        if isinstance(self.self_attn, MultiHeadSelfAttention):
            attn_out, present_key_value = self.self_attn(
                hidden_states,
                attention_bias=attention_mask,
                position_ids=position_ids,
                rotary_emb=self.rotary_emb,
                past_key_value=past_key_value,
                current_key_valid_mask=current_key_valid_mask,
                use_cache=use_cache,
            )
        else:
            if use_cache or past_key_value is not None:
                raise ValueError("DeltaNet does not implement decoder cache")
            attn_out = self.self_attn(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                rotary_emb=self.rotary_emb,
            )
            present_key_value = None
        hidden_states = residual + attn_out

        # FFN: Shared Dense + Optional MoE
        residual = hidden_states
        hidden_states = self.mlp_norm(hidden_states)

        # Shared FFN
        shared_out = self.shared_mlp(hidden_states)

        aux_loss = None
        if self.moe_mlp is not None:
            moe_out, aux_loss = self.moe_mlp(hidden_states)
            mlp_out = shared_out + moe_out
        else:
            mlp_out = shared_out

        hidden_states = residual + mlp_out
        return hidden_states, aux_loss, present_key_value


class LegacyIncrementalDecodeMixin:
    """Typed legacy-only prefill/decode adapter shared by standard and TP."""

    _tp_local_shard_cache_validated = False

    @property
    def cache_support(self):
        return cache_support_for_legacy_layers(
            self.layers,
            protocol_implemented=True,
            tp_local_shard_validated=self._tp_local_shard_cache_validated,
        )

    def _require_cache_runtime(self) -> tuple[int, ...]:
        scan = require_incremental_decode_support(
            self.layers,
            protocol_implemented=True,
            tp_local_shard_validated=self._tp_local_shard_cache_validated,
        )
        if self.training:
            raise RuntimeError(
                "incremental decode requires model.eval() inference mode"
            )
        return scan.cacheable_layer_indices

    def _model_device_dtype(self) -> tuple[torch.device, torch.dtype]:
        weight = self.embed_tokens.weight
        return weight.device, weight.dtype

    def create_state_owner(self, display_request_id: str) -> StateOwner:
        """Create one request owner, synchronized across a TP group."""

        local_owner: StateOwner | None = None
        validation_error: Exception | None = None
        try:
            local_owner = StateOwner.fresh(display_request_id)
        except Exception as error:
            validation_error = error
        self._synchronize_precompute_error(validation_error)
        assert local_owner is not None

        if not self._tp_local_shard_cache_validated:
            return local_owner

        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            return local_owner
        from qwen3_omni_pretrain.parallel import (
            get_tensor_model_parallel_group,
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        if get_tensor_model_parallel_world_size() <= 1:
            return local_owner
        device, _ = self._model_device_dtype()
        nonce_values = (
            tuple(local_owner.nonce)
            if get_tensor_model_parallel_rank() == 0
            else (0,) * len(local_owner.nonce)
        )
        nonce = torch.tensor(
            nonce_values,
            dtype=torch.int32,
            device=device,
        )
        dist.all_reduce(
            nonce,
            op=dist.ReduceOp.SUM,
            group=get_tensor_model_parallel_group(),
        )
        synchronized = StateOwner(
            display_request_id,
            bytes(int(value) for value in nonce.cpu().tolist()),
        )
        self._synchronize_precompute_error(None, owner=synchronized)
        return synchronized

    def _synchronize_precompute_error(
        self,
        error: Exception | None,
        *,
        owner: StateOwner | None = None,
    ) -> None:
        """Make TP validation and owner identity collective before compute."""

        if self._tp_local_shard_cache_validated:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                from qwen3_omni_pretrain.parallel import (
                    get_tensor_model_parallel_group,
                    get_tensor_model_parallel_world_size,
                )

                if get_tensor_model_parallel_world_size() > 1:
                    device, _ = self._model_device_dtype()
                    failed = torch.tensor(
                        [int(error is not None)],
                        dtype=torch.int32,
                        device=device,
                    )
                    dist.all_reduce(
                        failed,
                        op=dist.ReduceOp.MAX,
                        group=get_tensor_model_parallel_group(),
                    )
                    if bool(failed.item()):
                        if error is not None:
                            raise error
                        raise RuntimeError(
                            "a tensor-parallel peer failed cache preflight"
                        )
                    if owner is not None:
                        digest = hashlib.sha256(
                            owner.display_request_id.encode("utf-8")
                            + b"\0"
                            + owner.nonce
                        ).digest()
                        local_digest = torch.tensor(
                            tuple(digest),
                            dtype=torch.int32,
                            device=device,
                        )
                        gathered = [
                            torch.empty_like(local_digest)
                            for _ in range(
                                get_tensor_model_parallel_world_size()
                            )
                        ]
                        dist.all_gather(
                            gathered,
                            local_digest,
                            group=get_tensor_model_parallel_group(),
                        )
                        if any(
                            not torch.equal(gathered[0], candidate)
                            for candidate in gathered[1:]
                        ):
                            raise CacheCapabilityError(
                                CacheErrorCode.STATE_OWNER_MISMATCH,
                                "tensor-parallel ranks must share one state owner",
                            )
        if error is not None:
            raise error

    def _validate_query_boundary(
        self,
        *,
        key_valid_mask: torch.Tensor,
        position_batch,
        require_position: bool,
        past_storage_length: int,
    ) -> torch.Tensor:
        device, _ = self._model_device_dtype()
        if key_valid_mask.device != device:
            raise ValueError("model inputs and parameters must share a device")
        if bool((~key_valid_mask.any(dim=1)).any().item()):
            raise ValueError("every batch row must contain a valid token")
        total_storage_length = past_storage_length + key_valid_mask.shape[1]
        max_positions = self.thinker_cfg.max_position_embeddings
        if total_storage_length > max_positions:
            raise CacheCapabilityError(
                CacheErrorCode.CONTEXT_OVERFLOW,
                "cached storage length exceeds max_position_embeddings",
            )
        if position_batch is None:
            if require_position:
                raise ValueError("cache runtime requires an explicit PositionBatch")
            return torch.arange(
                key_valid_mask.shape[1],
                dtype=torch.long,
                device=device,
            ).unsqueeze(0).expand(key_valid_mask.shape[0], -1)
        try:
            return legacy_position_ids(
                position_batch,
                key_valid_mask,
                max_position_embeddings=max_positions,
            )
        except (TypeError, ValueError) as error:
            if "max_position_embeddings" in str(error):
                raise CacheCapabilityError(
                    CacheErrorCode.CONTEXT_OVERFLOW,
                    str(error),
                ) from error
            raise

    def _validate_payload_device_dtype(
        self,
        inputs: ModelPrefillInputs,
    ) -> None:
        device, dtype = self._model_device_dtype()
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
        if inputs.inputs_embeds is not None and inputs.inputs_embeds.shape[2] != (
            self.thinker_cfg.hidden_size
        ):
            raise ValueError("inputs_embeds hidden size does not match the model")
        if inputs.input_ids is not None and bool(
            (inputs.input_ids >= self.vocab_size).any().item()
        ):
            raise ValueError("input_ids contain an ID outside the model vocabulary")
        if inputs.media is not None:
            raise ValueError(
                "text Thinker does not accept raw media; use the vision/audio wrapper"
            )

    def _validate_cache_topology(
        self,
        state: DecoderState,
        expected_layers: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        for layer_index in expected_layers:
            cache = state.full_attention_kv[layer_index]
            attention = self.layers[layer_index].self_attn
            expected_heads = getattr(
                attention,
                "num_kv_heads_per_partition",
                getattr(attention, "num_kv_heads", None),
            )
            expected_head_dim = getattr(attention, "head_dim", None)
            if (
                type(expected_heads) is not int
                or type(expected_head_dim) is not int
            ):
                raise TypeError("cacheable attention topology is unavailable")
            if (
                cache.key.shape[1] != expected_heads
                or cache.key.shape[3] != expected_head_dim
                or cache.value.shape[3] != expected_head_dim
                or cache.key.dtype != dtype
                or cache.key.device != device
            ):
                raise ValueError(
                    f"cache topology for layer {layer_index} does not match the model"
                )

    def _run_typed_layers(
        self,
        *,
        hidden_states: torch.Tensor,
        current_key_valid_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: dict[int, AttentionKV] | None,
        use_cache: bool,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        dict[int, AttentionKV],
    ]:
        past_mask = None
        if past_key_values:
            past_mask = next(iter(past_key_values.values())).key_valid_mask
        attention_bias = _build_rectangular_causal_bias(
            past_key_valid_mask=past_mask,
            current_key_valid_mask=current_key_valid_mask,
            dtype=hidden_states.dtype,
        )
        total_aux_loss = None
        candidates: dict[int, AttentionKV] = {}
        for layer_index, layer in enumerate(self.layers):
            past = (
                None
                if past_key_values is None
                else past_key_values.get(layer_index)
            )
            hidden_states, layer_aux, present = layer(
                hidden_states,
                attention_bias,
                position_ids,
                past_key_value=past,
                current_key_valid_mask=current_key_valid_mask,
                use_cache=use_cache,
            )
            if use_cache:
                if present is None:
                    raise RuntimeError(
                        f"cacheable layer {layer_index} returned no KV state"
                    )
                candidates[layer_index] = present
            if layer_aux is not None:
                total_aux_loss = (
                    layer_aux
                    if total_aux_loss is None
                    else total_aux_loss + layer_aux
                )
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits, total_aux_loss, candidates

    @staticmethod
    def _prefill_cursor(
        position_ids: torch.Tensor,
        key_valid_mask: torch.Tensor,
    ) -> LegacyPositionCursor:
        batch_size, query_length = key_valid_mask.shape
        next_positions = torch.full(
            (batch_size,),
            query_length,
            dtype=torch.long,
            device=position_ids.device,
        )
        for row in range(batch_size):
            valid = position_ids[row].masked_select(key_valid_mask[row])
            next_positions[row] = torch.maximum(
                next_positions[row],
                valid.max() + 1,
            )
        return LegacyPositionCursor(next_positions)

    @staticmethod
    def _decode_cursor(
        old: LegacyPositionCursor,
        position_ids: torch.Tensor,
        key_valid_mask: torch.Tensor,
    ) -> LegacyPositionCursor:
        query_length = key_valid_mask.shape[1]
        next_positions = old.next_storage_position + query_length
        for row in range(key_valid_mask.shape[0]):
            valid = position_ids[row].masked_select(key_valid_mask[row])
            next_positions[row] = torch.maximum(
                next_positions[row],
                valid.max() + 1,
            )
        return LegacyPositionCursor(next_positions)

    @staticmethod
    def _validate_decode_positions(
        cursor: LegacyPositionCursor,
        position_ids: torch.Tensor,
        key_valid_mask: torch.Tensor,
    ) -> None:
        for row in range(key_valid_mask.shape[0]):
            valid = position_ids[row].masked_select(key_valid_mask[row])
            if valid[0] < cursor.next_storage_position[row]:
                raise ValueError(
                    "decode positions cannot move before the storage cursor"
                )
            if valid.numel() > 1 and bool((valid[1:] <= valid[:-1]).any().item()):
                raise ValueError("valid legacy decode positions must increase")

    def prefill(
        self,
        *,
        inputs: ModelPrefillInputs,
        owner: StateOwner,
        use_cache: bool,
    ) -> CausalLMOutput:
        validation_error: Exception | None = None
        expected_layers: tuple[int, ...] = ()
        position_ids: torch.Tensor | None = None
        try:
            if not isinstance(inputs, ModelPrefillInputs):
                raise TypeError("inputs must be ModelPrefillInputs")
            if not isinstance(owner, StateOwner):
                raise TypeError("owner must be StateOwner")
            if type(use_cache) is not bool:
                raise TypeError("use_cache must be a boolean")
            if use_cache:
                expected_layers = self._require_cache_runtime()
            self._validate_payload_device_dtype(inputs)
            position_ids = self._validate_query_boundary(
                key_valid_mask=inputs.key_valid_mask,
                position_batch=inputs.position_batch,
                require_position=use_cache,
                past_storage_length=0,
            )
            if use_cache and tuple(range(len(self.layers))) != expected_layers:
                raise RuntimeError("cache layer scan did not cover every layer")
        except Exception as error:
            validation_error = error
        self._synchronize_precompute_error(
            validation_error,
            owner=owner if isinstance(owner, StateOwner) else None,
        )
        assert isinstance(inputs, ModelPrefillInputs)
        assert isinstance(owner, StateOwner)
        assert position_ids is not None

        with torch.inference_mode():
            hidden_states = (
                self.embed_tokens(inputs.input_ids)
                if inputs.input_ids is not None
                else inputs.inputs_embeds
            )
            assert hidden_states is not None
            logits, aux_loss, candidates = self._run_typed_layers(
                hidden_states=hidden_states,
                current_key_valid_mask=inputs.key_valid_mask,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=use_cache,
            )
            state = None
            if use_cache:
                assert inputs.position_batch is not None
                position = DecoderPositionState(
                    cached=inputs.position_batch,
                    key_valid_mask=inputs.key_valid_mask,
                    continuation=self._prefill_cursor(
                        position_ids,
                        inputs.key_valid_mask,
                    ),
                )
                state = DecoderState(
                    owner=owner,
                    seen_tokens=inputs.key_valid_mask.sum(dim=1).to(torch.long),
                    position=position,
                    full_attention_kv=candidates,
                    swa_kv={},
                    processed_media=None,
                    gdn_state=None,
                    talker_state=None,
                    mtp_state=None,
                    codec_state=None,
                )
            return CausalLMOutput(
                logits=logits,
                loss=None,
                ce_loss=None,
                aux_loss=aux_loss,
                decoder_state=state,
                hidden_states=None,
            )

    def decode(
        self,
        *,
        inputs: ModelDecodeInputs,
        owner: StateOwner,
    ) -> CausalLMOutput:
        validation_error: Exception | None = None
        state: DecoderState | None = None
        expected_layers: tuple[int, ...] = ()
        position_ids: torch.Tensor | None = None
        try:
            if not isinstance(inputs, ModelDecodeInputs):
                raise TypeError("inputs must be ModelDecodeInputs")
            if not isinstance(owner, StateOwner):
                raise TypeError("owner must be StateOwner")
            state = inputs.decoder_state
            state.assert_owner(owner)
            expected_layers = self._require_cache_runtime()
            state.validate_full_attention_layers(expected_layers)
            if state.position is None:
                raise ValueError("decode state requires complete position history")
            if not isinstance(
                state.position.continuation,
                LegacyPositionCursor,
            ):
                raise ValueError("legacy Thinker requires a LegacyPositionCursor")
            if not torch.equal(
                state.seen_tokens,
                state.position.key_valid_mask.sum(dim=1).to(torch.long),
            ):
                raise ValueError("seen_tokens and cached valid positions disagree")
            device, dtype = self._model_device_dtype()
            if inputs.token_ids.device != device:
                raise ValueError("model input and parameters must share a device")
            if bool((inputs.token_ids >= self.vocab_size).any().item()):
                raise ValueError("token_ids contain an ID outside the model vocabulary")
            legacy_position_ids(
                state.position.cached,
                state.position.key_valid_mask,
                max_position_embeddings=self.thinker_cfg.max_position_embeddings,
            )
            self._validate_cache_topology(
                state,
                expected_layers,
                dtype,
                device,
            )
            position_ids = self._validate_query_boundary(
                key_valid_mask=inputs.current_key_valid_mask,
                position_batch=inputs.position_batch,
                require_position=True,
                past_storage_length=state.position.sequence_length,
            )
            self._validate_decode_positions(
                state.position.continuation,
                position_ids,
                inputs.current_key_valid_mask,
            )
        except Exception as error:
            validation_error = error
        self._synchronize_precompute_error(
            validation_error,
            owner=owner if isinstance(owner, StateOwner) else None,
        )
        assert isinstance(inputs, ModelDecodeInputs)
        assert isinstance(owner, StateOwner)
        assert state is not None and state.position is not None
        assert position_ids is not None

        with torch.inference_mode():
            hidden_states = self.embed_tokens(inputs.token_ids)
            logits, aux_loss, candidates = self._run_typed_layers(
                hidden_states=hidden_states,
                current_key_valid_mask=inputs.current_key_valid_mask,
                position_ids=position_ids,
                past_key_values=dict(state.full_attention_kv),
                use_cache=True,
            )
            continuation = self._decode_cursor(
                state.position.continuation,
                position_ids,
                inputs.current_key_valid_mask,
            )
            position = state.position.append(
                current=inputs.position_batch,
                current_key_valid_mask=inputs.current_key_valid_mask,
                continuation=continuation,
            )
            candidate = DecoderState(
                owner=state.owner,
                seen_tokens=(
                    state.seen_tokens
                    + inputs.current_key_valid_mask.sum(dim=1).to(torch.long)
                ),
                position=position,
                full_attention_kv=candidates,
                swa_kv=state.swa_kv,
                processed_media=state.processed_media,
                gdn_state=state.gdn_state,
                talker_state=state.talker_state,
                mtp_state=state.mtp_state,
                codec_state=state.codec_state,
            )
            return CausalLMOutput(
                logits=logits,
                loss=None,
                ce_loss=None,
                aux_loss=aux_loss,
                decoder_state=candidate,
                hidden_states=None,
            )


class Qwen3OmniMoeThinkerTextModel(
    LegacyIncrementalDecodeMixin,
    PreTrainedModel,
):
    config_class = Qwen3OmniMoeConfig
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Qwen3OmniMoeConfig):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.max_position_embeddings = config.max_position_embeddings

        thinker_cfg = config.thinker_config
        assert isinstance(thinker_cfg, Qwen3OmniMoeThinkerConfig)
        self.thinker_cfg = thinker_cfg

        self.embed_tokens = nn.Embedding(
            config.vocab_size, thinker_cfg.hidden_size
        )

        # head_dim 用于 RoPE
        self.head_dim = thinker_cfg.hidden_size // thinker_cfg.num_attention_heads

        # 从 config 中读取部分 RoPE 比例，默认 1.0（全维度）
        rope_partial_factor = getattr(config, "rope_partial_factor", 1.0)
        # 计算真正参与 RoPE 的维度
        rope_dim = int(self.head_dim * rope_partial_factor)
        # 保证非负且不超过 head_dim，偶数对齐
        rope_dim = max(0, min(rope_dim, self.head_dim))
        if rope_dim % 2 == 1:
            rope_dim -= 1

        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=thinker_cfg.max_position_embeddings,
            base=config.rope_theta,
            rope_dim=rope_dim,
        )

        self.use_flash_attention = getattr(thinker_cfg, "use_flash_attention", False)

        moe_layers = parse_layer_indices(
            thinker_cfg.moe_layer_indices,
            layer_count=thinker_cfg.num_hidden_layers,
            field="moe_layer_indices",
        )
        moe_layer_set = set(moe_layers) if moe_layers else None

        deltanet_layers = parse_layer_indices(
            thinker_cfg.deltanet_layer_indices,
            layer_count=thinker_cfg.num_hidden_layers,
            field="deltanet_layer_indices",
        )
        deltanet_layer_set = (
            set(deltanet_layers) if deltanet_layers else None
        )

        use_deltanet_global = getattr(thinker_cfg, "use_deltanet", False)
        self.gradient_checkpointing = getattr(thinker_cfg, "gradient_checkpointing", False)

        layers = []
        for layer_idx in range(thinker_cfg.num_hidden_layers):
            # 默认按 config.use_moe
            use_moe_layer = thinker_cfg.use_moe
            # 如果指定了 moe_layer_indices，则以它为准
            if moe_layer_set is not None:
                use_moe_layer = layer_idx in moe_layer_set
                
            # ---- Attention / DeltaNet 层类型选择 ----
            block_type = "attn"
            if use_deltanet_global:
                if deltanet_layer_set is not None:
                    # 显式指定哪些层用 DeltaNet
                    block_type = "deltanet" if layer_idx in deltanet_layer_set else "attn"
                else:
                    # 默认 3:1 模式：每 4 层中前 3 层 DeltaNet，最后 1 层标准 Attention
                    if (layer_idx % 4) in (0, 1, 2):
                        block_type = "deltanet"
                    else:
                        block_type = "attn"

            # 为当前层构造一个“局部Config”拷贝，覆盖 use_moe
            local_cfg = Qwen3OmniMoeThinkerConfig(**thinker_cfg.__dict__)
            local_cfg.use_moe = use_moe_layer
            local_cfg.block_type = block_type

            layers.append(ThinkerDecoderLayer(local_cfg, self.rotary_emb))

        self.layers = nn.ModuleList(layers)
        self.norm = RMSNorm(thinker_cfg.hidden_size)
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
        *,
        dtype: torch.dtype = torch.float32,
        past_key_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, tgt_len = input_shape
        if attention_mask is None:
            attention_mask = torch.ones(
                (bsz, tgt_len),
                dtype=torch.bool,
                device=device,
            )
        return _build_rectangular_causal_bias(
            past_key_valid_mask=past_key_valid_mask,
            current_key_valid_mask=attention_mask,
            dtype=dtype,
        )

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

        if attention_mask is None:
            current_key_valid_mask = torch.ones(
                (bsz, seq_len),
                dtype=torch.bool,
                device=device,
            )
        else:
            if not isinstance(attention_mask, torch.Tensor):
                raise TypeError("attention_mask must be a torch.Tensor")
            if attention_mask.shape != (bsz, seq_len):
                raise ValueError("attention_mask must have shape [B, T]")
            if attention_mask.device != device:
                raise ValueError("attention_mask and inputs must share a device")
            if not bool(
                ((attention_mask == 0) | (attention_mask == 1)).all().item()
            ):
                raise ValueError("attention_mask values must be 0 or 1")
            current_key_valid_mask = attention_mask.to(dtype=torch.bool)

        attention_mask_full = self._prepare_attention_mask(
            current_key_valid_mask,
            (bsz, seq_len),
            device,
            dtype=hidden_states.dtype,
        )

        all_hidden_states = [] if output_hidden_states else None
        total_aux_loss = None

        # 1. 堆叠所有解码层，顺便把 MoE aux loss 累加起来
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            if self.gradient_checkpointing and self.training:
                def layer_forward(x, pos_ids):
                    layer_hidden, layer_aux, _ = layer(
                        x,
                        attention_mask_full,
                        pos_ids,
                        current_key_valid_mask=current_key_valid_mask,
                    )
                    return layer_hidden, layer_aux

                hidden_states, layer_aux = torch.utils.checkpoint.checkpoint(
                    layer_forward,
                    hidden_states,
                    position_ids,
                    use_reentrant=False,
                )
            else:
                hidden_states, layer_aux, _ = layer(
                    hidden_states,
                    attention_mask_full,
                    position_ids,
                    current_key_valid_mask=current_key_valid_mask,
                )
            if layer_aux is not None:
                if total_aux_loss is None:
                    total_aux_loss = layer_aux
                else:
                    total_aux_loss = total_aux_loss + layer_aux

        # 2. 最后一层 RMSNorm + LM Head
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)  # [B, T, V]

        loss = None
        ce_loss = None
        aux_loss = None
        vocab_size = logits.size(-1)

        # 3. 自回归 CE loss
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()   # [B, T-1, V]
            shift_labels = labels[:, 1:].contiguous()       # [B, T-1]

            valid_mask = (shift_labels != -100)
            num_tokens = valid_mask.sum()

            loss_fct = nn.CrossEntropyLoss(
                ignore_index=-100,
                reduction="sum",
            )
            ce_loss_raw = loss_fct(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1),
            )

            if num_tokens > 0:
                ce_loss = ce_loss_raw / num_tokens
            else:
                ce_loss = ce_loss_raw * 0.0
            loss = ce_loss

        # 4. MoE aux loss 归一化 + 合并
        if total_aux_loss is not None:
            if labels is not None:
                if "num_tokens" not in locals():
                    valid_mask = (labels != -100)
                    num_tokens = valid_mask.sum()
                if num_tokens > 0:
                    aux_loss = total_aux_loss / num_tokens
                else:
                    aux_loss = total_aux_loss * 0.0
            else:
                aux_loss = total_aux_loss

            if loss is None:
                loss = self.thinker_cfg.moe_aux_loss_coef * aux_loss
            else:
                loss = loss + self.thinker_cfg.moe_aux_loss_coef * aux_loss

        # 5. 把 NaN/Inf 压到有限值
        if loss is not None:
            loss = torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=-1e4)
        if ce_loss is not None:
            ce_loss = torch.nan_to_num(ce_loss, nan=0.0, posinf=1e4, neginf=-1e4)
        if aux_loss is not None:
            aux_loss = torch.nan_to_num(aux_loss, nan=0.0, posinf=1e4, neginf=-1e4)

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
