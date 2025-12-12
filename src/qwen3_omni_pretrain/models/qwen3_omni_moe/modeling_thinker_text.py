# src/qwen3_omni_pretrain/models/qwen3_omni_moe/modeling_thinker_text.py

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from .configuration_qwen3_omni_moe import Qwen3OmniMoeConfig, Qwen3OmniMoeThinkerConfig
from .modules.moe import Qwen3OmniMoeMLP



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

            cos = self.cos_cached[position_ids]  # [B, T, rope_dim]
            sin = self.sin_cached[position_ids]  # [B, T, rope_dim]

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

            cos = self.cos_cached[position_ids]  # [B, T, rope_dim]
            sin = self.sin_cached[position_ids]  # [B, T, rope_dim]
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
            raise ValueError(
                f"hidden_size must be divisible by num_heads "
                f"(got hidden_size={self.hidden_size}, num_heads={self.num_heads}, "
                f"head_dim={self.head_dim})"
            )
        if self.num_kv_heads <= 0 or (self.num_heads % self.num_kv_heads) != 0:
            raise ValueError(
                f"num_heads must be a multiple of num_kv_heads for GQA/MQA "
                f"(got num_heads={self.num_heads}, num_kv_heads={self.num_kv_heads})"
            )

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
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        rotary_emb: Optional[RotaryEmbedding] = None,
    ) -> torch.Tensor:
        B, T, _ = hidden_states.size()
        
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

        # 4) 如果 num_kv_heads < num_heads，则复制 KV 以实现 GQA/MQA
        if self.num_kv_heads != self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = self._repeat_kv(k, n_rep)  # [B, nh, T, hd]
            v = self._repeat_kv(v, n_rep)  # [B, nh, T, hd]
            
        # 5) SDPA/FlashAttention
        if self.use_flash_attention:
            # 使用 PyTorch SDPA/FlashAttention 内核（自动选择最佳实现）
            attn_mask = attention_mask  # broadcast: [B, 1, 1, T] -> [B, nh, T, T]
            try:
                sdpa_ctx = None
                if hasattr(torch.nn, "attention") and hasattr(torch.nn.attention, "sdpa_kernel"):
                    sdpa_ctx = torch.nn.attention.sdpa_kernel(
                        enable_flash=True,
                        enable_math=True,
                        enable_mem_efficient=True,
                    )
                else:
                    sdpa_ctx = torch.backends.cuda.sdp_kernel(
                        enable_flash=True,
                        enable_math=True,
                        enable_mem_efficient=True,
                    )

                with sdpa_ctx:
                    out = F.scaled_dot_product_attention(
                        q, k, v,
                        attn_mask=attn_mask,
                        dropout_p=0.0,
                        is_causal=False,
                    )
            except Exception:
                out = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=0.0,
                    is_causal=False,
                )
        else:
            scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)  # [B, nh, T, T]

            if attention_mask is not None:
                # attention_mask: [B, 1, 1, T] 或 [B, 1, T, T]
                scores = scores + attention_mask

            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)  # [B, nh, T, hd]
            
        # --- NEW: apply gate at SDPA output (G1 position) ---
        if gate is not None:
            out = out * gate  # 形状兼容自动广播

        # 6) 合并 heads + 输出投影
        out = out.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        out = self.o_proj(out) # [B, T, hidden_size]
        return out

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
        assert hidden == self.hidden_size

        # [B, T, H] -> [B, H, T]
        x = x.transpose(1, 2)
        # 只在左侧 padding，保持因果性
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
            raise ValueError(
                f"[GatedDeltaNet] hidden_size must be divisible by num_heads "
                f"(got hidden_size={self.hidden_size}, num_heads={self.num_heads}, "
                f"head_dim={self.head_dim})"
            )

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
            q = rotary_emb(q, position_ids)  # [B, num_heads, T, head_dim]

        # 4) 计算时间尺度门控参数
        # dt_raw: [B, T, H_dt] -> [B, H_dt, T]
        dt_raw = self.dt_proj(hidden_states)  # [B, T, num_heads_for_dt]
        dt_raw = dt_raw.transpose(1, 2)       # [B, num_heads_for_dt, T]

        # softplus 保证 dt>0，再加上 dt_bias
        dt = F.softplus(dt_raw + self.dt_bias.view(1, -1, 1))  # [B, H_dt, T]
        # A_log 控制衰减率
        A = -self.A_log.exp().view(1, -1, 1)                   # [B, H_dt, 1]
        # 连续时间到离散时间的转换，这里是简化版本
        alpha = torch.exp(A * dt)  # [B, H_dt, T]
        beta = dt                  # [B, H_dt, T]
        
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
        
        # 6) 逐时间步更新 state 并计算输出
        # 我们用 per-head state_t: [B, num_heads, head_dim]
        # state = torch.zeros(
        #     B, self.num_heads, self.head_dim, device=device, dtype=dtype
        # )
        
        # 6.1 计算 p_t = cumprod(alpha)
        #     alpha: [B, nh, T]

        eps = 1e-6
        alpha = alpha.to(dtype)                 # 保持精度一致
        beta = beta.to(dtype)

        p = torch.cumprod(alpha, dim=2)        # [B, nh, T]

        # 6.2 计算贡献项 contrib = beta * v
        # v: [B, nh, T, hd]
        contrib = beta.unsqueeze(-1) * v       # [B, nh, T, hd]

        # 6.3 u = contrib / p_i
        inv_p = 1.0 / (p.unsqueeze(-1) + eps)  # [B, nh, T, 1]
        u = contrib * inv_p                    # [B, nh, T, hd]

        # 6.4 前缀和 s_t = sum_{i<=t} u_i
        s = torch.cumsum(u, dim=2)             # [B, nh, T, hd]

        # 6.5 state_t = p_t * s_t
        state = p.unsqueeze(-1) * s            # [B, nh, T, hd]

        # 7) 输出：q_t * state_t
        out = q * state                        # [B, nh, T, hd]

        # 8) 合并 heads + 输出投影
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

        # 解析 moe_layer_indices: 例如 "0,2,4" -> {0,2,4}
        moe_layer_set = None
        if getattr(thinker_cfg, "moe_layer_indices", None) is not None:
            indices_str = thinker_cfg.moe_layer_indices
            if isinstance(indices_str, str) and indices_str.strip():
                moe_layer_set = set(
                    int(x) for x in indices_str.split(",") if x.strip().isdigit()
                )
                
        # 解析 deltanet_layer_indices: 例如 "0,1,2,4,5,6"
        deltanet_layer_set = None
        if getattr(thinker_cfg, "deltanet_layer_indices", None) is not None:
            indices_str = thinker_cfg.deltanet_layer_indices
            if isinstance(indices_str, str) and indices_str.strip():
                deltanet_layer_set = set(
                    int(x) for x in indices_str.split(",") if x.strip().isdigit()
                )

        use_deltanet_global = getattr(thinker_cfg, "use_deltanet", False)

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

        # 1. 堆叠所有解码层，顺便把 MoE aux loss 累加起来
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

