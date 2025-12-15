# src/qwen3_omni_pretrain/parallel/layers.py
"""
Tensor Parallel versions of Attention and MLP layers.

These layers wrap the standard implementations with TP-aware linear layers,
allowing the same model architecture to run with or without tensor parallelism.

Usage:
    # Standard mode (no TP)
    attn = TensorParallelMultiHeadAttention(...)
    
    # TP mode (after initialize_model_parallel)
    from qwen3_omni_pretrain.parallel import initialize_model_parallel
    initialize_model_parallel(tensor_model_parallel_size=2)
    attn = TensorParallelMultiHeadAttention(...)  # Automatically uses TP
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from qwen3_omni_pretrain.parallel.initialize import (
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank,
    is_model_parallel_initialized,
)
from qwen3_omni_pretrain.parallel.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    copy_to_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
)


class TensorParallelMultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention with Tensor Parallelism support.
    
    When TP is enabled:
    - Q, K, V projections use ColumnParallelLinear (split heads across TP group)
    - O projection uses RowParallelLinear (reduce partial results)
    
    When TP is disabled:
    - Falls back to standard nn.Linear layers
    
    Args:
        hidden_size: Model hidden size
        num_heads: Total number of attention heads
        num_kv_heads: Number of key/value heads (for GQA)
        head_dim: Dimension per head (default: hidden_size // num_heads)
        use_flash_attention: Whether to use PyTorch SDPA
        headwise_attn_output_gate: Use head-wise output gate
        elementwise_attn_output_gate: Use element-wise output gate
        use_tensor_parallel: Force enable/disable TP (default: auto-detect)
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int = None,
        use_flash_attention: bool = True,
        headwise_attn_output_gate: bool = False,
        elementwise_attn_output_gate: bool = False,
        use_tensor_parallel: bool = None,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.use_flash_attention = use_flash_attention
        
        # Check dimensions
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(f"hidden_size must be divisible by num_heads")
        if self.num_kv_heads <= 0 or (self.num_heads % self.num_kv_heads) != 0:
            raise ValueError(f"num_heads must be a multiple of num_kv_heads for GQA")
        
        # Determine if TP should be used
        if use_tensor_parallel is None:
            use_tensor_parallel = is_model_parallel_initialized() and get_tensor_model_parallel_world_size() > 1
        self.use_tensor_parallel = use_tensor_parallel
        
        tp_world_size = get_tensor_model_parallel_world_size() if use_tensor_parallel else 1
        
        # Validate TP compatibility
        if use_tensor_parallel:
            if num_heads % tp_world_size != 0:
                raise ValueError(
                    f"num_heads ({num_heads}) must be divisible by TP world size ({tp_world_size})"
                )
            if num_kv_heads % tp_world_size != 0:
                raise ValueError(
                    f"num_kv_heads ({num_kv_heads}) must be divisible by TP world size ({tp_world_size})"
                )
        
        # Per-partition dimensions
        self.num_heads_per_partition = num_heads // tp_world_size
        self.num_kv_heads_per_partition = num_kv_heads // tp_world_size
        
        # QKV projections
        q_out_dim = num_heads * self.head_dim
        kv_out_dim = num_kv_heads * self.head_dim
        
        if use_tensor_parallel:
            # Column parallel: split output dimension
            self.q_proj = ColumnParallelLinear(
                hidden_size, q_out_dim, bias=False, gather_output=False
            )
            self.k_proj = ColumnParallelLinear(
                hidden_size, kv_out_dim, bias=False, gather_output=False
            )
            self.v_proj = ColumnParallelLinear(
                hidden_size, kv_out_dim, bias=False, gather_output=False
            )
            # Row parallel: reduce partial results
            self.o_proj = RowParallelLinear(
                q_out_dim, hidden_size, bias=False, input_is_parallel=True
            )
        else:
            # Standard linear layers
            self.q_proj = nn.Linear(hidden_size, q_out_dim, bias=False)
            self.k_proj = nn.Linear(hidden_size, kv_out_dim, bias=False)
            self.v_proj = nn.Linear(hidden_size, kv_out_dim, bias=False)
            self.o_proj = nn.Linear(q_out_dim, hidden_size, bias=False)
        
        # Output gating
        self.headwise_attn_output_gate = headwise_attn_output_gate
        self.elementwise_attn_output_gate = elementwise_attn_output_gate
        self.use_output_gate = headwise_attn_output_gate or elementwise_attn_output_gate
        
        if self.headwise_attn_output_gate:
            if use_tensor_parallel:
                self.gate_proj = ColumnParallelLinear(
                    hidden_size, num_heads, bias=True, gather_output=False
                )
            else:
                self.gate_proj = nn.Linear(hidden_size, num_heads, bias=True)
        elif self.elementwise_attn_output_gate:
            if use_tensor_parallel:
                self.gate_proj = ColumnParallelLinear(
                    hidden_size, hidden_size, bias=True, gather_output=False
                )
            else:
                self.gate_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        else:
            self.gate_proj = None
    
    @staticmethod
    def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat KV heads for GQA."""
        batch, num_kv_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_kv_heads, n_rep, slen, head_dim
        )
        return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        rotary_emb: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            hidden_states: Input tensor [B, T, H]
            attention_mask: Attention mask [B, 1, T, T] or [B, 1, 1, T]
            position_ids: Position IDs for RoPE [B, T]
            rotary_emb: RoPE embedding module
            
        Returns:
            Output tensor [B, T, H]
        """
        B, T, _ = hidden_states.size()
        
        # Compute output gate
        gate = None
        if self.use_output_gate:
            if self.headwise_attn_output_gate:
                gate_logits = self.gate_proj(hidden_states)
                gate = torch.sigmoid(gate_logits).view(B, T, self.num_heads_per_partition, 1)
                gate = gate.transpose(1, 2)  # [B, nh_local, T, 1]
            else:
                gate_logits = self.gate_proj(hidden_states)
                gate = torch.sigmoid(
                    gate_logits.view(B, T, self.num_heads_per_partition, self.head_dim)
                )
                gate = gate.transpose(1, 2)  # [B, nh_local, T, hd]
        
        # QKV projections
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape to multi-head format
        # With TP: [B, T, nh_local * hd] -> [B, nh_local, T, hd]
        q = q.view(B, T, self.num_heads_per_partition, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_kv_heads_per_partition, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_kv_heads_per_partition, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        if rotary_emb is not None and position_ids is not None:
            q = rotary_emb(q, position_ids)
            k = rotary_emb(k, position_ids)
        
        # Repeat KV for GQA
        if self.num_kv_heads_per_partition != self.num_heads_per_partition:
            n_rep = self.num_heads_per_partition // self.num_kv_heads_per_partition
            k = self._repeat_kv(k, n_rep)
            v = self._repeat_kv(v, n_rep)
        
        # Attention computation
        if self.use_flash_attention:
            attn_mask = attention_mask
            if attn_mask is not None and attn_mask.dtype != q.dtype:
                attn_mask = attn_mask.to(q.dtype)
            
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False,
            )
        else:
            scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                if attention_mask.dtype != scores.dtype:
                    attention_mask = attention_mask.to(scores.dtype)
                scores = scores + attention_mask
            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)
        
        # Apply gate
        if gate is not None:
            out = out * gate
        
        # Reshape and output projection
        out = out.transpose(1, 2).contiguous()
        out = out.view(B, T, self.num_heads_per_partition * self.head_dim)
        out = self.o_proj(out)
        
        return out


class TensorParallelMLP(nn.Module):
    """
    MLP layer with Tensor Parallelism support.
    
    Standard MLP: out = fc2(act(fc1(x)))
    With TP:
    - fc1 uses ColumnParallelLinear (split intermediate dimension)
    - fc2 uses RowParallelLinear (reduce partial results)
    
    Args:
        hidden_size: Model hidden size
        intermediate_size: MLP intermediate size
        activation: Activation function (default: SiLU)
        use_tensor_parallel: Force enable/disable TP (default: auto-detect)
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "silu",
        use_tensor_parallel: bool = None,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # Determine if TP should be used
        if use_tensor_parallel is None:
            use_tensor_parallel = is_model_parallel_initialized() and get_tensor_model_parallel_world_size() > 1
        self.use_tensor_parallel = use_tensor_parallel
        
        tp_world_size = get_tensor_model_parallel_world_size() if use_tensor_parallel else 1
        
        # Validate TP compatibility
        if use_tensor_parallel and intermediate_size % tp_world_size != 0:
            raise ValueError(
                f"intermediate_size ({intermediate_size}) must be divisible by "
                f"TP world size ({tp_world_size})"
            )
        
        if use_tensor_parallel:
            self.fc1 = ColumnParallelLinear(
                hidden_size, intermediate_size, bias=False, gather_output=False
            )
            self.fc2 = RowParallelLinear(
                intermediate_size, hidden_size, bias=False, input_is_parallel=True
            )
        else:
            self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        
        # Activation
        if activation == "silu":
            self.act = nn.SiLU()
        elif activation == "gelu":
            self.act = nn.GELU()
        elif activation == "relu":
            self.act = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: fc2(act(fc1(x)))"""
        return self.fc2(self.act(self.fc1(x)))


class TensorParallelGatedMLP(nn.Module):
    """
    Gated MLP (SwiGLU-style) with Tensor Parallelism support.
    
    Formula: out = down_proj(act(gate_proj(x)) * up_proj(x))
    
    With TP:
    - gate_proj, up_proj use ColumnParallelLinear
    - down_proj uses RowParallelLinear
    
    This is the common MLP variant used in LLaMA, Qwen, etc.
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "silu",
        use_tensor_parallel: bool = None,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # Determine if TP should be used
        if use_tensor_parallel is None:
            use_tensor_parallel = is_model_parallel_initialized() and get_tensor_model_parallel_world_size() > 1
        self.use_tensor_parallel = use_tensor_parallel
        
        tp_world_size = get_tensor_model_parallel_world_size() if use_tensor_parallel else 1
        
        if use_tensor_parallel and intermediate_size % tp_world_size != 0:
            raise ValueError(
                f"intermediate_size ({intermediate_size}) must be divisible by "
                f"TP world size ({tp_world_size})"
            )
        
        if use_tensor_parallel:
            # Gate and up projections (column parallel, no gather)
            self.gate_proj = ColumnParallelLinear(
                hidden_size, intermediate_size, bias=False, gather_output=False
            )
            self.up_proj = ColumnParallelLinear(
                hidden_size, intermediate_size, bias=False, gather_output=False
            )
            # Down projection (row parallel, reduces partial results)
            self.down_proj = RowParallelLinear(
                intermediate_size, hidden_size, bias=False, input_is_parallel=True
            )
        else:
            self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        
        # Activation
        if activation == "silu":
            self.act = nn.SiLU()
        elif activation == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: down_proj(act(gate_proj(x)) * up_proj(x))"""
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class TensorParallelExpertMLP(nn.Module):
    """
    Single expert MLP for MoE with Tensor Parallelism support.
    
    Same structure as TensorParallelMLP, but designed to be used within MoE layers.
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        use_tensor_parallel: bool = None,
    ):
        super().__init__()
        
        if use_tensor_parallel is None:
            use_tensor_parallel = is_model_parallel_initialized() and get_tensor_model_parallel_world_size() > 1
        self.use_tensor_parallel = use_tensor_parallel
        
        tp_world_size = get_tensor_model_parallel_world_size() if use_tensor_parallel else 1
        
        if use_tensor_parallel and intermediate_size % tp_world_size != 0:
            raise ValueError(
                f"intermediate_size ({intermediate_size}) must be divisible by "
                f"TP world size ({tp_world_size})"
            )
        
        if use_tensor_parallel:
            self.fc1 = ColumnParallelLinear(
                hidden_size, intermediate_size, bias=False, gather_output=False
            )
            self.fc2 = RowParallelLinear(
                intermediate_size, hidden_size, bias=False, input_is_parallel=True
            )
        else:
            self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        
        self.act = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))
