# src/qwen3_omni_pretrain/models/qwen3_omni_moe/modeling_thinker_text_tp.py
"""
Tensor Parallel version of Qwen3OmniMoeThinkerTextModel.

This module provides a TP-enabled variant of the Thinker model that can be used
when tensor parallelism is initialized. It is designed to be a drop-in replacement
for the standard model when `use_tensor_parallel=True` in the config.

Key differences from standard model:
1. Embedding: Uses VocabParallelEmbedding
2. Attention Q/K/V: Use ColumnParallelLinear
3. Attention O: Uses RowParallelLinear
4. MLP fc1: Uses ColumnParallelLinear
5. MLP fc2: Uses RowParallelLinear
6. LM Head: Uses ParallelLMHead (or tied with VocabParallelEmbedding)

Usage:
    from qwen3_omni_pretrain.parallel import initialize_model_parallel
    initialize_model_parallel(tensor_model_parallel_size=2)
    
    model = Qwen3OmniMoeThinkerTextModelTP(config)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from .configuration_qwen3_omni_moe import Qwen3OmniMoeConfig, Qwen3OmniMoeThinkerConfig
from .modeling_thinker_text import (
    RMSNorm,
    RotaryEmbedding,
    CausalConv1d,
    GatedDeltaNetAttention,
)

# Import parallel components
from qwen3_omni_pretrain.parallel import (
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank,
    is_model_parallel_initialized,
    VocabParallelEmbedding,
    ParallelLMHead,
    ColumnParallelLinear,
    RowParallelLinear,
    reduce_from_tensor_model_parallel_region,
)


class TensorParallelMultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention with Tensor Parallelism.
    
    Q, K, V projections use ColumnParallelLinear.
    O projection uses RowParallelLinear.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        use_flash_attention: bool = False,
        headwise_attn_output_gate: bool = False,
        elementwise_attn_output_gate: bool = False,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.use_flash_attention = use_flash_attention
        
        tp_world_size = get_tensor_model_parallel_world_size()
        
        # Check dimensions
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(f"hidden_size must be divisible by num_heads")
        if self.num_kv_heads <= 0 or (self.num_heads % self.num_kv_heads) != 0:
            raise ValueError(f"num_heads must be a multiple of num_kv_heads for GQA")
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
        
        # QKV projections with ColumnParallel (output dimension split)
        self.q_proj = ColumnParallelLinear(
            hidden_size, num_heads * head_dim, bias=False, gather_output=False
        )
        self.k_proj = ColumnParallelLinear(
            hidden_size, num_kv_heads * head_dim, bias=False, gather_output=False
        )
        self.v_proj = ColumnParallelLinear(
            hidden_size, num_kv_heads * head_dim, bias=False, gather_output=False
        )
        
        # Output projection with RowParallel (input dimension split)
        self.o_proj = RowParallelLinear(
            num_heads * head_dim, hidden_size, bias=False, input_is_parallel=True
        )
        
        # Output gating
        self.headwise_attn_output_gate = headwise_attn_output_gate
        self.elementwise_attn_output_gate = elementwise_attn_output_gate
        self.use_output_gate = headwise_attn_output_gate or elementwise_attn_output_gate
        
        if self.headwise_attn_output_gate:
            # Per-head gate: output per-partition heads
            self.gate_proj = ColumnParallelLinear(
                hidden_size, num_heads, bias=True, gather_output=False
            )
        elif self.elementwise_attn_output_gate:
            # Per-element gate: output per-partition hidden
            self.gate_proj = ColumnParallelLinear(
                hidden_size, hidden_size, bias=True, gather_output=False
            )
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
        rotary_emb: Optional[RotaryEmbedding] = None,
    ) -> torch.Tensor:
        B, T, _ = hidden_states.size()
        
        # Compute output gate
        gate = None
        if self.use_output_gate:
            if self.headwise_attn_output_gate:
                gate_logits = self.gate_proj(hidden_states)
                gate = torch.sigmoid(gate_logits).view(B, T, self.num_heads_per_partition, 1)
                gate = gate.transpose(1, 2)
            else:
                gate_logits = self.gate_proj(hidden_states)
                gate = torch.sigmoid(
                    gate_logits.view(B, T, self.num_heads_per_partition, self.head_dim)
                )
                gate = gate.transpose(1, 2)
        
        # QKV projections (already partitioned by ColumnParallel)
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape to multi-head format [B, T, nh_local * hd] -> [B, nh_local, T, hd]
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
            q_f = q.float()
            k_f = k.float()
            v_f = v.float()
            scores = torch.matmul(q_f, k_f.transpose(-1, -2)) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                if attention_mask.dtype != scores.dtype:
                    attention_mask = attention_mask.to(scores.dtype)
                scores = scores + attention_mask
            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn, v_f)
            out = out.to(v.dtype)
        
        # Apply gate
        if gate is not None:
            out = out * gate
        
        # Reshape and output projection
        out = out.transpose(1, 2).contiguous()
        out = out.view(B, T, self.num_heads_per_partition * self.head_dim)
        out = self.o_proj(out)
        
        return out


class TensorParallelMLP(nn.Module):
    """MLP with Tensor Parallelism."""
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        
        tp_world_size = get_tensor_model_parallel_world_size()
        if intermediate_size % tp_world_size != 0:
            raise ValueError(
                f"intermediate_size ({intermediate_size}) must be divisible by "
                f"TP world size ({tp_world_size})"
            )
        
        self.fc1 = ColumnParallelLinear(
            hidden_size, intermediate_size, bias=False, gather_output=False
        )
        self.fc2 = RowParallelLinear(
            intermediate_size, hidden_size, bias=False, input_is_parallel=True
        )
        self.act = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class TensorParallelExpertMLP(nn.Module):
    """Single expert MLP with Tensor Parallelism."""
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        
        tp_world_size = get_tensor_model_parallel_world_size()
        if intermediate_size % tp_world_size != 0:
            raise ValueError(
                f"intermediate_size ({intermediate_size}) must be divisible by "
                f"TP world size ({tp_world_size})"
            )
        
        self.fc1 = ColumnParallelLinear(
            hidden_size, intermediate_size, bias=False, gather_output=False
        )
        self.fc2 = RowParallelLinear(
            intermediate_size, hidden_size, bias=False, input_is_parallel=True
        )
        self.act = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class TensorParallelMoeMLP(nn.Module):
    """
    Sparse Top-k MoE MLP with Tensor Parallelism.
    
    Each expert is TP-parallelized. The router remains replicated.
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        *,
        use_shared_expert: bool = True,
        shared_intermediate_size: int = None,
        router_init_std: float = 1e-3,
        router_normalize_init: bool = True,
        renormalize_topk: bool = True,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_tok = max(1, min(num_experts_per_tok, num_experts))
        self.use_shared_expert = use_shared_expert
        self.renormalize_topk = renormalize_topk
        
        # Router is replicated (not parallelized)
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        with torch.no_grad():
            self.gate.weight.normal_(mean=0.0, std=float(router_init_std))
            if router_normalize_init:
                self.gate.weight.div_(torch.norm(self.gate.weight, dim=-1, keepdim=True) + 1e-6)
        
        # Experts with TP
        self.experts = nn.ModuleList([
            TensorParallelExpertMLP(hidden_size, intermediate_size)
            for _ in range(num_experts)
        ])
        
        # Shared expert with TP
        self.shared_expert = None
        if use_shared_expert:
            shared_int = shared_intermediate_size or intermediate_size
            self.shared_expert = TensorParallelExpertMLP(hidden_size, shared_int)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, H = x.shape
        nt = B * T
        x_flat = x.view(nt, H)
        
        # Router: keep matmul in bf16, softmax/topk in fp32 for stability
        gate_logits = self.gate(x_flat)
        gate_probs = torch.softmax(gate_logits.float(), dim=-1)

        if dist.is_initialized():
            rank = dist.get_rank()
            if rank == 0:
                print("[moe] router ok", flush=True)
        
        # Top-k selection
        k = self.num_experts_per_tok
        topk_vals, topk_idx = gate_probs.topk(k=k, dim=-1)
        
        # Expand indices
        topk_vals_flat = topk_vals.reshape(-1)
        topk_idx_flat = topk_idx.reshape(-1)
        token_indices = torch.arange(nt, device=x.device).unsqueeze(1).expand(nt, k).reshape(-1)
        
        # Aux loss
        E = self.num_experts
        importance = gate_probs.mean(dim=0)
        load = torch.zeros(E, device=x.device, dtype=gate_probs.dtype)
        ones = torch.ones_like(topk_idx_flat, dtype=gate_probs.dtype)
        load.index_add_(0, topk_idx_flat, ones)
        load = load / topk_idx_flat.numel()
        aux_loss = (importance * load).sum() * E
        
        # Sparse forward
        y_flat = torch.zeros((nt, H), device=x.device, dtype=torch.float32)
        
        for e_id, expert in enumerate(self.experts):
            mask = (topk_idx_flat == e_id)
            if not mask.any():
                continue
            
            sel_token_idx = token_indices[mask]
            sel_scores = topk_vals_flat[mask]
            
            # Keep input dtype aligned with expert weights (bf16 under ZeRO-3/TP)
            x_sel = x_flat[sel_token_idx]
            out_sel = expert(x_sel).float()

            if dist.is_initialized():
                rank = dist.get_rank()
                if rank == 0 and e_id == 0:
                    print("[moe] expert0 ok", flush=True)
            
            weighted_out = out_sel * sel_scores.unsqueeze(-1).float()
            if weighted_out.dtype != y_flat.dtype:
                weighted_out = weighted_out.to(y_flat.dtype)
            y_flat.index_add_(0, sel_token_idx, weighted_out)
        
        # Shared expert
        if self.shared_expert is not None:
            shared_out = self.shared_expert(x_flat).float()
            y_flat = y_flat + shared_out
        
        y = y_flat.view(B, T, H).to(x.dtype)
        return y, aux_loss


class TensorParallelThinkerDecoderLayer(nn.Module):
    """Decoder layer with Tensor Parallelism."""
    
    def __init__(self, config: Qwen3OmniMoeThinkerConfig, rotary_emb: RotaryEmbedding):
        super().__init__()
        
        self.hidden_size = config.hidden_size
        self.use_moe = getattr(config, "use_moe", False)
        self.num_experts = getattr(config, "num_experts", 0)
        self.num_experts_per_tok = getattr(config, "num_experts_per_tok", 1)
        self.block_type = getattr(config, "block_type", "attn")
        
        head_dim = config.hidden_size // config.num_attention_heads
        
        if self.block_type == "deltanet":
            # DeltaNet doesn't support TP currently, use standard implementation
            kernel_size = getattr(config, "deltanet_kernel_size", 3)
            num_heads_for_dt = getattr(config, "deltanet_num_heads", config.num_attention_heads)
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
            # TP-enabled attention
            self.self_attn = TensorParallelMultiHeadSelfAttention(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                head_dim=head_dim,
                use_flash_attention=getattr(config, "use_flash_attention", False),
                headwise_attn_output_gate=getattr(config, "headwise_attn_output_gate", False),
                elementwise_attn_output_gate=getattr(config, "elementwise_attn_output_gate", False),
            )
        
        self.attn_norm = RMSNorm(config.hidden_size)
        
        # Shared Dense FFN with TP
        self.shared_mlp = TensorParallelMLP(config.hidden_size, config.intermediate_size)
        
        # Optional MoE FFN with TP
        if self.use_moe and self.num_experts > 0 and self.num_experts_per_tok > 0:
            self.moe_mlp = TensorParallelMoeMLP(
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
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        attn_out = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            rotary_emb=self.rotary_emb,
        )
        hidden_states = residual + attn_out
        
        # FFN
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
        return hidden_states, aux_loss


class Qwen3OmniMoeThinkerTextModelTP(PreTrainedModel):
    """
    Tensor Parallel version of Qwen3OmniMoeThinkerTextModel.
    
    This model uses VocabParallelEmbedding for embeddings and
    TensorParallel* layers for attention and MLP.
    """
    
    config_class = Qwen3OmniMoeConfig
    
    def __init__(self, config: Qwen3OmniMoeConfig):
        super().__init__(config)
        
        if not is_model_parallel_initialized():
            raise RuntimeError(
                "Tensor parallel is not initialized. "
                "Call initialize_model_parallel() before creating this model."
            )
        
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.max_position_embeddings = config.max_position_embeddings
        
        thinker_cfg = config.thinker_config
        assert isinstance(thinker_cfg, Qwen3OmniMoeThinkerConfig)
        self.thinker_cfg = thinker_cfg
        
        tp_world_size = get_tensor_model_parallel_world_size()
        
        # Check vocab_size is divisible by TP size
        if config.vocab_size % tp_world_size != 0:
            raise ValueError(
                f"vocab_size ({config.vocab_size}) must be divisible by "
                f"TP world size ({tp_world_size})"
            )
        
        # Parallel embedding
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, thinker_cfg.hidden_size
        )
        
        # RoPE
        self.head_dim = thinker_cfg.hidden_size // thinker_cfg.num_attention_heads
        rope_partial_factor = getattr(config, "rope_partial_factor", 1.0)
        rope_dim = int(self.head_dim * rope_partial_factor)
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
        
        # Parse layer indices
        moe_layer_set = None
        if getattr(thinker_cfg, "moe_layer_indices", None) is not None:
            indices_str = thinker_cfg.moe_layer_indices
            if isinstance(indices_str, str) and indices_str.strip():
                moe_layer_set = set(
                    int(x) for x in indices_str.split(",") if x.strip().isdigit()
                )
        
        deltanet_layer_set = None
        if getattr(thinker_cfg, "deltanet_layer_indices", None) is not None:
            indices_str = thinker_cfg.deltanet_layer_indices
            if isinstance(indices_str, str) and indices_str.strip():
                deltanet_layer_set = set(
                    int(x) for x in indices_str.split(",") if x.strip().isdigit()
                )
        
        use_deltanet_global = getattr(thinker_cfg, "use_deltanet", False)
        self.gradient_checkpointing = getattr(thinker_cfg, "gradient_checkpointing", False)
        
        # Build layers
        layers = []
        for layer_idx in range(thinker_cfg.num_hidden_layers):
            use_moe_layer = thinker_cfg.use_moe
            if moe_layer_set is not None:
                use_moe_layer = layer_idx in moe_layer_set
            
            block_type = "attn"
            if use_deltanet_global:
                if deltanet_layer_set is not None:
                    block_type = "deltanet" if layer_idx in deltanet_layer_set else "attn"
                else:
                    if (layer_idx % 4) in (0, 1, 2):
                        block_type = "deltanet"
            
            local_cfg = Qwen3OmniMoeThinkerConfig(**thinker_cfg.__dict__)
            local_cfg.use_moe = use_moe_layer
            local_cfg.block_type = block_type
            
            layers.append(TensorParallelThinkerDecoderLayer(local_cfg, self.rotary_emb))
        
        self.layers = nn.ModuleList(layers)
        self.norm = RMSNorm(thinker_cfg.hidden_size)
        
        # Parallel LM head (tied with embedding)
        self.lm_head = ParallelLMHead(
            thinker_cfg.hidden_size,
            config.vocab_size,
            bias=False,
            tied_embedding=self.embed_tokens,
        )
        
        self.config.tie_word_embeddings = True
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
        
        causal_mask = torch.full(
            (tgt_len, tgt_len),
            fill_value=-float("inf"),
            device=device,
        )
        causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        
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
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
            device = hidden_states.device
            bsz, seq_len, _ = hidden_states.size()
        else:
            assert input_ids is not None, "input_ids or inputs_embeds must be provided"
            hidden_states = self.embed_tokens(input_ids)
            device = input_ids.device
            bsz, seq_len = input_ids.size()
        
        if position_ids is None:
            position_ids = torch.arange(
                seq_len, dtype=torch.long, device=device
            ).unsqueeze(0).expand(bsz, -1)
        
        attention_mask_full = self._prepare_attention_mask(
            attention_mask, (bsz, seq_len), device
        )
        
        all_hidden_states = [] if output_hidden_states else None
        total_aux_loss = None
        
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            
            if self.gradient_checkpointing and self.training:
                def layer_forward(x, pos_ids):
                    return layer(x, attention_mask_full, pos_ids)
                try:
                    hidden_states, layer_aux = torch.utils.checkpoint.checkpoint(
                        layer_forward,
                        hidden_states,
                        position_ids,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                except torch.utils.checkpoint.CheckpointError:
                    # Fallback: disable checkpoint for this layer to avoid mismatch errors
                    hidden_states, layer_aux = layer(
                        hidden_states, attention_mask_full, position_ids
                    )
            else:
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
        
        loss = None
        ce_loss = None
        aux_loss = None
        vocab_size = logits.size(-1)
        
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = labels[:, 1:].contiguous()
            
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
        
            if total_aux_loss is not None:
                if labels is not None:
                    if "num_tokens" not in locals():
                        valid_mask = (labels != -100)
                        num_tokens = valid_mask.sum()
                    if num_tokens > 0:
                        aux_loss = total_aux_loss.float() / num_tokens
                    else:
                        aux_loss = total_aux_loss.float() * 0.0
                else:
                    aux_loss = total_aux_loss.float()
            
            if loss is None:
                loss = self.thinker_cfg.moe_aux_loss_coef * aux_loss
            else:
                loss = loss + self.thinker_cfg.moe_aux_loss_coef * aux_loss
        
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


def create_thinker_model(config: Qwen3OmniMoeConfig, use_tensor_parallel: bool = None):
    """
    Factory function to create the appropriate Thinker model.
    
    If use_tensor_parallel is True or TP is initialized and config enables it,
    returns Qwen3OmniMoeThinkerTextModelTP. Otherwise returns the standard model.
    
    Args:
        config: Model configuration
        use_tensor_parallel: Force enable/disable TP. If None, auto-detect.
        
    Returns:
        Thinker model instance
    """
    from .modeling_thinker_text import Qwen3OmniMoeThinkerTextModel
    
    if use_tensor_parallel is None:
        # Check config and initialization status
        use_tp = getattr(config.thinker_config, "use_tensor_parallel", False)
        if use_tp and not is_model_parallel_initialized():
            print("[WARNING] use_tensor_parallel=True but model parallel not initialized. "
                  "Falling back to standard model.")
            use_tp = False
        use_tensor_parallel = use_tp and is_model_parallel_initialized()
    
    if use_tensor_parallel:
        if not is_model_parallel_initialized():
            raise RuntimeError(
                "Tensor parallel requested but not initialized. "
                "Call initialize_model_parallel() first."
            )
        return Qwen3OmniMoeThinkerTextModelTP(config)
    else:
        return Qwen3OmniMoeThinkerTextModel(config)
