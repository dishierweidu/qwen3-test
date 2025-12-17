# src/qwen3_omni_pretrain/parallel/embedding.py
"""
Vocabulary Parallel Embedding and LM Head

For large vocabulary models, the embedding and LM head can consume significant
memory. VocabParallelEmbedding splits the vocabulary across TP group, where
each GPU only stores vocab_size/tp_size embeddings.

For tied word embeddings (common in LLMs), ParallelLMHead reuses the embedding
weight for the output projection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from qwen3_omni_pretrain.parallel.initialize import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from qwen3_omni_pretrain.parallel.tensor_parallel import (
    reduce_from_tensor_model_parallel_region,
    gather_from_tensor_model_parallel_region,
)


class VocabParallelEmbedding(nn.Module):
    """
    Embedding layer with vocabulary parallelism.
    
    The vocabulary is split across TP group. Each GPU stores embeddings for
    vocab_size/tp_size tokens. During forward pass:
    1. Mask out-of-range token IDs
    2. Look up local embeddings
    3. All-reduce to combine (masked positions contribute zeros)
    
    Args:
        num_embeddings: Total vocabulary size
        embedding_dim: Embedding dimension
        padding_idx: Padding token index (if any)
        init_method: Optional initialization method
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int = None,
        init_method=None,
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        
        tp_world_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        
        # Compute local vocabulary range
        # We use a simple partition: each rank gets a contiguous chunk
        assert num_embeddings % tp_world_size == 0, \
            f"num_embeddings ({num_embeddings}) must be divisible by TP world size ({tp_world_size})"
        
        self.vocab_size_per_partition = num_embeddings // tp_world_size
        self.vocab_start_index = tp_rank * self.vocab_size_per_partition
        self.vocab_end_index = self.vocab_start_index + self.vocab_size_per_partition
        
        # Local embedding table
        self.weight = nn.Parameter(
            torch.empty(self.vocab_size_per_partition, embedding_dim)
        )
        
        # Initialize
        if init_method is None:
            nn.init.normal_(self.weight, mean=0.0, std=0.02)
        else:
            init_method(self.weight)
        
        # Handle padding index
        if padding_idx is not None:
            if self.vocab_start_index <= padding_idx < self.vocab_end_index:
                local_padding_idx = padding_idx - self.vocab_start_index
                with torch.no_grad():
                    self.weight[local_padding_idx].fill_(0)
    
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_: Input token IDs, shape [*, seq_len]
            
        Returns:
            Embeddings, shape [*, seq_len, embedding_dim]
        """
        tp_world_size = get_tensor_model_parallel_world_size()
        
        if tp_world_size == 1:
            # No parallelism, just do standard embedding
            return F.embedding(input_, self.weight, padding_idx=self.padding_idx)
        
        # Create mask for tokens in local vocabulary range
        input_mask = (input_ >= self.vocab_start_index) & (input_ < self.vocab_end_index)
        
        # Shift input to local indices (mask invalid ones to 0)
        masked_input = input_.clone()
        masked_input = masked_input - self.vocab_start_index
        masked_input[~input_mask] = 0  # Out-of-range tokens map to index 0
        
        # Local embedding lookup
        output_parallel = F.embedding(masked_input, self.weight)
        
        # Zero out embeddings for out-of-range tokens
        output_parallel = output_parallel * input_mask.unsqueeze(-1).to(output_parallel.dtype)
        
        # All-reduce to combine embeddings from all ranks
        output = reduce_from_tensor_model_parallel_region(output_parallel)
        
        return output


class ParallelLMHead(nn.Module):
    """
    Language model head with vocabulary parallelism.
    
    This is essentially a linear layer that projects hidden states to vocabulary
    logits. For tied embeddings, it can share weight with VocabParallelEmbedding.
    
    The weight is split along the vocabulary dimension (columns), so each GPU
    computes logits for vocab_size/tp_size tokens. Results are gathered at the end.
    
    Args:
        hidden_size: Input hidden size
        vocab_size: Total vocabulary size
        bias: Whether to use bias
        tied_embedding: Optional VocabParallelEmbedding to share weights with
    """
    
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        bias: bool = False,
        tied_embedding: VocabParallelEmbedding = None,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        tp_world_size = get_tensor_model_parallel_world_size()
        
        assert vocab_size % tp_world_size == 0, \
            f"vocab_size ({vocab_size}) must be divisible by TP world size ({tp_world_size})"
        
        self.vocab_size_per_partition = vocab_size // tp_world_size
        
        # Share weight with embedding if tied
        if tied_embedding is not None:
            self.weight = tied_embedding.weight
        else:
            self.weight = nn.Parameter(
                torch.empty(self.vocab_size_per_partition, hidden_size)
            )
            nn.init.normal_(self.weight, mean=0.0, std=0.02)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.vocab_size_per_partition))
        else:
            self.register_parameter("bias", None)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            hidden_states: Input hidden states, shape [*, seq_len, hidden_size]
            
        Returns:
            Logits, shape [*, seq_len, vocab_size]
        """
        tp_world_size = get_tensor_model_parallel_world_size()
        
        # Local matmul: [*, seq_len, hidden_size] @ [vocab_per_partition, hidden_size].T
        # = [*, seq_len, vocab_per_partition]
        output_parallel = F.linear(hidden_states, self.weight, self.bias)
        
        if tp_world_size == 1:
            return output_parallel
        
        # Use differentiable all-gather to preserve gradients
        output = gather_from_tensor_model_parallel_region(output_parallel)
        
        return output


def convert_embedding_to_parallel(
    embedding: nn.Embedding,
) -> VocabParallelEmbedding:
    """
    Convert a standard nn.Embedding to VocabParallelEmbedding.
    
    This is useful for initializing from pretrained weights.
    """
    tp_rank = get_tensor_model_parallel_rank()
    tp_world_size = get_tensor_model_parallel_world_size()
    
    num_embeddings = embedding.num_embeddings
    embedding_dim = embedding.embedding_dim
    padding_idx = embedding.padding_idx
    
    parallel_embedding = VocabParallelEmbedding(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        padding_idx=padding_idx,
    )
    
    # Copy partitioned weight
    per_partition = num_embeddings // tp_world_size
    start = tp_rank * per_partition
    end = start + per_partition
    
    with torch.no_grad():
        parallel_embedding.weight.copy_(embedding.weight[start:end, :])
    
    return parallel_embedding
