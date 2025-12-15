# src/qwen3_omni_pretrain/parallel/__init__.py
"""
3D Parallelism Module (TP + ZeRO + optional PP)

This module provides Tensor Parallelism (TP) support that can be combined with
DeepSpeed ZeRO for 3D parallel training. It is designed as an optional feature
that does not affect existing training workflows.

Usage:
    1. Initialize model parallel groups:
        from qwen3_omni_pretrain.parallel import initialize_model_parallel
        initialize_model_parallel(tensor_model_parallel_size=4)
    
    2. Use TP-enabled model layers:
        from qwen3_omni_pretrain.parallel import (
            ColumnParallelLinear,
            RowParallelLinear,
            VocabParallelEmbedding,
        )
    
    3. Use TP-enabled attention/MLP:
        from qwen3_omni_pretrain.parallel import (
            TensorParallelMultiHeadAttention,
            TensorParallelMLP,
        )
"""

from qwen3_omni_pretrain.parallel.initialize import (
    initialize_model_parallel,
    destroy_model_parallel,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_data_parallel_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    is_model_parallel_initialized,
    model_parallel_is_initialized,
)

from qwen3_omni_pretrain.parallel.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    copy_to_tensor_model_parallel_region,
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    scatter_to_tensor_model_parallel_region,
    convert_linear_to_column_parallel,
    convert_linear_to_row_parallel,
)

from qwen3_omni_pretrain.parallel.embedding import (
    VocabParallelEmbedding,
    ParallelLMHead,
    convert_embedding_to_parallel,
)

from qwen3_omni_pretrain.parallel.layers import (
    TensorParallelMultiHeadAttention,
    TensorParallelMLP,
    TensorParallelGatedMLP,
    TensorParallelExpertMLP,
)

__all__ = [
    # Initialization
    "initialize_model_parallel",
    "destroy_model_parallel",
    "get_tensor_model_parallel_group",
    "get_tensor_model_parallel_rank",
    "get_tensor_model_parallel_world_size",
    "get_data_parallel_group",
    "get_data_parallel_rank",
    "get_data_parallel_world_size",
    "is_model_parallel_initialized",
    "model_parallel_is_initialized",
    # Tensor Parallel Layers
    "ColumnParallelLinear",
    "RowParallelLinear",
    "VocabParallelEmbedding",
    "ParallelLMHead",
    # TP-enabled modules
    "TensorParallelMultiHeadAttention",
    "TensorParallelMLP",
    "TensorParallelGatedMLP",
    "TensorParallelExpertMLP",
    # Conversion utilities
    "convert_linear_to_column_parallel",
    "convert_linear_to_row_parallel",
    "convert_embedding_to_parallel",
    # Communication primitives
    "copy_to_tensor_model_parallel_region",
    "gather_from_tensor_model_parallel_region",
    "reduce_from_tensor_model_parallel_region",
    "scatter_to_tensor_model_parallel_region",
]
