# src/qwen3_omni_pretrain/parallel/initialize.py
"""
Model Parallelism Initialization

This module handles the initialization of process groups for:
- Tensor Model Parallelism (TP): splits model parameters across GPUs
- Data Parallelism (DP): splits data batches across GPUs
- Pipeline Parallelism (PP): splits model layers across GPUs (optional)

The parallelism follows a 3D decomposition:
    world_size = tensor_model_parallel_size * data_parallel_size * pipeline_model_parallel_size

Example with 8 GPUs, TP=2, DP=4, PP=1:
    GPU 0,1 form TP group 0, GPU 2,3 form TP group 1, ...
    GPU 0,2,4,6 form DP group 0, GPU 1,3,5,7 form DP group 1
"""

from typing import Optional
import torch
import torch.distributed as dist


# Global state for model parallelism
_MODEL_PARALLEL_GROUP: Optional[dist.ProcessGroup] = None
_TENSOR_MODEL_PARALLEL_GROUP: Optional[dist.ProcessGroup] = None
_DATA_PARALLEL_GROUP: Optional[dist.ProcessGroup] = None
_PIPELINE_MODEL_PARALLEL_GROUP: Optional[dist.ProcessGroup] = None

_TENSOR_MODEL_PARALLEL_WORLD_SIZE: int = 1
_TENSOR_MODEL_PARALLEL_RANK: int = 0
_DATA_PARALLEL_WORLD_SIZE: int = 1
_DATA_PARALLEL_RANK: int = 0
_PIPELINE_MODEL_PARALLEL_WORLD_SIZE: int = 1
_PIPELINE_MODEL_PARALLEL_RANK: int = 0

_MODEL_PARALLEL_INITIALIZED: bool = False


def is_model_parallel_initialized() -> bool:
    """Check if model parallel is initialized."""
    return _MODEL_PARALLEL_INITIALIZED


def model_parallel_is_initialized() -> bool:
    """Alias for is_model_parallel_initialized."""
    return is_model_parallel_initialized()


def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    *,
    backend: str = "nccl",
    timeout_minutes: int = 30,
) -> None:
    """
    Initialize model parallel groups.
    
    This function creates process groups for tensor model parallelism (TP),
    data parallelism (DP), and optionally pipeline parallelism (PP).
    
    Args:
        tensor_model_parallel_size: Number of GPUs for tensor model parallelism.
            Model parameters (weights) are split across these GPUs.
        pipeline_model_parallel_size: Number of GPUs for pipeline parallelism.
            Model layers are split across these GPUs.
        backend: Distributed backend (default: "nccl" for GPU training).
        timeout_minutes: Timeout for distributed operations.
    
    Example:
        # 8 GPUs with TP=2, DP=4
        initialize_model_parallel(tensor_model_parallel_size=2)
        
        # 8 GPUs with TP=2, PP=2, DP=2
        initialize_model_parallel(
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=2
        )
    """
    global _MODEL_PARALLEL_GROUP
    global _TENSOR_MODEL_PARALLEL_GROUP
    global _DATA_PARALLEL_GROUP
    global _PIPELINE_MODEL_PARALLEL_GROUP
    global _TENSOR_MODEL_PARALLEL_WORLD_SIZE
    global _TENSOR_MODEL_PARALLEL_RANK
    global _DATA_PARALLEL_WORLD_SIZE
    global _DATA_PARALLEL_RANK
    global _PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    global _PIPELINE_MODEL_PARALLEL_RANK
    global _MODEL_PARALLEL_INITIALIZED
    
    if _MODEL_PARALLEL_INITIALIZED:
        print("[WARNING] Model parallel already initialized, skipping re-initialization")
        return
    
    # Initialize torch.distributed if not already done
    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            timeout=torch.distributed.distributed_c10d._DEFAULT_PG_TIMEOUT
            if timeout_minutes <= 0 else 
            torch.distributed.timedelta(minutes=timeout_minutes)
        )
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    # Validate parallelism sizes
    tp_size = tensor_model_parallel_size
    pp_size = pipeline_model_parallel_size
    
    if tp_size <= 0 or pp_size <= 0:
        raise ValueError(
            f"tensor_model_parallel_size ({tp_size}) and "
            f"pipeline_model_parallel_size ({pp_size}) must be positive"
        )
    
    model_parallel_size = tp_size * pp_size
    if world_size % model_parallel_size != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by "
            f"tensor_model_parallel_size * pipeline_model_parallel_size = {model_parallel_size}"
        )
    
    dp_size = world_size // model_parallel_size
    
    if rank == 0:
        print(f">> Initializing Model Parallel:")
        print(f"   - World Size: {world_size}")
        print(f"   - Tensor Model Parallel Size (TP): {tp_size}")
        print(f"   - Pipeline Model Parallel Size (PP): {pp_size}")
        print(f"   - Data Parallel Size (DP): {dp_size}")
    
    # Build process groups
    # Layout: [DP, PP, TP] - innermost dimension is TP
    # Example with world_size=8, tp=2, pp=2, dp=2:
    #   GPU 0: dp=0, pp=0, tp=0
    #   GPU 1: dp=0, pp=0, tp=1
    #   GPU 2: dp=0, pp=1, tp=0
    #   GPU 3: dp=0, pp=1, tp=1
    #   GPU 4: dp=1, pp=0, tp=0
    #   GPU 5: dp=1, pp=0, tp=1
    #   GPU 6: dp=1, pp=1, tp=0
    #   GPU 7: dp=1, pp=1, tp=1
    
    num_tensor_model_parallel_groups = world_size // tp_size
    num_pipeline_model_parallel_groups = world_size // pp_size
    num_data_parallel_groups = world_size // dp_size
    
    # Create tensor model parallel groups
    # Each group contains GPUs that share the same (dp_rank, pp_rank)
    for i in range(num_tensor_model_parallel_groups):
        start = i * tp_size
        end = start + tp_size
        ranks = list(range(start, end))
        group = dist.new_group(ranks)
        if rank in ranks:
            _TENSOR_MODEL_PARALLEL_GROUP = group
            _TENSOR_MODEL_PARALLEL_WORLD_SIZE = tp_size
            _TENSOR_MODEL_PARALLEL_RANK = rank - start
    
    # Create data parallel groups
    # Each group contains GPUs that share the same (pp_rank, tp_rank)
    for pp in range(pp_size):
        for tp in range(tp_size):
            ranks = []
            for dp in range(dp_size):
                ranks.append(dp * model_parallel_size + pp * tp_size + tp)
            group = dist.new_group(ranks)
            if rank in ranks:
                _DATA_PARALLEL_GROUP = group
                _DATA_PARALLEL_WORLD_SIZE = dp_size
                _DATA_PARALLEL_RANK = ranks.index(rank)
    
    # Create pipeline model parallel groups (if PP > 1)
    if pp_size > 1:
        for dp in range(dp_size):
            for tp in range(tp_size):
                ranks = []
                for pp in range(pp_size):
                    ranks.append(dp * model_parallel_size + pp * tp_size + tp)
                group = dist.new_group(ranks)
                if rank in ranks:
                    _PIPELINE_MODEL_PARALLEL_GROUP = group
                    _PIPELINE_MODEL_PARALLEL_WORLD_SIZE = pp_size
                    _PIPELINE_MODEL_PARALLEL_RANK = ranks.index(rank)
    else:
        _PIPELINE_MODEL_PARALLEL_WORLD_SIZE = 1
        _PIPELINE_MODEL_PARALLEL_RANK = 0
    
    # Model parallel group (TP + PP combined)
    for dp in range(dp_size):
        start = dp * model_parallel_size
        end = start + model_parallel_size
        ranks = list(range(start, end))
        group = dist.new_group(ranks)
        if rank in ranks:
            _MODEL_PARALLEL_GROUP = group
    
    _MODEL_PARALLEL_INITIALIZED = True
    
    # Synchronize all processes
    dist.barrier()
    
    if rank == 0:
        print(f">> Model Parallel Initialization Complete")


def destroy_model_parallel() -> None:
    """Clean up model parallel groups."""
    global _MODEL_PARALLEL_GROUP
    global _TENSOR_MODEL_PARALLEL_GROUP
    global _DATA_PARALLEL_GROUP
    global _PIPELINE_MODEL_PARALLEL_GROUP
    global _TENSOR_MODEL_PARALLEL_WORLD_SIZE
    global _TENSOR_MODEL_PARALLEL_RANK
    global _DATA_PARALLEL_WORLD_SIZE
    global _DATA_PARALLEL_RANK
    global _PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    global _PIPELINE_MODEL_PARALLEL_RANK
    global _MODEL_PARALLEL_INITIALIZED
    
    _MODEL_PARALLEL_GROUP = None
    _TENSOR_MODEL_PARALLEL_GROUP = None
    _DATA_PARALLEL_GROUP = None
    _PIPELINE_MODEL_PARALLEL_GROUP = None
    _TENSOR_MODEL_PARALLEL_WORLD_SIZE = 1
    _TENSOR_MODEL_PARALLEL_RANK = 0
    _DATA_PARALLEL_WORLD_SIZE = 1
    _DATA_PARALLEL_RANK = 0
    _PIPELINE_MODEL_PARALLEL_WORLD_SIZE = 1
    _PIPELINE_MODEL_PARALLEL_RANK = 0
    _MODEL_PARALLEL_INITIALIZED = False


def get_tensor_model_parallel_group() -> Optional[dist.ProcessGroup]:
    """Get the tensor model parallel process group."""
    return _TENSOR_MODEL_PARALLEL_GROUP


def get_tensor_model_parallel_world_size() -> int:
    """Get the tensor model parallel world size."""
    if not _MODEL_PARALLEL_INITIALIZED:
        return 1
    return _TENSOR_MODEL_PARALLEL_WORLD_SIZE


def get_tensor_model_parallel_rank() -> int:
    """Get the rank within the tensor model parallel group."""
    if not _MODEL_PARALLEL_INITIALIZED:
        return 0
    return _TENSOR_MODEL_PARALLEL_RANK


def get_data_parallel_group() -> Optional[dist.ProcessGroup]:
    """Get the data parallel process group."""
    return _DATA_PARALLEL_GROUP


def get_data_parallel_world_size() -> int:
    """Get the data parallel world size."""
    if not _MODEL_PARALLEL_INITIALIZED:
        if dist.is_initialized():
            return dist.get_world_size()
        return 1
    return _DATA_PARALLEL_WORLD_SIZE


def get_data_parallel_rank() -> int:
    """Get the rank within the data parallel group."""
    if not _MODEL_PARALLEL_INITIALIZED:
        if dist.is_initialized():
            return dist.get_rank()
        return 0
    return _DATA_PARALLEL_RANK


def get_pipeline_model_parallel_group() -> Optional[dist.ProcessGroup]:
    """Get the pipeline model parallel process group."""
    return _PIPELINE_MODEL_PARALLEL_GROUP


def get_pipeline_model_parallel_world_size() -> int:
    """Get the pipeline model parallel world size."""
    if not _MODEL_PARALLEL_INITIALIZED:
        return 1
    return _PIPELINE_MODEL_PARALLEL_WORLD_SIZE


def get_pipeline_model_parallel_rank() -> int:
    """Get the rank within the pipeline model parallel group."""
    if not _MODEL_PARALLEL_INITIALIZED:
        return 0
    return _PIPELINE_MODEL_PARALLEL_RANK


def get_model_parallel_group() -> Optional[dist.ProcessGroup]:
    """Get the model parallel process group (TP + PP combined)."""
    return _MODEL_PARALLEL_GROUP
