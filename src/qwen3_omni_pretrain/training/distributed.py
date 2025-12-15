import os
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def ddp_available() -> bool:
    return "LOCAL_RANK" in os.environ and torch.cuda.is_available()


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def get_world_size() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    return get_rank() == 0


def ddp_init(backend: str = "nccl"):
    if not ddp_available():
        return

    local_rank = get_local_rank()
    torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(backend=backend)


def ddp_wrap_model(
    model: torch.nn.Module,
    *,
    find_unused_parameters: bool = True,
) -> torch.nn.Module:
    if not ddp_available():
        return model

    local_rank = get_local_rank()
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=find_unused_parameters,
    )
    return model


def ddp_cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


@contextmanager
def distributed_context(backend: str = "nccl"):
    """
    用法：
    with distributed_context():
        ... 训练逻辑 ...
    """
    ddp_init(backend=backend)
    try:
        yield
    finally:
        ddp_cleanup()


# =============================================================================
# Tensor Parallelism Support
# =============================================================================

def init_tensor_parallel(
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    backend: str = "nccl",
) -> None:
    """
    Initialize tensor parallelism on top of existing distributed setup.
    
    This function should be called AFTER torch.distributed is initialized
    (e.g., by ddp_init or DeepSpeed/Accelerate).
    
    Args:
        tensor_parallel_size: Number of GPUs for tensor model parallelism.
        pipeline_parallel_size: Number of GPUs for pipeline parallelism.
        backend: Distributed backend (default: "nccl").
        
    Example:
        # With 8 GPUs, use TP=2 (4 data parallel groups of 2 GPUs each)
        ddp_init()
        init_tensor_parallel(tensor_parallel_size=2)
    """
    from qwen3_omni_pretrain.parallel import (
        initialize_model_parallel,
        is_model_parallel_initialized,
    )
    
    if is_model_parallel_initialized():
        if is_main_process():
            print("[WARNING] Model parallel already initialized, skipping.")
        return
    
    # Ensure distributed is initialized first
    if not dist.is_initialized():
        ddp_init(backend=backend)
    
    if tensor_parallel_size > 1 or pipeline_parallel_size > 1:
        initialize_model_parallel(
            tensor_model_parallel_size=tensor_parallel_size,
            pipeline_model_parallel_size=pipeline_parallel_size,
            backend=backend,
        )
        
        if is_main_process():
            print(f">> Tensor Parallelism initialized: TP={tensor_parallel_size}, PP={pipeline_parallel_size}")


def cleanup_tensor_parallel() -> None:
    """Clean up tensor parallelism resources."""
    from qwen3_omni_pretrain.parallel import (
        destroy_model_parallel,
        is_model_parallel_initialized,
    )
    
    if is_model_parallel_initialized():
        destroy_model_parallel()


@contextmanager
def tensor_parallel_context(
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    backend: str = "nccl",
):
    """
    Context manager for tensor parallel training.
    
    Usage:
        with tensor_parallel_context(tensor_parallel_size=2):
            model = create_thinker_model(config, use_tensor_parallel=True)
            ... training logic ...
    """
    init_tensor_parallel(
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        backend=backend,
    )
    try:
        yield
    finally:
        cleanup_tensor_parallel()


@contextmanager
def distributed_tensor_parallel_context(
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    backend: str = "nccl",
):
    """
    Combined context manager for distributed + tensor parallel training.
    
    Usage:
        with distributed_tensor_parallel_context(tensor_parallel_size=2):
            model = create_thinker_model(config, use_tensor_parallel=True)
            ... training logic ...
    """
    ddp_init(backend=backend)
    init_tensor_parallel(
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        backend=backend,
    )
    try:
        yield
    finally:
        cleanup_tensor_parallel()
        ddp_cleanup()
