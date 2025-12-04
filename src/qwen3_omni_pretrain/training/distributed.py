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


def ddp_wrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if not ddp_available():
        return model

    local_rank = get_local_rank()
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
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
