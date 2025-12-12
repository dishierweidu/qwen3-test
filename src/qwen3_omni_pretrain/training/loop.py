# src/qwen3_omni_pretrain/training/loop.py

from typing import Callable, Optional, Dict, Any
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

try:
    # 不强依赖 DDP，但如果有就用
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.distributed as dist
except Exception:  # 本地单卡时也不报错
    DDP = None
    dist = None


def _move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    new_batch = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            new_batch[k] = v.to(device, non_blocking=True)
        else:
            new_batch[k] = v
    return new_batch


def _is_global_bad_loss(loss: torch.Tensor) -> bool:
    """
    在 DDP 下做一次 all_reduce，保证：
    - 只要有一个 rank 出现 NaN/Inf，所有 rank 都一起 skip 这个 batch，
      避免“有的 rank backward，有的 rank 不 backward”导致的 reduction 错位。
    """
    bad_local = (not torch.isfinite(loss))

    if dist is not None and dist.is_available() and dist.is_initialized():
        flag = torch.tensor(
            [1 if bad_local else 0],
            device=loss.device,
            dtype=torch.int32,
        )
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return flag.item() > 0
    else:
        return bad_local


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    device: torch.device,
    gradient_accumulation_steps: int = 1,
    log_step_fn: Optional[Callable[..., None]] = None,
    autocast_dtype: Optional[torch.dtype] = None,
    grad_scaler: Optional[torch.cuda.amp.GradScaler] = None,
    after_step_fn: Optional[Callable[..., None]] = None,
    should_stop_fn: Optional[Callable[[], bool]] = None,
) -> float:
    """
    通用训练循环：
    - batch 是一个 dict，直接 **batch 传给 model
    - 要求 model.forward 返回 dict，至少含 "loss"
      额外可选 "ce_loss" / "aux_loss" 用于日志

    支持：
    - 梯度累积 (gradient_accumulation_steps)
    - DDP 下 no_sync() 降低通信频率
    - NaN/Inf 保护（在所有 rank 上一致地跳过有问题的 batch）
    """
    model.train()
    grad_accum = max(1, gradient_accumulation_steps)

    # 判断是否是 DDP 包裹的模型
    is_ddp = (DDP is not None) and isinstance(model, DDP) or hasattr(model, "no_sync")
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0          # 累积“真实 loss”总和（未除 grad_accum）
    total_ce = 0.0
    total_aux = 0.0
    num_updates = 0           # 逻辑上的 optimizer.step() 次数

    for batch_idx, batch in enumerate(dataloader):
        if should_stop_fn is not None and should_stop_fn():
            break
        if batch_idx == 0:
            # 确认每个 rank 都拿到数据了
            if "input_ids" in batch:
                print(f"[Rank {rank}] got first batch, input_ids={batch['input_ids'].shape}")
            else:
                print(f"[Rank {rank}] got first batch, keys={list(batch.keys())}")

        batch = _move_batch_to_device(batch, device)

        # 本 micro-step 是否是一个“累计完成点”（要做 optimizer.step）
        is_update_step = ((batch_idx + 1) % grad_accum == 0) or ((batch_idx + 1) == len(dataloader))

        # DDP 下，只有在“真正 update step”才进行梯度同步；其他 micro-step 用 no_sync 减少 allreduce
        if is_ddp and not is_update_step:
            ctx = model.no_sync()
        else:
            ctx = nullcontext()

        amp_ctx = (
            torch.autocast(device_type=device.type, dtype=autocast_dtype)
            if autocast_dtype is not None
            else nullcontext()
        )

        with ctx:
            with amp_ctx:
                outputs = model(**batch)

                loss = outputs["loss"]
                ce_loss = outputs.get("ce_loss", None)
                aux_loss = outputs.get("aux_loss", None)

            # ✅ NaN/Inf 保护（在所有 rank 上统一判断）
            if _is_global_bad_loss(loss):
                print(
                    f"[train_one_epoch] NaN/Inf loss at batch {batch_idx + 1} "
                    f"(loss={loss}, "
                    f"ce={ce_loss if isinstance(ce_loss, torch.Tensor) else ce_loss}, "
                    f"aux={aux_loss if isinstance(aux_loss, torch.Tensor) else aux_loss}) — skip this batch."
                )
                optimizer.zero_grad(set_to_none=True)
                if grad_scaler is not None:
                    grad_scaler.update()
                continue

            # 梯度累积：缩放 loss
            scaled_loss = loss / grad_accum

            if grad_scaler is not None:
                grad_scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

        if is_update_step:
            # 可选：梯度裁剪
            if grad_scaler is not None:
                grad_scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            num_updates += 1

            # 统计用“真实 loss”（未除 grad_accum）
            true_loss = scaled_loss.item() * grad_accum
            total_loss += true_loss

            if isinstance(ce_loss, torch.Tensor):
                total_ce += ce_loss.detach().item()
            if isinstance(aux_loss, torch.Tensor):
                total_aux += aux_loss.detach().item()
                
            current_lr = optimizer.param_groups[0].get("lr", None) if optimizer.param_groups else None

            if log_step_fn is not None:
                log_step_fn(
                    step=num_updates,
                    loss=true_loss,
                    batch_idx=batch_idx,
                    ce_loss=ce_loss.detach().item() if isinstance(ce_loss, torch.Tensor) else None,
                    aux_loss=aux_loss.detach().item() if isinstance(aux_loss, torch.Tensor) else None,
                    lr=current_lr,
                )
            
            if after_step_fn is not None:
                after_step_fn(
                    step=num_updates,
                    loss=true_loss,
                    batch_idx=batch_idx,
                    ce_loss=ce_loss.detach().item() if isinstance(ce_loss, torch.Tensor) else None,
                    aux_loss=aux_loss.detach().item() if isinstance(aux_loss, torch.Tensor) else None,
                    lr=current_lr,
                    model=model,
                )

            if should_stop_fn is not None and should_stop_fn():
                break


    if num_updates == 0:
        return float("nan")

    avg_loss = total_loss / num_updates
    return avg_loss


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    autocast_dtype: Optional[torch.dtype] = None,
) -> float:
    """
    通用评估循环：
    - 同样直接 **batch 传给 model
    - 要求模型返回 "loss"
    """
    model.eval()
    total_loss = 0.0
    steps = 0

    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=autocast_dtype)
        if autocast_dtype is not None
        else nullcontext()
    )

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch = _move_batch_to_device(batch, device)

            with amp_ctx:
                outputs = model(**batch)
                loss = outputs["loss"]

            if not torch.isfinite(loss):
                print(f"[evaluate] NaN/Inf loss at batch {batch_idx}, skip.")
                continue

            total_loss += loss.item()
            steps += 1

    avg_loss = total_loss / max(1, steps)
    return avg_loss
