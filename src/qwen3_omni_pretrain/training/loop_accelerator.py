# src/qwen3_omni_pretrain/training/loop_accelerator.py
"""
Accelerator 专用训练循环
使用 HuggingFace Accelerate 库的统一接口进行训练
"""

from typing import Callable, Optional, Dict, Any, Tuple
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

try:
    from accelerate import Accelerator
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False
    Accelerator = None

from qwen3_omni_pretrain.parallel.initialize import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank,
)


def _move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """将 batch 移动到指定设备"""
    new_batch = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            new_batch[k] = v.to(device, non_blocking=True)
        else:
            new_batch[k] = v
    return new_batch


def _is_bad_loss(loss: torch.Tensor) -> bool:
    """检查 loss 是否为 NaN 或 Inf"""
    return not torch.isfinite(loss)


def _sanitize_gradients(parameters, clamp_value: float = 1e4) -> bool:
    """将梯度中的 NaN/Inf 置零，可选截断到 [-clamp_value, clamp_value]。
    返回是否发现非有限梯度。"""
    found_nonfinite = False
    for p in parameters:
        if p.grad is None:
            continue
        grad = p.grad
        if not torch.isfinite(grad).all():
            found_nonfinite = True
            grad.data = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        if clamp_value is not None:
            grad.data.clamp_(-clamp_value, clamp_value)
    return found_nonfinite


def _tp_broadcast_batch_inplace(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """确保同一 TP 组内使用同一份 batch（来自组内 src=0）。"""
    if (not dist.is_available()) or (not dist.is_initialized()):
        return batch
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size <= 1:
        return batch
    group = get_tensor_model_parallel_group()
    src_global = dist.get_global_rank(group, 0)

    for k, v in batch.items():
        if torch.is_tensor(v):
            if v.device != device:
                v = v.to(device, non_blocking=True)
                batch[k] = v
            dist.broadcast(v, src=src_global, group=group)
    return batch


def _tp_broadcast_batch_dynamic(
    batch: Optional[Dict[str, Any]],
    device: torch.device,
) -> Dict[str, Any]:
    """仅由 TP 组内 rank0 读取/提供 batch，其余 rank 动态广播元信息与数据，避免 StopIteration 不一致导致死锁。"""
    tp = get_tensor_model_parallel_world_size()
    if (not dist.is_available()) or (not dist.is_initialized()) or tp <= 1:
        return batch if batch is not None else {}

    group = get_tensor_model_parallel_group()
    src_global = dist.get_global_rank(group, 0)
    is_src = dist.get_rank() == src_global

    # 广播元信息
    meta = []
    if is_src:
        assert batch is not None
        for k, v in batch.items():
            if torch.is_tensor(v):
                meta.append((k, True, v.dtype, tuple(v.shape)))
            else:
                meta.append((k, False, v))
    obj = [meta]
    dist.broadcast_object_list(obj, src=src_global, group=group)
    meta = obj[0]

    out: Dict[str, Any] = {}
    for item in meta:
        k = item[0]
        is_tensor = item[1]
        if is_tensor:
            _, _, dtype, shape = item
            if is_src:
                t = batch[k]
                if t.device != device:
                    t = t.to(device, non_blocking=True)
            else:
                t = torch.empty(shape, device=device, dtype=dtype)
            dist.broadcast(t, src=src_global, group=group)
            out[k] = t
        else:
            _, _, v = item
            out[k] = v

    return out


def _evaluate_with_tp_broadcast(
    accelerator: "Accelerator",
    model: torch.nn.Module,
    step: int,
    eval_fn: Callable[[int, torch.nn.Module], Any],
    log_fn: Optional[Callable[..., None]] = None,
    is_first_step: bool = False,
):
    """Wrap eval_fn with synchronization to keep ranks aligned."""
    accelerator.wait_for_everyone()
    eval_fn(step, model)
    accelerator.wait_for_everyone()


def train_one_epoch_accelerator(
    accelerator: "Accelerator",
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    cfg: Any,  # TrainerThinkerConfig
    global_step_offset: int = 0,
    log_fn: Optional[Callable[..., None]] = None,
    eval_fn: Optional[Callable[..., Any]] = None,
    should_stop_fn: Optional[Callable[[], bool]] = None,
) -> Tuple[float, int]:
    """
    使用 Accelerator 的训练循环
    
    特点：
    - 自动处理混合精度
    - 自动处理梯度累积
    - 自动处理分布式同步
    - 统一的 DDP/DeepSpeed/FSDP 接口
    
    Args:
        accelerator: Accelerator 实例
        model: 模型（已经过 accelerator.prepare）
        dataloader: 数据加载器（已经过 accelerator.prepare）
        optimizer: 优化器（已经过 accelerator.prepare）
        scheduler: 学习率调度器
        cfg: 训练配置
        global_step_offset: 全局步数偏移（用于恢复训练）
        log_fn: 日志回调函数
        eval_fn: 评估回调函数
        should_stop_fn: 停止检查函数
    
    Returns:
        (avg_loss, num_updates): 平均损失和更新步数
    """
    model.train()
    
    total_loss = 0.0
    total_ce = 0.0
    total_aux = 0.0
    num_updates = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # 检查是否需要停止
        if should_stop_fn is not None and should_stop_fn():
            break
        
        # Debug: 打印第一个 batch 的形状和有效 label 数
        if batch_idx == 0 and accelerator.is_main_process:
            if "input_ids" in batch:
                msg = f"[Accelerator] First batch input_ids shape: {batch['input_ids'].shape}, device={batch['input_ids'].device}"
                if "labels" in batch:
                    valid = (batch["labels"] != -100).sum().item()
                    msg += f", valid_labels={valid}"
                print(msg)
            else:
                any_tensor = None
                for v in batch.values():
                    if isinstance(v, torch.Tensor):
                        any_tensor = v
                        break
                dev = any_tensor.device if any_tensor is not None else "n/a"
                print(f"[Accelerator] First batch keys: {list(batch.keys())}, device={dev}")
        
        # Accelerator 会自动处理设备放置，但确保 batch 在正确设备上
        # 注意：经过 accelerator.prepare 的 dataloader 已经处理了设备放置
        
        # 使用 accumulate 上下文管理器处理梯度累积
        # 这会自动处理 no_sync 和梯度同步
        with accelerator.accumulate(model):
            # 前向传播
            batch = _tp_broadcast_batch_inplace(batch, accelerator.device)

            # 可选：首个 batch 做 TP 组内一致性校验
            if batch_idx == 0 and dist.is_initialized() and get_tensor_model_parallel_world_size() > 1:
                if "input_ids" in batch:
                    group = get_tensor_model_parallel_group()
                    checksum = batch["input_ids"].to(torch.int64).sum()
                    gathered = [torch.zeros_like(checksum) for _ in range(get_tensor_model_parallel_world_size())]
                    dist.all_gather(gathered, checksum, group=group)
                    if dist.get_rank() == dist.get_global_rank(group, 0):
                        print("[TP] input_ids checksums:", [x.item() for x in gathered])

            outputs = model(**batch)

            if batch_idx < 5 and accelerator.is_main_process:
                micro_valid = None
                if "labels" in batch:
                    micro_valid = (batch["labels"] != -100).sum().item()
                micro_loss = outputs.get("loss", None)
                micro_ce = outputs.get("ce_loss", None)
                micro_aux = outputs.get("aux_loss", None)
                log_msg = f"[Accelerator] micro batch {batch_idx}"
                if micro_valid is not None:
                    log_msg += f" valid_labels={micro_valid}"
                if isinstance(micro_loss, torch.Tensor):
                    log_msg += f" loss={micro_loss.detach().float().item():.6f}"
                if isinstance(micro_ce, torch.Tensor):
                    log_msg += f" ce={micro_ce.detach().float().item():.6f}"
                if isinstance(micro_aux, torch.Tensor):
                    log_msg += f" aux={micro_aux.detach().float().item():.6f}"
                print(log_msg)

            if batch_idx == 0 and accelerator.is_main_process:
                loss_dbg = outputs.get("loss", None)
                ce_dbg = outputs.get("ce_loss", None)
                aux_dbg = outputs.get("aux_loss", None)
                logits = outputs.get("logits", None)

                msg = "[Accelerator] First batch loss stats:"
                if isinstance(loss_dbg, torch.Tensor):
                    msg += f" loss={loss_dbg.detach().float().item():.6f}"
                    msg += f" finite={torch.isfinite(loss_dbg).item()}"
                if isinstance(ce_dbg, torch.Tensor):
                    msg += f" ce={ce_dbg.detach().float().item():.6f}"
                    msg += f" ce_finite={torch.isfinite(ce_dbg).item()}"
                if isinstance(aux_dbg, torch.Tensor):
                    msg += f" aux={aux_dbg.detach().float().item():.6f}"
                    msg += f" aux_finite={torch.isfinite(aux_dbg).item()}"
                if isinstance(logits, torch.Tensor):
                    finite_ratio = torch.isfinite(logits).float().mean().item()
                    msg += f" logits_finite_ratio={finite_ratio:.4f}"
                    msg += f" logits_min={logits.min().item():.3f} logits_max={logits.max().item():.3f}"
                print(msg)
            
            loss = outputs["loss"]
            ce_loss = outputs.get("ce_loss", None)
            aux_loss = outputs.get("aux_loss", None)
            
            # NaN/Inf 检查 - 在分布式环境下需要同步
            bad_loss = _is_bad_loss(loss)
            if accelerator.num_processes > 1:
                bad_flag = torch.tensor([1 if bad_loss else 0], device=accelerator.device)
                bad_flag = accelerator.reduce(bad_flag, reduction="max")
                bad_loss = bad_flag.item() > 0
            
            if bad_loss:
                if accelerator.is_main_process:
                    print(f"[train] NaN/Inf loss at batch {batch_idx + 1}, skip.")
                    if "labels" in batch:
                        valid = (batch["labels"] != -100).sum().item()
                        print(f"[train] batch valid_labels={valid}")
                    log_dbg = []
                    for k in ("loss", "ce_loss", "aux_loss"):
                        v = outputs.get(k, None)
                        if isinstance(v, torch.Tensor):
                            log_dbg.append(f"{k}={v.detach().float().item():.4f}")
                    logits_dbg = outputs.get("logits", None)
                    if isinstance(logits_dbg, torch.Tensor):
                        finite_ratio = torch.isfinite(logits_dbg).float().mean().item()
                        log_dbg.append(f"logits_finite_ratio={finite_ratio:.4f}")
                        log_dbg.append(f"logits_min={logits_dbg.min().item():.3f}")
                        log_dbg.append(f"logits_max={logits_dbg.max().item():.3f}")
                    if log_dbg:
                        print("[train] dbg " + " ".join(log_dbg))
                optimizer.zero_grad()
                continue
            
            # 反向传播 - Accelerator 自动处理混合精度和梯度缩放
            accelerator.backward(loss)

            # 梯度清理：去掉 NaN/Inf，并裁剪极值，避免后续 grad_norm 变非有限
            had_nonfinite_grad = _sanitize_gradients(model.parameters(), clamp_value=1e4)
            if accelerator.num_processes > 1:
                flag_tensor = torch.tensor([1 if had_nonfinite_grad else 0], device=accelerator.device)
                flag_tensor = accelerator.reduce(flag_tensor, reduction="max")
                had_nonfinite_grad = flag_tensor.item() > 0
            if had_nonfinite_grad:
                if accelerator.is_main_process:
                    print(f"[train] Found non-finite grads at batch {batch_idx + 1}, zeroed/clamped. Skip step (synced across ranks).")
                optimizer.zero_grad()
                continue
            
            # 梯度裁剪 - 使用 accelerator.clip_grad_norm_
            if accelerator.sync_gradients:
                grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                grad_norm_tensor = torch.as_tensor(grad_norm, device=accelerator.device)
                if accelerator.num_processes > 1:
                    # 同步判断梯度范数是否有限
                    finite_flag = accelerator.reduce(torch.isfinite(grad_norm_tensor).int(), reduction="min")
                    finite_norm = finite_flag.item() > 0
                else:
                    finite_norm = torch.isfinite(grad_norm_tensor).item()
                if not finite_norm:
                    if accelerator.is_main_process:
                        print(f"[train] Non-finite grad norm at batch {batch_idx + 1}, skip step.")
                    optimizer.zero_grad()
                    continue
            
            # 优化器步进仅在同步梯度时执行
            if accelerator.sync_gradients:
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
        
        # 只在同步梯度后（即实际更新参数后）记录统计
        if accelerator.sync_gradients:
            num_updates += 1
            current_step = global_step_offset + num_updates
            
            # 统计损失
            step_loss = loss.detach().item()
            total_loss += step_loss
            
            if isinstance(ce_loss, torch.Tensor):
                total_ce += ce_loss.detach().item()
            if isinstance(aux_loss, torch.Tensor):
                total_aux += aux_loss.detach().item()
            
            # 获取当前学习率
            current_lr = None
            if optimizer.param_groups:
                current_lr = optimizer.param_groups[0].get("lr", None)
            
            # 日志回调
            if log_fn is not None:
                log_fn(
                    current_step,
                    step_loss,
                    ce_loss.detach().item() if isinstance(ce_loss, torch.Tensor) else None,
                    aux_loss.detach().item() if isinstance(aux_loss, torch.Tensor) else None,
                    current_lr,
                )
            
            # 评估回调（如 step-based eval/save）
            if eval_fn is not None:
                _evaluate_with_tp_broadcast(
                    accelerator=accelerator,
                    model=model,
                    step=current_step,
                    eval_fn=eval_fn,
                    log_fn=log_fn,
                    is_first_step=current_step == 0,
                )
            
            # 检查是否需要停止
            if should_stop_fn is not None and should_stop_fn():
                break
    
    if num_updates == 0:
        return float("nan"), 0
    
    avg_loss = total_loss / num_updates
    return avg_loss, num_updates


def evaluate_accelerator(
    accelerator: "Accelerator",
    model: torch.nn.Module,
    dataloader: DataLoader,
) -> float:
    """
    使用 Accelerator 的评估循环
    
    Args:
        accelerator: Accelerator 实例
        model: 模型
        dataloader: 评估数据加载器
    
    Returns:
        平均评估损失
    """
    accelerator.wait_for_everyone()
    model.eval()
    
    loss_sum = torch.zeros(1, device=accelerator.device)
    loss_count = torch.zeros(1, device=accelerator.device)

    tp_world = get_tensor_model_parallel_world_size() if dist.is_initialized() else 1
    tp_group = get_tensor_model_parallel_group() if dist.is_initialized() else None
    use_tp = tp_group is not None and tp_world > 1
    tp_src_global = dist.get_global_rank(tp_group, 0) if use_tp else 0
    is_tp_src = dist.get_rank() == tp_src_global if use_tp else True

    with torch.no_grad():
        if not use_tp:
            for batch_idx, batch in enumerate(dataloader):
                outputs = model(**batch)
                loss = outputs["loss"]

                if not torch.isfinite(loss):
                    if accelerator.is_main_process:
                        print(f"[eval] NaN/Inf loss at batch {batch_idx}, skip.")
                    continue

                loss_sum += loss.detach()
                loss_count += 1
        else:
            data_iter = iter(dataloader) if is_tp_src else None
            step_idx = 0
            while True:
                has_batch = torch.zeros(1, device=accelerator.device)
                if is_tp_src:
                    try:
                        batch = next(data_iter)
                        has_batch.fill_(1)
                    except StopIteration:
                        has_batch.fill_(0)
                dist.broadcast(has_batch, src=tp_src_global, group=tp_group)
                if has_batch.item() == 0:
                    break

                if not is_tp_src:
                    batch = None
                batch = _tp_broadcast_batch_dynamic(batch, accelerator.device)

                outputs = model(**batch)
                loss = outputs["loss"]

                if not torch.isfinite(loss):
                    if accelerator.is_main_process:
                        print(f"[eval] NaN/Inf loss at batch {step_idx}, skip.")
                    step_idx += 1
                    continue

                # 仅 TP 组 leader 累积损失，避免重复统计
                if get_tensor_model_parallel_rank() == 0:
                    loss_sum += loss.detach()
                    loss_count += 1
                step_idx += 1

    model.train()

    # 归约 sum 和 count
    global_sum = accelerator.reduce(loss_sum, reduction="sum")
    global_count = accelerator.reduce(loss_count, reduction="sum")

    accelerator.wait_for_everyone()

    if global_count.item() == 0:
        return float("nan")

    avg_loss = (global_sum / global_count).item()
    return avg_loss


def train_one_epoch_accelerator_simple(
    accelerator: "Accelerator",
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    max_grad_norm: float = 1.0,
) -> float:
    """
    简化版 Accelerator 训练循环（无回调）
    
    适用于简单训练场景
    """
    model.train()
    total_loss = 0.0
    num_steps = 0
    
    for batch in dataloader:
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs["loss"]
            
            if not torch.isfinite(loss):
                optimizer.zero_grad()
                continue
            
            accelerator.backward(loss)
            
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
        
        if accelerator.sync_gradients:
            total_loss += loss.detach().item()
            num_steps += 1
    
    return total_loss / max(num_steps, 1)
