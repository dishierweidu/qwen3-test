# src/qwen3_omni_pretrain/training/loop_accelerator.py
"""
Accelerator 专用训练循环
使用 HuggingFace Accelerate 库的统一接口进行训练
"""

from typing import Callable, Optional, Dict, Any, Tuple
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader

try:
    from accelerate import Accelerator
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False
    Accelerator = None


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
        
        # Debug: 打印第一个 batch 的形状
        if batch_idx == 0 and accelerator.is_main_process:
            if "input_ids" in batch:
                print(
                    f"[Accelerator] First batch input_ids shape: {batch['input_ids'].shape}, device={batch['input_ids'].device}"
                )
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
            outputs = model(**batch)
            
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
                optimizer.zero_grad()
                continue
            
            # 反向传播 - Accelerator 自动处理混合精度和梯度缩放
            accelerator.backward(loss)
            
            # 梯度裁剪 - 使用 accelerator.clip_grad_norm_
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 优化器步进
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
                eval_fn(current_step, model)
            
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
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            outputs = model(**batch)
            loss = outputs["loss"]
            
            if not torch.isfinite(loss):
                if accelerator.is_main_process:
                    print(f"[eval] NaN/Inf loss at batch {batch_idx}, skip.")
                continue
            
            # 收集所有进程的损失
            gathered_loss = accelerator.gather(loss.unsqueeze(0))
            total_loss += gathered_loss.mean().item()
            num_batches += 1
    
    model.train()
    
    if num_batches == 0:
        return float("nan")
    
    avg_loss = total_loss / num_batches
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
