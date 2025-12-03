# src/qwen3_omni_pretrain/training/loop.py

from typing import Callable, Optional, Dict, Any

import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_


def _move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    new_batch = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            new_batch[k] = v.to(device)
        else:
            new_batch[k] = v
    return new_batch


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    gradient_accumulation_steps: int = 1,
    log_step_fn: Optional[Callable[..., None]] = None,
) -> float:
    """
    通用训练循环：
    - batch 是一个 dict，直接 **batch 传给 model
    - 要求 model.forward 返回 dict，至少含 "loss"
      额外可选 "ce_loss" / "aux_loss" 用于日志
    """
    model.train()
    total_loss = 0.0
    steps = 0

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(dataloader):
        batch = _move_batch_to_device(batch, device)

        # 通用调用：支持 Stage1(Text) / Stage2(Omni) / 以后 Talker
        outputs = model(**batch)

        loss = outputs["loss"]
        ce_loss = outputs.get("ce_loss", None)
        aux_loss = outputs.get("aux_loss", None)
        
        # ✅ 如果出现 NaN/Inf，打印一次并跳过这个 batch，避免把整个 epoch 平均也搞成 NaN
        if torch.isnan(loss) or torch.isinf(loss):
            print(
                f"[train_one_epoch] NaN/Inf loss at batch {batch_idx} "
                f"(loss={loss.item()}, "
                f"ce={ce_loss.item() if ce_loss is not None else 'None'}, "
                f"aux={aux_loss.item() if aux_loss is not None else 'None'}) — skip this batch."
            )
            optimizer.zero_grad(set_to_none=True)
            continue

        # 梯度累积
        loss = loss / gradient_accumulation_steps
        loss.backward()

        total_loss += loss.item()
        steps += 1

        if steps % gradient_accumulation_steps == 0:
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        if log_step_fn is not None:
            # 这里打印的是“真实 loss”（未除以 GA）
            log_step_fn(
                step=steps,
                loss=loss.item() * gradient_accumulation_steps,
                batch_idx=batch_idx,
                ce_loss=ce_loss.item() if ce_loss is not None else None,
                aux_loss=aux_loss.item() if aux_loss is not None else None,
            )

    avg_loss = total_loss / max(1, len(dataloader))
    return avg_loss


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """
    通用评估循环：
    - 同样直接 **batch 传给 model
    - 要求模型返回 "loss"
    """
    model.eval()
    total_loss = 0.0
    steps = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = _move_batch_to_device(batch, device)

            outputs = model(**batch)
            loss = outputs["loss"]
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[evaluate] NaN/Inf loss at batch {batch}, skip.")
                continue

            total_loss += loss.item()
            steps += 1

    avg_loss = total_loss / max(1, steps)
    return avg_loss
