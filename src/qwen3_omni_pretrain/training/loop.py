# src/qwen3_omni_pretrain/training/loop.py

from typing import Dict, Any, Optional

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    device: torch.device,
    gradient_accumulation_steps: int = 1,
    log_step_fn: Optional[Any] = None,
    epoch: int = 0,
) -> float:
    model.train()
    total_loss = 0.0
    step = 0

    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", dynamic_ncols=True)
    for batch_idx, batch in enumerate(pbar):
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask", None),
            labels=batch["labels"],
        )
        loss = outputs["loss"]
        loss = loss / gradient_accumulation_steps
        loss.backward()

        total_loss += loss.item()
        step += 1

        if step % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        if log_step_fn is not None:
            log_step_fn(
                step=step,
                loss=loss.item() * gradient_accumulation_steps,
                batch_idx=batch_idx,
            )

        avg_loss = total_loss / step
        pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

    return total_loss / step


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    step = 0

    for batch in tqdm(dataloader, desc="Eval", dynamic_ncols=True):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask", None),
            labels=batch["labels"],
        )
        loss = outputs["loss"]

        total_loss += loss.item()
        step += 1

    return total_loss / step
