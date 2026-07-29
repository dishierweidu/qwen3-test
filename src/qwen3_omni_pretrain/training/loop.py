from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

try:
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.distributed as dist
except Exception:
    DDP = None
    dist = None


class NonFiniteTrainingError(FloatingPointError):
    """Raised when model outputs or gradients contain NaN/Inf values."""


def _move_batch_to_device(
    batch: Dict[str, Any], device: torch.device
) -> Dict[str, Any]:
    new_batch: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            new_batch[key] = value.to(device, non_blocking=True)
        else:
            new_batch[key] = value
    return new_batch


def _split_batch_metadata(
    batch: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Remove underscore-prefixed metadata before calling ``model.forward``."""
    model_batch: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}
    for key, value in batch.items():
        if key.startswith("_"):
            metadata[key] = value
        else:
            model_batch[key] = value
    return model_batch, metadata


def _dist_is_initialized() -> bool:
    return bool(
        dist is not None
        and dist.is_available()
        and dist.is_initialized()
    )


def _is_global_bad_flag(local_bad: bool, device: torch.device) -> bool:
    """All-reduce a local failure flag so every rank takes the same path."""
    if not _dist_is_initialized():
        return bool(local_bad)
    flag = torch.tensor(
        [1 if local_bad else 0],
        device=device,
        dtype=torch.int32,
    )
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def _is_global_bad_loss(loss: torch.Tensor) -> bool:
    """Compatibility wrapper retained for existing trainer imports."""
    return _is_global_bad_flag(
        not bool(torch.isfinite(loss).all().item()), loss.device
    )


def _find_non_finite_output(outputs: Mapping[str, Any]) -> Optional[str]:
    """Return the first checked output field containing NaN/Inf values."""
    for key in ("loss", "ce_loss", "aux_loss", "logits"):
        value = outputs.get(key)
        if isinstance(value, torch.Tensor) and not bool(
            torch.isfinite(value).all().item()
        ):
            count = int((~torch.isfinite(value)).sum().item())
            return f"output {key!r} contains {count} non-finite value(s)"
    return None


def _find_non_finite_gradient(model: torch.nn.Module) -> Optional[str]:
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        if not bool(torch.isfinite(gradient).all().item()):
            count = int((~torch.isfinite(gradient)).sum().item())
            return f"gradient {name!r} contains {count} non-finite value(s)"
    return None


def _format_metadata(metadata: Mapping[str, Any]) -> str:
    if not metadata:
        return ""
    parts = []
    if "_sample_ids" in metadata:
        parts.append(f"sample_ids={metadata['_sample_ids']!r}")
    if "_media_errors" in metadata and metadata["_media_errors"]:
        parts.append(f"media_errors={metadata['_media_errors']!r}")
    remaining = {
        key: value
        for key, value in metadata.items()
        if key not in {"_sample_ids", "_media_errors"}
    }
    if remaining:
        parts.append(f"metadata={remaining!r}")
    return ", ".join(parts)


def _raise_non_finite(
    *,
    reason: Optional[str],
    global_bad: bool,
    batch_idx: int,
    metadata: Mapping[str, Any],
) -> None:
    if not global_bad:
        return
    local_reason = reason or "another distributed rank reported non-finite values"
    metadata_text = _format_metadata(metadata)
    suffix = f", {metadata_text}" if metadata_text else ""
    raise NonFiniteTrainingError(
        f"Non-finite training state at batch {batch_idx + 1}: "
        f"{local_reason}{suffix}"
    )


def _autocast_context(
    device: torch.device, autocast_dtype: Optional[torch.dtype]
):
    if autocast_dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=autocast_dtype)


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
    Train for one epoch and fail consistently across ranks on NaN/Inf.

    Underscore-prefixed batch keys are treated as metadata and are never
    forwarded to the model. They are included in numerical failure messages.
    """
    model.train()
    grad_accum = max(1, gradient_accumulation_steps)
    is_ddp = bool(
        ((DDP is not None) and isinstance(model, DDP))
        or hasattr(model, "no_sync")
    )
    rank = dist.get_rank() if _dist_is_initialized() else 0

    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_ce = 0.0
    total_aux = 0.0
    num_updates = 0

    for batch_idx, raw_batch in enumerate(dataloader):
        if should_stop_fn is not None and should_stop_fn():
            break
        if batch_idx == 0:
            if "input_ids" in raw_batch:
                print(
                    f"[Rank {rank}] got first batch, "
                    f"input_ids={raw_batch['input_ids'].shape}"
                )
            else:
                print(
                    f"[Rank {rank}] got first batch, "
                    f"keys={list(raw_batch.keys())}"
                )

        moved_batch = _move_batch_to_device(raw_batch, device)
        batch, metadata = _split_batch_metadata(moved_batch)
        is_update_step = (
            ((batch_idx + 1) % grad_accum == 0)
            or ((batch_idx + 1) == len(dataloader))
        )
        sync_context = (
            model.no_sync()
            if is_ddp and not is_update_step
            else nullcontext()
        )

        with sync_context:
            with _autocast_context(device, autocast_dtype):
                outputs = model(**batch)
            if not isinstance(outputs, Mapping) or "loss" not in outputs:
                raise TypeError("model.forward must return a mapping containing 'loss'")

            loss = outputs["loss"]
            if not isinstance(loss, torch.Tensor):
                raise TypeError("outputs['loss'] must be a torch.Tensor")
            ce_loss = outputs.get("ce_loss")
            aux_loss = outputs.get("aux_loss")

            output_reason = _find_non_finite_output(outputs)
            global_output_bad = _is_global_bad_flag(
                output_reason is not None, loss.device
            )
            if global_output_bad:
                optimizer.zero_grad(set_to_none=True)
                _raise_non_finite(
                    reason=output_reason,
                    global_bad=True,
                    batch_idx=batch_idx,
                    metadata=metadata,
                )

            scaled_loss = loss / grad_accum
            if grad_scaler is not None:
                grad_scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

        if is_update_step:
            if grad_scaler is not None:
                grad_scaler.unscale_(optimizer)

            gradient_reason = _find_non_finite_gradient(model)
            global_gradient_bad = _is_global_bad_flag(
                gradient_reason is not None, loss.device
            )
            if global_gradient_bad:
                optimizer.zero_grad(set_to_none=True)
                _raise_non_finite(
                    reason=gradient_reason,
                    global_bad=True,
                    batch_idx=batch_idx,
                    metadata=metadata,
                )

            clip_grad_norm_(model.parameters(), max_norm=1.0)
            if grad_scaler is not None:
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            num_updates += 1
            true_loss = scaled_loss.detach().item() * grad_accum
            total_loss += true_loss
            if isinstance(ce_loss, torch.Tensor):
                total_ce += ce_loss.detach().item()
            if isinstance(aux_loss, torch.Tensor):
                total_aux += aux_loss.detach().item()

            current_lr = (
                optimizer.param_groups[0].get("lr")
                if optimizer.param_groups
                else None
            )
            if log_step_fn is not None:
                log_step_fn(
                    step=num_updates,
                    loss=true_loss,
                    batch_idx=batch_idx,
                    ce_loss=(
                        ce_loss.detach().item()
                        if isinstance(ce_loss, torch.Tensor)
                        else None
                    ),
                    aux_loss=(
                        aux_loss.detach().item()
                        if isinstance(aux_loss, torch.Tensor)
                        else None
                    ),
                    lr=current_lr,
                )
            if after_step_fn is not None:
                after_step_fn(
                    step=num_updates,
                    loss=true_loss,
                    batch_idx=batch_idx,
                    ce_loss=(
                        ce_loss.detach().item()
                        if isinstance(ce_loss, torch.Tensor)
                        else None
                    ),
                    aux_loss=(
                        aux_loss.detach().item()
                        if isinstance(aux_loss, torch.Tensor)
                        else None
                    ),
                    lr=current_lr,
                    model=model,
                )
            if should_stop_fn is not None and should_stop_fn():
                break

    if num_updates == 0:
        return float("nan")
    return total_loss / num_updates


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    autocast_dtype: Optional[torch.dtype] = None,
) -> float:
    """Evaluate and raise on non-finite output instead of skipping samples."""
    model.eval()
    total_loss = 0.0
    steps = 0

    with torch.no_grad():
        for batch_idx, raw_batch in enumerate(dataloader):
            moved_batch = _move_batch_to_device(raw_batch, device)
            batch, metadata = _split_batch_metadata(moved_batch)
            with _autocast_context(device, autocast_dtype):
                outputs = model(**batch)
            if not isinstance(outputs, Mapping) or "loss" not in outputs:
                raise TypeError("model.forward must return a mapping containing 'loss'")
            loss = outputs["loss"]
            reason = _find_non_finite_output(outputs)
            global_bad = _is_global_bad_flag(reason is not None, loss.device)
            _raise_non_finite(
                reason=reason,
                global_bad=global_bad,
                batch_idx=batch_idx,
                metadata=metadata,
            )
            total_loss += loss.detach().item()
            steps += 1

    return total_loss / max(1, steps)
