import os
from typing import Optional, Tuple

import torch
from transformers.modeling_utils import load_sharded_checkpoint


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
	"""Return the underlying model when wrapped by DDP/EMA/etc."""
	return model.module if hasattr(model, "module") else model


def save_checkpoint(
	checkpoint_dir: str,
	model: torch.nn.Module,
	optimizer: Optional[torch.optim.Optimizer],
	scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
	scaler: Optional[torch.cuda.amp.GradScaler],
	epoch: int,
	global_step: int,
	best_val_loss: float,
) -> str:
	"""Persist model (HF format) plus training state into checkpoint_dir."""
	os.makedirs(checkpoint_dir, exist_ok=True)

	# Save model weights/config in HF format
	to_save = _unwrap_model(model)
	to_save.save_pretrained(checkpoint_dir, safe_serialization=False)

	# Save trainer state
	state = {
		"epoch": epoch,
		"global_step": global_step,
		"best_val_loss": best_val_loss,
		"optimizer": optimizer.state_dict() if optimizer is not None else None,
		"scheduler": scheduler.state_dict() if scheduler is not None else None,
		"scaler": scaler.state_dict() if scaler is not None else None,
	}
	torch.save(state, os.path.join(checkpoint_dir, "trainer_state.pt"))
	return checkpoint_dir


def load_checkpoint(
	checkpoint_dir: str,
	model: torch.nn.Module,
	optimizer: Optional[torch.optim.Optimizer] = None,
	scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
	scaler: Optional[torch.cuda.amp.GradScaler] = None,
	map_location: Optional[torch.device] = None,
) -> Tuple[int, int, float]:
	"""
	Load model + optimizer/scheduler/scaler state from checkpoint_dir.
	Returns (start_epoch, global_step, best_val_loss).
	If trainer_state is missing, defaults to (0, 0, inf).
	"""
	model_path_bin = os.path.join(checkpoint_dir, "pytorch_model.bin")
	model_index = os.path.join(checkpoint_dir, "pytorch_model.bin.index.json")
	if os.path.exists(model_path_bin):
		state_dict = torch.load(model_path_bin, map_location=map_location)
		_unwrap_model(model).load_state_dict(state_dict, strict=False)
	elif os.path.exists(model_index):
		# HF sharded checkpoint
		load_sharded_checkpoint(_unwrap_model(model), checkpoint_dir, strict=False)
	else:
		raise FileNotFoundError(f"No pytorch_model.bin or shard index under {checkpoint_dir}")

	trainer_state_path = os.path.join(checkpoint_dir, "trainer_state.pt")
	if os.path.exists(trainer_state_path):
		trainer_state = torch.load(trainer_state_path, map_location=map_location)

		if optimizer is not None and trainer_state.get("optimizer") is not None:
			optimizer.load_state_dict(trainer_state["optimizer"])

		if scheduler is not None and trainer_state.get("scheduler") is not None:
			scheduler.load_state_dict(trainer_state["scheduler"])

		if scaler is not None and trainer_state.get("scaler") is not None:
			scaler.load_state_dict(trainer_state["scaler"])

		start_epoch = int(trainer_state.get("epoch", 0))
		global_step = int(trainer_state.get("global_step", 0))
		best_val_loss = float(trainer_state.get("best_val_loss", float("inf")))
		return start_epoch, global_step, best_val_loss

	# No trainer_state → start fresh from loaded weights
	return 0, 0, float("inf")
