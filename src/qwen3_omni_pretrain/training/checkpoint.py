import os
import shutil
import warnings
import tempfile
import glob
from typing import Optional, Tuple, TYPE_CHECKING

import torch
import torch.distributed as dist
from transformers.modeling_utils import load_sharded_checkpoint

if TYPE_CHECKING:
    from accelerate import Accelerator


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
	"""Return the underlying model when wrapped by DDP/EMA/etc."""
	return model.module if hasattr(model, "module") else model


def verify_checkpoint_integrity(checkpoint_dir: str, check_model: bool = True) -> bool:
	"""
	验证 checkpoint 的完整性。
	
	Args:
		checkpoint_dir: checkpoint 目录
		check_model: 是否检查模型文件（较慢）
	
	Returns:
		True 如果 checkpoint 完整，否则 False
	"""
	try:
		# 检查 trainer_state.pt
		trainer_state_path = os.path.join(checkpoint_dir, "trainer_state.pt")
		if os.path.exists(trainer_state_path):
			# 尝试加载以验证文件完整性
			torch.load(trainer_state_path, map_location="cpu")
		
		if check_model:
			# 检查模型文件
			model_bin = os.path.join(checkpoint_dir, "pytorch_model.bin")
			model_index = os.path.join(checkpoint_dir, "pytorch_model.bin.index.json")
			
			if os.path.exists(model_bin):
				# 单个模型文件，尝试加载
				torch.load(model_bin, map_location="cpu")
			elif os.path.exists(model_index):
				# 分片模型，检查所有分片文件
				import json
				with open(model_index, 'r') as f:
					index = json.load(f)
				# 获取所有分片文件
				shard_files = set(index.get("weight_map", {}).values())
				for shard_file in shard_files:
					shard_path = os.path.join(checkpoint_dir, shard_file)
					if not os.path.exists(shard_path):
						return False
					# 尝试加载分片以验证完整性
					torch.load(shard_path, map_location="cpu")
		
		return True
	except Exception as e:
		warnings.warn(f"[checkpoint] Integrity check failed: {e}")
		return False


def atomic_save_checkpoint(
	checkpoint_dir: str,
	model: torch.nn.Module,
	optimizer: Optional[torch.optim.Optimizer],
	scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
	scaler: Optional[torch.cuda.amp.GradScaler],
	epoch: int,
	global_step: int,
	best_val_loss: float,
	verify: bool = True,
	keep_backup: bool = True,
) -> str:
	"""
	原子性保存 checkpoint，防止保存中断导致文件损坏。
	
	流程：
	1. 保存到临时目录
	2. 验证临时目录中的文件完整性
	3. 如果目标目录已存在，重命名为 .backup
	4. 将临时目录 rename 到目标位置
	5. 删除备份（如果不保留）
	
	Args:
		checkpoint_dir: 目标保存目录
		model: 模型
		optimizer: 优化器
		scheduler: 学习率调度器
		scaler: GradScaler
		epoch: 当前 epoch
		global_step: 全局步数
		best_val_loss: 最佳验证损失
		verify: 是否验证 checkpoint 完整性（推荐 True）
		keep_backup: 是否保留上一个 checkpoint 的备份
	
	Returns:
		保存路径
	"""
	parent_dir = os.path.dirname(checkpoint_dir)
	os.makedirs(parent_dir, exist_ok=True)
	
	# 创建临时目录（与目标在同一文件系统，确保 rename 是原子操作）
	temp_dir = checkpoint_dir + ".tmp"
	backup_dir = checkpoint_dir + ".backup"
	
	# 清理可能存在的临时目录
	if os.path.exists(temp_dir):
		shutil.rmtree(temp_dir)
	
	os.makedirs(temp_dir, exist_ok=True)
	
	try:
		# 保存到临时目录
		to_save = _unwrap_model(model)
		to_save.save_pretrained(temp_dir, safe_serialization=False)
		
		# 保存训练状态
		state = {
			"epoch": epoch,
			"global_step": global_step,
			"best_val_loss": best_val_loss,
			"optimizer": optimizer.state_dict() if optimizer is not None else None,
			"scheduler": scheduler.state_dict() if scheduler is not None else None,
			"scaler": scaler.state_dict() if scaler is not None else None,
		}
		torch.save(state, os.path.join(temp_dir, "trainer_state.pt"))
		
		# 验证完整性（只检查 trainer_state，模型检查太慢）
		if verify:
			if not verify_checkpoint_integrity(temp_dir, check_model=False):
				raise RuntimeError(f"Checkpoint verification failed for {temp_dir}")
		
		# 原子替换：先备份旧的，再 rename 新的
		if os.path.exists(backup_dir):
			shutil.rmtree(backup_dir)
		
		if os.path.exists(checkpoint_dir):
			os.rename(checkpoint_dir, backup_dir)
		
		os.rename(temp_dir, checkpoint_dir)
		
		# 成功后删除备份（如果不保留）
		if not keep_backup and os.path.exists(backup_dir):
			shutil.rmtree(backup_dir)
		
		return checkpoint_dir
		
	except Exception as e:
		# 清理临时目录
		if os.path.exists(temp_dir):
			shutil.rmtree(temp_dir)
		# 如果有备份，恢复它
		if os.path.exists(backup_dir) and not os.path.exists(checkpoint_dir):
			os.rename(backup_dir, checkpoint_dir)
		raise RuntimeError(f"Failed to save checkpoint: {e}") from e


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
	"""Persist model (HF format) plus training state into checkpoint_dir.
	
	使用原子写入机制，防止保存中断导致文件损坏。
	"""
	return atomic_save_checkpoint(
		checkpoint_dir=checkpoint_dir,
		model=model,
		optimizer=optimizer,
		scheduler=scheduler,
		scaler=scaler,
		epoch=epoch,
		global_step=global_step,
		best_val_loss=best_val_loss,
		verify=True,
		keep_backup=True,
	)


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
	
	如果 checkpoint 损坏，会自动尝试加载 .backup 备份。
	"""
	# 检查是否有备份可用
	backup_dir = checkpoint_dir + ".backup"
	dirs_to_try = [checkpoint_dir]
	if os.path.exists(backup_dir):
		dirs_to_try.append(backup_dir)
	
	last_error = None
	for try_dir in dirs_to_try:
		try:
			return _load_checkpoint_impl(try_dir, model, optimizer, scheduler, scaler, map_location)
		except (RuntimeError, FileNotFoundError) as e:
			last_error = e
			if try_dir == checkpoint_dir and os.path.exists(backup_dir):
				warnings.warn(
					f"[checkpoint] Failed to load from {try_dir}: {e}\n"
					f"Trying backup: {backup_dir}"
				)
			continue
	
	# 所有尝试都失败
	raise RuntimeError(
		f"Failed to load checkpoint from {checkpoint_dir} (and backup if exists): {last_error}"
	) from last_error


def _load_checkpoint_impl(
	checkpoint_dir: str,
	model: torch.nn.Module,
	optimizer: Optional[torch.optim.Optimizer] = None,
	scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
	scaler: Optional[torch.cuda.amp.GradScaler] = None,
	map_location: Optional[torch.device] = None,
) -> Tuple[int, int, float]:
	"""内部实现：加载 checkpoint"""
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
			try:
				optimizer.load_state_dict(trainer_state["optimizer"])
			except (ValueError, RuntimeError) as e:
				warnings.warn(
					f"[checkpoint] optimizer state mismatch, skip loading optimizer: {e}"
				)

		if scheduler is not None and trainer_state.get("scheduler") is not None:
			try:
				scheduler.load_state_dict(trainer_state["scheduler"])
			except (ValueError, RuntimeError) as e:
				warnings.warn(
					f"[checkpoint] scheduler state mismatch, skip loading scheduler: {e}"
				)

		if scaler is not None and trainer_state.get("scaler") is not None:
			try:
				scaler.load_state_dict(trainer_state["scaler"])
			except (ValueError, RuntimeError) as e:
				warnings.warn(
					f"[checkpoint] scaler state mismatch, skip loading scaler: {e}"
				)

		start_epoch = int(trainer_state.get("epoch", 0))
		global_step = int(trainer_state.get("global_step", 0))
		best_val_loss = float(trainer_state.get("best_val_loss", float("inf")))
		return start_epoch, global_step, best_val_loss

	# No trainer_state → start fresh from loaded weights
	return 0, 0, float("inf")


# =============================================================================
# Accelerator 专用检查点函数
# =============================================================================

def save_checkpoint_accelerator(
	accelerator: "Accelerator",
	checkpoint_dir: str,
	epoch: int,
	global_step: int,
	best_val_loss: float,
	tokenizer=None,
) -> str:
	"""
	使用 Accelerator 保存检查点（原子写入）
	
	Accelerator.save_state() 会自动保存:
	- 模型权重 (支持 DDP/DeepSpeed/FSDP)
	- 优化器状态
	- 学习率调度器状态
	- GradScaler 状态 (如果使用混合精度)
	- 随机数状态
	
	使用临时目录 + rename 实现原子写入，防止保存中断导致文件损坏。
	
	Args:
		accelerator: Accelerator 实例
		checkpoint_dir: 检查点保存目录
		epoch: 当前 epoch
		global_step: 全局步数
		best_val_loss: 最佳验证损失
		tokenizer: tokenizer（可选）
	
	Returns:
		保存路径
	"""
	# 等待所有进程
	accelerator.wait_for_everyone()
	
	temp_dir = checkpoint_dir + ".tmp"
	backup_dir = checkpoint_dir + ".backup"
	
	# 清理可能存在的临时目录（仅主进程）
	if accelerator.is_main_process:
		if os.path.exists(temp_dir):
			shutil.rmtree(temp_dir)
	
	accelerator.wait_for_everyone()
	os.makedirs(temp_dir, exist_ok=True)
	
	# 保存 accelerator 状态到临时目录
	accelerator.save_state(temp_dir)
	
	# 主进程保存额外的训练状态信息
	if accelerator.is_main_process:
		state = {
			"epoch": epoch,
			"global_step": global_step,
			"best_val_loss": best_val_loss,
		}
		torch.save(state, os.path.join(temp_dir, "trainer_state.pt"))
		
		# 保存 tokenizer
		if tokenizer is not None:
			tokenizer.save_pretrained(temp_dir)
		
		# 验证 trainer_state 完整性
		try:
			torch.load(os.path.join(temp_dir, "trainer_state.pt"), map_location="cpu")
		except Exception as e:
			shutil.rmtree(temp_dir)
			raise RuntimeError(f"Checkpoint verification failed: {e}")
		
		# 原子替换
		if os.path.exists(backup_dir):
			shutil.rmtree(backup_dir)
		if os.path.exists(checkpoint_dir):
			os.rename(checkpoint_dir, backup_dir)
		os.rename(temp_dir, checkpoint_dir)
	
	accelerator.wait_for_everyone()
	return checkpoint_dir


def load_checkpoint_accelerator(
	accelerator: "Accelerator",
	checkpoint_dir: str,
) -> Tuple[int, int, float]:
	"""
	使用 Accelerator 加载检查点
	
	Accelerator.load_state() 会自动加载:
	- 模型权重
	- 优化器状态
	- 学习率调度器状态
	- GradScaler 状态
	- 随机数状态
	
	如果 checkpoint 损坏，会自动尝试加载 .backup 备份。
	
	Args:
		accelerator: Accelerator 实例
		checkpoint_dir: 检查点目录
	
	Returns:
		(start_epoch, global_step, best_val_loss)
	"""
	backup_dir = checkpoint_dir + ".backup"
	dirs_to_try = [checkpoint_dir]
	if os.path.exists(backup_dir):
		dirs_to_try.append(backup_dir)
	
	last_error = None
	for try_dir in dirs_to_try:
		try:
			# 加载 accelerator 状态
			accelerator.load_state(try_dir)
			
			# 加载额外的训练状态
			trainer_state_path = os.path.join(try_dir, "trainer_state.pt")
			if os.path.exists(trainer_state_path):
				state = torch.load(trainer_state_path, map_location="cpu")
				start_epoch = int(state.get("epoch", 0))
				global_step = int(state.get("global_step", 0))
				best_val_loss = float(state.get("best_val_loss", float("inf")))
				return start_epoch, global_step, best_val_loss
			
			return 0, 0, float("inf")
		except (RuntimeError, FileNotFoundError) as e:
			last_error = e
			if try_dir == checkpoint_dir and os.path.exists(backup_dir):
				warnings.warn(
					f"[checkpoint] Failed to load from {try_dir}: {e}\n"
					f"Trying backup: {backup_dir}"
				)
			continue
	
	raise RuntimeError(
		f"Failed to load checkpoint from {checkpoint_dir} (and backup if exists): {last_error}"
	) from last_error


def save_model_only_accelerator(
	accelerator: "Accelerator",
	model: torch.nn.Module,
	save_dir: str,
	safe_serialization: bool = True,
) -> str:
	"""
	仅保存模型权重（用于推理/发布）
	
	Args:
		accelerator: Accelerator 实例
		model: 模型
		save_dir: 保存目录
		safe_serialization: 是否使用 safetensors 格式
	
	Returns:
		保存路径
	"""
	accelerator.wait_for_everyone()

	rank = dist.get_rank() if dist.is_initialized() else 0
	unwrapped_model = accelerator.unwrap_model(model)
	print(f"[rank{rank}] >>> save_only_model: before state_dict", flush=True)
	state_dict = accelerator.get_state_dict(unwrapped_model)
	print(f"[rank{rank}] >>> save_only_model: after state_dict", flush=True)

	accelerator.wait_for_everyone()

	if accelerator.is_main_process:
		os.makedirs(save_dir, exist_ok=True)
		print("[rank0] >>> save_only_model: before save_pretrained", flush=True)
		unwrapped_model.save_pretrained(
			save_dir,
			state_dict=state_dict,
			safe_serialization=safe_serialization,
			max_shard_size="2GB",
		)
		print("[rank0] >>> save_only_model: after save_pretrained", flush=True)

	accelerator.wait_for_everyone()
	print(f"[rank{rank}] >>> save_only_model: done", flush=True)
	return save_dir

