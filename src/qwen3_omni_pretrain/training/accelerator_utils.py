# src/qwen3_omni_pretrain/training/accelerator_utils.py
"""
Accelerate 工具模块 - 提供统一的 Accelerator 初始化和配置管理
支持 DDP、DeepSpeed ZeRO、FSDP 三种分布式策略
"""

import os
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union, Tuple

import torch
from torch.utils.data import DataLoader

try:
    from accelerate import Accelerator, DistributedType
    from accelerate.utils import (
        DeepSpeedPlugin,
        FullyShardedDataParallelPlugin,
        ProjectConfiguration,
    )
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False
    Accelerator = None
    DistributedType = None
    DeepSpeedPlugin = None
    FullyShardedDataParallelPlugin = None
    ProjectConfiguration = None

from qwen3_omni_pretrain.utils.config_utils import load_yaml


@dataclass
class AcceleratorConfig:
    """Accelerator 配置数据类"""
    use_accelerator: bool = False
    mixed_precision: str = "bf16"  # "no", "fp16", "bf16"
    distributed_type: str = "MULTI_GPU"  # "NO", "MULTI_GPU", "DEEPSPEED", "FSDP"
    
    # DeepSpeed 配置
    deepspeed_config_path: Optional[str] = None
    
    # FSDP 配置
    fsdp_config_path: Optional[str] = None
    
    # 通用配置
    gradient_accumulation_steps: int = 8
    gradient_checkpointing: bool = False
    
    # 日志配置
    log_with: Optional[str] = "tensorboard"  # "tensorboard", "wandb", "all", None
    project_dir: str = "./runs"
    
    # 高级选项
    dispatch_batches: Optional[bool] = None
    split_batches: bool = False
    even_batches: bool = True
    step_scheduler_with_optimizer: bool = True


def load_accelerator_config(config_path: str) -> AcceleratorConfig:
    """从 YAML 文件加载 Accelerator 配置"""
    if not os.path.exists(config_path):
        return AcceleratorConfig()
    
    cfg = load_yaml(config_path)
    return AcceleratorConfig(
        use_accelerator=bool(cfg.get("use_accelerator", False)),
        mixed_precision=str(cfg.get("mixed_precision", "bf16")),
        distributed_type=str(cfg.get("distributed_type", "MULTI_GPU")),
        deepspeed_config_path=cfg.get("deepspeed_config_path"),
        fsdp_config_path=cfg.get("fsdp_config_path"),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 8)),
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", False)),
        log_with=cfg.get("log_with", "tensorboard"),
        project_dir=str(cfg.get("project_dir", "./runs")),
        dispatch_batches=cfg.get("dispatch_batches"),
        split_batches=bool(cfg.get("split_batches", False)),
        even_batches=bool(cfg.get("even_batches", True)),
        step_scheduler_with_optimizer=bool(cfg.get("step_scheduler_with_optimizer", True)),
    )


def _build_deepspeed_plugin(config_path: Optional[str], gradient_accumulation_steps: int) -> Optional["DeepSpeedPlugin"]:
    """构建 DeepSpeed 插件"""
    if not ACCELERATE_AVAILABLE or DeepSpeedPlugin is None:
        return None
    
    if config_path and os.path.exists(config_path):
        # 从 JSON 文件加载 DeepSpeed 配置
        with open(config_path, "r") as f:
            ds_config = json.load(f)
        return DeepSpeedPlugin(
            hf_ds_config=ds_config,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
    
    # 从 YAML 配置构建
    if config_path and config_path.endswith(".yaml"):
        yaml_cfg = load_yaml(config_path)
        return DeepSpeedPlugin(
            zero_stage=yaml_cfg.get("zero_stage", 3),
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_clipping=yaml_cfg.get("gradient_clipping", 1.0),
            offload_optimizer_device=yaml_cfg.get("offload_optimizer_device", "none"),
            offload_param_device=yaml_cfg.get("offload_param_device", "none"),
            zero3_init_flag=yaml_cfg.get("zero3_init_flag", True),
            zero3_save_16bit_model=yaml_cfg.get("zero3_save_16bit_model", True),
        )
    
    # 默认 ZeRO-3 配置
    return DeepSpeedPlugin(
        zero_stage=3,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_clipping=1.0,
        zero3_init_flag=True,
        zero3_save_16bit_model=True,
    )


def _build_fsdp_plugin(config_path: Optional[str]) -> Optional["FullyShardedDataParallelPlugin"]:
    """构建 FSDP 插件"""
    if not ACCELERATE_AVAILABLE or FullyShardedDataParallelPlugin is None:
        return None
    
    from torch.distributed.fsdp import ShardingStrategy, BackwardPrefetch
    from torch.distributed.fsdp.api import StateDictType
    
    # 策略映射
    sharding_strategy_map = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
        "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
    }
    
    backward_prefetch_map = {
        "BACKWARD_PRE": BackwardPrefetch.BACKWARD_PRE,
        "BACKWARD_POST": BackwardPrefetch.BACKWARD_POST,
        "NO_PREFETCH": None,
    }
    
    state_dict_type_map = {
        "FULL_STATE_DICT": StateDictType.FULL_STATE_DICT,
        "SHARDED_STATE_DICT": StateDictType.SHARDED_STATE_DICT,
        "LOCAL_STATE_DICT": StateDictType.LOCAL_STATE_DICT,
    }
    
    if config_path and os.path.exists(config_path):
        yaml_cfg = load_yaml(config_path)
        
        sharding_strategy = sharding_strategy_map.get(
            yaml_cfg.get("fsdp_sharding_strategy", "FULL_SHARD"),
            ShardingStrategy.FULL_SHARD
        )
        backward_prefetch = backward_prefetch_map.get(
            yaml_cfg.get("fsdp_backward_prefetch", "BACKWARD_PRE"),
            BackwardPrefetch.BACKWARD_PRE
        )
        state_dict_type = state_dict_type_map.get(
            yaml_cfg.get("fsdp_state_dict_type", "SHARDED_STATE_DICT"),
            StateDictType.SHARDED_STATE_DICT
        )
        
        return FullyShardedDataParallelPlugin(
            sharding_strategy=sharding_strategy,
            backward_prefetch=backward_prefetch,
            state_dict_type=state_dict_type,
            cpu_offload=yaml_cfg.get("fsdp_offload_params", False),
            sync_module_states=yaml_cfg.get("fsdp_sync_module_states", True),
            forward_prefetch=yaml_cfg.get("fsdp_forward_prefetch", True),
            use_orig_params=yaml_cfg.get("fsdp_use_orig_params", True),
            activation_checkpointing=yaml_cfg.get("fsdp_activation_checkpointing", False),
        )
    
    # 默认 FSDP 配置
    return FullyShardedDataParallelPlugin(
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        state_dict_type=StateDictType.SHARDED_STATE_DICT,
        sync_module_states=True,
        forward_prefetch=True,
        use_orig_params=True,
    )


def create_accelerator(
    accelerator_config: AcceleratorConfig,
    project_name: str = "qwen3_omni",
) -> "Accelerator":
    """
    根据配置创建 Accelerator 实例
    
    Args:
        accelerator_config: Accelerator 配置对象
        project_name: 项目名称（用于日志目录）
    
    Returns:
        配置好的 Accelerator 实例
    """
    if not ACCELERATE_AVAILABLE:
        raise ImportError(
            "accelerate is not installed. Please install it with: pip install accelerate>=0.30.0"
        )
    
    # 项目配置
    project_config = ProjectConfiguration(
        project_dir=accelerator_config.project_dir,
        logging_dir=os.path.join(accelerator_config.project_dir, project_name),
    )
    
    # 根据分布式类型构建插件
    deepspeed_plugin = None
    fsdp_plugin = None
    
    dist_type = accelerator_config.distributed_type.upper()
    
    if dist_type == "DEEPSPEED":
        deepspeed_plugin = _build_deepspeed_plugin(
            accelerator_config.deepspeed_config_path,
            accelerator_config.gradient_accumulation_steps,
        )
    elif dist_type == "FSDP":
        fsdp_plugin = _build_fsdp_plugin(accelerator_config.fsdp_config_path)
    
    # 创建 Accelerator
    # 注意: dispatch_batches 和 even_batches 参数在某些版本中可能不支持
    accelerator_kwargs = {
        "mixed_precision": accelerator_config.mixed_precision,
        "gradient_accumulation_steps": accelerator_config.gradient_accumulation_steps,
        "deepspeed_plugin": deepspeed_plugin,
        "fsdp_plugin": fsdp_plugin,
        "log_with": accelerator_config.log_with if accelerator_config.log_with else None,
        "project_config": project_config,
        "split_batches": accelerator_config.split_batches,
        "step_scheduler_with_optimizer": accelerator_config.step_scheduler_with_optimizer,
    }
    
    # 可选参数（根据 accelerate 版本）
    try:
        accelerator = Accelerator(
            **accelerator_kwargs,
            dispatch_batches=accelerator_config.dispatch_batches,
            even_batches=accelerator_config.even_batches,
        )
    except TypeError:
        # 旧版本 accelerate 可能不支持这些参数
        accelerator = Accelerator(**accelerator_kwargs)
    
    return accelerator


def prepare_model_optimizer_dataloader(
    accelerator: "Accelerator",
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader] = None,
) -> Tuple[torch.nn.Module, torch.optim.Optimizer, Any, DataLoader, Optional[DataLoader]]:
    """
    使用 Accelerator 封装模型、优化器和数据加载器
    
    Returns:
        (model, optimizer, scheduler, train_dataloader, val_dataloader)
    """
    if val_dataloader is not None:
        model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, val_dataloader
        )
    else:
        model, optimizer, train_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader
        )
    
    if scheduler is not None:
        scheduler = accelerator.prepare(scheduler)
    
    return model, optimizer, scheduler, train_dataloader, val_dataloader


def save_accelerator_checkpoint(
    accelerator: "Accelerator",
    checkpoint_dir: str,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    tokenizer=None,
) -> str:
    """
    使用 Accelerator 保存检查点
    
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
    
    # 保存 accelerator 状态（包含 model, optimizer, scheduler, scaler）
    accelerator.save_state(checkpoint_dir)
    
    # 主进程保存额外信息
    if accelerator.is_main_process:
        # 保存 trainer 状态
        state = {
            "epoch": epoch,
            "global_step": global_step,
            "best_val_loss": best_val_loss,
        }
        torch.save(state, os.path.join(checkpoint_dir, "trainer_state.pt"))
        
        # 保存 tokenizer
        if tokenizer is not None:
            tokenizer.save_pretrained(checkpoint_dir)
    
    accelerator.wait_for_everyone()
    return checkpoint_dir


def load_accelerator_checkpoint(
    accelerator: "Accelerator",
    checkpoint_dir: str,
) -> Tuple[int, int, float]:
    """
    使用 Accelerator 加载检查点
    
    Returns:
        (start_epoch, global_step, best_val_loss)
    """
    # 加载 accelerator 状态
    accelerator.load_state(checkpoint_dir)
    
    # 加载 trainer 状态
    trainer_state_path = os.path.join(checkpoint_dir, "trainer_state.pt")
    if os.path.exists(trainer_state_path):
        state = torch.load(trainer_state_path, map_location="cpu")
        start_epoch = int(state.get("epoch", 0))
        global_step = int(state.get("global_step", 0))
        best_val_loss = float(state.get("best_val_loss", float("inf")))
        return start_epoch, global_step, best_val_loss
    
    return 0, 0, float("inf")


def get_accelerator_device(accelerator: "Accelerator") -> torch.device:
    """获取 Accelerator 管理的设备"""
    return accelerator.device


def is_main_process_accelerator(accelerator: "Accelerator") -> bool:
    """检查当前进程是否为主进程"""
    return accelerator.is_main_process


def print_on_main(accelerator: "Accelerator", *args, **kwargs):
    """仅在主进程打印"""
    if accelerator.is_main_process:
        print(*args, **kwargs)


def unwrap_model(accelerator: "Accelerator", model: torch.nn.Module) -> torch.nn.Module:
    """获取未封装的原始模型"""
    return accelerator.unwrap_model(model)
