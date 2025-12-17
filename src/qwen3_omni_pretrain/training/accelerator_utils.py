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
import torch.distributed as dist
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
from qwen3_omni_pretrain.parallel.initialize import (
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank,
)


@dataclass
class AcceleratorConfig:
    """Accelerator 配置数据类"""
    use_accelerator: bool = False
    mixed_precision: str = "bf16"  # "no", "fp16", "bf16"
    distributed_type: str = "MULTI_GPU"  # "NO", "MULTI_GPU", "DEEPSPEED", "FSDP"
    
    # DeepSpeed 配置
    deepspeed_config_path: Optional[str] = None
    deepspeed_config: Optional[Dict[str, Any]] = None  # 内嵌的 deepspeed 配置
    
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
        deepspeed_config=cfg.get("deepspeed_config"),  # 内嵌的 deepspeed 配置
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


def _build_deepspeed_plugin(
    config_path: Optional[str], 
    gradient_accumulation_steps: int,
    deepspeed_config: Optional[Dict[str, Any]] = None
) -> Optional["DeepSpeedPlugin"]:
    """构建 DeepSpeed 插件
    
    优先级：
    1. config_path (JSON 文件路径)
    2. deepspeed_config (内嵌配置字典)
    3. 默认 ZeRO-2 配置 (适用于 TP 训练)
    """
    if not ACCELERATE_AVAILABLE or DeepSpeedPlugin is None:
        return None
    
    # 优先级 1：从 JSON 文件加载
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            ds_config = json.load(f)
        return DeepSpeedPlugin(
            hf_ds_config=ds_config,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
    
    # 优先级 2：从内嵌配置构建 (例如 accelerate 配置文件中的 deepspeed_config)
    if deepspeed_config and isinstance(deepspeed_config, dict):
        return DeepSpeedPlugin(
            zero_stage=deepspeed_config.get("zero_stage", 2),
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_clipping=deepspeed_config.get("gradient_clipping", 1.0),
            offload_optimizer_device=deepspeed_config.get("offload_optimizer_device", "none"),
            offload_param_device=deepspeed_config.get("offload_param_device", "none"),
            zero3_init_flag=deepspeed_config.get("zero3_init_flag", False),
            zero3_save_16bit_model=deepspeed_config.get("zero3_save_16bit_model", True),
        )
    
    # 优先级 3：从 YAML 配置构建 (旧路径)
    if config_path and config_path.endswith(".yaml"):
        yaml_cfg = load_yaml(config_path)
        return DeepSpeedPlugin(
            zero_stage=yaml_cfg.get("zero_stage", 2),
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_clipping=yaml_cfg.get("gradient_clipping", 1.0),
            offload_optimizer_device=yaml_cfg.get("offload_optimizer_device", "none"),
            offload_param_device=yaml_cfg.get("offload_param_device", "none"),
            zero3_init_flag=yaml_cfg.get("zero3_init_flag", False),
            zero3_save_16bit_model=yaml_cfg.get("zero3_save_16bit_model", True),
        )
    
    # 默认配置：使用 ZeRO-2 (兼容 TP)
    return DeepSpeedPlugin(
        zero_stage=2,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_clipping=1.0,
        zero3_init_flag=False,
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
    
    # 检查是否由 accelerate launch 启动
    # 如果是，不要创建自己的插件，让 accelerate 使用其配置文件
    launched_by_accelerate = os.environ.get("ACCELERATE_USE_DEEPSPEED", "false").lower() == "true" or \
                             os.environ.get("USE_DEEPSPEED", "false").lower() == "true" or \
                             os.environ.get("RANK") is not None
    
    # 项目配置
    project_config = ProjectConfiguration(
        project_dir=accelerator_config.project_dir,
        logging_dir=os.path.join(accelerator_config.project_dir, project_name),
    )
    
    # 根据分布式类型构建插件
    deepspeed_plugin = None
    fsdp_plugin = None
    
    dist_type = accelerator_config.distributed_type.upper()
    
    # 只有在非 accelerate launch 环境下才构建插件
    # 否则让 accelerate 使用其自己的配置
    if not launched_by_accelerate:
        if dist_type == "DEEPSPEED":
            deepspeed_plugin = _build_deepspeed_plugin(
                accelerator_config.deepspeed_config_path,
                accelerator_config.gradient_accumulation_steps,
                accelerator_config.deepspeed_config,  # 传递内嵌配置
            )
        elif dist_type == "FSDP":
            fsdp_plugin = _build_fsdp_plugin(accelerator_config.fsdp_config_path)
    
    # DDP 参数：MoE/TP 可能存在未参与 loss 的参数，需要开启 find_unused_parameters
    ddp_kwargs = None
    try:
        from accelerate.utils import DistributedDataParallelKwargs
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    except Exception:
        ddp_kwargs = None

    # 创建 Accelerator
    # 注意: dispatch_batches 和 even_batches 参数在某些版本中可能不支持
    accelerator_kwargs = {
        "gradient_accumulation_steps": accelerator_config.gradient_accumulation_steps,
        "log_with": accelerator_config.log_with if accelerator_config.log_with else None,
        "project_config": project_config,
        "split_batches": accelerator_config.split_batches,
        "step_scheduler_with_optimizer": accelerator_config.step_scheduler_with_optimizer,
    }
    
    # 只有在非 accelerate launch 环境下才设置这些参数
    if not launched_by_accelerate:
        accelerator_kwargs["mixed_precision"] = accelerator_config.mixed_precision
        accelerator_kwargs["deepspeed_plugin"] = deepspeed_plugin
        accelerator_kwargs["fsdp_plugin"] = fsdp_plugin
    
    if ddp_kwargs is not None:
        accelerator_kwargs["kwargs_handlers"] = [ddp_kwargs]
    
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


def save_tp_sharded_checkpoint(
    accelerator: "Accelerator",
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    checkpoint_dir: str,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    tokenizer=None,
) -> str:
    """TP 安全的分片检查点保存：每个 rank 写自己 shard，rank0 写 manifest。"""

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    rank = dist.get_rank() if dist.is_initialized() else 0
    world = accelerator.num_processes
    tp_size = get_tensor_model_parallel_world_size() if dist.is_initialized() else 1
    tp_rank = get_tensor_model_parallel_rank() if dist.is_initialized() else 0

    # 所有 rank 都获取 state_dict，确保 TP 内部 collective 对齐
    unwrapped = accelerator.unwrap_model(model)
    print(f"[rank{rank}] >>> save: before state_dict (tp_rank={tp_rank})", flush=True)
    model_sd = unwrapped.state_dict()
    optim_sd = optimizer.state_dict() if optimizer is not None else None
    sched_sd = scheduler.state_dict() if scheduler is not None else None
    print(f"[rank{rank}] >>> save: after state_dict", flush=True)

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        # 写单文件 train.pt，包含模型权重与基本训练状态
        train_payload = {
            "model": model_sd,
            "epoch": epoch,
            "global_step": global_step,
            "best_val_loss": best_val_loss,
        }
        if optim_sd is not None:
            train_payload["optimizer"] = optim_sd
        if sched_sd is not None:
            train_payload["scheduler"] = sched_sd

        torch.save(train_payload, os.path.join(checkpoint_dir, "train.pt"))

        if tokenizer is not None:
            tokenizer.save_pretrained(checkpoint_dir)

    accelerator.wait_for_everyone()
    print(f"[rank{rank}] >>> save: done", flush=True)
    return checkpoint_dir


def save_accelerator_checkpoint(
    accelerator: "Accelerator",
    checkpoint_dir: str,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    tokenizer=None,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> str:
    """通用保存：TP>1 走分片，其余保持原行为。"""

    accelerator.wait_for_everyone()

    tp = get_tensor_model_parallel_world_size() if dist.is_initialized() else 1
    if tp > 1:
        assert model is not None and optimizer is not None, "TP checkpoint需要 model 与 optimizer"
        return save_tp_sharded_checkpoint(
            accelerator=accelerator,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            checkpoint_dir=checkpoint_dir,
            epoch=epoch,
            global_step=global_step,
            best_val_loss=best_val_loss,
            tokenizer=tokenizer,
        )

    # 非 TP：使用 Accelerate 内建保存
    accelerator.save_state(checkpoint_dir)

    if accelerator.is_main_process:
        state = {
            "epoch": epoch,
            "global_step": global_step,
            "best_val_loss": best_val_loss,
        }
        torch.save(state, os.path.join(checkpoint_dir, "trainer_state.pt"))
        if tokenizer is not None:
            tokenizer.save_pretrained(checkpoint_dir)

    accelerator.wait_for_everyone()
    return checkpoint_dir


def load_accelerator_checkpoint(
    accelerator: "Accelerator",
    checkpoint_dir: str,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> Tuple[int, int, float]:
    """
    加载检查点：TP>1 且存在 train.pt 时走自定义加载，否则使用 accelerator.load_state。
    """
    accelerator.wait_for_everyone()

    tp = get_tensor_model_parallel_world_size() if dist.is_initialized() else 1
    train_pt = os.path.join(checkpoint_dir, "train.pt")

    if tp > 1 and os.path.exists(train_pt) and model is not None:
        payload = torch.load(train_pt, map_location="cpu")
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.load_state_dict(payload.get("model", {}), strict=False)

        if optimizer is not None and payload.get("optimizer") is not None:
            optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None and payload.get("scheduler") is not None:
            scheduler.load_state_dict(payload["scheduler"])

        start_epoch = int(payload.get("epoch", 0))
        global_step = int(payload.get("global_step", 0))
        best_val_loss = float(payload.get("best_val_loss", float("inf")))

        accelerator.wait_for_everyone()
        return start_epoch, global_step, best_val_loss

    # 非 TP 或未找到 train.pt，回退到 Accelerator 内置状态加载
    accelerator.load_state(checkpoint_dir)
    
    trainer_state_path = os.path.join(checkpoint_dir, "trainer_state.pt")
    if os.path.exists(trainer_state_path):
        state = torch.load(trainer_state_path, map_location="cpu")
        start_epoch = int(state.get("epoch", 0))
        global_step = int(state.get("global_step", 0))
        best_val_loss = float(state.get("best_val_loss", float("inf")))
        accelerator.wait_for_everyone()
        return start_epoch, global_step, best_val_loss
    
    accelerator.wait_for_everyone()
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
