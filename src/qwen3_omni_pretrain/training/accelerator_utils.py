# src/qwen3_omni_pretrain/training/accelerator_utils.py
"""
Accelerate 工具模块 - 提供统一的 Accelerator 初始化和配置管理
支持 DDP、DeepSpeed ZeRO、FSDP 三种分布式策略
"""

import os
import json
import hashlib
import pickle
import re
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Mapping, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from qwen3_omni_pretrain.architecture.checkpoint_metadata import (
    CheckpointArtifactKind,
    CheckpointMetadata,
    ModelTopology,
    load_checkpoint_metadata,
    write_checkpoint_metadata,
)
from qwen3_omni_pretrain.architecture.profiles import (
    ArchitectureProfile,
    CompatibilityLevel,
)
from qwen3_omni_pretrain.architecture.summary import ArchitectureSummary
from qwen3_omni_pretrain.training.checkpoint import (
    _publish_accelerator_checkpoint_transaction,
    _raise_if_accelerator_checkpoint_failed,
)

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


_TP_MANIFEST_FILENAME = "tp_shards.json"
_TP_MANIFEST_SCHEMA_VERSION = 1
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _state_mapping_schema_sha256(
    state: Mapping[str, object],
) -> str:
    entries: list[dict[str, object]] = []
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"model state entry {name!r} is not a tensor"
            )
        logical_shape = getattr(value, "ds_shape", value.shape)
        entries.append(
            {
                "name": name,
                "shape": [int(dimension) for dimension in logical_shape],
                "dtype": str(value.dtype),
            }
        )
    payload = json.dumps(
        entries,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _distributed_rank_layout(
    accelerator: "Accelerator",
) -> tuple[int, int, int, int]:
    initialized = dist.is_available() and dist.is_initialized()
    global_rank = (
        dist.get_rank()
        if initialized
        else int(get_tensor_model_parallel_rank())
    )
    world_size = (
        dist.get_world_size()
        if initialized
        else int(getattr(accelerator, "num_processes", 1))
    )
    tp_world_size = int(get_tensor_model_parallel_world_size())
    tp_rank = int(get_tensor_model_parallel_rank())
    if world_size < 1 or tp_world_size < 1:
        raise ValueError("checkpoint world sizes must be positive")
    if not 0 <= global_rank < world_size:
        raise ValueError("global rank is outside checkpoint world size")
    if not 0 <= tp_rank < tp_world_size:
        raise ValueError("TP rank is outside TP degree")
    if world_size % tp_world_size != 0:
        raise ValueError("world size must be divisible by TP degree")
    return global_rank, world_size, tp_rank, tp_world_size


def _write_tp_manifest(
    directory: str,
    *,
    world_size: int,
    tp_world_size: int,
    shards: list[dict[str, object]],
) -> None:
    ordered = sorted(shards, key=lambda item: int(item["global_rank"]))
    if len(ordered) != world_size:
        raise ValueError("TP checkpoint has an incomplete shard set")
    if [entry["global_rank"] for entry in ordered] != list(
        range(world_size)
    ):
        raise ValueError("TP checkpoint has an incomplete shard set")
    for entry in ordered:
        global_rank = int(entry["global_rank"])
        expected_tp_rank = global_rank % tp_world_size
        if int(entry["tp_rank"]) != expected_tp_rank:
            raise ValueError("TP rank mapping is inconsistent")
    payload = {
        "schema_version": _TP_MANIFEST_SCHEMA_VERSION,
        "world_size": world_size,
        "tp_world_size": tp_world_size,
        "shards": ordered,
    }
    destination = os.path.join(directory, _TP_MANIFEST_FILENAME)
    with open(destination, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")


def _load_and_validate_tp_manifest(
    directory: str,
    *,
    world_size: int,
    tp_world_size: int,
) -> list[dict[str, object]]:
    path = os.path.join(directory, _TP_MANIFEST_FILENAME)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid TP shard manifest: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("invalid TP shard manifest")
    expected_keys = {
        "schema_version",
        "world_size",
        "tp_world_size",
        "shards",
    }
    if set(raw) != expected_keys:
        raise ValueError("invalid TP shard manifest keys")
    if raw["schema_version"] != _TP_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported TP shard manifest schema")
    if raw["world_size"] != world_size:
        raise ValueError("checkpoint world size does not match")
    if raw["tp_world_size"] != tp_world_size:
        raise ValueError("checkpoint TP degree does not match")
    shards = raw["shards"]
    if not isinstance(shards, list) or len(shards) != world_size:
        raise ValueError("TP checkpoint has an incomplete shard set")
    expected_entry_keys = {
        "global_rank",
        "tp_rank",
        "filename",
        "sha256",
        "model_schema_sha256",
    }
    by_rank: dict[int, dict[str, object]] = {}
    for entry in shards:
        if not isinstance(entry, dict) or set(entry) != expected_entry_keys:
            raise ValueError("invalid TP shard entry")
        global_rank = entry["global_rank"]
        tp_rank = entry["tp_rank"]
        filename = entry["filename"]
        if type(global_rank) is not int or global_rank in by_rank:
            raise ValueError("TP checkpoint has an incomplete shard set")
        if type(tp_rank) is not int:
            raise ValueError("invalid TP rank")
        if (
            not isinstance(filename, str)
            or filename != f"tp-shard-rank-{global_rank:05d}.pt"
        ):
            raise ValueError("invalid TP shard filename")
        if tp_rank != global_rank % tp_world_size:
            raise ValueError("checkpoint TP rank mapping does not match")
        for digest_key in ("sha256", "model_schema_sha256"):
            digest = entry[digest_key]
            if (
                not isinstance(digest, str)
                or _SHA256_PATTERN.fullmatch(digest) is None
            ):
                raise ValueError(
                    f"invalid TP shard {digest_key}"
                )
        shard_path = os.path.join(directory, filename)
        if not os.path.isfile(shard_path):
            raise ValueError("TP checkpoint has an incomplete shard set")
        if _file_sha256(shard_path) != entry["sha256"]:
            raise ValueError("TP shard content hash does not match")
        by_rank[global_rank] = entry
    if sorted(by_rank) != list(range(world_size)):
        raise ValueError("TP checkpoint has an incomplete shard set")
    return [by_rank[index] for index in range(world_size)]


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
    *,
    metadata: CheckpointMetadata,
) -> str:
    """Save one immutable shard per global rank, then publish a manifest."""
    if optimizer is None:
        raise ValueError("TP training checkpoint requires optimizer")
    rank, world_size, tp_rank, tp_world_size = (
        _distributed_rank_layout(accelerator)
    )
    state_error: BaseException | None = None
    try:
        unwrapped = accelerator.unwrap_model(model)
        model_sd = unwrapped.state_dict()
        optim_sd = (
            optimizer.state_dict() if optimizer is not None else None
        )
        sched_sd = (
            scheduler.state_dict() if scheduler is not None else None
        )
        model_schema_sha256 = _state_mapping_schema_sha256(model_sd)
    except BaseException as exc:
        state_error = exc
    _raise_if_accelerator_checkpoint_failed(
        accelerator,
        state_error,
        "state collection",
    )

    local_record: dict[str, object] = {}
    gathered_records: list[dict[str, object]] = []

    def write_payload(temp_dir: str) -> None:
        os.makedirs(temp_dir, exist_ok=True)
        filename = f"tp-shard-rank-{rank:05d}.pt"
        shard_path = os.path.join(temp_dir, filename)
        payload: dict[str, object] = {
            "model": model_sd,
            "epoch": epoch,
            "global_step": global_step,
            "best_val_loss": best_val_loss,
        }
        if optim_sd is not None:
            payload["optimizer"] = optim_sd
        if sched_sd is not None:
            payload["scheduler"] = sched_sd
        torch.save(payload, shard_path)
        local_record.update(
            {
                "global_rank": rank,
                "tp_rank": tp_rank,
                "filename": filename,
                "sha256": _file_sha256(shard_path),
                "model_schema_sha256": model_schema_sha256,
            }
        )

    def gather_manifest_records(temp_dir: str) -> None:
        del temp_dir
        if dist.is_available() and dist.is_initialized():
            records: list[dict[str, object] | None] = [
                None
            ] * world_size
            dist.all_gather_object(records, dict(local_record))
            gathered_records.extend(
                record for record in records if record is not None
            )
        else:
            gathered_records.append(dict(local_record))

    def finalize_manifest(temp_dir: str) -> None:
        _write_tp_manifest(
            temp_dir,
            world_size=world_size,
            tp_world_size=tp_world_size,
            shards=gathered_records,
        )
        if tokenizer is not None:
            tokenizer.save_pretrained(temp_dir)

    result = _publish_accelerator_checkpoint_transaction(
        accelerator,
        checkpoint_dir,
        metadata,
        write_payload,
        rank_zero_finalizer=finalize_manifest,
        all_rank_finalizer=gather_manifest_records,
        metadata_writer=write_checkpoint_metadata,
    )
    print(f"[rank{rank}] >>> save: done", flush=True)
    return result


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
    *,
    metadata: CheckpointMetadata,
) -> str:
    """通用保存：TP>1 走分片，其余保持原行为。"""

    tp = int(get_tensor_model_parallel_world_size())
    if tp > 1:
        if model is None:
            raise ValueError("TP checkpoint requires model")
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
            metadata=metadata,
        )

    def write_payload(temp_dir: str) -> None:
        accelerator.save_state(temp_dir)
        if accelerator.is_main_process:
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_val_loss": best_val_loss,
                },
                os.path.join(temp_dir, "trainer_state.pt"),
            )
            if tokenizer is not None:
                tokenizer.save_pretrained(temp_dir)

    return _publish_accelerator_checkpoint_transaction(
        accelerator,
        checkpoint_dir,
        metadata,
        write_payload,
        metadata_writer=write_checkpoint_metadata,
    )


def _accelerator_sum_flag(
    accelerator: "Accelerator",
    flag: bool,
) -> int:
    """Return an all-rank count for a local boolean."""
    world_size = int(getattr(accelerator, "num_processes", 1))
    if world_size <= 1:
        return int(flag)
    status = torch.tensor(
        [int(flag)],
        dtype=torch.int64,
        device=getattr(
            accelerator,
            "device",
            torch.device("cpu"),
        ),
    )
    reduce = getattr(accelerator, "reduce", None)
    if callable(reduce):
        return int(reduce(status, reduction="sum").item())
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(status, op=dist.ReduceOp.SUM)
        return int(status.item())
    raise RuntimeError(
        "Accelerator checkpoint candidate coordination is unavailable"
    )


def load_accelerator_checkpoint(
    accelerator: "Accelerator",
    checkpoint_dir: str,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    *,
    expected_profile: ArchitectureProfile | None = None,
    expected_compatibility: CompatibilityLevel | None = None,
    expected_architecture: ArchitectureSummary | None = None,
    expected_artifact_kind: CheckpointArtifactKind | None = None,
    expected_topology: ModelTopology | None = None,
    expected_tokenizer_sha256: str | None = None,
    legacy_config: object | None = None,
) -> Tuple[int, int, float]:
    """Load a complete standard or rank-local TP checkpoint generation."""
    accelerator.wait_for_everyone()
    backup_dir = checkpoint_dir + ".backup"
    candidates = [checkpoint_dir]
    if _accelerator_sum_flag(
        accelerator,
        os.path.isdir(backup_dir),
    ) > 0:
        candidates.append(backup_dir)
    last_error: BaseException | None = None
    world_size = int(getattr(accelerator, "num_processes", 1))

    for directory in candidates:
        try:
            identity_error: BaseException | None = None
            try:
                load_checkpoint_metadata(
                    directory,
                    expected_profile=expected_profile,
                    expected_compatibility=expected_compatibility,
                    expected_architecture=expected_architecture,
                    expected_artifact_kind=expected_artifact_kind,
                    expected_topology=expected_topology,
                    expected_tokenizer_sha256=expected_tokenizer_sha256,
                    legacy_config=legacy_config,
                )
            except BaseException as exc:
                identity_error = exc
            if world_size <= 1:
                if identity_error is not None:
                    raise identity_error
            else:
                _raise_if_accelerator_checkpoint_failed(
                    accelerator,
                    identity_error,
                    "identity validation",
                )

            manifest_path = os.path.join(
                directory,
                _TP_MANIFEST_FILENAME,
            )
            manifest_count = _accelerator_sum_flag(
                accelerator,
                os.path.isfile(manifest_path),
            )
            if manifest_count not in (0, world_size):
                raise RuntimeError(
                    "TP manifest availability differs across ranks"
                )
            if manifest_count == world_size:
                if model is None:
                    raise ValueError(
                        "TP checkpoint restore requires a model"
                    )
                rank, world_size, tp_rank, tp_world_size = (
                    _distributed_rank_layout(accelerator)
                )
                validation_error: BaseException | None = None
                try:
                    shards = _load_and_validate_tp_manifest(
                        directory,
                        world_size=world_size,
                        tp_world_size=tp_world_size,
                    )
                    local_entry = shards[rank]
                    if local_entry["tp_rank"] != tp_rank:
                        raise ValueError(
                            "checkpoint TP rank does not match local rank"
                        )
                    current_schema = _state_mapping_schema_sha256(
                        accelerator.unwrap_model(model).state_dict()
                    )
                    if (
                        local_entry["model_schema_sha256"]
                        != current_schema
                    ):
                        raise ValueError(
                            "checkpoint parameter schema does not match "
                            "the local TP model"
                        )
                except BaseException as exc:
                    validation_error = exc
                try:
                    _raise_if_accelerator_checkpoint_failed(
                        accelerator,
                        validation_error,
                        "manifest validation",
                    )
                except RuntimeError as exc:
                    raise ValueError(str(exc)) from validation_error

                payload_error: BaseException | None = None
                payload: Mapping[str, object] | None = None
                try:
                    loaded_payload = torch.load(
                        os.path.join(
                            directory,
                            str(local_entry["filename"]),
                        ),
                        map_location="cpu",
                    )
                    if not isinstance(loaded_payload, Mapping):
                        raise ValueError(
                            "checkpoint shard payload is not a mapping"
                        )
                    payload = loaded_payload
                    model_state = payload.get("model", {})
                    if (
                        not isinstance(model_state, Mapping)
                        or _state_mapping_schema_sha256(model_state)
                        != local_entry["model_schema_sha256"]
                    ):
                        raise ValueError(
                            "checkpoint parameter schema is invalid"
                        )
                    if optimizer is not None and "optimizer" not in payload:
                        raise ValueError(
                            "TP training checkpoint is missing optimizer "
                            "state"
                        )
                except BaseException as exc:
                    payload_error = exc
                _raise_if_accelerator_checkpoint_failed(
                    accelerator,
                    payload_error,
                    "rank-local payload validation",
                )

                restore_error: BaseException | None = None
                restored: tuple[int, int, float] | None = None
                try:
                    assert payload is not None
                    model_state = payload["model"]
                    accelerator.unwrap_model(model).load_state_dict(
                        model_state,
                        strict=True,
                    )
                    if (
                        optimizer is not None
                        and payload.get("optimizer") is not None
                    ):
                        optimizer.load_state_dict(payload["optimizer"])
                    if (
                        scheduler is not None
                        and payload.get("scheduler") is not None
                    ):
                        scheduler.load_state_dict(payload["scheduler"])
                    restored = (
                        int(payload.get("epoch", 0)),
                        int(payload.get("global_step", 0)),
                        float(
                            payload.get(
                                "best_val_loss",
                                float("inf"),
                            )
                        ),
                    )
                except BaseException as exc:
                    restore_error = exc
                _raise_if_accelerator_checkpoint_failed(
                    accelerator,
                    restore_error,
                    "rank-local restore",
                )
                assert restored is not None
                accelerator.wait_for_everyone()
                return restored

            standard_error: BaseException | None = None
            restored = None
            try:
                accelerator.load_state(directory)
                trainer_state_path = os.path.join(
                    directory,
                    "trainer_state.pt",
                )
                if os.path.exists(trainer_state_path):
                    state = torch.load(
                        trainer_state_path,
                        map_location="cpu",
                    )
                    restored = (
                        int(state.get("epoch", 0)),
                        int(state.get("global_step", 0)),
                        float(
                            state.get(
                                "best_val_loss",
                                float("inf"),
                            )
                        ),
                    )
                else:
                    restored = (0, 0, float("inf"))
            except BaseException as exc:
                standard_error = exc
            _raise_if_accelerator_checkpoint_failed(
                accelerator,
                standard_error,
                "restore",
            )
            assert restored is not None
            accelerator.wait_for_everyone()
            return restored
        except (
            ValueError,
            TypeError,
            RuntimeError,
            FileNotFoundError,
            EOFError,
            pickle.UnpicklingError,
            OSError,
        ) as exc:
            last_error = exc
            if directory == checkpoint_dir and len(candidates) > 1:
                import warnings

                warnings.warn(
                    f"[checkpoint] Failed to load from {directory}: {exc}\n"
                    f"Trying backup: {backup_dir}"
                )
                continue
            raise
    raise RuntimeError(
        f"Failed to load checkpoint {checkpoint_dir}: {last_error}"
    ) from last_error


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
