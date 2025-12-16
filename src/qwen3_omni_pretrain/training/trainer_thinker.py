# src/qwen3_omni_pretrain/training/trainer_thinker.py

import os
import time
import datetime
import signal
import json
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, Union, List, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard.writer import SummaryWriter
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

try:
    import deepspeed
    deepspeed.init_distributed(
        dist_backend="nccl", 
        timeout=datetime.timedelta(hours=2)  # <--- 关键！强制传入 2小时
    )
except ImportError:
    deepspeed = None

# Accelerate 支持
try:
    from accelerate import Accelerator, DistributedType
    from accelerate.utils import is_deepspeed_available
    from qwen3_omni_pretrain.training.accelerator_utils import (
        ACCELERATE_AVAILABLE,
        AcceleratorConfig,
        load_accelerator_config,
        create_accelerator,
        prepare_model_optimizer_dataloader,
        save_accelerator_checkpoint,
        load_accelerator_checkpoint,
        print_on_main,
        unwrap_model,
    )
except ImportError:
    ACCELERATE_AVAILABLE = False
    Accelerator = None
    DistributedType = None
    is_deepspeed_available = None

from qwen3_omni_pretrain.training.distributed import (
    distributed_context,
    ddp_available,
    get_local_rank,
    is_main_process,
    ddp_wrap_model,
)

from qwen3_omni_pretrain.data.datasets.omni_webdataset import OmniJsonlDataset
from qwen3_omni_pretrain.data.collators import OmniStage2Collator
from qwen3_omni_pretrain.utils.config_utils import load_yaml
from qwen3_omni_pretrain.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeConfig,
)
from qwen3_omni_pretrain.models.qwen3_omni_moe.modeling_thinker_text import (
    Qwen3OmniMoeThinkerTextModel,
)
from qwen3_omni_pretrain.models.qwen3_omni_moe.modeling_thinker_vision_audio import (
    Qwen3OmniMoeThinkerVisionAudioModel,
)
from qwen3_omni_pretrain.data.datasets.text_dataset import TextJsonlDataset, PackedTokenDataset
from qwen3_omni_pretrain.data.collators import TextCausalLMCollator, PackedCausalLMCollator
from qwen3_omni_pretrain.training.loop import train_one_epoch, evaluate, _move_batch_to_device, _is_global_bad_loss
from qwen3_omni_pretrain.training.checkpoint import save_checkpoint, load_checkpoint
# from qwen3_omni_pretrain.utils.seed import set_seed


@dataclass
class TrainerThinkerConfig:
    train_corpus_paths: Union[str, List[str]]
    val_corpus_paths: Union[str, List[str]]
    
    experiment_name: str
    model_config_path: str
    max_seq_length: int
    batch_size: int
    num_workers: int
    shuffle: bool
    num_epochs: int
    max_steps: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    logging_steps: int
    eval_steps: int
    save_steps: int
    output_dir: str
    fp16: bool
    bf16: bool
    seed: int = 42
    ddp: bool = False
    ddp_find_unused_parameters: bool = True
    fp8: bool = False
    int8_optimizer: bool = False
    resume_from_checkpoint: Optional[str] = None

    gradient_checkpointing: bool = False
    deepspeed: Optional[str] = None
    
    # Accelerator 配置
    use_accelerator: bool = False
    accelerator_config_path: Optional[str] = None
    
    use_packed_dataset: bool = False
    packed_train_bin_path: Optional[str] = None
    packed_val_bin_path: Optional[str] = None
    packed_seq_length: Optional[int] = None
    
@dataclass
class Stage2TrainConfig:
    stage1_init_ckpt: str
    model_config_path: str
    train_corpus_path: str
    val_corpus_path: str
    image_root: str
    audio_root: str

    output_dir: str

    resume_from_checkpoint: Optional[str] = None

    num_epochs: int = 1
    batch_size: int = 2
    max_seq_length: int = 1024
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    gradient_accumulation_steps: int = 1
    logging_steps: int = 10
    eval_steps: int = 200
    num_workers: int = 4
    seed: int = 42


def _build_text_dataset(paths, tokenizer, max_seq_length):
    """
    支持：
    - paths 是 str：单一 jsonl
    - paths 是 [str, ...]：多个 jsonl，用 ConcatDataset 串联
    """
    if isinstance(paths, (list, tuple)):
        datasets = [
            TextJsonlDataset(p, tokenizer, max_seq_length)
            for p in paths
        ]
        if len(datasets) == 1:
            return datasets[0]
        return ConcatDataset(datasets)
    else:
        return TextJsonlDataset(paths, tokenizer, max_seq_length)


def build_trainer_config(yaml_path: str) -> TrainerThinkerConfig:
    cfg = load_yaml(yaml_path)

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    
    train_paths = data_cfg.get("train_corpus_paths", data_cfg.get("train_corpus_path"))
    val_paths = data_cfg.get("val_corpus_paths", data_cfg.get("val_corpus_path"))
    
    use_packed_dataset = bool(data_cfg.get("use_packed_dataset", False))
    packed_train_bin_path = data_cfg.get("packed_train_bin_path", None)
    packed_val_bin_path = data_cfg.get("packed_val_bin_path", None)
    packed_seq_length = int(
        data_cfg.get(
            "packed_seq_length",
            data_cfg.get("max_seq_length", 2048),
        )
    )


    return TrainerThinkerConfig(
        experiment_name=str(cfg["experiment_name"]),
        model_config_path=str(model_cfg["config_path"]),
        train_corpus_paths=train_paths,
        val_corpus_paths=val_paths,
        max_seq_length=int(data_cfg.get("max_seq_length", 2048)),
        batch_size=int(data_cfg.get("batch_size", 4)),
        num_workers=int(data_cfg.get("num_workers", 4)),
        shuffle=bool(data_cfg.get("shuffle", True)),
        num_epochs=int(train_cfg["num_epochs"]),
        max_steps=int(train_cfg.get("max_steps", -1)),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 1)),
        learning_rate=float(train_cfg["learning_rate"]),            # 这里强制转 float
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.03)),
        logging_steps=int(train_cfg.get("logging_steps", 20)),
        eval_steps=int(train_cfg.get("eval_steps", 1000)),
        save_steps=int(train_cfg.get("save_steps", 1000)),
        output_dir=str(train_cfg["output_dir"]) + time.strftime("-%Y%m%d-%H%M%S"),
        fp16=bool(train_cfg.get("fp16", False)),
        bf16=bool(train_cfg.get("bf16", False)),
        fp8=bool(train_cfg.get("fp8", False)),
        int8_optimizer=bool(train_cfg.get("int8_optimizer", False)),
        seed=int(train_cfg.get("seed", 42)),
        ddp=bool(train_cfg.get("ddp", False)),
        ddp_find_unused_parameters=bool(train_cfg.get("ddp_find_unused_parameters", True)),
        resume_from_checkpoint=train_cfg.get("resume_from_checkpoint"),
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", False)),
        deepspeed=train_cfg.get("deepspeed"),
        # Accelerator 配置
        use_accelerator=bool(train_cfg.get("use_accelerator", False)),
        accelerator_config_path=train_cfg.get("accelerator_config_path"),
        use_packed_dataset=use_packed_dataset,
        packed_train_bin_path=packed_train_bin_path,
        packed_val_bin_path=packed_val_bin_path,
        packed_seq_length=packed_seq_length,
    )


def set_seed(seed: int):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =============================================================================
# Accelerator 训练模式实现
# =============================================================================
def _train_with_accelerator(
    cfg: TrainerThinkerConfig,
    tokenizer_name_or_path: str,
    enable_tensorboard: bool,
    log_dir: str,
    resume_from_checkpoint: Optional[str],
    use_tensor_parallel: bool = False,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
):
    """
    使用 HuggingFace Accelerate 进行训练
    支持 DDP、DeepSpeed ZeRO、FSDP 三种分布式策略的统一接口
    可选支持 Tensor Parallelism (TP + ZeRO)
    """
    from qwen3_omni_pretrain.training.accelerator_utils import (
        load_accelerator_config,
        create_accelerator,
        save_accelerator_checkpoint,
        load_accelerator_checkpoint,
    )
    from qwen3_omni_pretrain.training.loop_accelerator import (
        train_one_epoch_accelerator,
        evaluate_accelerator,
    )
    
    # 1. 加载 Accelerator 配置
    if cfg.accelerator_config_path:
        accel_cfg = load_accelerator_config(cfg.accelerator_config_path)
        # 统一以训练配置为准，避免多处累积步不一致
        accel_cfg.gradient_accumulation_steps = cfg.gradient_accumulation_steps
    else:
        # 使用默认配置
        accel_cfg = AcceleratorConfig(
            use_accelerator=True,
            mixed_precision="bf16" if cfg.bf16 else ("fp16" if cfg.fp16 else "no"),
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            gradient_checkpointing=cfg.gradient_checkpointing,
            log_with="tensorboard" if enable_tensorboard else None,
            project_dir=log_dir,
        )
    
    # 2. 创建 Accelerator
    accelerator = create_accelerator(accel_cfg, project_name=cfg.experiment_name)
    if accelerator.device.type != "cuda":
        raise RuntimeError(
            f"Accelerator picked device={accelerator.device}; expected CUDA. Check CUDA visibility / accelerate config."
        )
    torch.cuda.set_device(accelerator.local_process_index % torch.cuda.device_count())
    
    # 主进程打印配置
    if accelerator.is_main_process:
        print(f">> Using Accelerator Mode")
        print(f"   - distributed_type: {accelerator.distributed_type}")
        print(f"   - mixed_precision: {accelerator.mixed_precision}")
        print(f"   - num_processes: {accelerator.num_processes}")
        print(f"   - gradient_accumulation_steps: {accelerator.gradient_accumulation_steps}")
    
    # 3. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 4. 模型配置 & 初始化
    model_cfg_dict = load_yaml(cfg.model_config_path)
    model_cfg = Qwen3OmniMoeConfig(**model_cfg_dict)
    
    # =========================================================================
    # Tensor Parallelism 初始化 (在模型创建之前)
    # =========================================================================
    if use_tensor_parallel and tensor_parallel_size > 1:
        from qwen3_omni_pretrain.training.distributed import init_tensor_parallel
        init_tensor_parallel(
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
        )
        # 更新模型配置以启用 TP
        model_cfg.thinker_config.use_tensor_parallel = True
        model_cfg.thinker_config.tensor_parallel_size = tensor_parallel_size
        if accelerator.is_main_process:
            print(f">> Tensor Parallelism enabled: TP={tensor_parallel_size}, PP={pipeline_parallel_size}")
    
    # 对于 DeepSpeed ZeRO-3，使用 zero.Init 上下文
    init_ctx = nullcontext()
    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        # DeepSpeed ZeRO-3 初始化
        if accelerator.state.deepspeed_plugin is not None:
            from accelerate.utils import is_deepspeed_available
            if is_deepspeed_available():
                import deepspeed
                ds_config = accelerator.state.deepspeed_plugin.deepspeed_config
                if isinstance(ds_config, dict):
                    def _to_int(value, default):
                        try:
                            return int(value)
                        except (TypeError, ValueError):
                            return default

                    micro_bs = ds_config.get("train_micro_batch_size_per_gpu") or ds_config.get("train_batch_size_per_gpu")
                    # Force gradient accumulation to align with Accelerator config to avoid DS/Accelerate mismatch
                    grad_acc = accelerator.gradient_accumulation_steps

                    micro_bs = _to_int(micro_bs, cfg.batch_size)
                    grad_acc = _to_int(grad_acc, accelerator.gradient_accumulation_steps)

                    ds_config["train_micro_batch_size_per_gpu"] = micro_bs
                    ds_config["gradient_accumulation_steps"] = grad_acc
                    ds_config["train_batch_size"] = micro_bs * accelerator.num_processes * grad_acc

                init_ctx = deepspeed.zero.Init(config_dict_or_path=ds_config)
                if accelerator.is_main_process:
                    print(">> Initializing model with DeepSpeed ZeRO-3 Context")
    
    # 选择模型类型：TP 模型或标准模型
    with init_ctx:
        if use_tensor_parallel and tensor_parallel_size > 1:
            from qwen3_omni_pretrain.models.qwen3_omni_moe.modeling_thinker_text_tp import (
                Qwen3OmniMoeThinkerTextModelTP,
            )
            model = Qwen3OmniMoeThinkerTextModelTP(model_cfg)
        else:
            model = Qwen3OmniMoeThinkerTextModel(model_cfg)

    # 在非 DeepSpeed 模式下，尽早将参数迁移到 GPU 并降精度，降低 CPU 常驻占用
    if accelerator.distributed_type != DistributedType.DEEPSPEED:
        target_dtype = None
        if accelerator.mixed_precision == "bf16":
            target_dtype = torch.bfloat16
        elif accelerator.mixed_precision == "fp16":
            target_dtype = torch.float16

        model.to(device=accelerator.device, dtype=target_dtype)

    if accelerator.is_main_process:
        first_param_device = next(model.parameters()).device
        print(f">> Model initial device: {first_param_device}")
    
    # 激活检查点（TP 场景下默认关闭以避免 checkpoint 错配）
    enable_ckpt = (cfg.gradient_checkpointing or accel_cfg.gradient_checkpointing) and not (
        use_tensor_parallel and tensor_parallel_size > 1
    )
    if enable_ckpt:
        if hasattr(model, "thinker_cfg"):
            model.thinker_cfg.gradient_checkpointing = True
        try:
            model.gradient_checkpointing_enable()
        except ValueError:
            if hasattr(model, "gradient_checkpointing"):
                model.gradient_checkpointing = True
            if accelerator.is_main_process:
                print(
                    ">> Gradient checkpointing flag enabled (manual) for model without built-in support"
                )
    else:
        if hasattr(model, "gradient_checkpointing"):
            model.gradient_checkpointing = False
        if hasattr(model, "thinker_cfg"):
            model.thinker_cfg.gradient_checkpointing = False
        if accelerator.is_main_process:
            print(">> Gradient checkpointing disabled for TP run to avoid checkpoint mismatch")
    
    # 5. 数据集
    if cfg.use_packed_dataset:
        assert cfg.packed_train_bin_path is not None
        assert cfg.packed_val_bin_path is not None
        train_dataset = PackedTokenDataset(
            bin_path=cfg.packed_train_bin_path,
            seq_length=cfg.packed_seq_length or cfg.max_seq_length,
        )
        val_dataset = PackedTokenDataset(
            bin_path=cfg.packed_val_bin_path,
            seq_length=cfg.packed_seq_length or cfg.max_seq_length,
        )
        collator = PackedCausalLMCollator(pad_token_id=tokenizer.pad_token_id)
    else:
        train_dataset = _build_text_dataset(cfg.train_corpus_paths, tokenizer, cfg.max_seq_length)
        val_dataset = _build_text_dataset(cfg.val_corpus_paths, tokenizer, cfg.max_seq_length)
        collator = TextCausalLMCollator(tokenizer, cfg.max_seq_length)

    # 6. DataLoader - 为 TP 分组的 DP 切分数据
    train_sampler = None
    val_sampler = None
    if use_tensor_parallel and tensor_parallel_size > 1 and torch.distributed.is_initialized():
        dp_world_size = torch.distributed.get_world_size() // tensor_parallel_size
        dp_rank = torch.distributed.get_rank() // tensor_parallel_size
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=dp_world_size,
            rank=dp_rank,
            shuffle=cfg.shuffle,
            drop_last=False,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=dp_world_size,
            rank=dp_rank,
            shuffle=False,
            drop_last=False,
        )

    loader_kwargs = {
        "num_workers": cfg.num_workers,
        "collate_fn": collator,
        "pin_memory": True,
    }
    if cfg.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None and cfg.shuffle),
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        sampler=val_sampler,
        shuffle=False,
        **loader_kwargs,
    )
    
    # 7. 计算总步数
    num_update_steps_per_epoch = len(train_loader) // accelerator.gradient_accumulation_steps
    if cfg.max_steps > 0:
        num_training_steps = cfg.max_steps
        cfg.num_epochs = (cfg.max_steps // num_update_steps_per_epoch) + 1
    else:
        num_training_steps = num_update_steps_per_epoch * cfg.num_epochs
    num_warmup_steps = int(num_training_steps * cfg.warmup_ratio)
    
    # 8. 优化器
    no_decay = ["bias", "norm", "layernorm", "ln"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n.lower() for nd in no_decay)],
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n.lower() for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=cfg.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    
    # 9. Scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    # 10. Accelerator prepare (封装 model, optimizer, scheduler, dataloaders)
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )
    
    # 11. 恢复检查点
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    resume_path = resume_from_checkpoint or cfg.resume_from_checkpoint
    
    if resume_path:
        start_epoch, global_step, best_val_loss = load_accelerator_checkpoint(accelerator, resume_path)
        if accelerator.is_main_process:
            print(f"Resumed from {resume_path}: epoch={start_epoch}, step={global_step}, best_val_loss={best_val_loss}")
    
    # 12. TensorBoard (通过 Accelerator 的 tracker)
    if enable_tensorboard and accelerator.is_main_process:
        run_name = f"{cfg.experiment_name}_stage1" + time.strftime("-%Y%m%d-%H%M%S")
        def _sanitize_for_tracker(obj: Any) -> Dict[str, Any]:
            safe_dict: Dict[str, Any] = {}
            for k, v in vars(obj).items():
                if isinstance(v, (int, float, bool, str)):
                    safe_dict[k] = v
                elif v is None:
                    continue
                else:
                    safe_dict[k] = str(v)
            return safe_dict

        tracker_cfg = _sanitize_for_tracker(cfg)
        accelerator.init_trackers(run_name, config=tracker_cfg)
    
    # 13. 训练状态
    current_epoch = start_epoch
    
    def save_full_checkpoint(tag: str):
        """保存完整检查点（所有 rank 都调用，内部仅 rank0 写 manifest/tokenizer）。"""
        save_path = os.path.join(cfg.output_dir, tag)
        save_accelerator_checkpoint(
            accelerator=accelerator,
            checkpoint_dir=save_path,
            epoch=current_epoch,
            global_step=global_step,
            best_val_loss=best_val_loss,
            tokenizer=tokenizer,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        if accelerator.is_main_process:
            print(f"Saved checkpoint to {save_path}")
    
    # --- graceful stop handling ---
    stop_requested = False
    received_signal = None

    def _request_stop(sig, frame):
        nonlocal stop_requested, received_signal
        if stop_requested:
            return
        stop_requested = True
        received_signal = sig
        if accelerator.is_main_process:
            print(f"Received signal {sig}, will save and exit after current step.")

    old_sigint = signal.getsignal(signal.SIGINT)
    old_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    def _maybe_exit_gracefully():
        nonlocal stop_requested, received_signal
        if not stop_requested:
            return
        accelerator.wait_for_everyone()
        tag = f"interrupted_step_{global_step}"
        if accelerator.is_main_process:
            print(f"Saving emergency checkpoint: {tag}")
        save_full_checkpoint(tag)
        accelerator.wait_for_everyone()
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)
        raise SystemExit(0)
    
    # 14. 训练循环
    train_start_time = time.time()
    
    try:
        for epoch in range(start_epoch, cfg.num_epochs):
            current_epoch = epoch
            
            if accelerator.is_main_process:
                print(f"\nEpoch {epoch} / {cfg.num_epochs - 1}")
            
            # 训练一个 epoch
            train_loss, steps_this_epoch = train_one_epoch_accelerator(
                accelerator=accelerator,
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                cfg=cfg,
                global_step_offset=global_step,
                log_fn=lambda step, loss, ce, aux, lr: _log_step(
                    accelerator, step, loss, ce, aux, lr, cfg.logging_steps, train_start_time, num_training_steps
                ),
                eval_fn=lambda step, m: _eval_and_save(
                    accelerator, m, val_loader, step, cfg, save_full_checkpoint, best_val_loss
                ),
                should_stop_fn=lambda: stop_requested,
            )
            
            global_step += steps_this_epoch
            
            if accelerator.is_main_process:
                print(f"Epoch {epoch} train_loss={train_loss:.4f}")
                accelerator.log({"epoch/train_loss": train_loss}, step=global_step)
            
            # 评估
            val_loss = evaluate_accelerator(accelerator, model, val_loader)
            
            _maybe_exit_gracefully()
            
            if accelerator.is_main_process:
                print(f"Epoch {epoch} val_loss={val_loss:.4f}")
                accelerator.log({"epoch/val_loss": val_loss, "epoch/epoch": epoch}, step=global_step)
            
            # 所有 rank 同步是否保存最佳
            save_best_flag = torch.tensor(1 if val_loss < best_val_loss else 0, device=accelerator.device)
            save_best_flag = accelerator.reduce(save_best_flag, reduction="max")
            if save_best_flag.item() > 0:
                best_val_loss = val_loss if val_loss < best_val_loss else best_val_loss
                save_full_checkpoint(f"best_epoch{epoch}")
        
        # 保存 latest（所有 rank 都进入，内部仅 rank0 写磁盘）
        save_full_checkpoint("latest")
    
    except KeyboardInterrupt:
        if accelerator.is_main_process:
            print("KeyboardInterrupt, saving emergency checkpoint...")
            save_full_checkpoint(f"interrupted_step_{global_step}")
        raise
    finally:
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)
        accelerator.end_training()


def _log_step(accelerator, step, loss, ce_loss, aux_loss, lr, logging_steps, start_time, total_steps):
    """训练步骤日志"""
    if not accelerator.is_main_process:
        return
    if step % logging_steps == 0:
        msg = f"[step {step}] loss={loss:.4f}"
        if ce_loss is not None:
            msg += f" ce={ce_loss:.4f}"
        if aux_loss is not None:
            msg += f" aux={aux_loss:.4f}"
        if lr is not None:
            msg += f" lr={lr:.6f}"
        
        elapsed = time.time() - start_time
        if total_steps > 0 and step > 0:
            eta = (elapsed / step) * (total_steps - step)
            hrs, rem = divmod(eta, 3600)
            mins, secs = divmod(rem, 60)
            msg += f" eta={int(hrs):02d}:{int(mins):02d}:{int(secs):02d}"
        
        print(msg)
    
    accelerator.log({
        "train/loss": loss,
        "train/ce_loss": ce_loss,
        "train/aux_loss": aux_loss,
        "train/lr": lr,
    }, step=step)


def _eval_and_save(accelerator, model, val_loader, step, cfg, save_fn, best_val_loss):
    """评估并保存最佳模型"""
    from qwen3_omni_pretrain.training.loop_accelerator import evaluate_accelerator
    
    if cfg.eval_steps > 0 and step % cfg.eval_steps == 0:
        val_loss = evaluate_accelerator(accelerator, model, val_loader)
        
        if accelerator.is_main_process:
            print(f"[step {step}] val_loss={val_loss:.4f}")
            accelerator.log({"step/val_loss": val_loss}, step=step)

        save_flag = torch.tensor(1 if val_loss < best_val_loss else 0, device=accelerator.device)
        save_flag = accelerator.reduce(save_flag, reduction="max")
        if save_flag.item() > 0:
            best_val_loss = val_loss if val_loss < best_val_loss else best_val_loss
            save_fn(f"best_step_{step}")

    return best_val_loss


def train_thinker_stage1(
    config_path: str,
    tokenizer_name_or_path: str = "Qwen/Qwen2.5-7B",
    enable_tensorboard: bool = False,
    log_dir: str = "./runs",
    resume_from_checkpoint: Optional[str] = None,
    deepspeed_config: Optional[str] = None,
    accelerator_config_path: Optional[str] = None,
    use_tensor_parallel: bool = False,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
):
    # 1. 加载配置
    cfg = build_trainer_config(config_path)
    if deepspeed_config:
        cfg.deepspeed = deepspeed_config
    if accelerator_config_path:
        cfg.accelerator_config_path = accelerator_config_path
        cfg.use_accelerator = True
    os.makedirs(cfg.output_dir, exist_ok=True)

    set_seed(cfg.seed)

    # =========================================================================
    # 分布式策略优先级：Accelerator > DeepSpeed > DDP
    # =========================================================================
    use_accelerator = cfg.use_accelerator and ACCELERATE_AVAILABLE
    use_deepspeed = cfg.deepspeed is not None and not use_accelerator
    use_ddp = (cfg.ddp and ddp_available()) and (not use_deepspeed) and (not use_accelerator)

    # =========================================================================
    # Accelerator 模式
    # =========================================================================
    if use_accelerator:
        _train_with_accelerator(
            cfg, tokenizer_name_or_path, enable_tensorboard, log_dir, resume_from_checkpoint,
            use_tensor_parallel=use_tensor_parallel,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
        )
        return

    # DeepSpeed 有自己的分布式初始化；DDP 仅在未使用 DeepSpeed 时启用

    # -------------------------------------------------------------------------
    # [FIX] 预加载 DeepSpeed Config，用于后续的 zero.Init
    # -------------------------------------------------------------------------
    ds_config_dict = None
    if use_deepspeed:
        with open(cfg.deepspeed, "r") as f:
            ds_config_dict = json.load(f)

    # 把后面的所有逻辑包进 distributed_context，这样会自动 init/destroy process_group
    with distributed_context():
        # 2. 设备 & rank
        if use_deepspeed:
            local_rank = get_local_rank()
            device = torch.device(f"cuda:{local_rank}")
        elif use_ddp:
            local_rank = get_local_rank()
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 3. tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            trust_remote_code=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Precision setup
        precision_flags = [cfg.fp16, cfg.bf16, getattr(cfg, "fp8", False)]
        if sum(bool(x) for x in precision_flags) > 1:
            raise ValueError("Only one of fp16/bf16/fp8 can be enabled at the same time.")

        autocast_dtype = None
        grad_scaler = None
        if getattr(cfg, "fp8", False):
            if not hasattr(torch, "float8_e4m3fn"):
                raise RuntimeError("fp8 requested but torch.float8_e4m3fn is unavailable in this build.")
            if device.type != "cuda":
                raise RuntimeError("fp8 training is only supported on CUDA devices.")
            autocast_dtype = torch.float8_e4m3fn
        elif cfg.fp16:
            autocast_dtype = torch.float16 if device.type == "cuda" else None
            grad_scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
        elif cfg.bf16:
            autocast_dtype = torch.bfloat16 if device.type == "cuda" else None

        # 4. 模型配置 & 模型
        model_cfg_dict = load_yaml(cfg.model_config_path)
        model_cfg = Qwen3OmniMoeConfig(**model_cfg_dict)

        # -------------------------------------------------------------------------
        # [FIX] 使用 ZeRO-3 Init 初始化模型 (防止 CPU OOM)
        # -------------------------------------------------------------------------
        if use_deepspeed:
            if is_main_process():
                print(">> Initializing model with DeepSpeed ZeRO-3 Context (Partitioning params)...")
            # 传入 config 字典，Init 会自动按照 fp16/bf16 和 ZeRO 策略切分参数
            init_ctx = deepspeed.zero.Init(config_dict_or_path=ds_config_dict)
        else:
            init_ctx = nullcontext()

        with init_ctx:
            model = Qwen3OmniMoeThinkerTextModel(model_cfg)
        
        # 即使使用了 zero.Init，to(device) 也是安全的（通常只移动 buffer）
        if not use_deepspeed: 
             model.to(device) # DeepSpeed下通常不需要完整to(device)，但保留也无妨

        model_engine = None
        # DDP 封装
        if use_ddp:
            model = ddp_wrap_model(
                model,
                find_unused_parameters=cfg.ddp_find_unused_parameters,
            )

        # 激活检查点
        if cfg.gradient_checkpointing:
            # 注意：使用了 DeepSpeed 后，直接修改 internal model config 可能需要小心
            # 但 Qwen3OmniMoeThinkerTextModel 应该把 config 暴露出来了
            if hasattr(model, "thinker_cfg"):
                model.thinker_cfg.gradient_checkpointing = True
            elif hasattr(model, "module") and hasattr(model.module, "thinker_cfg"):
                model.module.thinker_cfg.gradient_checkpointing = True

        # 5. 数据集 (保持原逻辑)
        if cfg.use_packed_dataset:
            assert cfg.packed_train_bin_path is not None, "use_packed_dataset=True 但 packed_train_bin_path 为空"
            assert cfg.packed_val_bin_path is not None, "use_packed_dataset=True 但 packed_val_bin_path 为空"

            train_dataset = PackedTokenDataset(
                bin_path=cfg.packed_train_bin_path,
                seq_length=cfg.packed_seq_length or cfg.max_seq_length,
            )
            val_dataset = PackedTokenDataset(
                bin_path=cfg.packed_val_bin_path,
                seq_length=cfg.packed_seq_length or cfg.max_seq_length,
            )
            collator = PackedCausalLMCollator(
                pad_token_id=tokenizer.pad_token_id,
            )
        else:
            train_dataset = _build_text_dataset(cfg.train_corpus_paths, tokenizer, cfg.max_seq_length)
            val_dataset = _build_text_dataset(cfg.val_corpus_paths, tokenizer, cfg.max_seq_length)
            collator = TextCausalLMCollator(tokenizer, cfg.max_seq_length)

        # DistributedSampler
        if use_ddp or (dist.is_available() and dist.is_initialized()):
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=False,
            )
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=False,
            )
        else:
            train_sampler = None
            val_sampler = None

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None and cfg.shuffle),
            num_workers=cfg.num_workers,
            collate_fn=collator,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=collator,
            pin_memory=True,
        )

        # 计算总步数 (用于 Scheduler)
        if cfg.max_steps > 0:
            num_training_steps = cfg.max_steps
        else:
            num_training_steps = len(train_loader) * cfg.num_epochs // max(cfg.gradient_accumulation_steps, 1)
        num_warmup_steps = int(num_training_steps * cfg.warmup_ratio)

        # 6. 优化器 & scheduler 配置
        optimizer = None
        scheduler = None

        no_decay = ["bias", "norm", "layernorm", "ln"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if not any(nd in n.lower() for nd in no_decay)
                ],
                "weight_decay": cfg.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if any(nd in n.lower() for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        if use_deepspeed:
            if deepspeed is None:
                raise ImportError("DeepSpeed is not installed")
            
            zero_opt = ds_config_dict.get("zero_optimization", {})
            offload_opt = zero_opt.get("offload_optimizer", {})
            
            # 检查是否配置了 device 为 cpu 或 nvme
            offload_device = offload_opt.get("device")
            is_offload_enabled = offload_device in ["cpu", "nvme"]

            if is_offload_enabled:
                # 场景 A: 70B模型 或 30B+CPU内存 (开启了 Offload) -> 必须用 CPUAdam
                from deepspeed.ops.adam import DeepSpeedCPUAdam
                OptimizerClass = DeepSpeedCPUAdam
                if is_main_process():
                    print(f">> [Auto-Select] Detected Offload (device={offload_device}). Using 'DeepSpeedCPUAdam'.")
            else:
                # 场景 B: 30B模型+纯显存 (关闭了 Offload) -> 推荐用 FusedAdam (更快)
                from deepspeed.ops.adam import FusedAdam
                OptimizerClass = FusedAdam
                if is_main_process():
                    print(">> [Auto-Select] Detected Pure GPU training. Using 'FusedAdam'.")

            # [Step 3] 实例化选中的优化器类
            optimizer = OptimizerClass(
                optimizer_grouped_parameters,
                lr=cfg.learning_rate,
                betas=(0.9, 0.95),
                eps=1e-8,
                bias_correction=True,
                # 注意: FusedAdam 参数叫 adam_w_mode, CPUAdam 叫 adamw_mode
                # 这里做一个小的参数适配
                **({"adam_w_mode": True} if OptimizerClass.__name__ == "FusedAdam" else {"adamw_mode": True}),
                amsgrad=False
            )
            
            # 3. 将两者传给 deepspeed.initialize
            # 这样 DeepSpeed 引擎内部就会接管这个 scheduler 的 step
            model_engine, optimizer, _, scheduler = deepspeed.initialize(
                model=model,
                model_parameters=model.parameters(),
                optimizer=optimizer,      # <--- 传入手动创建的优化器
                lr_scheduler=scheduler,   # <--- 传入手动创建的调度器
                config=ds_config_dict,
            )
            
            model = model_engine
            device = torch.device(f"cuda:{model_engine.local_rank}")

        else:
            # DDP 模式 (保持不变)
            if cfg.int8_optimizer:
                try:
                    import bitsandbytes as bnb
                    optimizer = bnb.optim.AdamW8bit(
                        optimizer_grouped_parameters,
                        lr=cfg.learning_rate,
                        betas=(0.9, 0.95),
                        eps=1e-8,
                    )
                except ImportError:
                    raise ImportError("int8_optimizer=True requires bitsandbytes.")
            else:
                optimizer = torch.optim.AdamW(
                    optimizer_grouped_parameters,
                    lr=cfg.learning_rate,
                    betas=(0.9, 0.95),
                    eps=1e-8,
                )

            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )

        # -------------------------------------------------------------------------
        # Checkpoint Resume (修复 DeepSpeed Resume 逻辑)
        # -------------------------------------------------------------------------
        start_epoch = 0
        global_step = 0
        best_val_loss = float("inf")
        resume_path = resume_from_checkpoint or cfg.resume_from_checkpoint
        
        if resume_path:
            if use_deepspeed:
                # DeepSpeed 的 load_checkpoint 需要传入目录路径
                # 它会自动加载 model, optimizer, scheduler (如果 scheduler 是传给 initialize 的)
                # 因为我们 scheduler 是后挂的，所以只能加载 model 和 optimizer 状态
                load_path, client_state = model_engine.load_checkpoint(resume_path)
                if load_path is None:
                    if is_main_process():
                        print(f"[Warn] DeepSpeed failed to load checkpoint from {resume_path}")
                else:
                    if is_main_process():
                        print(f"DeepSpeed Resumed from {load_path}")
                    # 尝试恢复 step 信息
                    if client_state:
                         global_step = client_state.get('step', global_step)
                         start_epoch = client_state.get('epoch', start_epoch)
                         best_val_loss = client_state.get('best_val_loss', best_val_loss)
                    
                    # 恢复 scheduler 状态
                    # 注意：如果 DeepSpeed 没有管理 scheduler，需要手动 load
                    scheduler_state = os.path.join(resume_path, "scheduler.pt")
                    if os.path.exists(scheduler_state):
                        scheduler.load_state_dict(torch.load(scheduler_state, map_location="cpu"))

            else:
                start_epoch, global_step, best_val_loss = load_checkpoint(
                    resume_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=grad_scaler,
                    map_location=device,
                )
                if is_main_process():
                    print(
                        f"Resumed from {resume_path}: start_epoch={start_epoch}, "
                        f"global_step={global_step}, best_val_loss={best_val_loss}"
                    )

        # 7. 训练循环
        writer = None
        if enable_tensorboard and is_main_process():
            run_name = f"{cfg.experiment_name}_stage1" + time.strftime("-%Y%m%d-%H%M%S")
            writer = SummaryWriter(log_dir=os.path.join(log_dir, run_name))

        # 记录当前 epoch（用于 checkpoint 元数据）
        current_epoch = start_epoch

        def save_full_checkpoint(tag: str):
            """Persist full trainer state with atomic write protection."""
            if not is_main_process():
                # DeepSpeed 只有 rank 0 负责协调，但所有进程都要参与 save_checkpoint 调用
                if use_deepspeed:
                     pass # DeepSpeed 要求所有 rank 调用，见下文
                else:
                     return

            save_path = os.path.join(cfg.output_dir, tag)
            temp_path = save_path + ".tmp"
            backup_path = save_path + ".backup"
            
            client_state = {
                'step': global_step,
                'epoch': current_epoch,
                'best_val_loss': best_val_loss
            }

            if use_deepspeed and model_engine is not None:
                # DeepSpeed 模式：使用临时目录进行原子写入
                # 清理可能存在的临时目录
                if is_main_process() and os.path.exists(temp_path):
                    import shutil
                    shutil.rmtree(temp_path)
                
                # 同步所有进程
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()
                
                # DeepSpeed save_checkpoint 需要所有进程调用
                model_engine.save_checkpoint(temp_path, client_state=client_state)
                
                # Scheduler 如果没托管给 DS，需要单独存
                if is_main_process():
                    tokenizer.save_pretrained(temp_path)
                    torch.save(scheduler.state_dict(), os.path.join(temp_path, "scheduler.pt"))
                    
                    # 验证 checkpoint 完整性
                    scheduler_path = os.path.join(temp_path, "scheduler.pt")
                    try:
                        torch.load(scheduler_path, map_location="cpu")
                    except Exception as e:
                        import shutil
                        shutil.rmtree(temp_path)
                        raise RuntimeError(f"Checkpoint verification failed: {e}")
                    
                    # 原子替换：备份旧的，rename 新的
                    import shutil
                    if os.path.exists(backup_path):
                        shutil.rmtree(backup_path)
                    if os.path.exists(save_path):
                        os.rename(save_path, backup_path)
                    os.rename(temp_path, save_path)
                    
                    # 保留备份以防万一（下次保存时会被清理）
            else:
                # DDP 模式仅主进程保存（save_checkpoint 已经实现原子写入）
                if is_main_process():
                    save_checkpoint(
                        checkpoint_dir=save_path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=grad_scaler,
                        epoch=current_epoch,
                        global_step=global_step,
                        best_val_loss=best_val_loss,
                    )
                    tokenizer.save_pretrained(save_path)

            if is_main_process() and writer is not None:
                writer.add_text("ckpt/path", save_path, global_step)
                writer.flush()

        # --- graceful stop handling ---
        stop_requested = False
        received_signal = None

        def _request_stop(sig, frame):
            nonlocal stop_requested, received_signal
            if stop_requested:
                return
            stop_requested = True
            received_signal = sig
            if is_main_process():
                print(f"Received signal {sig}, will barrier, save, then exit after current step.")

        old_sigint = signal.getsignal(signal.SIGINT)
        old_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, _request_stop)
        signal.signal(signal.SIGTERM, _request_stop)

        def _maybe_exit_gracefully():
            nonlocal stop_requested, received_signal
            if not stop_requested:
                return
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
            # DeepSpeed 下所有 rank 都要调 save
            tag = f"interrupted_step_{global_step}"
            if is_main_process():
                print(f"Saving emergency checkpoint: {tag} (signal={received_signal})")
            
            save_full_checkpoint(tag)
            
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
            # 恢复信号处理后退出
            signal.signal(signal.SIGINT, old_sigint)
            signal.signal(signal.SIGTERM, old_sigterm)
            raise SystemExit(0)
        
        total_updates = num_training_steps
        train_start_time = time.time()
        epoch_start_time = time.time()

        def _fmt_secs(seconds: float) -> str:
            seconds = max(0.0, float(seconds))
            mins, secs = divmod(seconds, 60)
            hrs, mins = divmod(mins, 60)
            return f"{int(hrs):02d}:{int(mins):02d}:{int(secs):02d}"

        step_offset = global_step

        def log_step_fn(step: int, loss: float, batch_idx: int, ce_loss=None, aux_loss=None, lr=None):
            nonlocal global_step
            nonlocal epoch_start_time
            global_step = step_offset + step
            # ✅ 只在主进程打 log
            if not is_main_process():
                return
            if global_step % cfg.logging_steps == 0:
                msg = f"[step {global_step}] loss={loss:.4f}"
                if ce_loss is not None:
                    msg += f" ce={ce_loss:.4f}"
                if aux_loss is not None:
                    msg += f" aux={aux_loss:.4f}"
                if lr is not None:
                    msg += f" lr={lr:.6f}"

                elapsed_total = time.time() - train_start_time
                eta_total = None
                if total_updates > 0:
                    remaining = max(total_updates - global_step, 0)
                    avg_step = elapsed_total / max(global_step, 1)
                    eta_total = remaining * avg_step

                elapsed_epoch = time.time() - epoch_start_time
                progress_epoch = (batch_idx + 1) / max(len(train_loader), 1)
                eta_epoch = None
                if progress_epoch > 0:
                    eta_epoch = (elapsed_epoch / progress_epoch) - elapsed_epoch

                if eta_epoch is not None:
                    msg += f" eta_epoch={_fmt_secs(eta_epoch)}"
                if eta_total is not None:
                    msg += f" eta_total={_fmt_secs(eta_total)}"
                print(msg)

            if writer is not None:
                writer.add_scalar("train/loss", loss, global_step)
                if ce_loss is not None:
                    writer.add_scalar("train/ce_loss", ce_loss, global_step)
                if aux_loss is not None:
                    writer.add_scalar("train/aux_loss", aux_loss, global_step)
                if lr is not None:
                    writer.add_scalar("train/lr", lr, global_step)
                if global_step % (cfg.logging_steps * 5) == 0:
                    writer.flush()

        def after_step_fn(step: int, loss: float, batch_idx: int, ce_loss=None, aux_loss=None, lr=None, model=None):
            nonlocal best_val_loss, global_step
            global_step = step_offset + step

            # step-based checkpointing
            if cfg.save_steps > 0 and step % cfg.save_steps == 0:
                save_full_checkpoint(f"step_{global_step}")

            # optional step-based evaluation
            if cfg.eval_steps > 0 and step % cfg.eval_steps == 0:
                # 只有主进程打印，但 evaluate 内部应该是所有 rank 参与（如果 DataLoader 是分布式的）
                # 这里为了简单，假设 evaluate 是全 rank 同步的，只在 rank 0 打印
                val_loss = evaluate(model, val_loader, device)
                model.train()
                
                if is_main_process():
                    print(f"[step {step}] val_loss={val_loss:.4f}")
                    if writer is not None:
                        writer.add_scalar("step/val_loss", val_loss, step)
                        writer.flush()

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_full_checkpoint(f"best_step_{global_step}")

            _maybe_exit_gracefully()

        def train_one_epoch_deepspeed(
            model_engine,
            dataloader,
            device,
            autocast_dtype=None,
            log_step_fn=None,
            after_step_fn=None,
            should_stop_fn=None,
        ) -> float:
            model_engine.train()
            total_loss = 0.0
            total_ce = 0.0
            total_aux = 0.0
            num_updates = 0

            # Rank check for logging
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            
            amp_ctx = (
                torch.autocast(device_type=device.type, dtype=autocast_dtype)
                if autocast_dtype is not None
                else nullcontext()
            )

            for batch_idx, batch in enumerate(dataloader):
                if should_stop_fn is not None and should_stop_fn():
                    break

                if batch_idx == 0 and rank == 0:
                    if "input_ids" in batch:
                        print(f"[Rank {rank}] got first batch, input_ids={batch['input_ids'].shape}")
                    else:
                        print(f"[Rank {rank}] got first batch, keys={list(batch.keys())}")

                batch = _move_batch_to_device(batch, device)

                # DeepSpeed 自动处理 mixed precision，通常不需要外层 autocast
                # 但如果你的模型内部没有 cast 逻辑，保险起见保留
                with amp_ctx:
                    outputs = model_engine(**batch)
                    loss = outputs["loss"]
                    ce_loss = outputs.get("ce_loss", None)
                    aux_loss = outputs.get("aux_loss", None)

                if _is_global_bad_loss(loss):
                    if rank == 0:
                        print(
                            f"[train_one_epoch_ds] NaN/Inf loss at batch {batch_idx + 1} "
                            f"(loss={loss}) — skip this batch."
                        )
                    # DeepSpeed 遇到 NaN 会导致 overflow warning，通常不需要手动 zero_grad
                    # 但可以显式跳过
                    continue

                model_engine.backward(loss)
                model_engine.step()

                # DeepSpeed 内部处理了 gradient accumulation，
                # 所以每次 step 之后检查一下是否真的发生了 optimizer update
                # model_engine.is_gradient_accumulation_boundary() 用来判断
                
                if model_engine.is_gradient_accumulation_boundary():
                    num_updates += 1
                    step_loss = loss.detach().item()
                    total_loss += step_loss

                    if isinstance(ce_loss, torch.Tensor):
                        total_ce += ce_loss.detach().item()
                    if isinstance(aux_loss, torch.Tensor):
                        total_aux += aux_loss.detach().item()

                    current_lr = None
                    if hasattr(model_engine, "get_lr"):
                        lrs = model_engine.get_lr()
                        if isinstance(lrs, (list, tuple)) and len(lrs) > 0:
                            current_lr = lrs[0]

                    if log_step_fn is not None:
                        log_step_fn(
                            step=num_updates,
                            loss=step_loss,
                            batch_idx=batch_idx,
                            ce_loss=ce_loss.detach().item() if isinstance(ce_loss, torch.Tensor) else None,
                            aux_loss=aux_loss.detach().item() if isinstance(aux_loss, torch.Tensor) else None,
                            lr=current_lr,
                        )

                    if after_step_fn is not None:
                        after_step_fn(
                            step=num_updates,
                            loss=step_loss,
                            batch_idx=batch_idx,
                            ce_loss=ce_loss.detach().item() if isinstance(ce_loss, torch.Tensor) else None,
                            aux_loss=aux_loss.detach().item() if isinstance(aux_loss, torch.Tensor) else None,
                            lr=current_lr,
                            model=model_engine,
                        )

                if should_stop_fn is not None and should_stop_fn():
                    break

            if num_updates == 0:
                return float("nan")

            avg_loss = total_loss / num_updates
            return avg_loss

        try:
            for epoch in range(start_epoch, cfg.num_epochs):
                current_epoch = epoch
                step_offset = global_step
                # DDP sampler 每个 epoch 需要设置一下 epoch
                if isinstance(train_sampler, DistributedSampler) and dist.is_available() and dist.is_initialized():
                    train_sampler.set_epoch(epoch)

                if is_main_process():
                    print(f"Epoch {epoch} / {cfg.num_epochs - 1}")

                if use_deepspeed and model_engine is not None:
                    train_loss = train_one_epoch_deepspeed(
                        model_engine=model_engine,
                        dataloader=train_loader,
                        device=device,
                        autocast_dtype=autocast_dtype,
                        log_step_fn=log_step_fn,
                        after_step_fn=after_step_fn,
                        should_stop_fn=lambda: stop_requested,
                    )
                else:
                    train_loss = train_one_epoch(
                        model=model,
                        dataloader=train_loader,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        device=device,
                        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
                        log_step_fn=log_step_fn,
                        autocast_dtype=autocast_dtype,
                        grad_scaler=grad_scaler,
                        after_step_fn=after_step_fn,
                        should_stop_fn=lambda: stop_requested,
                    )
                
                if is_main_process():
                    print(f"Epoch {epoch} train_loss={train_loss:.4f}")
                    if writer is not None:
                        writer.add_scalar("epoch/train_loss", train_loss, global_step)
                        writer.flush()

                # Evaluate
                val_loss = evaluate(model, val_loader, device)

                _maybe_exit_gracefully()

                if is_main_process():
                    print(f"Epoch {epoch} val_loss={val_loss:.4f}")

                    if writer is not None:
                        writer.add_scalar("epoch/val_loss", val_loss, global_step)
                        writer.add_scalar("epoch/epoch", epoch, global_step)
                        writer.flush()

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_full_checkpoint(f"best_epoch{epoch}")

            if is_main_process():
                print("Saving final model to latest")
                save_full_checkpoint("latest")

        except KeyboardInterrupt:
            if is_main_process():
                print("KeyboardInterrupt received, saving emergency checkpoint...")
                save_full_checkpoint(f"interrupted_step_{global_step}")
            raise
        finally:
            signal.signal(signal.SIGINT, old_sigint)
            signal.signal(signal.SIGTERM, old_sigterm)
            if writer is not None:
                writer.close()

def train_thinker_stage2(
    cfg: Union[Stage2TrainConfig, Dict[str, Any]],
    tokenizer_name_or_path: str,
    enable_tensorboard: bool = False,
    log_dir: str = "./runs",
    resume_from_checkpoint: Optional[str] = None,
):
    """
    Stage2: Omni 多模态 AuT 训练入口。
    """
    import yaml

    # --- 统一把 cfg 变成 dict ---
    if isinstance(cfg, dict):
        c = cfg
    else:
        # dataclass 或其它对象，转成 dict
        c = vars(cfg)

    # 安全取配置，给一些默认值
    seed = c.get("seed", 42)
    set_seed(seed)
    stage1_init_ckpt = c["stage1_init_ckpt"]
    model_config_path = c["model"]["model_config_path"]
    train_corpus_path = c["data"]["train_corpus_path"]
    val_corpus_path = c["data"]["val_corpus_path"]
    image_root = c["data"]["image_root"]
    audio_root = c["data"]["audio_root"]
    output_dir = c["train"]["output_dir"] + time.strftime("-%Y%m%d-%H%M%S")
    resume_path = resume_from_checkpoint or c.get("resume_from_checkpoint") or c.get("train", {}).get("resume_from_checkpoint")

    train_cfg = c.get("train", {}) if isinstance(c, dict) else {}

    num_epochs = c.get("num_epochs", train_cfg.get("num_epochs", 1))
    batch_size = c.get("batch_size", train_cfg.get("batch_size", 2))
    max_seq_length = c.get("max_seq_length", train_cfg.get("max_seq_length", 1024))
    learning_rate = c.get("learning_rate", train_cfg.get("learning_rate", 1e-4))
    weight_decay = c.get("weight_decay", train_cfg.get("weight_decay", 0.01))
    warmup_ratio = c.get("warmup_ratio", train_cfg.get("warmup_ratio", 0.03))
    grad_accum = c.get("gradient_accumulation_steps", train_cfg.get("gradient_accumulation_steps", 1))
    logging_steps = c.get("logging_steps", train_cfg.get("logging_steps", 10))
    num_workers = c.get("num_workers", train_cfg.get("num_workers", 4))
    fp16 = bool(c.get("fp16", train_cfg.get("fp16", False)))
    bf16 = bool(c.get("bf16", train_cfg.get("bf16", False)))
    fp8 = bool(c.get("fp8", train_cfg.get("fp8", False)))
    int8_optimizer = bool(c.get("int8_optimizer", train_cfg.get("int8_optimizer", False)))
    # --- 以上 cfg 处理结束 ---

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # 1. tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        use_fast=True,
        local_files_only=True,
    )

    precision_flags = [fp16, bf16, fp8]
    if sum(bool(x) for x in precision_flags) > 1:
        raise ValueError("Only one of fp16/bf16/fp8 can be enabled for Stage2.")

    autocast_dtype = None
    grad_scaler = None
    if fp8:
        if not hasattr(torch, "float8_e4m3fn"):
            raise RuntimeError("fp8 requested but torch.float8_e4m3fn is unavailable in this build.")
        if device.type != "cuda":
            raise RuntimeError("fp8 training is only supported on CUDA devices.")
        autocast_dtype = torch.float8_e4m3fn
    elif fp16:
        autocast_dtype = torch.float16 if torch.cuda.is_available() else None
        grad_scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    elif bf16:
        autocast_dtype = torch.bfloat16 if torch.cuda.is_available() else None

    # 2. model config
    with open(model_config_path, "r", encoding="utf-8") as f:
        model_conf_dict = yaml.safe_load(f)
    model_config = Qwen3OmniMoeConfig(**model_conf_dict)

    # 3. model
    model = Qwen3OmniMoeThinkerVisionAudioModel(model_config)
    # ✅ 从 Stage1 ckpt 初始化 Thinker 权重（仅在不从已有 stage2 ckpt 恢复时）
    if stage1_init_ckpt and not resume_path:
        print(f"Loading Stage1 checkpoint from {stage1_init_ckpt}")
        base_thinker = Qwen3OmniMoeThinkerTextModel.from_pretrained(
            stage1_init_ckpt
        )
        missing, unexpected = model.thinker.load_state_dict(
            base_thinker.state_dict(), strict=False
        )
        print(f"Loaded Stage1 weights into Thinker. missing={len(missing)}, unexpected={len(unexpected)}")

    model.to(device)  # type: ignore[arg-type]

    # 4. dataset & dataloader
    train_dataset = OmniJsonlDataset(
        jsonl_path=train_corpus_path,
        image_root=image_root,
        audio_root=audio_root,
    )
    val_dataset = OmniJsonlDataset(
        jsonl_path=val_corpus_path,
        image_root=image_root,
        audio_root=audio_root,
    )

    collator = OmniStage2Collator(
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
    )

    # 5. optimizer & scheduler (简单版)
    if int8_optimizer:
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise ImportError(
                "int8_optimizer=True requires bitsandbytes to be installed."
            ) from exc
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    total_updates_per_epoch = max(
        1, len(train_loader) // grad_accum
    )
    total_updates = total_updates_per_epoch * num_epochs
    warmup_steps = int(total_updates * warmup_ratio)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, step / max(1, warmup_steps))
    )

    os.makedirs(output_dir, exist_ok=True)
    best_val_loss = float("inf")

    # Resume training if checkpoint provided
    start_epoch = 0
    global_step = 0
    if resume_path:
        start_epoch, global_step, best_val_loss = load_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=grad_scaler,
            map_location=device,
        )
        print(
            f"Resumed Stage2 from {resume_path}: start_epoch={start_epoch}, "
            f"global_step={global_step}, best_val_loss={best_val_loss}"
        )

    writer = None
    if enable_tensorboard and is_main_process():
        run_name = os.path.join(log_dir, f"{c.get('experiment_name', 'thinker_stage2')}_stage2" + time.strftime("-%Y%m%d-%H%M%S"))
        writer = SummaryWriter(log_dir=run_name)

    train_start_time = time.time()
    epoch_start_time = time.time()
    step_offset = global_step

    def _fmt_secs(seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        mins, secs = divmod(seconds, 60)
        hrs, mins = divmod(mins, 60)
        return f"{int(hrs):02d}:{int(mins):02d}:{int(secs):02d}"

    def log_step_fn(step: int, loss: float, batch_idx: int, ce_loss=None, aux_loss=None, lr=None):
        nonlocal global_step
        nonlocal epoch_start_time
        global_step = step_offset + step
        if global_step % logging_steps == 0:
            msg = f"[step {global_step}] loss={loss:.4f}"
            if ce_loss is not None:
                msg += f" ce={ce_loss:.4f}"
            if aux_loss is not None:
                msg += f" aux={aux_loss:.4f}"
            if lr is not None:
                msg += f" lr={lr:.6f}"

            elapsed_total = time.time() - train_start_time
            remaining = max(total_updates - global_step, 0)
            avg_step = elapsed_total / max(global_step, 1)
            eta_total = remaining * avg_step

            elapsed_epoch = time.time() - epoch_start_time
            progress_epoch = (batch_idx + 1) / max(len(train_loader), 1)
            eta_epoch = (elapsed_epoch / progress_epoch - elapsed_epoch) if progress_epoch > 0 else None

            if eta_epoch is not None:
                msg += f" eta_epoch={_fmt_secs(eta_epoch)}"
            if eta_total is not None:
                msg += f" eta_total={_fmt_secs(eta_total)}"
            print(msg)

        if writer is not None:
            writer.add_scalar("train/loss", loss, global_step)
            if ce_loss is not None:
                writer.add_scalar("train/ce_loss", ce_loss, global_step)
            if aux_loss is not None:
                writer.add_scalar("train/aux_loss", aux_loss, global_step)
            if lr is not None:
                writer.add_scalar("train/lr", lr, global_step)
            writer.flush()

    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch} / {num_epochs - 1}")
        epoch_start_time = time.time()
        step_offset = global_step

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            gradient_accumulation_steps=grad_accum,
            log_step_fn=log_step_fn,
            autocast_dtype=autocast_dtype,
            grad_scaler=grad_scaler,
        )
        print(f"Epoch {epoch} train_loss={train_loss:.4f}")
        if writer is not None:
            writer.add_scalar("epoch/train_loss", train_loss, global_step)
            writer.flush()

        val_loss = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
            autocast_dtype=autocast_dtype,
        )
        print(f"Epoch {epoch} val_loss={val_loss:.4f}")
        if writer is not None:
            writer.add_scalar("epoch/val_loss", val_loss, global_step)
            writer.add_scalar("epoch/epoch", epoch, global_step)
            writer.flush()

        # save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(output_dir, f"best_epoch{epoch}")
            print(f"Saving best model to {save_path}")
            save_checkpoint(
                checkpoint_dir=save_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=grad_scaler,
                epoch=epoch + 1,
                global_step=global_step,
                best_val_loss=best_val_loss,
            )
            tokenizer.save_pretrained(save_path)

            if writer is not None:
                writer.add_scalar("ckpt/best_val_loss", best_val_loss, global_step)
                writer.add_text("ckpt/path", save_path, global_step)
                writer.flush()

        # save latest
        latest_path = os.path.join(output_dir, "latest")
        print(f"Saving latest model to {latest_path}")
        save_checkpoint(
            checkpoint_dir=latest_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=grad_scaler,
            epoch=epoch + 1,
            global_step=global_step,
            best_val_loss=best_val_loss,
        )
        tokenizer.save_pretrained(latest_path)

    if writer is not None:
        writer.close()
