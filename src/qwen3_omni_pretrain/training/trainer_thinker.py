# src/qwen3_omni_pretrain/training/trainer_thinker.py

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Union, List, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard.writer import SummaryWriter
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

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
from qwen3_omni_pretrain.data.datasets.text_dataset import TextJsonlDataset
from qwen3_omni_pretrain.data.collators import TextCausalLMCollator
from qwen3_omni_pretrain.training.loop import train_one_epoch, evaluate
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
    fp8: bool = False
    int8_optimizer: bool = False
    resume_from_checkpoint: Optional[str] = None
    
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
        resume_from_checkpoint=train_cfg.get("resume_from_checkpoint"),
    )


def set_seed(seed: int):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_thinker_stage1(
    config_path: str,
    tokenizer_name_or_path: str = "Qwen/Qwen2.5-7B",
    enable_tensorboard: bool = False,
    log_dir: str = "./runs",
    resume_from_checkpoint: Optional[str] = None,
):
    # 1. 加载配置
    cfg = build_trainer_config(config_path)
    os.makedirs(cfg.output_dir, exist_ok=True)

    set_seed(cfg.seed)

    # 用 cfg.ddp 控制是否开启 DDP（再叠加 ddp_available 判断）
    use_ddp = cfg.ddp and ddp_available()

    # 把后面的所有逻辑包进 distributed_context，这样会自动 init/destroy process_group
    with distributed_context():
        # 2. 设备 & rank
        if use_ddp:
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
        model = Qwen3OmniMoeThinkerTextModel(model_cfg)
        model.to(device)  # type: ignore[arg-type]

        # ✅ DDP 包裹模型
        model = ddp_wrap_model(model) if use_ddp else model

        # 5. 数据集
        train_dataset = _build_text_dataset(cfg.train_corpus_paths, tokenizer, cfg.max_seq_length)
        val_dataset = _build_text_dataset(cfg.val_corpus_paths, tokenizer, cfg.max_seq_length)

        collator = TextCausalLMCollator(tokenizer, cfg.max_seq_length)

        # ✅ DistributedSampler
        if use_ddp:
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
            # sampler 存在时必须 shuffle=False
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

        # 6. 优化器 & scheduler
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
        if cfg.int8_optimizer:
            try:
                import bitsandbytes as bnb
            except ImportError as exc:
                raise ImportError(
                    "int8_optimizer=True requires bitsandbytes to be installed."
                ) from exc

            optimizer = bnb.optim.AdamW8bit(
                optimizer_grouped_parameters,
                lr=cfg.learning_rate,
                betas=(0.9, 0.95),
                eps=1e-8,
            )
        else:
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=cfg.learning_rate,
                betas=(0.9, 0.95),
                eps=1e-8,
            )

        if cfg.max_steps > 0:
            num_training_steps = cfg.max_steps
        else:
            num_training_steps = len(train_loader) * cfg.num_epochs // max(cfg.gradient_accumulation_steps, 1)

        num_warmup_steps = int(num_training_steps * cfg.warmup_ratio)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        # 如果提供了 resume 路径，加载训练状态
        start_epoch = 0
        global_step = 0
        best_val_loss = float("inf")
        resume_path = resume_from_checkpoint or cfg.resume_from_checkpoint
        if resume_path:
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

        best_val_loss = float("inf")

        def save_checkpoint(tag: str):
            # 主进程保存即可，避免并发写
            if not is_main_process():
                return
            save_path = os.path.join(cfg.output_dir, tag)
            os.makedirs(save_path, exist_ok=True)
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(save_path, safe_serialization=False)
            tokenizer.save_pretrained(save_path)
            if writer is not None:
                writer.add_text("ckpt/path", save_path, global_step)
                writer.flush()
        
        total_updates = num_training_steps
        train_start_time = time.time()
        epoch_start_time = time.time()

        def _fmt_secs(seconds: float) -> str:
            seconds = max(0.0, float(seconds))
            mins, secs = divmod(seconds, 60)
            hrs, mins = divmod(mins, 60)
            return f"{int(hrs):02d}:{int(mins):02d}:{int(secs):02d}"

        def log_step_fn(step: int, loss: float, batch_idx: int, ce_loss=None, aux_loss=None, lr=None):
            nonlocal global_step
            nonlocal epoch_start_time
            global_step += 1
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

                # ETA 估算
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
                writer.flush()

        def after_step_fn(step: int, loss: float, batch_idx: int, ce_loss=None, aux_loss=None, lr=None, model=None):
            nonlocal best_val_loss, global_step
            global_step = step

            # step-based checkpointing
            if cfg.save_steps > 0 and step % cfg.save_steps == 0:
                save_checkpoint(f"step_{step}")

            # optional step-based evaluation
            if cfg.eval_steps > 0 and step % cfg.eval_steps == 0:
                if is_main_process():
                    val_loss = evaluate(model, val_loader, device)
                    model.train()
                    print(f"[step {step}] val_loss={val_loss:.4f}")

                    if writer is not None:
                        writer.add_scalar("step/val_loss", val_loss, step)
                        writer.flush()

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(f"best_step_{step}")

        try:
            for epoch in range(cfg.num_epochs):
                # DDP sampler 每个 epoch 需要设置一下 epoch，保证 shuffle 不同
                if use_ddp and isinstance(train_sampler, DistributedSampler):
                    train_sampler.set_epoch(epoch)

                if is_main_process():
                    print(f"Epoch {epoch} / {cfg.num_epochs - 1}")

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
                )
                if is_main_process():
                    print(f"Epoch {epoch} train_loss={train_loss:.4f}")

                    if writer is not None:
                        writer.add_scalar("epoch/train_loss", train_loss, global_step)
                        writer.flush()

                # ✅ 所有 rank 都跑 evaluate（这样 val_sampler 分片才有意义）
                val_loss = evaluate(model, val_loader, device)

                # 只在主进程打印 & 保存
                if is_main_process():
                    print(f"Epoch {epoch} val_loss={val_loss:.4f}")

                    if writer is not None:
                        writer.add_scalar("epoch/val_loss", val_loss, global_step)
                        writer.add_scalar("epoch/epoch", epoch, global_step)
                        writer.flush()

                    # 取出真正的 HF 模型（DDP 包裹时在 model.module 里）
                    model_to_save = model.module if hasattr(model, "module") else model

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(f"best_epoch{epoch}")

            # 最终再保存一版 latest
            if is_main_process():
                final_path = os.path.join(cfg.output_dir, "latest")
                os.makedirs(final_path, exist_ok=True)
                print(f"Saving final model to {final_path}")
                model_to_save = model.module if hasattr(model, "module") else model
                model_to_save.save_pretrained(final_path, safe_serialization=False)
                tokenizer.save_pretrained(final_path)

        except KeyboardInterrupt:
            # 训练被打断时尽量保留一个可用 ckpt
            if is_main_process():
                print("KeyboardInterrupt received, saving emergency checkpoint...")
                save_checkpoint(f"interrupted_step_{global_step}")
            raise
        finally:
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
    tokenizer_path = _ensure_local_path(tokenizer_name_or_path, "tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
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

    global_step = 0
    writer = None
    if enable_tensorboard and is_main_process():
        run_name = os.path.join(log_dir, f"{c.get('experiment_name', 'thinker_stage2')}_stage2" + time.strftime("-%Y%m%d-%H%M%S"))
        writer = SummaryWriter(log_dir=run_name)

    train_start_time = time.time()
    epoch_start_time = time.time()

    def _fmt_secs(seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        mins, secs = divmod(seconds, 60)
        hrs, mins = divmod(mins, 60)
        return f"{int(hrs):02d}:{int(mins):02d}:{int(secs):02d}"

    def log_step_fn(step: int, loss: float, batch_idx: int, ce_loss=None, aux_loss=None, lr=None):
        nonlocal global_step
        nonlocal epoch_start_time
        global_step += 1
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
