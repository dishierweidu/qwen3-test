# src/qwen3_omni_pretrain/training/trainer_thinker.py

import os
from dataclasses import dataclass
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from qwen3_omni_pretrain.utils.config_utils import load_yaml
from qwen3_omni_pretrain.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeConfig,
)
from qwen3_omni_pretrain.models.qwen3_omni_moe.modeling_thinker_text import (
    Qwen3OmniMoeThinkerTextModel,
)
from qwen3_omni_pretrain.data.datasets.text_dataset import TextJsonlDataset
from qwen3_omni_pretrain.data.collators import TextCausalLMCollator
from qwen3_omni_pretrain.training.loop import train_one_epoch, evaluate


@dataclass
class TrainerThinkerConfig:
    experiment_name: str
    model_config_path: str
    train_corpus_path: str
    val_corpus_path: str
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
    seed: int
    ddp: bool


def build_trainer_config(yaml_path: str) -> TrainerThinkerConfig:
    cfg = load_yaml(yaml_path)

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]

    return TrainerThinkerConfig(
        experiment_name=str(cfg["experiment_name"]),
        model_config_path=str(model_cfg["config_path"]),
        train_corpus_path=str(data_cfg["train_corpus_path"]),
        val_corpus_path=str(data_cfg["val_corpus_path"]),
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
        output_dir=str(train_cfg["output_dir"]),
        fp16=bool(train_cfg.get("fp16", False)),
        bf16=bool(train_cfg.get("bf16", False)),
        seed=int(train_cfg.get("seed", 42)),
        ddp=bool(train_cfg.get("ddp", False)),
    )


def set_seed(seed: int):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_thinker_stage1(config_path: str, tokenizer_name_or_path: str = "Qwen/Qwen2.5-7B"):
    # 1. 加载配置
    cfg = build_trainer_config(config_path)
    os.makedirs(cfg.output_dir, exist_ok=True)

    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 构建 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. 构建模型配置 & 模型
    model_cfg_dict = load_yaml(cfg.model_config_path)
    model_cfg = Qwen3OmniMoeConfig(**model_cfg_dict)
    model = Qwen3OmniMoeThinkerTextModel(model_cfg)

    model.to(device)

    # 4. 数据
    train_dataset = TextJsonlDataset(cfg.train_corpus_path, tokenizer, cfg.max_seq_length)
    val_dataset = TextJsonlDataset(cfg.val_corpus_path, tokenizer, cfg.max_seq_length)

    collator = TextCausalLMCollator(tokenizer, cfg.max_seq_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
        collate_fn=collator,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collator,
        pin_memory=True,
    )

    # 5. 优化器 & scheduler
    no_decay = ["bias", "norm", "layernorm", "ln"]  # 粗略的划分
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
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=cfg.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    if cfg.max_steps > 0:
        num_training_steps = cfg.max_steps
    else:
        num_training_steps = len(train_loader) * cfg.num_epochs // cfg.gradient_accumulation_steps

    num_warmup_steps = int(num_training_steps * cfg.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # 6. 训练循环
    global_step = 0

    def log_step_fn(step: int, loss: float, batch_idx: int):
        nonlocal global_step
        global_step += 1
        if global_step % cfg.logging_steps == 0:
            print(f"[step {global_step}] loss={loss:.4f}")

    best_val_loss = float("inf")

    for epoch in range(cfg.num_epochs):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            log_step_fn=log_step_fn,
            epoch=epoch,
        )
        print(f"Epoch {epoch} train_loss={train_loss:.4f}")

        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch} val_loss={val_loss:.4f}")

        # 保存最好的一版
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(cfg.output_dir, f"best_epoch{epoch}")
            os.makedirs(save_path, exist_ok=True)
            print(f"Saving best model to {save_path}")
            model.save_pretrained(save_path, safe_serialization=False)
            tokenizer.save_pretrained(save_path)

    # 最终再保存一版 latest
    final_path = os.path.join(cfg.output_dir, "latest")
    os.makedirs(final_path, exist_ok=True)
    print(f"Saving final model to {final_path}")
    model.save_pretrained(final_path, safe_serialization=False)
    tokenizer.save_pretrained(final_path)
