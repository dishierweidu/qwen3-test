#!/usr/bin/env python
"""Quick unused-gradient check.

Runs a tiny forward/backward and reports parameters whose .grad is None.
This helps decide whether DDP can safely run with find_unused_parameters=False.

Example:
  python scripts/check_unused_grads.py \
    --config configs/train/stage1_text_only.yaml \
    --tokenizer_name_or_path Qwen/Qwen2.5-7B \
    --device cuda
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

import torch
from transformers import AutoTokenizer

from qwen3_omni_pretrain.utils.config_utils import load_yaml
from qwen3_omni_pretrain.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeConfig,
)
from qwen3_omni_pretrain.models.qwen3_omni_moe.modeling_thinker_text import (
    Qwen3OmniMoeThinkerTextModel,
)


@torch.no_grad()
def _make_dummy_batch(
    tokenizer, max_seq_length: int, batch_size: int, device: torch.device
) -> Dict[str, torch.Tensor]:
    text = "hello"  # minimal tokenized input
    enc = tokenizer(
        [text] * batch_size,
        padding="max_length",
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # LM-style labels
    labels = input_ids.clone()

    batch: Dict[str, torch.Tensor] = {
        "input_ids": input_ids,
        "labels": labels,
    }
    if attention_mask is not None:
        batch["attention_mask"] = attention_mask
    return batch


def _backward_once(model: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    # enable grads
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None

    out = model(**batch)
    loss = getattr(out, "loss", None)
    if loss is None:
        raise RuntimeError("Model output has no .loss; cannot run backward.")

    loss.backward()
    return loss.detach()


def _find_none_grads(model: torch.nn.Module) -> Tuple[int, int, List[str]]:
    none_names: List[str] = []
    total = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        total += 1
        if p.grad is None:
            none_names.append(name)
    return total, len(none_names), none_names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Train YAML config path")
    ap.add_argument("--tokenizer_name_or_path", required=True)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--max_seq_length", type=int, default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    model_cfg_path = cfg["model"]["config_path"]
    max_seq_length = args.max_seq_length or int(cfg.get("data", {}).get("max_seq_length", 2048))

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_cfg_dict = load_yaml(model_cfg_path)
    model_cfg = Qwen3OmniMoeConfig(**model_cfg_dict)
    model = Qwen3OmniMoeThinkerTextModel(model_cfg)
    model.to(device)
    model.train()

    # create dummy batch with grad enabled
    batch = _make_dummy_batch(tokenizer, max_seq_length=max_seq_length, batch_size=args.batch_size, device=device)
    # _make_dummy_batch is no_grad; turn grad back on for forward/backward
    for k, v in list(batch.items()):
        batch[k] = v.detach()

    loss = _backward_once(model, batch)
    total, none_count, none_names = _find_none_grads(model)

    print(f"loss: {loss.item():.6f}")
    print(f"trainable params: {total}")
    print(f"grad is None: {none_count} ({(none_count / max(total, 1)) * 100:.2f}%)")
    if none_count:
        # keep output bounded
        show = none_names[:50]
        print("first none-grad params:")
        for n in show:
            print(f"  - {n}")
        if len(none_names) > len(show):
            print(f"  ... and {len(none_names) - len(show)} more")

    # exit code hint for automation
    raise SystemExit(0 if none_count == 0 else 2)


if __name__ == "__main__":
    main()
