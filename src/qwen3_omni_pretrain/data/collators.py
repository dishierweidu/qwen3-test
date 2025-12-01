# src/qwen3_omni_pretrain/data/collators.py

from typing import List, Dict, Any

import torch
from transformers import PreTrainedTokenizerBase


class TextCausalLMCollator:
    """
    纯文本 Causal LM collator:
    - 将样本中的 "text" tokenize
    - pad 到 batch 最大长度
    - labels = input_ids.copy(), pad 部分设为 -100
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_seq_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [b["text"] for b in batch]
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
