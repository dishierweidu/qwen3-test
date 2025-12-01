# src/qwen3_omni_pretrain/data/datasets/text_dataset.py

import json
from typing import List, Dict

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class TextJsonlDataset(Dataset):
    """
    每行: {"text": "..."} 的 jsonl 文本语料
    """

    def __init__(self, path: str, tokenizer: PreTrainedTokenizerBase, max_seq_length: int = 2048):
        self.path = path
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        self.samples: List[str] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get("text", "")
                if text:
                    self.samples.append(text)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        # 在 collator 里再做 tokenize & pad
        return {"text": self.samples[idx]}
