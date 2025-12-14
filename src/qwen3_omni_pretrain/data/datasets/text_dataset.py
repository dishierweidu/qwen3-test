# src/qwen3_omni_pretrain/data/datasets/text_dataset.py

import numpy as np
import json
from typing import List, Dict

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class TextJsonlDataset(Dataset):
    """
    每行: {"text": "..."} 的 jsonl 文本语料。

    为了限制常驻 RAM 占用，不再把所有文本读入内存，而是只记录
    每条样本在文件中的偏移量，按需 seek 读取并 tokenize。
    """

    def __init__(self, path: str, tokenizer: PreTrainedTokenizerBase, max_seq_length: int = 2048):
        self.path = path
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        self.offsets: List[int] = []
        with open(path, "rb") as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = obj.get("text", "")
                if text:
                    self.offsets.append(pos)

    def __len__(self) -> int:
        return len(self.offsets)

    def _read_line(self, offset: int) -> str:
        with open(self.path, "rb") as f:
            f.seek(offset)
            line = f.readline()
        return line.decode("utf-8", errors="ignore")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        offset = self.offsets[idx]
        line = self._read_line(offset).strip()

        obj = json.loads(line)
        text = obj.get("text", "")

        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_length,
            padding=False,              # 不在这里 pad，留给 collator
            return_attention_mask=True,
        )

        input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(encoded["attention_mask"], dtype=torch.long)

        return {
            "input_ids": input_ids,         # [T]
            "attention_mask": attention_mask,  # [T]
        }
        
class PackedTokenDataset(Dataset):
    """
    离线 pre-tokenization 后的 packed token 数据集：
      - 从 <prefix>.bin 中用 np.memmap 把所有 token 读进来（只读）
      - 固定 seq_length 切成 [N, seq_length] 的样本
      - 每个样本只返回 input_ids，labels/attention_mask 交给 collator 处理

    bin 文件格式：
      - int32 连续 token 流
      - 元信息（pad/eos/seq_length 等）存放在 <prefix>.meta.json，仅作参考
    """

    def __init__(self, bin_path: str, seq_length: int):
        super().__init__()
        self.bin_path = bin_path
        self.seq_length = int(seq_length)

        assert self.seq_length > 0, "seq_length must be positive."

        # memmap，不会一次性把所有数据 load 进内存
        self.tokens = np.memmap(self.bin_path, dtype=np.int32, mode="r")
        self.total_tokens = int(self.tokens.shape[0])

        # 直接向下取整，丢掉末尾不足一个 seq 的部分
        self.num_sequences = self.total_tokens // self.seq_length

        if self.num_sequences == 0:
            raise ValueError(
                f"PackedTokenDataset: total_tokens={self.total_tokens} < seq_length={self.seq_length}"
            )

        print(
            f"[PackedTokenDataset] {self.bin_path}: "
            f"total_tokens={self.total_tokens}, "
            f"seq_length={self.seq_length}, "
            f"num_sequences={self.num_sequences}"
        )

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx < 0 or idx >= self.num_sequences:
            raise IndexError(f"Index {idx} out of range for {self.num_sequences} sequences.")

        start = idx * self.seq_length
        end = start + self.seq_length

        # 注意：memmap 切片还是视图，这里转成独立 np.array，避免后续修改影响底层
        np_slice = np.array(self.tokens[start:end], dtype=np.int64)
        input_ids = torch.from_numpy(np_slice)  # [seq_length]

        return {"input_ids": input_ids}
