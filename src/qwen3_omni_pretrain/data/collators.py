# src/qwen3_omni_pretrain/data/collators.py

from typing import List, Dict, Any, Optional

import torch
from torch.nn.utils.rnn import pad_sequence

from PIL import Image
import torchaudio


class TextCausalLMCollator:
    """
    Stage1: 纯文本 Causal LM collator。
    """

    def __init__(self, tokenizer, max_seq_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts = []
        for ex in batch:
            # 假设 ex["text"] 存原始文本；也可以根据你的 text_dataset 改这里
            txt = ex.get("text", "")
            texts.append(txt)

        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class OmniStage2Collator:
    """
    Stage2: 文本 + 图像 + 音频统一 collator。

    输出：
      - input_ids, attention_mask, labels
      - pixel_values: [B,3,H,W]
      - audio_values: [B, max_audio_len]
      - has_image, has_audio: [B] 0/1 mask
    """

    def __init__(
        self,
        tokenizer,
        max_seq_length: int = 2048,
        image_size: int = 224,
        max_audio_len: int = 32000,  # 2 秒 @ 16kHz
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.image_size = image_size
        self.max_audio_len = max_audio_len

    def _load_image(self, path: str) -> torch.Tensor:
        # 简单 resize + ToTensor [0,1]
        img = Image.open(path).convert("RGB")
        img = img.resize((self.image_size, self.image_size))
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
        img = img.view(self.image_size, self.image_size, 3)
        img = img.permute(2, 0, 1).float() / 255.0  # [3,H,W]
        return img

    def _load_audio(self, path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(path)  # [C, T]
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)  # 转 mono

        wav = torchaudio.functional.resample(wav, sr, 16000)  # 重采样到 16k
        wav = wav.squeeze(0)  # [T]

        T = wav.size(0)
        if T >= self.max_audio_len:
            wav = wav[: self.max_audio_len]
        else:
            pad = self.max_audio_len - T
            wav = torch.nn.functional.pad(wav, (0, pad))

        return wav  # [max_audio_len]

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts: List[str] = []
        images: List[torch.Tensor] = []
        audios: List[torch.Tensor] = []
        has_image_flags: List[int] = []
        has_audio_flags: List[int] = []

        for ex in batch:
            input_text = ex.get("input_text", "") or ""
            target_text = ex.get("target_text", "") or ""

            # 简单拼接：input + target
            full_text = input_text + target_text
            texts.append(full_text)

            # image
            if ex.get("image_path"):
                try:
                    img = self._load_image(ex["image_path"])
                    has_image_flags.append(1)
                except Exception:
                    img = torch.zeros(3, self.image_size, self.image_size)
                    has_image_flags.append(0)
            else:
                img = torch.zeros(3, self.image_size, self.image_size)
                has_image_flags.append(0)
            images.append(img)

            # audio
            if ex.get("audio_path"):
                try:
                    wav = self._load_audio(ex["audio_path"])
                    has_audio_flags.append(1)
                except Exception:
                    wav = torch.zeros(self.max_audio_len)
                    has_audio_flags.append(0)
            else:
                wav = torch.zeros(self.max_audio_len)
                has_audio_flags.append(0)
            audios.append(wav)

        # 文本 tokenization
        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        input_ids = tokenized["input_ids"]          # [B, T]
        attention_mask = tokenized["attention_mask"]
        labels = input_ids.clone()                 # 简单 LM：预测整句

        pixel_values = torch.stack(images, dim=0)  # [B, 3, H, W]
        audio_values = torch.stack(audios, dim=0)  # [B, max_audio_len]
        has_image = torch.tensor(has_image_flags, dtype=torch.long)
        has_audio = torch.tensor(has_audio_flags, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "audio_values": audio_values,
            "has_image": has_image,
            "has_audio": has_audio,
        }
