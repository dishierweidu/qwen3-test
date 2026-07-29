from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
import torchaudio


class MediaLoadError(RuntimeError):
    """Raised when a sample references media that cannot be decoded."""

    def __init__(
        self,
        *,
        modality: str,
        path: str,
        sample_id: str,
        cause: BaseException,
    ) -> None:
        self.modality = modality
        self.path = path
        self.sample_id = sample_id
        self.cause = cause
        super().__init__(
            f"Failed to load {modality} for sample {sample_id!r} from {path!r}: "
            f"{type(cause).__name__}: {cause}"
        )

    def to_dict(self) -> Dict[str, str]:
        return {
            "sample_id": self.sample_id,
            "modality": self.modality,
            "path": self.path,
            "error_type": type(self.cause).__name__,
            "error": str(self.cause),
        }


@dataclass(frozen=True)
class MediaLoadResult:
    tensor: torch.Tensor
    present: int
    error: Optional[Dict[str, str]] = None


class Stage2MediaLoader:
    def __init__(
        self,
        image_size: int = 224,
        max_audio_len: int = 32000,
        *,
        skip_bad_media: bool = False,
    ) -> None:
        self.image_size = int(image_size)
        self.max_audio_len = int(max_audio_len)
        self.skip_bad_media = bool(skip_bad_media)

    def _zero(self, modality: str) -> torch.Tensor:
        if modality == "image":
            return torch.zeros(3, self.image_size, self.image_size)
        if modality == "audio":
            return torch.zeros(self.max_audio_len)
        raise ValueError(f"unsupported modality: {modality}")

    def _load_image(self, path: str) -> torch.Tensor:
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = image.resize((self.image_size, self.image_size))
            data = bytearray(image.tobytes())
        tensor = torch.frombuffer(data, dtype=torch.uint8).clone()
        tensor = tensor.view(self.image_size, self.image_size, 3)
        return tensor.permute(2, 0, 1).float().div_(255.0)

    def _load_audio(self, path: str) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(path)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, 16000
            )
        waveform = waveform.squeeze(0)
        if waveform.size(0) >= self.max_audio_len:
            return waveform[: self.max_audio_len]
        return torch.nn.functional.pad(
            waveform, (0, self.max_audio_len - waveform.size(0))
        )

    def load_optional(
        self,
        *,
        modality: str,
        path: Optional[str],
        sample_id: str,
    ) -> MediaLoadResult:
        if modality not in {"image", "audio"}:
            raise ValueError(f"unsupported modality: {modality}")
        if path is None:
            return MediaLoadResult(self._zero(modality), 0)
        loader = self._load_image if modality == "image" else self._load_audio
        try:
            return MediaLoadResult(loader(path), 1)
        except Exception as cause:
            error = MediaLoadError(
                modality=modality,
                path=path,
                sample_id=sample_id,
                cause=cause,
            )
            if not self.skip_bad_media:
                raise error from cause
            return MediaLoadResult(
                self._zero(modality),
                0,
                error.to_dict(),
            )


class TextCausalLMCollator:
    """Pad already-tokenized causal-LM examples and mask padding labels."""

    def __init__(self, tokenizer, max_seq_length: int = 2048):
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        self.pad_token_id = int(pad_id)
        self.max_seq_length = int(max_seq_length)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        bsz = len(batch)
        input_ids = torch.full(
            (bsz, self.max_seq_length), self.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros(
            (bsz, self.max_seq_length), dtype=torch.long
        )
        labels = torch.full((bsz, self.max_seq_length), -100, dtype=torch.long)

        for i, ex in enumerate(batch):
            ids = ex["input_ids"]
            mask = ex["attention_mask"]
            if not torch.is_tensor(ids):
                ids = torch.tensor(ids, dtype=torch.long)
            if not torch.is_tensor(mask):
                mask = torch.tensor(mask, dtype=torch.long)
            ids = ids[: self.max_seq_length]
            mask = mask[: self.max_seq_length]
            seq_len = ids.size(0)
            input_ids[i, :seq_len] = ids
            attention_mask[i, :seq_len] = mask
            label_ids = ids.clone()
            label_ids[label_ids == self.pad_token_id] = -100
            labels[i, :seq_len] = label_ids

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class OmniStage2Collator:
    """
    Build Stage-2 text/image/audio batches.

    Only target text contributes to LM loss. Missing optional media is valid.
    Referenced but unreadable media raises by default; callers must opt in to
    ``skip_bad_media`` to convert such media to an absent-modality marker.
    """

    def __init__(
        self,
        tokenizer,
        max_seq_length: int = 2048,
        image_size: int = 224,
        max_audio_len: int = 32000,
        *,
        skip_bad_media: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_length = int(max_seq_length)
        self.media_loader = Stage2MediaLoader(
            image_size=image_size,
            max_audio_len=max_audio_len,
            skip_bad_media=skip_bad_media,
        )
        self.image_size = self.media_loader.image_size
        self.max_audio_len = self.media_loader.max_audio_len
        self.skip_bad_media = self.media_loader.skip_bad_media

        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("Tokenizer must define pad_token_id or eos_token_id")
        self.pad_token_id = int(pad_id)

    def _encode_text(self, text: str) -> List[int]:
        encoded = self.tokenizer(text, add_special_tokens=False)
        ids = encoded["input_ids"]
        if torch.is_tensor(ids):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return [int(token_id) for token_id in ids]

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        sequences: List[List[int]] = []
        sequence_labels: List[List[int]] = []
        images: List[torch.Tensor] = []
        audios: List[torch.Tensor] = []
        has_image_flags: List[int] = []
        has_audio_flags: List[int] = []
        sample_ids: List[str] = []
        media_errors: List[Dict[str, str]] = []

        for index, example in enumerate(batch):
            sample_id = str(example.get("id", f"batch-index-{index}"))
            sample_ids.append(sample_id)
            prompt_ids = self._encode_text(example.get("input_text", "") or "")
            target_ids = self._encode_text(example.get("target_text", "") or "")

            # Reserve sequence capacity for supervised target tokens. When the
            # prompt is too long, keep its most recent tokens rather than
            # silently truncating the entire target and producing zero loss.
            kept_target = target_ids[: self.max_seq_length]
            prompt_budget = self.max_seq_length - len(kept_target)
            kept_prompt = prompt_ids[-prompt_budget:] if prompt_budget > 0 else []
            full_ids = kept_prompt + kept_target
            labels = ([-100] * len(kept_prompt)) + kept_target
            sequences.append(full_ids)
            sequence_labels.append(labels)

            image_path = example.get("image_path")
            image_result = self.media_loader.load_optional(
                modality="image",
                path=str(image_path) if image_path else None,
                sample_id=sample_id,
            )
            images.append(image_result.tensor)
            has_image_flags.append(image_result.present)
            if image_result.error is not None:
                media_errors.append(image_result.error)

            audio_path = example.get("audio_path")
            audio_result = self.media_loader.load_optional(
                modality="audio",
                path=str(audio_path) if audio_path else None,
                sample_id=sample_id,
            )
            audios.append(audio_result.tensor)
            has_audio_flags.append(audio_result.present)
            if audio_result.error is not None:
                media_errors.append(audio_result.error)

        width = min(
            self.max_seq_length,
            max((len(ids) for ids in sequences), default=0),
        )
        input_ids = torch.full(
            (len(batch), width), self.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros((len(batch), width), dtype=torch.long)
        labels = torch.full((len(batch), width), -100, dtype=torch.long)
        for row, (ids, row_labels) in enumerate(zip(sequences, sequence_labels)):
            length = min(len(ids), width)
            if length == 0:
                continue
            input_ids[row, :length] = torch.tensor(ids[:length], dtype=torch.long)
            attention_mask[row, :length] = 1
            labels[row, :length] = torch.tensor(
                row_labels[:length], dtype=torch.long
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": torch.stack(images, dim=0),
            "audio_values": torch.stack(audios, dim=0),
            "has_image": torch.tensor(has_image_flags, dtype=torch.long),
            "has_audio": torch.tensor(has_audio_flags, dtype=torch.long),
            "_sample_ids": sample_ids,
            "_media_errors": media_errors,
        }


class PackedCausalLMCollator:
    """Stack fixed-length packed causal-LM examples."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = int(pad_token_id)

    def __call__(self, batch):
        input_ids = torch.stack(
            [item["input_ids"].long() for item in batch], dim=0
        )
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "attention_mask": torch.ones_like(input_ids, dtype=torch.long),
        }
