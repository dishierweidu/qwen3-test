from __future__ import annotations

from collections.abc import Sequence
import math
from numbers import Real

import torch
from torch import nn
from torch.nn import functional as F

from qwen3_omni_pretrain.multimodal.encoders.vision import (
    _pack_feature_rows,
    _positive_integer,
    _source_from_item,
    _validate_decoded_items,
    _validate_sources,
)
from qwen3_omni_pretrain.multimodal.io import DecodedMedia
from qwen3_omni_pretrain.multimodal.modalities import MediaModality
from qwen3_omni_pretrain.multimodal.types import (
    MediaSequence,
    MediaSource,
)


def _validate_waveforms(
    waveforms: object,
) -> tuple[torch.Tensor, ...]:
    if (
        not isinstance(waveforms, Sequence)
        or isinstance(waveforms, (str, bytes, bytearray))
    ):
        raise TypeError("waveforms must be a sequence of tensors")
    if not waveforms:
        raise ValueError("waveforms must be non-empty")
    validated: list[torch.Tensor] = []
    for waveform in waveforms:
        if not isinstance(waveform, torch.Tensor):
            raise TypeError("waveforms must contain torch.Tensor values")
        if waveform.ndim != 1:
            raise ValueError("each waveform must be a 1-D tensor")
        if waveform.numel() == 0:
            raise ValueError("each waveform must be non-empty")
        if not waveform.is_floating_point():
            raise TypeError("waveforms must have a floating-point dtype")
        if not bool(torch.isfinite(waveform).all().item()):
            raise ValueError("waveforms must contain finite values")
        validated.append(waveform)
    return tuple(validated)


def _validate_offsets(
    offsets: object,
    count: int,
) -> tuple[float, ...]:
    if offsets is None:
        return (0.0,) * count
    if (
        not isinstance(offsets, Sequence)
        or isinstance(offsets, (str, bytes, bytearray))
    ):
        raise TypeError(
            "timeline offsets must be a sequence or None"
        )
    if len(offsets) != count:
        raise ValueError(
            "timeline offset count must equal waveform count"
        )
    validated: list[float] = []
    for offset in offsets:
        if isinstance(offset, bool) or not isinstance(offset, Real):
            raise TypeError("timeline offsets must be real numbers")
        numeric = float(offset)
        if not math.isfinite(numeric) or numeric < 0:
            raise ValueError(
                "timeline offsets must be finite and non-negative"
            )
        validated.append(numeric)
    return tuple(validated)


class AudioWindowEncoder(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        window_size: int,
        hop_size: int,
        sample_rate: int,
    ) -> None:
        super().__init__()
        self.hidden_size = _positive_integer(
            hidden_size,
            "hidden_size",
        )
        self.window_size = _positive_integer(
            window_size,
            "window_size",
        )
        self.hop_size = _positive_integer(
            hop_size,
            "hop_size",
        )
        self.sample_rate = _positive_integer(
            sample_rate,
            "sample_rate",
        )
        if self.hop_size > self.window_size:
            raise ValueError("hop_size must not exceed window_size")
        self.proj = nn.Linear(
            self.window_size,
            self.hidden_size,
            bias=False,
        )
        self.norm = nn.LayerNorm(self.hidden_size)

    def _window_count(self, length: int) -> int:
        uncovered = max(length - self.window_size, 0)
        return max(
            1,
            1
            + (
                uncovered + self.hop_size - 1
            )
            // self.hop_size,
        )

    def encode_waveform(
        self,
        waveform: torch.Tensor,
    ) -> torch.Tensor:
        waveform = _validate_waveforms([waveform])[0]
        weight = self.proj.weight
        waveform = waveform.to(
            device=weight.device,
            dtype=weight.dtype,
        )
        window_count = self._window_count(waveform.numel())
        required_length = (
            (window_count - 1) * self.hop_size + self.window_size
        )
        waveform = F.pad(
            waveform,
            (0, required_length - waveform.numel()),
        )
        windows = waveform.unfold(
            0,
            self.window_size,
            self.hop_size,
        )
        return self.norm(self.proj(windows))

    def from_waveforms(
        self,
        waveforms: Sequence[torch.Tensor],
        *,
        sources: tuple[MediaSource, ...],
        timeline_offsets_seconds: Sequence[float] | None = None,
    ) -> MediaSequence:
        waveforms = _validate_waveforms(waveforms)
        sources = _validate_sources(sources, len(waveforms))
        offsets = _validate_offsets(
            timeline_offsets_seconds,
            len(waveforms),
        )

        rows = [self.encode_waveform(waveform) for waveform in waveforms]
        embeddings, attention_mask = _pack_feature_rows(rows)
        maximum = embeddings.shape[1]
        timestamp_rows: list[torch.Tensor] = []
        for row, offset in zip(rows, offsets):
            timestamps = (
                torch.arange(
                    row.shape[0],
                    device=embeddings.device,
                    dtype=torch.float32,
                )
                * (float(self.hop_size) / float(self.sample_rate))
                + offset
            )
            timestamp_rows.append(
                F.pad(
                    timestamps,
                    (0, maximum - timestamps.shape[0]),
                )
            )
        sequence = MediaSequence(
            embeddings=embeddings,
            attention_mask=attention_mask,
            modality=MediaModality.AUDIO,
            sources=sources,
            grid=None,
            timestamps=torch.stack(timestamp_rows, dim=0),
            seconds_per_grid=None,
        )
        sequence.validate()
        return sequence

    def forward(
        self,
        items: Sequence[DecodedMedia],
    ) -> MediaSequence:
        items = _validate_decoded_items(items, MediaModality.AUDIO)
        for item in items:
            item_rate = item.metadata.get("sample_rate")
            if item_rate != self.sample_rate:
                raise ValueError(
                    "decoded audio sample_rate must match the encoder"
                )
        return self.from_waveforms(
            [item.tensor.squeeze(0) for item in items],
            sources=tuple(_source_from_item(item) for item in items),
            timeline_offsets_seconds=tuple(
                float(item.request.timeline_offset_seconds)
                for item in items
            ),
        )


__all__ = ["AudioWindowEncoder"]
