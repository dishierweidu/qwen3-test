from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral

import torch
from torch import nn
from torch.nn import functional as F

from qwen3_omni_pretrain.multimodal.io import DecodedMedia
from qwen3_omni_pretrain.multimodal.modalities import MediaModality
from qwen3_omni_pretrain.multimodal.types import (
    MediaGrid,
    MediaSequence,
    MediaSource,
)


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    numeric = int(value)
    if numeric <= 0:
        raise ValueError(f"{name} must be positive")
    return numeric


def _validate_sources(
    sources: object,
    expected_count: int,
) -> tuple[MediaSource, ...]:
    if not isinstance(sources, tuple):
        raise TypeError("sources must be a tuple")
    if len(sources) != expected_count:
        raise ValueError(
            "source count must equal the media batch size"
        )
    if any(not isinstance(source, MediaSource) for source in sources):
        raise TypeError("sources must contain MediaSource values")
    item_keys = {
        (source.sample_index, source.item_index) for source in sources
    }
    source_keys = {
        (source.sample_index, source.source_id) for source in sources
    }
    if len(item_keys) != len(sources) or len(source_keys) != len(sources):
        raise ValueError("sources must be unique within each sample")
    return sources


def _validate_decoded_items(
    items: object,
    modality: MediaModality,
) -> tuple[DecodedMedia, ...]:
    if (
        not isinstance(items, Sequence)
        or isinstance(items, (str, bytes, bytearray))
    ):
        raise TypeError("items must be a sequence of DecodedMedia")
    if not items:
        raise ValueError("items must be non-empty")
    validated: list[DecodedMedia] = []
    for item in items:
        if not isinstance(item, DecodedMedia):
            raise TypeError("items must contain DecodedMedia values")
        item.__post_init__()
        if item.request.modality is not modality:
            raise ValueError(
                f"expected {modality.value} modality items"
            )
        validated.append(item)
    return tuple(validated)


def _validate_pixel_batch(
    pixels: object,
    *,
    in_channels: int,
) -> torch.Tensor:
    if not isinstance(pixels, torch.Tensor):
        raise TypeError("pixels must be a torch.Tensor")
    if pixels.ndim != 4:
        raise ValueError("pixels must have shape [B, C, H, W]")
    if pixels.shape[0] <= 0:
        raise ValueError("pixels must contain a non-empty batch")
    if pixels.shape[1] != in_channels:
        raise ValueError(
            f"pixel channel count must equal in_channels={in_channels}"
        )
    if pixels.shape[2] <= 0 or pixels.shape[3] <= 0:
        raise ValueError("pixel height and width must be positive")
    if not pixels.is_floating_point():
        raise TypeError("pixels must have a floating-point dtype")
    if not bool(torch.isfinite(pixels).all().item()):
        raise ValueError("pixels must contain finite values")
    return pixels


def _pack_feature_rows(
    rows: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    maximum = max(row.shape[0] for row in rows)
    embeddings = torch.stack(
        [
            F.pad(row, (0, 0, 0, maximum - row.shape[0]))
            for row in rows
        ],
        dim=0,
    )
    positions = torch.arange(maximum, device=embeddings.device)
    lengths = torch.tensor(
        [row.shape[0] for row in rows],
        device=embeddings.device,
    )
    attention_mask = positions.unsqueeze(0) < lengths.unsqueeze(1)
    return embeddings, attention_mask


def _source_from_item(item: DecodedMedia) -> MediaSource:
    request = item.request
    return MediaSource(
        request.sample_index,
        request.item_index,
        request.source_id,
    )


class PatchVisionEncoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        hidden_size: int,
        patch_size: int,
    ) -> None:
        super().__init__()
        self.in_channels = _positive_integer(
            in_channels,
            "in_channels",
        )
        self.hidden_size = _positive_integer(
            hidden_size,
            "hidden_size",
        )
        self.patch_size = _positive_integer(
            patch_size,
            "patch_size",
        )
        self.patch_embed = nn.Conv2d(
            self.in_channels,
            self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )
        self.norm = nn.LayerNorm(self.hidden_size)

    def _encode_single(
        self,
        pixels: torch.Tensor,
    ) -> tuple[torch.Tensor, MediaGrid]:
        if pixels.ndim != 3:
            raise ValueError("one image must have shape [C, H, W]")
        if pixels.shape[0] != self.in_channels:
            raise ValueError(
                "image channel count does not match the patch encoder"
            )
        if pixels.shape[1] <= 0 or pixels.shape[2] <= 0:
            raise ValueError("image height and width must be positive")
        if not pixels.is_floating_point():
            raise TypeError("image pixels must have a floating-point dtype")
        if not bool(torch.isfinite(pixels).all().item()):
            raise ValueError("image pixels must contain finite values")

        weight = self.patch_embed.weight
        pixels = pixels.to(
            device=weight.device,
            dtype=weight.dtype,
        )
        height, width = pixels.shape[-2:]
        pad_height = (-height) % self.patch_size
        pad_width = (-width) % self.patch_size
        padded = F.pad(
            pixels,
            (0, pad_width, 0, pad_height),
        ).unsqueeze(0)
        features = self.patch_embed(padded)
        grid = MediaGrid(
            1,
            features.shape[2],
            features.shape[3],
        )
        tokens = features.flatten(2).transpose(1, 2).squeeze(0)
        return self.norm(tokens), grid

    def encode_images(self, pixels: torch.Tensor) -> torch.Tensor:
        pixels = _validate_pixel_batch(
            pixels,
            in_channels=self.in_channels,
        )
        rows = [self._encode_single(image)[0] for image in pixels]
        return torch.stack(rows, dim=0)

    @staticmethod
    def _sequence(
        rows: Sequence[torch.Tensor],
        *,
        sources: tuple[MediaSource, ...],
        grids: tuple[MediaGrid, ...],
    ) -> MediaSequence:
        embeddings, attention_mask = _pack_feature_rows(rows)
        sequence = MediaSequence(
            embeddings=embeddings,
            attention_mask=attention_mask,
            modality=MediaModality.IMAGE,
            sources=sources,
            grid=grids,
            timestamps=None,
            seconds_per_grid=None,
        )
        sequence.validate()
        return sequence

    def from_tensor_batch(
        self,
        pixels: torch.Tensor,
        *,
        sources: tuple[MediaSource, ...],
    ) -> MediaSequence:
        pixels = _validate_pixel_batch(
            pixels,
            in_channels=self.in_channels,
        )
        sources = _validate_sources(sources, pixels.shape[0])
        encoded = [self._encode_single(image) for image in pixels]
        return self._sequence(
            [row for row, _ in encoded],
            sources=sources,
            grids=tuple(grid for _, grid in encoded),
        )

    def forward(
        self,
        items: Sequence[DecodedMedia],
    ) -> MediaSequence:
        items = _validate_decoded_items(items, MediaModality.IMAGE)
        sources = _validate_sources(
            tuple(_source_from_item(item) for item in items),
            len(items),
        )
        if any(item.tensor.shape[0] != self.in_channels for item in items):
            raise ValueError(
                "image channel count does not match the patch encoder"
            )
        encoded = [self._encode_single(item.tensor) for item in items]
        return self._sequence(
            [row for row, _ in encoded],
            sources=sources,
            grids=tuple(grid for _, grid in encoded),
        )


class TemporalVideoEncoder(nn.Module):
    def __init__(self, *, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = _positive_integer(
            hidden_size,
            "hidden_size",
        )
        self.temporal_proj = nn.Linear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
        )
        self.temporal_norm = nn.LayerNorm(self.hidden_size)

    def forward(
        self,
        items: Sequence[DecodedMedia],
        *,
        patch_encoder: PatchVisionEncoder,
    ) -> MediaSequence:
        if not isinstance(patch_encoder, PatchVisionEncoder):
            raise TypeError(
                "patch_encoder must be a PatchVisionEncoder"
            )
        if patch_encoder.hidden_size != self.hidden_size:
            raise ValueError(
                "patch and temporal encoder hidden sizes must match"
            )
        items = _validate_decoded_items(items, MediaModality.VIDEO)
        sources = _validate_sources(
            tuple(_source_from_item(item) for item in items),
            len(items),
        )
        if any(
            item.tensor.shape[1] != patch_encoder.in_channels
            for item in items
        ):
            raise ValueError(
                "video channel count does not match the patch encoder"
            )

        rows: list[torch.Tensor] = []
        grids: list[MediaGrid] = []
        timestamp_rows: list[torch.Tensor] = []
        temporal_weight = self.temporal_proj.weight
        for item in items:
            encoded_frames = [
                patch_encoder._encode_single(frame)
                for frame in item.tensor
            ]
            spatial_grid = encoded_frames[0][1]
            patch_rows = [
                frame_tokens for frame_tokens, _ in encoded_frames
            ]
            features = torch.cat(patch_rows, dim=0).to(
                device=temporal_weight.device,
                dtype=temporal_weight.dtype,
            )
            features = self.temporal_norm(
                self.temporal_proj(features)
            )
            rows.append(features)
            grids.append(
                MediaGrid(
                    int(item.length),
                    spatial_grid.height,
                    spatial_grid.width,
                )
            )

            assert item.timestamps is not None
            global_timestamps = item.timestamps.to(
                device=features.device,
                dtype=torch.float32,
            ) + float(item.request.timeline_offset_seconds)
            timestamp_rows.append(
                global_timestamps.repeat_interleave(
                    spatial_grid.height * spatial_grid.width
                )
            )

        embeddings, attention_mask = _pack_feature_rows(rows)
        maximum = embeddings.shape[1]
        timestamps = torch.stack(
            [
                F.pad(row, (0, maximum - row.shape[0]))
                for row in timestamp_rows
            ],
            dim=0,
        )
        sequence = MediaSequence(
            embeddings=embeddings,
            attention_mask=attention_mask,
            modality=MediaModality.VIDEO,
            sources=sources,
            grid=tuple(grids),
            timestamps=timestamps,
            seconds_per_grid=tuple(
                item.seconds_per_grid for item in items
            ),
        )
        sequence.validate()
        return sequence


__all__ = [
    "PatchVisionEncoder",
    "TemporalVideoEncoder",
]
