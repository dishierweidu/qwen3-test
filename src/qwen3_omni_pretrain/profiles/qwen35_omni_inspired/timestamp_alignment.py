from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from qwen3_omni_pretrain.multimodal.modalities import MediaModality
from qwen3_omni_pretrain.multimodal.positions import (
    TMRoPEConfig,
    TMRoPEPositionBuilder,
)
from qwen3_omni_pretrain.multimodal.sequence_assembler import (
    ExpandedMediaRow,
    ExpandedMediaSample,
    ExpansionToken,
    MediaExpansionGroup,
    MediaPlaceholder,
    MediaTokenRef,
)
from qwen3_omni_pretrain.multimodal.types import SequenceSpanKind


def _valid_count(placeholder: MediaPlaceholder) -> int:
    mask = placeholder.sequence.attention_mask[placeholder.sequence_row].bool()
    count = int(mask.sum().item())
    if count <= 0:
        raise ValueError("timestamp expansion requires non-empty media")
    expected = torch.arange(mask.shape[0], device=mask.device) < count
    if not torch.equal(mask, expected):
        raise ValueError("media attention mask must be a right-padded prefix")
    return count


def _token_ids(tokenizer: object, text: str) -> tuple[int, ...]:
    if callable(tokenizer):
        encoded = tokenizer(text, add_special_tokens=False)
        if isinstance(encoded, Mapping):
            encoded = encoded.get("input_ids")
    else:
        encode = getattr(tokenizer, "encode", None)
        if not callable(encode):
            raise TypeError("timestamp tokenizer must be callable or define encode")
        encoded = encode(text, add_special_tokens=False)
    if isinstance(encoded, torch.Tensor):
        encoded = encoded.detach().cpu().tolist()
    if not isinstance(encoded, (list, tuple)) or not encoded:
        raise ValueError("timestamp tokenizer must return non-empty non-negative IDs")
    if encoded and isinstance(encoded[0], (list, tuple)):
        if len(encoded) != 1:
            raise ValueError("timestamp tokenizer returned a batched result")
        encoded = encoded[0]
    if not encoded or any(
        type(token_id) is not int or token_id < 0 for token_id in encoded
    ):
        raise ValueError("timestamp tokenizer must return non-empty non-negative IDs")
    return tuple(int(token_id) for token_id in encoded)


@dataclass(frozen=True)
class _TemporalUnit:
    timestamp: float
    placeholder: MediaPlaceholder
    unit_local_index: int
    token_indices: tuple[int, ...]

    @property
    def modality_order(self) -> int:
        return {
            MediaModality.AUDIO: 0,
            MediaModality.VIDEO: 1,
            MediaModality.IMAGE: 2,
        }[self.placeholder.modality]


class Qwen35TimestampExpansionPolicy:
    """Insert ordinary timestamp text before each sample-local temporal unit."""

    __slots__ = ("tokenizer", "timestamp_format", "timestamp_images")

    def __init__(
        self,
        *,
        tokenizer: object,
        timestamp_format: str = "[{seconds:.2f}s]",
        timestamp_images: bool = False,
    ) -> None:
        if tokenizer is None:
            raise TypeError("tokenizer is required")
        if not isinstance(timestamp_format, str) or "{seconds" not in timestamp_format:
            raise ValueError("timestamp_format must contain {seconds")
        try:
            timestamp_format.format(seconds=0.0)
        except (KeyError, ValueError, IndexError) as exc:
            raise ValueError("timestamp_format is invalid") from exc
        if type(timestamp_images) is not bool:
            raise TypeError("timestamp_images must be a boolean")
        self.tokenizer = tokenizer
        self.timestamp_format = timestamp_format
        self.timestamp_images = timestamp_images

    def _units(self, placeholder: MediaPlaceholder) -> tuple[_TemporalUnit, ...]:
        count = _valid_count(placeholder)
        sequence = placeholder.sequence
        row = placeholder.sequence_row
        if placeholder.modality is MediaModality.IMAGE:
            if not self.timestamp_images:
                return ()
            return (_TemporalUnit(0.0, placeholder, 0, tuple(range(count))),)
        if sequence.timestamps is None:
            raise ValueError("audio/video timestamp expansion requires timestamps")
        timestamps = sequence.timestamps[row, :count]
        if not timestamps.is_floating_point():
            raise TypeError("audio/video timestamps must have floating dtype")
        if placeholder.modality is MediaModality.AUDIO:
            return tuple(
                _TemporalUnit(
                    float(timestamps[index].item()),
                    placeholder,
                    index,
                    (index,),
                )
                for index in range(count)
            )
        grid = sequence.grid[row] if sequence.grid is not None else None
        if grid is None:
            raise ValueError("video timestamp expansion requires a complete grid")
        spatial = grid.height * grid.width
        if grid.token_count != count:
            raise ValueError("video grid token count must match valid media tokens")
        units = []
        for frame in range(grid.temporal):
            start = frame * spatial
            end = start + spatial
            frame_timestamps = timestamps[start:end]
            if not bool((frame_timestamps == frame_timestamps[0]).all().item()):
                raise ValueError("all patches in one video frame need one timestamp")
            units.append(
                _TemporalUnit(
                    float(frame_timestamps[0].item()),
                    placeholder,
                    frame,
                    tuple(range(start, end)),
                )
            )
        return tuple(units)

    @staticmethod
    def _identity(placeholder: MediaPlaceholder) -> ExpandedMediaRow:
        source = placeholder.source
        return ExpandedMediaRow(
            tuple(
                ExpansionToken(
                    token_id=placeholder.sentinel_token_id,
                    kind=SequenceSpanKind.MEDIA,
                    media_ref=MediaTokenRef(source, index),
                    source=source,
                )
                for index in range(_valid_count(placeholder))
            )
        )

    def _timestamp_tokens(self, unit: _TemporalUnit) -> tuple[ExpansionToken, ...]:
        text = self.timestamp_format.format(seconds=unit.timestamp)
        source = unit.placeholder.source
        return tuple(
            ExpansionToken(
                token_id=token_id,
                kind=SequenceSpanKind.TIMESTAMP,
                source=source,
            )
            for token_id in _token_ids(self.tokenizer, text)
        )

    @staticmethod
    def _media_tokens(unit: _TemporalUnit) -> tuple[ExpansionToken, ...]:
        source = unit.placeholder.source
        return tuple(
            ExpansionToken(
                token_id=unit.placeholder.sentinel_token_id,
                kind=SequenceSpanKind.MEDIA,
                media_ref=MediaTokenRef(source, index),
                source=source,
            )
            for index in unit.token_indices
        )

    def _expand_group(self, group: MediaExpansionGroup) -> dict[int, ExpandedMediaRow]:
        placeholders = group.placeholders
        modalities = {placeholder.modality for placeholder in placeholders}
        if modalities == {MediaModality.IMAGE} and not self.timestamp_images:
            return {
                placeholder.text_position: self._identity(placeholder)
                for placeholder in placeholders
            }
        if MediaModality.IMAGE in modalities and len(modalities) > 1:
            raise ValueError("timestamp expansion cannot jointly mix images with AV")
        units = [unit for placeholder in placeholders for unit in self._units(placeholder)]
        units.sort(
            key=lambda unit: (
                unit.timestamp,
                unit.placeholder.source.item_index,
                unit.unit_local_index,
                unit.modality_order,
            )
        )
        tokens = tuple(
            token
            for unit in units
            for token in (*self._timestamp_tokens(unit), *self._media_tokens(unit))
        )
        replacements = {
            placeholders[0].text_position: ExpandedMediaRow(tokens),
        }
        replacements.update(
            {
                placeholder.text_position: ExpandedMediaRow(())
                for placeholder in placeholders[1:]
            }
        )
        return replacements

    def expand_sample(
        self,
        *,
        sample_index: int,
        groups: tuple[MediaExpansionGroup, ...],
    ) -> ExpandedMediaSample:
        if type(sample_index) is not int or sample_index < 0:
            raise ValueError("sample_index must be a non-negative integer")
        if not isinstance(groups, tuple):
            raise TypeError("groups must be a tuple")
        replacements: dict[int, ExpandedMediaRow] = {}
        for group in groups:
            if not isinstance(group, MediaExpansionGroup):
                raise TypeError("groups must contain MediaExpansionGroup values")
            if any(
                placeholder.source.sample_index != sample_index
                for placeholder in group.placeholders
            ):
                raise ValueError("timestamp expansion is strictly sample-local")
            additions = self._expand_group(group)
            if set(additions) & set(replacements):
                raise ValueError("timestamp expansion groups overlap")
            replacements.update(additions)
        return ExpandedMediaSample(replacements)


def build_qwen35_position_builder() -> TMRoPEPositionBuilder:
    return TMRoPEPositionBuilder(
        TMRoPEConfig(
            temporal_seconds_per_id=0.16,
            rotary_sections=(24, 20, 20),
            interleaved=True,
        )
    )


__all__ = [
    "Qwen35TimestampExpansionPolicy",
    "build_qwen35_position_builder",
]
