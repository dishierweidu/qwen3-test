from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from numbers import Real
from typing import Protocol

import torch

from qwen3_omni_pretrain.multimodal.modalities import MediaModality
from qwen3_omni_pretrain.multimodal.time_quantization import (
    quantize_timestamps_half_up,
)
from qwen3_omni_pretrain.multimodal.types import (
    AssembledSequence,
    MediaSource,
    PositionBatch,
    SequenceSpan,
    SequenceSpanKind,
)


LEGACY_AXIS_NAMES = ("sequence",)
THREE_AXIS_NAMES = ("temporal", "height", "width")


class PositionBuilder(Protocol):
    def build(self, assembled: AssembledSequence) -> PositionBatch:
        ...


def _validate_rotary_sections(value: object) -> tuple[int, int, int]:
    if not isinstance(value, tuple):
        raise TypeError("rotary_sections must be a tuple")
    if len(value) != 3:
        raise ValueError(
            "rotary_sections must contain three positive integers"
        )
    if any(type(section) is not int for section in value):
        raise TypeError("rotary_sections must contain integers")
    if any(section <= 0 for section in value):
        raise ValueError(
            "rotary_sections must contain positive integers"
        )
    return value


def _validate_positive_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return numeric


@dataclass(frozen=True)
class Qwen3DisjointPositionConfig:
    position_id_per_seconds: float = 13.0
    rotary_sections: tuple[int, int, int] = (24, 20, 20)

    def __post_init__(self) -> None:
        _validate_positive_real(
            self.position_id_per_seconds,
            "position_id_per_seconds",
        )
        _validate_rotary_sections(self.rotary_sections)


@dataclass(frozen=True)
class TMRoPEConfig:
    temporal_seconds_per_id: float
    rotary_sections: tuple[int, int, int]
    interleaved: bool = True

    def __post_init__(self) -> None:
        _validate_positive_real(
            self.temporal_seconds_per_id,
            "temporal_seconds_per_id",
        )
        _validate_rotary_sections(self.rotary_sections)
        if type(self.interleaved) is not bool:
            raise TypeError("interleaved must be a boolean")


def _validated_prefix_lengths(
    assembled: object,
) -> tuple[AssembledSequence, tuple[int, ...]]:
    if not isinstance(assembled, AssembledSequence):
        raise TypeError("assembled must be an AssembledSequence")
    assembled.validate()
    mask = assembled.attention_mask
    if mask.ndim != 2 or mask.shape[0] == 0 or mask.shape[1] == 0:
        raise ValueError(
            "assembled attention_mask must have non-empty shape [B, S]"
        )
    lengths: list[int] = []
    for row in range(mask.shape[0]):
        row_mask = mask[row].to(dtype=torch.bool)
        length = int(torch.count_nonzero(row_mask).item())
        if length <= 0:
            raise ValueError(
                "every assembled row must contain a non-empty prefix"
            )
        expected = (
            torch.arange(mask.shape[1], device=mask.device) < length
        )
        if not torch.equal(row_mask, expected):
            raise ValueError(
                "assembled attention_mask rows must be right-padded prefixes"
            )
        lengths.append(length)
    return assembled, tuple(lengths)


def _spans_by_sample(
    assembled: AssembledSequence,
) -> dict[int, tuple[SequenceSpan, ...]]:
    grouped: dict[int, list[SequenceSpan]] = defaultdict(list)
    for span in assembled.spans:
        grouped[span.sample_index].append(span)
    return {
        sample_index: tuple(sorted(spans, key=lambda span: span.start))
        for sample_index, spans in grouped.items()
    }


def _media_spans_by_source(
    assembled: AssembledSequence,
) -> dict[MediaSource, tuple[SequenceSpan, ...]]:
    grouped: dict[MediaSource, list[SequenceSpan]] = defaultdict(list)
    for span in assembled.spans:
        if span.kind is SequenceSpanKind.MEDIA:
            assert span.source is not None
            grouped[span.source].append(span)
    return {
        source: tuple(sorted(spans, key=lambda span: span.start))
        for source, spans in grouped.items()
    }


def _validated_result(
    *,
    position_ids: torch.Tensor,
    rope_deltas: torch.Tensor,
    axis_names: tuple[str, ...],
    attention_mask: torch.Tensor,
) -> PositionBatch:
    result = PositionBatch(
        position_ids=position_ids,
        rope_deltas=rope_deltas,
        axis_names=axis_names,
    )
    result.validate(attention_mask)
    return result


class LegacyPositionBuilder:
    def build(self, assembled: AssembledSequence) -> PositionBatch:
        assembled, lengths = _validated_prefix_lengths(assembled)
        mask = assembled.attention_mask
        batch_size, sequence_length = mask.shape
        positions = torch.zeros(
            1,
            batch_size,
            sequence_length,
            dtype=torch.long,
            device=mask.device,
        )
        for row, length in enumerate(lengths):
            positions[0, row, :length] = torch.arange(
                length,
                dtype=torch.long,
                device=mask.device,
            )
            if length > 1 and bool(
                (
                    positions[0, row, 1:length]
                    < positions[0, row, : length - 1]
                )
                .any()
                .item()
            ):
                raise ValueError(
                    "legacy valid positions must be monotonic"
                )
        deltas = torch.zeros(
            batch_size,
            1,
            dtype=torch.long,
            device=mask.device,
        )
        return _validated_result(
            position_ids=positions,
            rope_deltas=deltas,
            axis_names=LEGACY_AXIS_NAMES,
            attention_mask=mask,
        )


def _validate_qwen_media_sources(
    assembled: AssembledSequence,
) -> None:
    for source, spans in _media_spans_by_source(assembled).items():
        if len(spans) != 1:
            raise ValueError(
                "Qwen3 disjoint positions require one complete MEDIA span "
                f"per source: {source.source_id}"
            )
        span = spans[0]
        indices = span.source_token_indices
        assert indices is not None
        expected_count = (
            span.grid.token_count
            if span.grid is not None
            else span.end - span.start
        )
        if indices != tuple(range(expected_count)):
            raise ValueError(
                "Qwen3 disjoint source indices must be complete and "
                "canonical"
            )
        if span.modality is MediaModality.IMAGE:
            if span.grid is None or span.grid.temporal != 1:
                raise ValueError(
                    "Qwen3 images require a temporal-one complete grid"
                )
        elif span.modality is MediaModality.VIDEO:
            if span.grid is None:
                raise ValueError("Qwen3 videos require a complete grid")
            if (
                span.grid.temporal > 1
                and span.seconds_per_grid is None
            ):
                raise ValueError(
                    "multi-frame Qwen3 video requires seconds_per_grid"
                )
        elif span.modality is MediaModality.AUDIO:
            if span.grid is not None:
                raise ValueError("Qwen3 audio must not carry a grid")
        else:  # pragma: no cover - SequenceSpan validates the enum.
            raise ValueError("unsupported Qwen3 media modality")


class Qwen3DisjointPositionBuilder:
    def __init__(
        self,
        config: Qwen3DisjointPositionConfig,
    ) -> None:
        if not isinstance(config, Qwen3DisjointPositionConfig):
            raise TypeError(
                "config must be Qwen3DisjointPositionConfig"
            )
        self.config = config

    def _media_block(
        self,
        *,
        span: SequenceSpan,
        continuation: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        length = span.end - span.start
        if span.modality is MediaModality.AUDIO:
            sequential = continuation + torch.arange(
                length,
                dtype=torch.float32,
                device=device,
            )
            return sequential.unsqueeze(0).expand(3, -1)

        grid = span.grid
        indices = span.source_token_indices
        assert grid is not None
        assert indices is not None
        source_indices = torch.tensor(
            indices,
            dtype=torch.long,
            device=device,
        )
        spatial_size = grid.height * grid.width
        frame = torch.div(
            source_indices,
            spatial_size,
            rounding_mode="floor",
        )
        within_frame = source_indices % spatial_size
        height = torch.div(
            within_frame,
            grid.width,
            rounding_mode="floor",
        )
        width = within_frame % grid.width

        if span.modality is MediaModality.IMAGE:
            temporal = continuation + frame.to(torch.float32)
        else:
            seconds_value = (
                1.0
                if span.seconds_per_grid is None
                else span.seconds_per_grid
            )
            seconds = torch.tensor(
                seconds_value,
                dtype=torch.float32,
                device=device,
            )
            rate = torch.tensor(
                self.config.position_id_per_seconds,
                dtype=torch.float32,
                device=device,
            )
            temporal = frame.to(torch.float32) * seconds
            temporal = temporal * rate
            temporal = temporal + continuation
        return torch.stack(
            (
                temporal,
                continuation + height.to(torch.float32),
                continuation + width.to(torch.float32),
            )
        )

    def build(self, assembled: AssembledSequence) -> PositionBatch:
        assembled, lengths = _validated_prefix_lengths(assembled)
        _validate_qwen_media_sources(assembled)
        mask = assembled.attention_mask
        batch_size, sequence_length = mask.shape
        spans_by_sample = _spans_by_sample(assembled)
        batch_has_media = any(
            span.kind is SequenceSpanKind.MEDIA
            for span in assembled.spans
        )
        positions = torch.zeros(
            3,
            batch_size,
            sequence_length,
            dtype=torch.float32,
            device=mask.device,
        )
        deltas = torch.zeros(
            batch_size,
            1,
            dtype=torch.float32,
            device=mask.device,
        )
        one = torch.tensor(1.0, dtype=torch.float32, device=mask.device)

        if not batch_has_media:
            for row, length in enumerate(lengths):
                sequential = torch.arange(
                    length,
                    dtype=torch.float32,
                    device=mask.device,
                )
                positions[:, row, :length] = sequential.unsqueeze(0)
                raw_max = sequential[-1]
                if length < sequence_length:
                    raw_max = torch.maximum(raw_max, one)
                deltas[row, 0] = (
                    raw_max
                    + one
                    - torch.tensor(
                        length,
                        dtype=torch.float32,
                        device=mask.device,
                    )
                )
            return _validated_result(
                position_ids=positions,
                rope_deltas=deltas,
                axis_names=THREE_AXIS_NAMES,
                attention_mask=mask,
            )

        for row, length in enumerate(lengths):
            maximum = torch.tensor(
                -1.0,
                dtype=torch.float32,
                device=mask.device,
            )
            for span in spans_by_sample[row]:
                continuation = maximum + one
                span_length = span.end - span.start
                if span.kind in (
                    SequenceSpanKind.TEXT,
                    SequenceSpanKind.TIMESTAMP,
                ):
                    sequential = continuation + torch.arange(
                        span_length,
                        dtype=torch.float32,
                        device=mask.device,
                    )
                    block = sequential.unsqueeze(0).expand(3, -1)
                else:
                    block = self._media_block(
                        span=span,
                        continuation=continuation,
                        device=mask.device,
                    )
                positions[:, row, span.start : span.end] = block
                maximum = torch.maximum(maximum, block.max())
            deltas[row, 0] = (
                maximum
                + one
                - torch.tensor(
                    length,
                    dtype=torch.float32,
                    device=mask.device,
                )
            )
        return _validated_result(
            position_ids=positions,
            rope_deltas=deltas,
            axis_names=THREE_AXIS_NAMES,
            attention_mask=mask,
        )


@dataclass(frozen=True)
class _TMSourceInfo:
    modality: MediaModality
    spans: tuple[SequenceSpan, ...]
    token_buckets: torch.LongTensor | None


def _validate_tm_sources(
    assembled: AssembledSequence,
    config: TMRoPEConfig,
) -> dict[MediaSource, _TMSourceInfo]:
    result: dict[MediaSource, _TMSourceInfo] = {}
    for source, spans in _media_spans_by_source(assembled).items():
        first = spans[0]
        modality = first.modality
        assert modality is not None
        if modality is MediaModality.IMAGE:
            if len(spans) != 1:
                raise ValueError(
                    "experimental TM-RoPE rejects split image sources"
                )
            if (
                first.grid is None
                or first.grid.temporal != 1
                or first.timestamps is not None
            ):
                raise ValueError(
                    "experimental images require an untimestamped "
                    "temporal-one grid"
                )
            result[source] = _TMSourceInfo(modality, spans, None)
            continue

        if modality is MediaModality.AUDIO and first.grid is not None:
            raise ValueError("experimental audio must not carry a grid")
        if modality is MediaModality.VIDEO and first.grid is None:
            raise ValueError("experimental video requires a complete grid")
        if any(span.timestamps is None for span in spans):
            raise ValueError(
                "experimental audio/video requires explicit timestamps"
            )
        timestamp_parts = [
            span.timestamps
            for span in spans
            if span.timestamps is not None
        ]
        timestamps = torch.cat(timestamp_parts)
        if not timestamps.is_floating_point():
            raise TypeError(
                "experimental timestamps must have floating dtype"
            )
        if (
            timestamps.numel() > 1
            and bool(
                (timestamps[1:] < timestamps[:-1]).any().item()
            )
        ):
            raise ValueError(
                "source timestamps must be globally nondecreasing"
            )

        if modality is MediaModality.VIDEO:
            grid = first.grid
            assert grid is not None
            if timestamps.shape != (grid.token_count,):
                raise ValueError(
                    "video timestamp count must match its complete grid"
                )
            per_frame = timestamps.reshape(
                grid.temporal,
                grid.height * grid.width,
            )
            frame_timestamps = per_frame[:, 0]
            if bool(
                (per_frame != frame_timestamps.unsqueeze(1)).any().item()
            ):
                raise ValueError(
                    "all video patches in one frame must share a timestamp"
                )
            frame_buckets = quantize_timestamps_half_up(
                frame_timestamps,
                config.temporal_seconds_per_id,
            )
            token_buckets = frame_buckets.repeat_interleave(
                grid.height * grid.width
            )
        else:
            token_buckets = quantize_timestamps_half_up(
                timestamps,
                config.temporal_seconds_per_id,
            )
        result[source] = _TMSourceInfo(
            modality,
            spans,
            token_buckets,
        )
    return result


class TMRoPEPositionBuilder:
    def __init__(self, config: TMRoPEConfig) -> None:
        if not isinstance(config, TMRoPEConfig):
            raise TypeError("config must be TMRoPEConfig")
        self.config = config

    def build(self, assembled: AssembledSequence) -> PositionBatch:
        assembled, lengths = _validated_prefix_lengths(assembled)
        source_info = _validate_tm_sources(assembled, self.config)
        mask = assembled.attention_mask
        batch_size, sequence_length = mask.shape
        spans_by_sample = _spans_by_sample(assembled)
        positions = torch.zeros(
            3,
            batch_size,
            sequence_length,
            dtype=torch.float32,
            device=mask.device,
        )
        deltas = torch.zeros(
            batch_size,
            1,
            dtype=torch.float32,
            device=mask.device,
        )
        one = torch.tensor(1.0, dtype=torch.float32, device=mask.device)

        for row, length in enumerate(lengths):
            text_cursor = torch.tensor(
                0.0,
                dtype=torch.float32,
                device=mask.device,
            )
            maximum = torch.tensor(
                -1.0,
                dtype=torch.float32,
                device=mask.device,
            )
            timeline_anchor: torch.Tensor | None = None
            for span in spans_by_sample[row]:
                continuation = torch.maximum(
                    text_cursor,
                    maximum + one,
                )
                span_length = span.end - span.start
                if span.kind in (
                    SequenceSpanKind.TEXT,
                    SequenceSpanKind.TIMESTAMP,
                ):
                    sequential = continuation + torch.arange(
                        span_length,
                        dtype=torch.float32,
                        device=mask.device,
                    )
                    block = sequential.unsqueeze(0).expand(3, -1)
                    text_cursor = continuation + torch.tensor(
                        span_length,
                        dtype=torch.float32,
                        device=mask.device,
                    )
                else:
                    source = span.source
                    assert source is not None
                    info = source_info[source]
                    indices = span.source_token_indices
                    assert indices is not None
                    source_indices = torch.tensor(
                        indices,
                        dtype=torch.long,
                        device=mask.device,
                    )
                    if info.modality is MediaModality.IMAGE:
                        grid = span.grid
                        assert grid is not None
                        height = torch.div(
                            source_indices,
                            grid.width,
                            rounding_mode="floor",
                        )
                        width = source_indices % grid.width
                        block = torch.stack(
                            (
                                continuation.expand(span_length),
                                continuation
                                + height.to(torch.float32),
                                continuation
                                + width.to(torch.float32),
                            )
                        )
                    else:
                        if timeline_anchor is None:
                            timeline_anchor = continuation
                        buckets = info.token_buckets
                        assert buckets is not None
                        selected = buckets.index_select(
                            0,
                            source_indices,
                        ).to(dtype=torch.float32)
                        temporal = timeline_anchor + selected
                        if info.modality is MediaModality.AUDIO:
                            block = temporal.unsqueeze(0).expand(3, -1)
                        else:
                            grid = span.grid
                            assert grid is not None
                            spatial_size = grid.height * grid.width
                            within_frame = source_indices % spatial_size
                            height = torch.div(
                                within_frame,
                                grid.width,
                                rounding_mode="floor",
                            )
                            width = within_frame % grid.width
                            block = torch.stack(
                                (
                                    temporal,
                                    timeline_anchor
                                    + height.to(torch.float32),
                                    timeline_anchor
                                    + width.to(torch.float32),
                                )
                            )
                positions[:, row, span.start : span.end] = block
                maximum = torch.maximum(maximum, block.max())
            deltas[row, 0] = (
                maximum
                + one
                - torch.tensor(
                    length,
                    dtype=torch.float32,
                    device=mask.device,
                )
            )
        return _validated_result(
            position_ids=positions,
            rope_deltas=deltas,
            axis_names=THREE_AXIS_NAMES,
            attention_mask=mask,
        )


__all__ = [
    "LEGACY_AXIS_NAMES",
    "THREE_AXIS_NAMES",
    "LegacyPositionBuilder",
    "PositionBuilder",
    "Qwen3DisjointPositionBuilder",
    "Qwen3DisjointPositionConfig",
    "TMRoPEConfig",
    "TMRoPEPositionBuilder",
]
