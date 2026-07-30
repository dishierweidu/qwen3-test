from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from numbers import Real

import torch

from qwen3_omni_pretrain.multimodal.modalities import MediaModality


_INTEGER_DTYPES = frozenset(
    {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
)


def _require_tensor(value: object, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    return value


def _is_mask_dtype(dtype: torch.dtype) -> bool:
    return dtype is torch.bool or dtype in _INTEGER_DTYPES


def _is_numeric_non_complex(tensor: torch.Tensor) -> bool:
    return tensor.is_floating_point() or tensor.dtype in _INTEGER_DTYPES


def _validate_mask_dtype(mask: torch.Tensor, name: str) -> None:
    if not _is_mask_dtype(mask.dtype):
        raise TypeError(f"{name} must have a boolean or integer dtype")


def _validate_positive_number(value: object, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise ValueError(f"{name} must be a finite positive number")


@dataclass(frozen=True)
class MediaSource:
    sample_index: int
    item_index: int
    source_id: str

    def __post_init__(self) -> None:
        for name in ("sample_index", "item_index"):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if not isinstance(self.source_id, str):
            raise TypeError("source_id must be a string")
        if not self.source_id.strip():
            raise ValueError("source_id must be non-empty")


@dataclass(frozen=True)
class MediaGrid:
    temporal: int
    height: int
    width: int

    def __post_init__(self) -> None:
        for name in ("temporal", "height", "width"):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def token_count(self) -> int:
        return self.temporal * self.height * self.width


@dataclass(frozen=True)
class MediaSequence:
    embeddings: torch.Tensor
    attention_mask: torch.Tensor
    modality: MediaModality
    sources: tuple[MediaSource, ...]
    grid: tuple[MediaGrid | None, ...] | None = None
    timestamps: torch.Tensor | None = None
    seconds_per_grid: tuple[float | None, ...] | None = None

    def validate(self) -> None:
        embeddings = _require_tensor(self.embeddings, "embeddings")
        attention_mask = _require_tensor(
            self.attention_mask, "attention_mask"
        )
        if embeddings.ndim != 3:
            raise ValueError("embeddings must have shape [B, M, H]")
        if attention_mask.shape != embeddings.shape[:2]:
            raise ValueError("attention_mask must have shape [B, M]")
        _validate_mask_dtype(attention_mask, "attention_mask")
        if not isinstance(self.modality, MediaModality):
            raise TypeError("modality must be MediaModality")
        if not isinstance(self.sources, tuple):
            raise TypeError("sources must be a tuple")

        batch_size = embeddings.shape[0]
        if len(self.sources) != batch_size:
            raise ValueError("one MediaSource is required per batch row")
        if any(
            not isinstance(source, MediaSource) for source in self.sources
        ):
            raise TypeError("sources must contain MediaSource values")

        item_keys = {
            (source.sample_index, source.item_index)
            for source in self.sources
        }
        if len(item_keys) != len(self.sources):
            raise ValueError(
                "(sample_index, item_index) must be unique batch-wide"
            )
        source_keys = {
            (source.sample_index, source.source_id)
            for source in self.sources
        }
        if len(source_keys) != len(self.sources):
            raise ValueError(
                "(sample_index, source_id) must be unique within a sample"
            )

        if self.grid is None:
            if self.modality in (
                MediaModality.IMAGE,
                MediaModality.VIDEO,
            ):
                raise ValueError(
                    "grid is required for every image/video batch row"
                )
        else:
            if not isinstance(self.grid, tuple):
                raise TypeError("grid must be a tuple")
            if len(self.grid) != batch_size:
                raise ValueError("grid tuple length must equal batch size")
            for row, grid in enumerate(self.grid):
                if grid is not None and not isinstance(grid, MediaGrid):
                    raise TypeError(
                        "grid entries must be MediaGrid or None"
                    )
                if self.modality in (
                    MediaModality.IMAGE,
                    MediaModality.VIDEO,
                ):
                    if grid is None:
                        raise ValueError(
                            "grid is required for every image/video "
                            "batch row"
                        )
                    valid_token_count = int(
                        torch.count_nonzero(attention_mask[row]).item()
                    )
                    if grid.token_count != valid_token_count:
                        raise ValueError(
                            "grid token count must equal valid tokens "
                            "for every image/video row"
                        )

        if self.seconds_per_grid is not None:
            if not isinstance(self.seconds_per_grid, tuple):
                raise TypeError("seconds_per_grid must be a tuple")
            if len(self.seconds_per_grid) != batch_size:
                raise ValueError(
                    "seconds_per_grid tuple length must equal batch size"
                )
            for value in self.seconds_per_grid:
                if value is not None:
                    _validate_positive_number(value, "seconds_per_grid")

        if self.timestamps is not None:
            timestamps = _require_tensor(self.timestamps, "timestamps")
            if timestamps.shape != embeddings.shape[:2]:
                raise ValueError("timestamps must have shape [B, M]")
            if not _is_numeric_non_complex(timestamps):
                raise TypeError(
                    "timestamps must have a numeric non-complex dtype"
                )
            valid_mask = attention_mask.to(
                device=timestamps.device, dtype=torch.bool
            )
            valid_timestamps = timestamps.masked_select(valid_mask)
            if not bool(torch.isfinite(valid_timestamps).all().item()):
                raise ValueError("valid timestamps must be finite")
            if bool((valid_timestamps < 0).any().item()):
                raise ValueError("valid timestamps must be non-negative")
            for row in range(batch_size):
                row_timestamps = timestamps[row].masked_select(
                    valid_mask[row]
                )
                if (
                    row_timestamps.numel() > 1
                    and bool(
                        (row_timestamps[1:] < row_timestamps[:-1])
                        .any()
                        .item()
                    )
                ):
                    raise ValueError(
                        "valid timestamps must be monotonic within a row"
                    )


class SequenceSpanKind(str, Enum):
    TEXT = "text"
    MEDIA = "media"
    TIMESTAMP = "timestamp"


@dataclass(frozen=True)
class SequenceSpan:
    sample_index: int
    start: int
    end: int
    kind: SequenceSpanKind
    modality: MediaModality | None
    grid: MediaGrid | None
    timestamps: torch.Tensor | None
    seconds_per_grid: float | None
    source: MediaSource | None
    source_token_indices: tuple[int, ...] | None

    def validate(self) -> None:
        if type(self.sample_index) is not int:
            raise TypeError("span sample_index must be an integer")
        if self.sample_index < 0:
            raise ValueError("span sample_index must be non-negative")
        if type(self.start) is not int or type(self.end) is not int:
            raise TypeError("span start and end must be integers")
        if self.start < 0 or self.end <= self.start:
            raise ValueError(
                "span bounds must satisfy 0 <= start < end"
            )
        if not isinstance(self.kind, SequenceSpanKind):
            raise TypeError("span kind must be SequenceSpanKind")
        if (
            self.modality is not None
            and not isinstance(self.modality, MediaModality)
        ):
            raise TypeError("span modality must be MediaModality or None")
        if self.grid is not None and not isinstance(self.grid, MediaGrid):
            raise TypeError("span grid must be MediaGrid or None")
        if self.source is not None:
            if not isinstance(self.source, MediaSource):
                raise TypeError("span source must be MediaSource or None")
            if self.source.sample_index != self.sample_index:
                raise ValueError(
                    "span source sample_index must match span sample_index"
                )

        span_length = self.end - self.start
        if self.timestamps is not None:
            timestamps = _require_tensor(
                self.timestamps, "span timestamps"
            )
            if timestamps.shape != (span_length,):
                raise ValueError(
                    "span timestamps must have one value per output token"
                )
            if not _is_numeric_non_complex(timestamps):
                raise TypeError(
                    "span timestamps must have a numeric "
                    "non-complex dtype"
                )
            if not bool(torch.isfinite(timestamps).all().item()):
                raise ValueError("span timestamps must be finite")
            if bool((timestamps < 0).any().item()):
                raise ValueError("span timestamps must be non-negative")
            if (
                timestamps.numel() > 1
                and bool(
                    (timestamps[1:] < timestamps[:-1]).any().item()
                )
            ):
                raise ValueError("span timestamps must be monotonic")

        if self.kind is SequenceSpanKind.TEXT:
            if any(
                value is not None
                for value in (
                    self.modality,
                    self.grid,
                    self.timestamps,
                    self.seconds_per_grid,
                    self.source,
                    self.source_token_indices,
                )
            ):
                raise ValueError(
                    "text spans cannot contain media metadata"
                )
            return

        if self.kind is SequenceSpanKind.TIMESTAMP:
            if self.source_token_indices is not None:
                raise ValueError(
                    "timestamp spans cannot contain "
                    "source_token_indices"
                )
            if self.seconds_per_grid is not None:
                raise ValueError(
                    "seconds_per_grid is only valid for video media "
                    "spans"
                )
            return

        if self.source is None:
            raise ValueError("media spans require a complete source")
        if self.modality is None:
            raise ValueError("media spans require a modality")
        if (
            self.modality in (MediaModality.IMAGE, MediaModality.VIDEO)
            and self.grid is None
        ):
            raise ValueError(
                "image/video media spans require a complete source grid"
            )
        if not isinstance(self.source_token_indices, tuple):
            raise ValueError(
                "media spans require source_token_indices"
            )
        if len(self.source_token_indices) != span_length:
            raise ValueError(
                "source_token_indices must contain one index per "
                "output token"
            )

        previous_index: int | None = None
        for index in self.source_token_indices:
            if type(index) is not int:
                raise TypeError(
                    "source_token_indices must contain integers"
                )
            if index < 0:
                raise ValueError(
                    "source_token_indices must be non-negative"
                )
            if previous_index is not None and index <= previous_index:
                raise ValueError(
                    "source_token_indices must be strictly increasing"
                )
            if self.grid is not None and index >= self.grid.token_count:
                raise ValueError(
                    "source_token_indices must be below the complete "
                    "source grid token count"
                )
            previous_index = index

        if self.seconds_per_grid is not None:
            if self.modality is not MediaModality.VIDEO:
                raise ValueError(
                    "seconds_per_grid is only valid for video media "
                    "spans"
                )
            _validate_positive_number(
                self.seconds_per_grid, "seconds_per_grid"
            )


@dataclass(frozen=True)
class AssembledSequence:
    expanded_input_ids: torch.LongTensor
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor | None
    spans: tuple[SequenceSpan, ...]

    def validate(self) -> None:
        expanded_input_ids = _require_tensor(
            self.expanded_input_ids, "expanded_input_ids"
        )
        inputs_embeds = _require_tensor(
            self.inputs_embeds, "inputs_embeds"
        )
        attention_mask = _require_tensor(
            self.attention_mask, "attention_mask"
        )
        if expanded_input_ids.ndim != 2:
            raise ValueError(
                "expanded_input_ids must have shape [B, S]"
            )
        if expanded_input_ids.dtype is not torch.long:
            raise TypeError("expanded_input_ids must have dtype torch.long")
        batch_size, sequence_length = expanded_input_ids.shape
        if inputs_embeds.ndim != 3 or inputs_embeds.shape[:2] != (
            batch_size,
            sequence_length,
        ):
            raise ValueError("inputs_embeds must have shape [B, S, H]")
        if attention_mask.shape != (batch_size, sequence_length):
            raise ValueError("attention_mask must have shape [B, S]")
        _validate_mask_dtype(attention_mask, "attention_mask")
        if self.labels is not None:
            labels = _require_tensor(self.labels, "labels")
            if labels.shape != (batch_size, sequence_length):
                raise ValueError("labels must have shape [B, S]")
        if not isinstance(self.spans, tuple):
            raise TypeError("spans must be a tuple")

        valid_mask = attention_mask.to(dtype=torch.bool)
        coverage = torch.zeros_like(valid_mask)
        media_by_source: dict[MediaSource, list[SequenceSpan]] = {}
        item_sources: dict[tuple[int, int], MediaSource] = {}
        id_sources: dict[tuple[int, str], MediaSource] = {}

        for span in self.spans:
            if not isinstance(span, SequenceSpan):
                raise TypeError("spans must contain SequenceSpan values")
            span.validate()
            if span.sample_index >= batch_size:
                raise ValueError(
                    "span sample_index is outside the assembled batch"
                )
            if span.end > sequence_length:
                raise ValueError(
                    "span bounds are outside the assembled sequence"
                )

            row_coverage = coverage[span.sample_index, span.start : span.end]
            if bool(row_coverage.any().item()):
                raise ValueError("assembled spans must not overlap")
            row_mask = valid_mask[
                span.sample_index, span.start : span.end
            ]
            if not bool(row_mask.all().item()):
                raise ValueError(
                    "assembled spans cannot cover masked positions"
                )
            coverage[
                span.sample_index, span.start : span.end
            ] = True

            if span.kind is SequenceSpanKind.MEDIA:
                assert span.source is not None
                media_by_source.setdefault(span.source, []).append(span)
                item_key = (
                    span.source.sample_index,
                    span.source.item_index,
                )
                if (
                    item_key in item_sources
                    and item_sources[item_key] != span.source
                ):
                    raise ValueError(
                        "one complete source is required per "
                        "(sample_index, item_index)"
                    )
                item_sources[item_key] = span.source
                id_key = (
                    span.source.sample_index,
                    span.source.source_id,
                )
                if (
                    id_key in id_sources
                    and id_sources[id_key] != span.source
                ):
                    raise ValueError(
                        "one complete source is required per "
                        "(sample_index, source_id)"
                    )
                id_sources[id_key] = span.source

        if not torch.equal(coverage, valid_mask):
            raise ValueError(
                "assembled span coverage must equal all valid tokens"
            )

        for source, source_spans in media_by_source.items():
            ordered_spans = sorted(source_spans, key=lambda span: span.start)
            complete_grid = ordered_spans[0].grid
            modality = ordered_spans[0].modality
            seconds_per_grid = ordered_spans[0].seconds_per_grid
            if any(span.grid != complete_grid for span in ordered_spans):
                raise ValueError(
                    "all fragments must retain the same complete source grid"
                )
            if any(span.modality is not modality for span in ordered_spans):
                raise ValueError(
                    "all fragments from one source must retain its modality"
                )
            if any(
                span.seconds_per_grid != seconds_per_grid
                for span in ordered_spans
            ):
                raise ValueError(
                    "all fragments from one source must retain "
                    "seconds_per_grid"
                )

            indices = tuple(
                index
                for span in ordered_spans
                for index in (span.source_token_indices or ())
            )
            valid_token_count = (
                complete_grid.token_count
                if complete_grid is not None
                else len(indices)
            )
            if indices != tuple(range(valid_token_count)):
                raise ValueError(
                    "media source indices must preserve complete "
                    "row-major coverage"
                )


@dataclass(frozen=True)
class PositionBatch:
    position_ids: torch.Tensor
    rope_deltas: torch.Tensor
    axis_names: tuple[str, ...]

    def validate(self, attention_mask: torch.Tensor) -> None:
        position_ids = _require_tensor(self.position_ids, "position_ids")
        rope_deltas = _require_tensor(self.rope_deltas, "rope_deltas")
        attention_mask = _require_tensor(
            attention_mask, "attention_mask"
        )
        if position_ids.ndim != 3:
            raise ValueError(
                "position_ids must have shape [axes, B, S]"
            )
        axis_count, batch_size, sequence_length = position_ids.shape
        if attention_mask.shape != (batch_size, sequence_length):
            raise ValueError("attention_mask must have shape [B, S]")
        _validate_mask_dtype(attention_mask, "attention_mask")
        if rope_deltas.shape != (batch_size, 1):
            raise ValueError("rope_deltas must have shape [B, 1]")
        if (
            position_ids.dtype != rope_deltas.dtype
            or not _is_numeric_non_complex(position_ids)
            or not _is_numeric_non_complex(rope_deltas)
        ):
            raise TypeError(
                "position_ids and rope_deltas must have matching "
                "numeric non-complex dtypes"
            )
        if not isinstance(self.axis_names, tuple):
            raise TypeError("axis_names must be a tuple")
        if (
            axis_count == 0
            or len(self.axis_names) != axis_count
            or len(set(self.axis_names)) != axis_count
            or any(
                not isinstance(name, str) or not name
                for name in self.axis_names
            )
        ):
            raise ValueError(
                "axis_names must be unique and match the axis count"
            )
        if (
            not bool(torch.isfinite(position_ids).all().item())
            or not bool(torch.isfinite(rope_deltas).all().item())
        ):
            raise ValueError(
                "position_ids and rope_deltas must contain finite values"
            )

        valid_mask = (
            attention_mask.to(
                device=position_ids.device, dtype=torch.bool
            )
            .unsqueeze(0)
            .expand_as(position_ids)
        )
        if bool(
            (position_ids.masked_select(valid_mask) < 0).any().item()
        ):
            raise ValueError("valid position_ids must be non-negative")
        if bool(
            (
                position_ids.masked_select(~valid_mask)
                != 0
            )
            .any()
            .item()
        ):
            raise ValueError(
                "position_ids must be zero at masked positions"
            )


__all__ = [
    "AssembledSequence",
    "MediaGrid",
    "MediaSequence",
    "MediaSource",
    "PositionBatch",
    "SequenceSpan",
    "SequenceSpanKind",
]
