from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
import math
from numbers import Real
from types import MappingProxyType
from typing import Protocol

import torch
from torch.nn import functional as F

from qwen3_omni_pretrain.multimodal.modalities import MediaModality
from qwen3_omni_pretrain.multimodal.time_quantization import (
    quantize_timestamps_half_up,
)
from qwen3_omni_pretrain.multimodal.tokenization.schema import (
    ResolvedMultimodalTokens,
)
from qwen3_omni_pretrain.multimodal.types import (
    AssembledSequence,
    MediaGrid,
    MediaSequence,
    MediaSource,
    SequenceSpan,
    SequenceSpanKind,
)


_INTEGER_DTYPES = frozenset(
    {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
)


def _exact_non_negative_integer(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


@dataclass(frozen=True)
class MediaTokenRef:
    source: MediaSource
    token_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.source, MediaSource):
            raise TypeError("media ref source must be MediaSource")
        _exact_non_negative_integer(
            self.token_index,
            "media ref token_index",
        )


@dataclass(frozen=True)
class MediaPlaceholder:
    text_position: int
    sentinel_token_id: int
    modality: MediaModality
    sequence: MediaSequence
    sequence_row: int

    def __post_init__(self) -> None:
        _exact_non_negative_integer(
            self.text_position,
            "placeholder text_position",
        )
        _exact_non_negative_integer(
            self.sentinel_token_id,
            "placeholder sentinel_token_id",
        )
        if not isinstance(self.modality, MediaModality):
            raise TypeError("placeholder modality must be MediaModality")
        if not isinstance(self.sequence, MediaSequence):
            raise TypeError("placeholder sequence must be MediaSequence")
        _exact_non_negative_integer(
            self.sequence_row,
            "placeholder sequence_row",
        )
        if self.sequence_row >= self.sequence.embeddings.shape[0]:
            raise ValueError("placeholder sequence_row is out of range")

    @property
    def source(self) -> MediaSource:
        return self.sequence.sources[self.sequence_row]


@dataclass(frozen=True)
class MediaExpansionGroup:
    placeholders: tuple[MediaPlaceholder, ...]
    first_text_position: int
    last_text_position: int

    def __post_init__(self) -> None:
        if not isinstance(self.placeholders, tuple):
            raise TypeError("group placeholders must be a tuple")
        if (
            not self.placeholders
            or any(
                not isinstance(value, MediaPlaceholder)
                for value in self.placeholders
            )
        ):
            raise ValueError(
                "group placeholders must contain at least one placeholder"
            )
        _exact_non_negative_integer(
            self.first_text_position,
            "group first_text_position",
        )
        _exact_non_negative_integer(
            self.last_text_position,
            "group last_text_position",
        )
        if (
            self.first_text_position
            != self.placeholders[0].text_position
            or self.last_text_position
            != self.placeholders[-1].text_position
        ):
            raise ValueError(
                "group bounds must match its first and last placeholders"
            )


@dataclass(frozen=True)
class ExpansionToken:
    token_id: int
    kind: SequenceSpanKind
    media_ref: MediaTokenRef | None = None
    source: MediaSource | None = None

    def __post_init__(self) -> None:
        _exact_non_negative_integer(
            self.token_id,
            "expansion token_id",
        )
        if not isinstance(self.kind, SequenceSpanKind):
            raise TypeError("expansion kind must be SequenceSpanKind")
        if self.media_ref is not None and not isinstance(
            self.media_ref,
            MediaTokenRef,
        ):
            raise TypeError("media_ref must be MediaTokenRef or None")
        if self.source is not None and not isinstance(
            self.source,
            MediaSource,
        ):
            raise TypeError("source must be MediaSource or None")


@dataclass(frozen=True)
class ExpandedMediaRow:
    tokens: tuple[ExpansionToken, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.tokens, tuple):
            raise TypeError("expanded row tokens must be a tuple")
        if any(
            not isinstance(token, ExpansionToken)
            for token in self.tokens
        ):
            raise TypeError(
                "expanded row tokens must contain ExpansionToken values"
            )


@dataclass(frozen=True)
class ExpandedMediaSample:
    replacements: Mapping[int, ExpandedMediaRow]

    def __post_init__(self) -> None:
        if not isinstance(self.replacements, Mapping):
            raise TypeError("replacements must be a mapping")


class EmbeddingLookup(Protocol):
    def __call__(self, token_ids: torch.LongTensor) -> torch.Tensor:
        ...


class MediaExpansionPolicy(Protocol):
    def expand_sample(
        self,
        *,
        sample_index: int,
        groups: tuple[MediaExpansionGroup, ...],
    ) -> ExpandedMediaSample:
        ...


class AssembledLengthError(ValueError):
    def __init__(
        self,
        *,
        sample_index: int,
        assembled_length: int,
        max_assembled_length: int,
        retained_text_tokens: int,
        inserted_expansion_tokens: int,
    ) -> None:
        self.sample_index = sample_index
        self.assembled_length = assembled_length
        self.max_assembled_length = max_assembled_length
        self.retained_text_tokens = retained_text_tokens
        self.inserted_expansion_tokens = inserted_expansion_tokens
        super().__init__(
            "assembled length exceeds hard limit: "
            f"sample_index={sample_index}, "
            f"assembled_length={assembled_length}, "
            f"max_assembled_length={max_assembled_length}, "
            f"retained_text_tokens={retained_text_tokens}, "
            "inserted_expansion_tokens="
            f"{inserted_expansion_tokens}"
        )


def _prefix_count(mask: torch.Tensor, name: str) -> int:
    values = mask.to(dtype=torch.bool)
    count = int(torch.count_nonzero(values).item())
    if count <= 0:
        raise ValueError(f"{name} must contain a non-empty prefix")
    expected = (
        torch.arange(values.shape[0], device=values.device) < count
    )
    if not torch.equal(values, expected):
        raise ValueError(f"{name} must be a right-padded prefix mask")
    return count


def _identity_row(
    placeholder: MediaPlaceholder,
) -> ExpandedMediaRow:
    source = placeholder.source
    count = _prefix_count(
        placeholder.sequence.attention_mask[
            placeholder.sequence_row
        ],
        "media attention_mask row",
    )
    return ExpandedMediaRow(
        tuple(
            ExpansionToken(
                token_id=placeholder.sentinel_token_id,
                kind=SequenceSpanKind.MEDIA,
                media_ref=MediaTokenRef(source, token_index),
                source=source,
            )
            for token_index in range(count)
        )
    )


class IdentityMediaExpansion:
    def expand_sample(
        self,
        *,
        sample_index: int,
        groups: tuple[MediaExpansionGroup, ...],
    ) -> ExpandedMediaSample:
        _exact_non_negative_integer(sample_index, "sample_index")
        if not isinstance(groups, tuple):
            raise TypeError("groups must be a tuple")
        replacements: dict[int, ExpandedMediaRow] = {}
        for group in groups:
            if not isinstance(group, MediaExpansionGroup):
                raise TypeError(
                    "groups must contain MediaExpansionGroup values"
                )
            for placeholder in group.placeholders:
                replacements[placeholder.text_position] = _identity_row(
                    placeholder
                )
        return ExpandedMediaSample(replacements)


class TimestampInterleaveExpansion:
    __slots__ = ("seconds_per_bucket", "bucket_token_ids")

    def __init__(
        self,
        *,
        seconds_per_bucket: float,
        bucket_token_ids: Mapping[int, int],
    ) -> None:
        if isinstance(seconds_per_bucket, bool) or not isinstance(
            seconds_per_bucket,
            Real,
        ):
            raise TypeError("seconds_per_bucket must be a real number")
        step = float(seconds_per_bucket)
        if not math.isfinite(step) or step <= 0:
            raise ValueError(
                "seconds_per_bucket must be finite and positive"
            )
        if not isinstance(bucket_token_ids, Mapping):
            raise TypeError("bucket_token_ids must be a mapping")
        copied: dict[int, int] = {}
        for bucket, token_id in bucket_token_ids.items():
            copied[
                _exact_non_negative_integer(bucket, "bucket key")
            ] = _exact_non_negative_integer(
                token_id,
                "bucket token ID",
            )
        self.seconds_per_bucket = step
        self.bucket_token_ids = MappingProxyType(copied)

    def _expand_av_group(
        self,
        group: MediaExpansionGroup,
    ) -> dict[int, ExpandedMediaRow]:
        refs: list[MediaTokenRef] = []
        timestamps: list[torch.Tensor] = []
        sentinel_ids: list[int] = []
        for placeholder in group.placeholders:
            sequence = placeholder.sequence
            if sequence.timestamps is None:
                raise ValueError(
                    "timestamp interleave requires AV timestamps"
                )
            if not sequence.timestamps.is_floating_point():
                raise TypeError(
                    "timestamp interleave requires floating timestamps"
                )
            count = _prefix_count(
                sequence.attention_mask[placeholder.sequence_row],
                "media attention_mask row",
            )
            source = placeholder.source
            for token_index in range(count):
                refs.append(MediaTokenRef(source, token_index))
                timestamps.append(
                    sequence.timestamps[
                        placeholder.sequence_row,
                        token_index,
                    ]
                )
                sentinel_ids.append(placeholder.sentinel_token_id)

        timestamp_values = torch.stack(
            [
                timestamp.to(dtype=torch.float32)
                for timestamp in timestamps
            ]
        )
        buckets = quantize_timestamps_half_up(
            timestamp_values,
            self.seconds_per_bucket,
        )
        order = sorted(
            range(len(refs)),
            key=lambda index: (
                float(timestamp_values[index].item()),
                refs[index].source.item_index,
                refs[index].token_index,
            ),
        )

        tokens: list[ExpansionToken] = []
        previous_bucket: int | None = None
        for index in order:
            bucket = int(buckets[index].item())
            ref = refs[index]
            if bucket != previous_bucket:
                if bucket not in self.bucket_token_ids:
                    raise ValueError(
                        f"timestamp bucket {bucket} has no mapped token"
                    )
                tokens.append(
                    ExpansionToken(
                        token_id=self.bucket_token_ids[bucket],
                        kind=SequenceSpanKind.TIMESTAMP,
                        source=ref.source,
                    )
                )
                previous_bucket = bucket
            tokens.append(
                ExpansionToken(
                    token_id=sentinel_ids[index],
                    kind=SequenceSpanKind.MEDIA,
                    media_ref=ref,
                    source=ref.source,
                )
            )

        replacements = {
            group.placeholders[0].text_position: ExpandedMediaRow(
                tuple(tokens)
            )
        }
        replacements.update(
            {
                placeholder.text_position: ExpandedMediaRow(())
                for placeholder in group.placeholders[1:]
            }
        )
        return replacements

    def expand_sample(
        self,
        *,
        sample_index: int,
        groups: tuple[MediaExpansionGroup, ...],
    ) -> ExpandedMediaSample:
        _exact_non_negative_integer(sample_index, "sample_index")
        if not isinstance(groups, tuple):
            raise TypeError("groups must be a tuple")
        replacements: dict[int, ExpandedMediaRow] = {}
        for group in groups:
            if not isinstance(group, MediaExpansionGroup):
                raise TypeError(
                    "groups must contain MediaExpansionGroup values"
                )
            modalities = {
                placeholder.modality
                for placeholder in group.placeholders
            }
            has_image = MediaModality.IMAGE in modalities
            has_av = bool(
                modalities
                & {MediaModality.AUDIO, MediaModality.VIDEO}
            )
            if has_image and has_av:
                raise ValueError(
                    "one expansion group cannot mix image with AV"
                )
            if has_image:
                for placeholder in group.placeholders:
                    replacements[
                        placeholder.text_position
                    ] = _identity_row(placeholder)
            else:
                replacements.update(self._expand_av_group(group))
        return ExpandedMediaSample(replacements)


@dataclass(frozen=True)
class _SourceRecord:
    sequence: MediaSequence
    sequence_row: int
    source: MediaSource
    modality: MediaModality
    valid_count: int
    grid: MediaGrid | None
    seconds_per_grid: float | None

    def ref(self, token_index: int) -> MediaTokenRef:
        return MediaTokenRef(self.source, token_index)


@dataclass
class _SamplePlan:
    sample_index: int
    valid_length: int
    placeholders: tuple[MediaPlaceholder, ...]
    groups: tuple[MediaExpansionGroup, ...]
    expansion: ExpandedMediaSample | None = None
    retained_text_tokens: int = 0
    inserted_expansion_tokens: int = 0


@dataclass(frozen=True)
class _SymbolicAtom:
    kind: SequenceSpanKind
    token_id: int
    text_position: int | None = None
    expansion_token: ExpansionToken | None = None


@dataclass(frozen=True)
class _MaterializedAtom:
    kind: SequenceSpanKind
    token_id: int
    embedding: torch.Tensor
    label: int | None
    source: MediaSource | None = None
    record: _SourceRecord | None = None
    token_index: int | None = None


class SequenceAssembler:
    @staticmethod
    def _validate_tokens_and_scalars(
        *,
        tokens: object,
        pad_token_id: object,
        joint_separator_token_ids: object,
        max_assembled_length: object,
    ) -> tuple[
        ResolvedMultimodalTokens,
        int,
        frozenset[int],
        int,
        frozenset[int],
    ]:
        if not isinstance(tokens, ResolvedMultimodalTokens):
            raise TypeError("tokens must be ResolvedMultimodalTokens")
        resolved_values: list[int] = []
        for field in fields(ResolvedMultimodalTokens):
            value = getattr(tokens, field.name)
            if value is None:
                if field.name in {
                    "image_pad",
                    "video_pad",
                    "audio_pad",
                }:
                    raise TypeError(
                        f"tokens.{field.name} must be an integer"
                    )
                continue
            resolved_values.append(
                _exact_non_negative_integer(
                    value,
                    f"tokens.{field.name}",
                )
            )
        if len(set(resolved_values)) != len(resolved_values):
            raise ValueError(
                "all present resolved token IDs must be distinct"
            )
        sentinel_ids = frozenset(
            (tokens.image_pad, tokens.video_pad, tokens.audio_pad)
        )
        pad = _exact_non_negative_integer(
            pad_token_id,
            "pad_token_id",
        )
        if pad in sentinel_ids:
            raise ValueError(
                "pad_token_id must differ from media sentinel IDs"
            )
        if type(max_assembled_length) is not int:
            raise TypeError("max_assembled_length must be an integer")
        if max_assembled_length <= 0:
            raise ValueError("max_assembled_length must be positive")
        if type(joint_separator_token_ids) is not frozenset:
            raise TypeError(
                "joint_separator_token_ids must be a frozenset"
            )
        separators: set[int] = set()
        for value in joint_separator_token_ids:
            separators.add(
                _exact_non_negative_integer(
                    value,
                    "joint separator token ID",
                )
            )
        if separators & (set(sentinel_ids) | {pad}):
            raise ValueError(
                "joint separators must be disjoint from pad and sentinels"
            )
        return (
            tokens,
            pad,
            frozenset(separators),
            max_assembled_length,
            sentinel_ids,
        )

    @staticmethod
    def _validate_text_tensors(
        *,
        input_ids: object,
        text_embeddings: object,
        attention_mask: object,
        labels: object,
        pad_token_id: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        tuple[int, ...],
    ]:
        for value, name in (
            (input_ids, "input_ids"),
            (text_embeddings, "text_embeddings"),
            (attention_mask, "attention_mask"),
        ):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
        assert isinstance(input_ids, torch.Tensor)
        assert isinstance(text_embeddings, torch.Tensor)
        assert isinstance(attention_mask, torch.Tensor)
        if input_ids.ndim != 2 or input_ids.numel() == 0:
            raise ValueError(
                "input_ids must be non-empty with shape [B, T]"
            )
        if input_ids.dtype is not torch.long:
            raise TypeError("input_ids must have dtype torch.long")
        if bool((input_ids < 0).any().item()):
            raise ValueError("input_ids must be non-negative")
        batch_size, text_length = input_ids.shape
        if (
            text_embeddings.ndim != 3
            or text_embeddings.shape[:2]
            != (batch_size, text_length)
            or text_embeddings.shape[2] <= 0
        ):
            raise ValueError(
                "text_embeddings must have shape [B, T, H] with H > 0"
            )
        if not text_embeddings.is_floating_point():
            raise TypeError(
                "text_embeddings must have a floating non-complex dtype"
            )
        if attention_mask.shape != (batch_size, text_length):
            raise ValueError("attention_mask must have shape [B, T]")
        if (
            attention_mask.dtype is not torch.bool
            and attention_mask.dtype not in _INTEGER_DTYPES
        ):
            raise TypeError(
                "attention_mask must have boolean or integer dtype"
            )
        if not bool(
            (
                (attention_mask == 0) | (attention_mask == 1)
            )
            .all()
            .item()
        ):
            raise ValueError("attention_mask values must be binary")
        if (
            input_ids.device != text_embeddings.device
            or input_ids.device != attention_mask.device
        ):
            raise ValueError(
                "coupled text tensors must share one device"
            )

        validated_labels: torch.Tensor | None
        if labels is None:
            validated_labels = None
        else:
            if not isinstance(labels, torch.Tensor):
                raise TypeError("labels must be a torch.Tensor or None")
            if labels.shape != (batch_size, text_length):
                raise ValueError("labels must have shape [B, T]")
            if labels.dtype is not torch.long:
                raise TypeError("labels must have dtype torch.long")
            if labels.device != input_ids.device:
                raise ValueError(
                    "coupled text tensors must share one device"
                )
            if bool(((labels < 0) & (labels != -100)).any().item()):
                raise ValueError(
                    "labels must contain only -100 or non-negative IDs"
                )
            validated_labels = labels

        valid_lengths: list[int] = []
        for row in range(batch_size):
            count = _prefix_count(
                attention_mask[row],
                f"attention_mask row {row}",
            )
            valid_lengths.append(count)
            if bool(
                (
                    input_ids[row, count:] != pad_token_id
                )
                .any()
                .item()
            ):
                raise ValueError(
                    "masked input IDs must equal pad_token_id"
                )
            if (
                validated_labels is not None
                and bool(
                    (
                        validated_labels[row, count:] != -100
                    )
                    .any()
                    .item()
                )
            ):
                raise ValueError("masked labels must equal -100")
        return (
            input_ids,
            text_embeddings,
            attention_mask,
            validated_labels,
            tuple(valid_lengths),
        )

    @staticmethod
    def _collect_sources(
        *,
        media_sequences: object,
        batch_size: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[
        dict[MediaSource, _SourceRecord],
        dict[tuple[int, int], _SourceRecord],
    ]:
        if (
            not isinstance(media_sequences, Sequence)
            or isinstance(
                media_sequences,
                (str, bytes, bytearray),
            )
        ):
            raise TypeError(
                "media_sequences must be a sequence of MediaSequence"
            )
        by_source: dict[MediaSource, _SourceRecord] = {}
        by_item: dict[tuple[int, int], _SourceRecord] = {}
        id_keys: set[tuple[int, str]] = set()
        for sequence in media_sequences:
            if not isinstance(sequence, MediaSequence):
                raise TypeError(
                    "media_sequences must contain MediaSequence values"
                )
            sequence.validate()
            if sequence.embeddings.shape[2] != hidden_size:
                raise ValueError(
                    "media and text hidden sizes must match exactly"
                )
            if sequence.embeddings.dtype != dtype:
                raise TypeError(
                    "media and text embedding dtypes must match exactly"
                )
            if sequence.embeddings.device != device:
                raise ValueError(
                    "media and text embedding devices must match exactly"
                )
            for row, source in enumerate(sequence.sources):
                if source.sample_index >= batch_size:
                    raise ValueError(
                        "media source sample_index is outside text batch"
                    )
                valid_count = _prefix_count(
                    sequence.attention_mask[row],
                    "media attention_mask row",
                )
                item_key = (source.sample_index, source.item_index)
                id_key = (source.sample_index, source.source_id)
                if item_key in by_item:
                    raise ValueError(
                        "duplicate (sample_index, item_index)"
                    )
                if id_key in id_keys:
                    raise ValueError(
                        "duplicate (sample_index, source_id)"
                    )
                grid = (
                    None
                    if sequence.grid is None
                    else sequence.grid[row]
                )
                seconds_per_grid = (
                    None
                    if sequence.seconds_per_grid is None
                    else sequence.seconds_per_grid[row]
                )
                record = _SourceRecord(
                    sequence=sequence,
                    sequence_row=row,
                    source=source,
                    modality=sequence.modality,
                    valid_count=valid_count,
                    grid=grid,
                    seconds_per_grid=seconds_per_grid,
                )
                by_source[source] = record
                by_item[item_key] = record
                id_keys.add(id_key)

        for sample_index in range(batch_size):
            item_indices = sorted(
                source.item_index
                for source in by_source
                if source.sample_index == sample_index
            )
            if item_indices != list(range(len(item_indices))):
                raise ValueError(
                    "media item indices must be exactly 0..N-1"
                )
        return by_source, by_item

    @staticmethod
    def _build_groups(
        *,
        placeholders: tuple[MediaPlaceholder, ...],
        row_input_ids: torch.Tensor,
        joint_separator_token_ids: frozenset[int],
    ) -> tuple[MediaExpansionGroup, ...]:
        if not placeholders:
            return ()
        grouped: list[list[MediaPlaceholder]] = [
            [placeholders[0]]
        ]
        for placeholder in placeholders[1:]:
            previous = grouped[-1][-1]
            between = row_input_ids[
                previous.text_position + 1 : placeholder.text_position
            ]
            joins = all(
                int(token_id.item()) in joint_separator_token_ids
                for token_id in between
            )
            if joins:
                grouped[-1].append(placeholder)
            else:
                grouped.append([placeholder])
        return tuple(
            MediaExpansionGroup(
                placeholders=tuple(values),
                first_text_position=values[0].text_position,
                last_text_position=values[-1].text_position,
            )
            for values in grouped
        )

    @classmethod
    def _build_plans(
        cls,
        *,
        input_ids: torch.Tensor,
        valid_lengths: tuple[int, ...],
        by_item: Mapping[tuple[int, int], _SourceRecord],
        tokens: ResolvedMultimodalTokens,
        separators: frozenset[int],
    ) -> tuple[_SamplePlan, ...]:
        sentinel_modalities = {
            tokens.image_pad: MediaModality.IMAGE,
            tokens.video_pad: MediaModality.VIDEO,
            tokens.audio_pad: MediaModality.AUDIO,
        }
        plans: list[_SamplePlan] = []
        for sample_index, valid_length in enumerate(valid_lengths):
            sentinel_positions = [
                (
                    position,
                    sentinel_modalities[int(input_ids[sample_index, position])],
                )
                for position in range(valid_length)
                if int(input_ids[sample_index, position])
                in sentinel_modalities
            ]
            sample_records = [
                record
                for key, record in by_item.items()
                if key[0] == sample_index
            ]
            if len(sentinel_positions) != len(sample_records):
                if len(sentinel_positions) > len(sample_records):
                    raise ValueError(
                        "media sentinel has no matching media item"
                    )
                raise ValueError(
                    "extra media item has no matching sentinel"
                )
            placeholders: list[MediaPlaceholder] = []
            for item_index, (position, modality) in enumerate(
                sentinel_positions
            ):
                record = by_item.get((sample_index, item_index))
                if record is None:
                    raise ValueError(
                        "media sentinel has no matching item index"
                    )
                if record.modality is not modality:
                    raise ValueError(
                        "media sentinel modality does not match item"
                    )
                placeholders.append(
                    MediaPlaceholder(
                        text_position=position,
                        sentinel_token_id=int(
                            input_ids[sample_index, position]
                        ),
                        modality=modality,
                        sequence=record.sequence,
                        sequence_row=record.sequence_row,
                    )
                )
            placeholder_tuple = tuple(placeholders)
            groups = cls._build_groups(
                placeholders=placeholder_tuple,
                row_input_ids=input_ids[sample_index, :valid_length],
                joint_separator_token_ids=separators,
            )
            plans.append(
                _SamplePlan(
                    sample_index=sample_index,
                    valid_length=valid_length,
                    placeholders=placeholder_tuple,
                    groups=groups,
                    retained_text_tokens=(
                        valid_length - len(placeholder_tuple)
                    ),
                )
            )
        return tuple(plans)

    @staticmethod
    def _validate_expansion(
        *,
        plan: _SamplePlan,
        result: object,
        by_source: Mapping[MediaSource, _SourceRecord],
        tokens: ResolvedMultimodalTokens,
        pad_token_id: int,
        sentinel_ids: frozenset[int],
    ) -> ExpandedMediaSample:
        if not isinstance(result, ExpandedMediaSample):
            raise TypeError(
                "policy must return ExpandedMediaSample"
            )
        replacements = result.replacements
        if not isinstance(replacements, Mapping):
            raise TypeError(
                "ExpandedMediaSample.replacements must be a mapping"
            )
        for key, row in replacements.items():
            _exact_non_negative_integer(
                key,
                "replacement text position",
            )
            if not isinstance(row, ExpandedMediaRow):
                raise TypeError(
                    "replacement values must be ExpandedMediaRow"
                )
            if not isinstance(row.tokens, tuple):
                raise TypeError(
                    "ExpandedMediaRow.tokens must be a tuple"
                )
            if any(
                not isinstance(token, ExpansionToken)
                for token in row.tokens
            ):
                raise TypeError(
                    "replacement rows must contain ExpansionToken values"
                )
        expected_positions = {
            placeholder.text_position
            for placeholder in plan.placeholders
        }
        if set(replacements) != expected_positions:
            raise ValueError(
                "replacement keys must exactly match placeholder positions"
            )

        for group in plan.groups:
            group_sources = {
                placeholder.source
                for placeholder in group.placeholders
            }
            expected_refs = [
                MediaTokenRef(source, token_index)
                for source in group_sources
                for token_index in range(
                    by_source[source].valid_count
                )
            ]
            output_refs: list[MediaTokenRef] = []
            effective_sources: dict[int, list[MediaSource]] = {}
            for placeholder in group.placeholders:
                row = replacements[placeholder.text_position]
                effective_sources[placeholder.text_position] = []
                for expansion_token in row.tokens:
                    token_id = _exact_non_negative_integer(
                        expansion_token.token_id,
                        "expansion token_id",
                    )
                    if expansion_token.kind is SequenceSpanKind.TEXT:
                        raise ValueError(
                            "policies may not emit TEXT expansion tokens"
                        )
                    if (
                        expansion_token.kind
                        is SequenceSpanKind.MEDIA
                    ):
                        ref = expansion_token.media_ref
                        if not isinstance(ref, MediaTokenRef):
                            raise ValueError(
                                "MEDIA expansion tokens require media_ref"
                            )
                        record = by_source.get(ref.source)
                        if (
                            record is None
                            or ref.source not in group_sources
                        ):
                            raise ValueError(
                                "media_ref is outside its approved group"
                            )
                        if ref.token_index >= record.valid_count:
                            raise ValueError(
                                "media_ref token_index is out of range"
                            )
                        if (
                            expansion_token.source is not None
                            and expansion_token.source != ref.source
                        ):
                            raise ValueError(
                                "MEDIA token source must equal media_ref "
                                "source"
                            )
                        expected_token_id = tokens.sentinel_for(
                            record.modality
                        )
                        if token_id != expected_token_id:
                            raise ValueError(
                                "MEDIA token_id must equal its source "
                                "modality sentinel"
                            )
                        output_refs.append(ref)
                        effective_sources[
                            placeholder.text_position
                        ].append(ref.source)
                    elif (
                        expansion_token.kind
                        is SequenceSpanKind.TIMESTAMP
                    ):
                        if expansion_token.media_ref is not None:
                            raise ValueError(
                                "TIMESTAMP tokens forbid media_ref"
                            )
                        source = expansion_token.source
                        if (
                            not isinstance(source, MediaSource)
                            or source not in group_sources
                            or source not in by_source
                        ):
                            raise ValueError(
                                "TIMESTAMP source must belong to the "
                                "approved group"
                            )
                        if (
                            token_id == pad_token_id
                            or token_id in sentinel_ids
                        ):
                            raise ValueError(
                                "TIMESTAMP token_id must be an ordinary "
                                "non-pad, non-sentinel ID"
                            )
                        effective_sources[
                            placeholder.text_position
                        ].append(source)
                    else:
                        raise TypeError(
                            "expansion kind must be MEDIA or TIMESTAMP"
                        )

            own_shape = all(
                all(
                    source == placeholder.source
                    for source in effective_sources[
                        placeholder.text_position
                    ]
                )
                for placeholder in group.placeholders
            )
            first_position = group.first_text_position
            joint_shape = (
                all(
                    source in group_sources
                    for source in effective_sources[first_position]
                )
                and all(
                    len(
                        replacements[
                            placeholder.text_position
                        ].tokens
                    )
                    == 0
                    for placeholder in group.placeholders[1:]
                )
            )
            if not own_shape and not joint_shape:
                raise ValueError(
                    "group refs may stay with their own placeholder or "
                    "move jointly to the first placeholder only"
                )
            if Counter(output_refs) != Counter(expected_refs):
                raise ValueError(
                    "output media refs must be an exact permutation"
                )
            for source in group_sources:
                indices = [
                    ref.token_index
                    for ref in output_refs
                    if ref.source == source
                ]
                if indices != list(
                    range(by_source[source].valid_count)
                ):
                    raise ValueError(
                        "each source must preserve canonical token order"
                    )
        return result

    @staticmethod
    def _symbolic_atoms(
        *,
        plan: _SamplePlan,
        input_ids: torch.Tensor,
    ) -> list[_SymbolicAtom]:
        assert plan.expansion is not None
        replacements = plan.expansion.replacements
        atoms: list[_SymbolicAtom] = []
        for position in range(plan.valid_length):
            replacement = replacements.get(position)
            if replacement is None:
                atoms.append(
                    _SymbolicAtom(
                        kind=SequenceSpanKind.TEXT,
                        token_id=int(
                            input_ids[plan.sample_index, position]
                        ),
                        text_position=position,
                    )
                )
            else:
                atoms.extend(
                    _SymbolicAtom(
                        kind=token.kind,
                        token_id=token.token_id,
                        expansion_token=token,
                    )
                    for token in replacement.tokens
                )
        return atoms

    @staticmethod
    def _lookup_timestamp_embeddings(
        *,
        atoms: Sequence[_SymbolicAtom],
        embedding_lookup: EmbeddingLookup | None,
        input_device: torch.device,
        text_dtype: torch.dtype,
        text_device: torch.device,
        hidden_size: int,
    ) -> list[torch.Tensor]:
        timestamp_ids = [
            atom.token_id
            for atom in atoms
            if atom.kind is SequenceSpanKind.TIMESTAMP
        ]
        if not timestamp_ids:
            return []
        if embedding_lookup is None or not callable(embedding_lookup):
            raise ValueError(
                "timestamp expansion requires embedding_lookup"
            )
        token_ids = torch.tensor(
            timestamp_ids,
            dtype=torch.long,
            device=input_device,
        )
        embeddings = embedding_lookup(token_ids)
        if not isinstance(embeddings, torch.Tensor):
            raise TypeError(
                "embedding_lookup must return a torch.Tensor"
            )
        if not embeddings.is_floating_point():
            raise TypeError(
                "timestamp lookup embeddings must have floating dtype"
            )
        if embeddings.shape != (len(timestamp_ids), hidden_size):
            raise ValueError(
                "timestamp lookup must return "
                "[timestamp_token_count, hidden_size]"
            )
        if embeddings.dtype != text_dtype:
            raise TypeError(
                "timestamp lookup dtype must match text embeddings"
            )
        if embeddings.device != text_device:
            raise ValueError(
                "timestamp lookup device must match text embeddings"
            )
        return list(embeddings.unbind(0))

    @staticmethod
    def _materialize_atoms(
        *,
        plan: _SamplePlan,
        atoms: Sequence[_SymbolicAtom],
        text_embeddings: torch.Tensor,
        labels: torch.Tensor | None,
        by_source: Mapping[MediaSource, _SourceRecord],
        timestamp_embeddings: Sequence[torch.Tensor],
    ) -> list[_MaterializedAtom]:
        materialized: list[_MaterializedAtom] = []
        timestamp_index = 0
        for atom in atoms:
            if atom.kind is SequenceSpanKind.TEXT:
                assert atom.text_position is not None
                materialized.append(
                    _MaterializedAtom(
                        kind=atom.kind,
                        token_id=atom.token_id,
                        embedding=text_embeddings[
                            plan.sample_index,
                            atom.text_position,
                        ],
                        label=(
                            None
                            if labels is None
                            else int(
                                labels[
                                    plan.sample_index,
                                    atom.text_position,
                                ]
                            )
                        ),
                    )
                )
                continue

            expansion_token = atom.expansion_token
            assert expansion_token is not None
            if atom.kind is SequenceSpanKind.MEDIA:
                ref = expansion_token.media_ref
                assert ref is not None
                record = by_source[ref.source]
                materialized.append(
                    _MaterializedAtom(
                        kind=atom.kind,
                        token_id=atom.token_id,
                        embedding=record.sequence.embeddings[
                            record.sequence_row,
                            ref.token_index,
                        ],
                        label=None if labels is None else -100,
                        source=ref.source,
                        record=record,
                        token_index=ref.token_index,
                    )
                )
            else:
                source = expansion_token.source
                assert source is not None
                record = by_source[source]
                materialized.append(
                    _MaterializedAtom(
                        kind=atom.kind,
                        token_id=atom.token_id,
                        embedding=timestamp_embeddings[
                            timestamp_index
                        ],
                        label=None if labels is None else -100,
                        source=source,
                        record=record,
                    )
                )
                timestamp_index += 1
        return materialized

    @staticmethod
    def _build_spans(
        *,
        sample_index: int,
        atoms: Sequence[_MaterializedAtom],
    ) -> tuple[SequenceSpan, ...]:
        spans: list[SequenceSpan] = []
        start = 0
        while start < len(atoms):
            first = atoms[start]
            end = start + 1
            while end < len(atoms):
                candidate = atoms[end]
                if candidate.kind is not first.kind:
                    break
                if first.kind is SequenceSpanKind.TEXT:
                    end += 1
                    continue
                if candidate.source != first.source:
                    break
                if first.kind is SequenceSpanKind.TIMESTAMP:
                    end += 1
                    continue
                previous = atoms[end - 1]
                assert previous.token_index is not None
                assert candidate.token_index is not None
                if candidate.token_index != previous.token_index + 1:
                    break
                end += 1

            if first.kind is SequenceSpanKind.TEXT:
                span = SequenceSpan(
                    sample_index=sample_index,
                    start=start,
                    end=end,
                    kind=first.kind,
                    modality=None,
                    grid=None,
                    timestamps=None,
                    seconds_per_grid=None,
                    source=None,
                    source_token_indices=None,
                )
            elif first.kind is SequenceSpanKind.TIMESTAMP:
                assert first.record is not None
                assert first.source is not None
                span = SequenceSpan(
                    sample_index=sample_index,
                    start=start,
                    end=end,
                    kind=first.kind,
                    modality=first.record.modality,
                    grid=None,
                    timestamps=None,
                    seconds_per_grid=None,
                    source=first.source,
                    source_token_indices=None,
                )
            else:
                assert first.record is not None
                assert first.source is not None
                indices = tuple(
                    int(atom.token_index)
                    for atom in atoms[start:end]
                    if atom.token_index is not None
                )
                timestamps = None
                if first.record.sequence.timestamps is not None:
                    index_tensor = torch.tensor(
                        indices,
                        dtype=torch.long,
                        device=(
                            first.record.sequence.timestamps.device
                        ),
                    )
                    timestamps = first.record.sequence.timestamps[
                        first.record.sequence_row
                    ].index_select(0, index_tensor)
                span = SequenceSpan(
                    sample_index=sample_index,
                    start=start,
                    end=end,
                    kind=first.kind,
                    modality=first.record.modality,
                    grid=first.record.grid,
                    timestamps=timestamps,
                    seconds_per_grid=first.record.seconds_per_grid,
                    source=first.source,
                    source_token_indices=indices,
                )
            spans.append(span)
            start = end
        return tuple(spans)

    def assemble(
        self,
        *,
        input_ids: torch.LongTensor,
        text_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None,
        media_sequences: Sequence[MediaSequence],
        tokens: ResolvedMultimodalTokens,
        expansion_policy: MediaExpansionPolicy | None = None,
        embedding_lookup: EmbeddingLookup | None = None,
        pad_token_id: int,
        joint_separator_token_ids: frozenset[int] = frozenset(),
        max_assembled_length: int,
    ) -> AssembledSequence:
        (
            tokens,
            pad_token_id,
            separators,
            max_assembled_length,
            sentinel_ids,
        ) = self._validate_tokens_and_scalars(
            tokens=tokens,
            pad_token_id=pad_token_id,
            joint_separator_token_ids=joint_separator_token_ids,
            max_assembled_length=max_assembled_length,
        )
        (
            input_ids,
            text_embeddings,
            attention_mask,
            labels,
            valid_lengths,
        ) = self._validate_text_tensors(
            input_ids=input_ids,
            text_embeddings=text_embeddings,
            attention_mask=attention_mask,
            labels=labels,
            pad_token_id=pad_token_id,
        )
        batch_size = input_ids.shape[0]
        hidden_size = text_embeddings.shape[2]
        by_source, by_item = self._collect_sources(
            media_sequences=media_sequences,
            batch_size=batch_size,
            hidden_size=hidden_size,
            dtype=text_embeddings.dtype,
            device=text_embeddings.device,
        )
        plans = self._build_plans(
            input_ids=input_ids,
            valid_lengths=valid_lengths,
            by_item=by_item,
            tokens=tokens,
            separators=separators,
        )

        policy = (
            IdentityMediaExpansion()
            if expansion_policy is None
            else expansion_policy
        )
        expand_sample = getattr(policy, "expand_sample", None)
        if not callable(expand_sample):
            raise TypeError(
                "expansion_policy must define callable expand_sample"
            )
        for plan in plans:
            result = expand_sample(
                sample_index=plan.sample_index,
                groups=plan.groups,
            )
            plan.expansion = self._validate_expansion(
                plan=plan,
                result=result,
                by_source=by_source,
                tokens=tokens,
                pad_token_id=pad_token_id,
                sentinel_ids=sentinel_ids,
            )
            plan.inserted_expansion_tokens = sum(
                len(row.tokens)
                for row in plan.expansion.replacements.values()
            )
            assembled_length = (
                plan.retained_text_tokens
                + plan.inserted_expansion_tokens
            )
            if assembled_length > max_assembled_length:
                raise AssembledLengthError(
                    sample_index=plan.sample_index,
                    assembled_length=assembled_length,
                    max_assembled_length=max_assembled_length,
                    retained_text_tokens=plan.retained_text_tokens,
                    inserted_expansion_tokens=(
                        plan.inserted_expansion_tokens
                    ),
                )

        row_ids: list[torch.Tensor] = []
        row_embeddings: list[torch.Tensor] = []
        row_labels: list[torch.Tensor] = []
        all_spans: list[SequenceSpan] = []
        for plan in plans:
            symbolic = self._symbolic_atoms(
                plan=plan,
                input_ids=input_ids,
            )
            timestamp_embeddings = (
                self._lookup_timestamp_embeddings(
                    atoms=symbolic,
                    embedding_lookup=embedding_lookup,
                    input_device=input_ids.device,
                    text_dtype=text_embeddings.dtype,
                    text_device=text_embeddings.device,
                    hidden_size=hidden_size,
                )
            )
            atoms = self._materialize_atoms(
                plan=plan,
                atoms=symbolic,
                text_embeddings=text_embeddings,
                labels=labels,
                by_source=by_source,
                timestamp_embeddings=timestamp_embeddings,
            )
            row_ids.append(
                torch.tensor(
                    [atom.token_id for atom in atoms],
                    dtype=torch.long,
                    device=input_ids.device,
                )
            )
            row_embeddings.append(
                torch.stack(
                    [atom.embedding for atom in atoms],
                    dim=0,
                )
            )
            if labels is not None:
                row_labels.append(
                    torch.tensor(
                        [int(atom.label) for atom in atoms],
                        dtype=torch.long,
                        device=input_ids.device,
                    )
                )
            all_spans.extend(
                self._build_spans(
                    sample_index=plan.sample_index,
                    atoms=atoms,
                )
            )

        maximum = max(row.shape[0] for row in row_ids)
        expanded_input_ids = torch.stack(
            [
                F.pad(
                    row,
                    (0, maximum - row.shape[0]),
                    value=pad_token_id,
                )
                for row in row_ids
            ],
            dim=0,
        )
        inputs_embeds = torch.stack(
            [
                F.pad(
                    row,
                    (0, 0, 0, maximum - row.shape[0]),
                )
                for row in row_embeddings
            ],
            dim=0,
        )
        positions = torch.arange(
            maximum,
            device=input_ids.device,
        )
        lengths = torch.tensor(
            [row.shape[0] for row in row_ids],
            device=input_ids.device,
        )
        output_mask = positions.unsqueeze(0) < lengths.unsqueeze(1)
        output_labels = None
        if labels is not None:
            output_labels = torch.stack(
                [
                    F.pad(
                        row,
                        (0, maximum - row.shape[0]),
                        value=-100,
                    )
                    for row in row_labels
                ],
                dim=0,
            )

        result = AssembledSequence(
            expanded_input_ids=expanded_input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=output_mask,
            labels=output_labels,
            spans=tuple(all_spans),
        )
        result.validate()
        return result


__all__ = [
    "AssembledLengthError",
    "EmbeddingLookup",
    "ExpandedMediaRow",
    "ExpandedMediaSample",
    "ExpansionToken",
    "IdentityMediaExpansion",
    "MediaExpansionGroup",
    "MediaExpansionPolicy",
    "MediaPlaceholder",
    "MediaTokenRef",
    "SequenceAssembler",
    "TimestampInterleaveExpansion",
]
