from dataclasses import FrozenInstanceError

import pytest
import torch

from qwen3_omni_pretrain.multimodal.modalities import MediaModality
from qwen3_omni_pretrain.multimodal.types import (
    AssembledSequence,
    MediaGrid,
    MediaSequence,
    MediaSource,
    PositionBatch,
    SequenceSpan,
    SequenceSpanKind,
)


def make_audio_sequence(
    *,
    attention_mask: torch.Tensor | None = None,
    timestamps: torch.Tensor | None = None,
    sources: tuple[MediaSource, ...] | None = None,
    grid: tuple[MediaGrid | None, ...] | None = None,
    seconds_per_grid: tuple[float | None, ...] | None = None,
) -> MediaSequence:
    if attention_mask is None:
        attention_mask = torch.ones(1, 3, dtype=torch.bool)
    batch_size, token_count = attention_mask.shape
    if sources is None:
        sources = tuple(
            MediaSource(sample_index, 0, f"audio-{sample_index}")
            for sample_index in range(batch_size)
        )
    return MediaSequence(
        embeddings=torch.zeros(batch_size, token_count, 8),
        attention_mask=attention_mask,
        modality=MediaModality.AUDIO,
        sources=sources,
        grid=grid,
        timestamps=timestamps,
        seconds_per_grid=seconds_per_grid,
    )


def make_span(
    *,
    sample_index: int = 0,
    start: int = 0,
    end: int = 1,
    kind: SequenceSpanKind = SequenceSpanKind.TEXT,
    modality: MediaModality | None = None,
    grid: MediaGrid | None = None,
    timestamps: torch.Tensor | None = None,
    seconds_per_grid: float | None = None,
    source: MediaSource | None = None,
    source_token_indices: tuple[int, ...] | None = None,
) -> SequenceSpan:
    return SequenceSpan(
        sample_index=sample_index,
        start=start,
        end=end,
        kind=kind,
        modality=modality,
        grid=grid,
        timestamps=timestamps,
        seconds_per_grid=seconds_per_grid,
        source=source,
        source_token_indices=source_token_indices,
    )


def make_media_span(
    *,
    sample_index: int = 0,
    start: int = 0,
    end: int = 2,
    modality: MediaModality = MediaModality.IMAGE,
    grid: MediaGrid | None = None,
    timestamps: torch.Tensor | None = None,
    seconds_per_grid: float | None = None,
    source: MediaSource | None = None,
    source_token_indices: tuple[int, ...] = (0, 1),
) -> SequenceSpan:
    if source is None:
        source = MediaSource(sample_index, 0, f"{modality.value}-0")
    if grid is None and modality in (
        MediaModality.IMAGE,
        MediaModality.VIDEO,
    ):
        grid = MediaGrid(1, 1, 2)
    return make_span(
        sample_index=sample_index,
        start=start,
        end=end,
        kind=SequenceSpanKind.MEDIA,
        modality=modality,
        grid=grid,
        timestamps=timestamps,
        seconds_per_grid=seconds_per_grid,
        source=source,
        source_token_indices=source_token_indices,
    )


def make_assembled(
    *,
    attention_mask: torch.Tensor | None = None,
    spans: tuple[SequenceSpan, ...] | None = None,
    expanded_input_ids: torch.Tensor | None = None,
    inputs_embeds: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
) -> AssembledSequence:
    if attention_mask is None:
        attention_mask = torch.ones(1, 2, dtype=torch.bool)
    batch_size, sequence_length = attention_mask.shape
    if expanded_input_ids is None:
        expanded_input_ids = torch.zeros(
            batch_size, sequence_length, dtype=torch.long
        )
    if inputs_embeds is None:
        inputs_embeds = torch.zeros(batch_size, sequence_length, 4)
    if spans is None:
        spans = tuple(
            make_span(
                sample_index=sample_index,
                start=0,
                end=sequence_length,
            )
            for sample_index in range(batch_size)
        )
    return AssembledSequence(
        expanded_input_ids=expanded_input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        labels=labels,
        spans=spans,
    )


def test_media_contracts_are_exported_from_the_multimodal_package():
    from qwen3_omni_pretrain import multimodal

    assert (
        multimodal.MediaSource,
        multimodal.MediaGrid,
        multimodal.MediaSequence,
        multimodal.SequenceSpanKind,
        multimodal.SequenceSpan,
        multimodal.AssembledSequence,
        multimodal.PositionBatch,
    ) == (
        MediaSource,
        MediaGrid,
        MediaSequence,
        SequenceSpanKind,
        SequenceSpan,
        AssembledSequence,
        PositionBatch,
    )


def test_contract_dataclasses_are_frozen():
    source = MediaSource(0, 0, "audio-0")

    with pytest.raises(FrozenInstanceError):
        source.source_id = "changed"


@pytest.mark.parametrize("field", ["sample_index", "item_index"])
@pytest.mark.parametrize("value", [True, 1.0, "1"])
def test_media_source_rejects_bool_and_non_integer_indices(field, value):
    values = {
        "sample_index": 0,
        "item_index": 0,
        "source_id": "audio-0",
    }
    values[field] = value

    with pytest.raises(TypeError, match=field):
        MediaSource(**values)


@pytest.mark.parametrize("field", ["sample_index", "item_index"])
def test_media_source_rejects_negative_indices(field):
    values = {
        "sample_index": 0,
        "item_index": 0,
        "source_id": "audio-0",
    }
    values[field] = -1

    with pytest.raises(ValueError, match=field):
        MediaSource(**values)


@pytest.mark.parametrize("source_id", ["", "   "])
def test_media_source_rejects_empty_ids(source_id):
    with pytest.raises(ValueError, match="source_id"):
        MediaSource(0, 0, source_id)


def test_media_source_rejects_non_string_ids():
    with pytest.raises(TypeError, match="source_id"):
        MediaSource(0, 0, 12)


@pytest.mark.parametrize("field", ["temporal", "height", "width"])
@pytest.mark.parametrize("value", [True, 1.0, "1"])
def test_media_grid_rejects_bool_and_non_integer_axes(field, value):
    values = {"temporal": 1, "height": 2, "width": 3}
    values[field] = value

    with pytest.raises(TypeError, match=field):
        MediaGrid(**values)


@pytest.mark.parametrize("field", ["temporal", "height", "width"])
@pytest.mark.parametrize("value", [-1, 0])
def test_media_grid_rejects_non_positive_axes(field, value):
    values = {"temporal": 1, "height": 2, "width": 3}
    values[field] = value

    with pytest.raises(ValueError, match=field):
        MediaGrid(**values)


def test_media_grid_reports_complete_source_token_count():
    assert MediaGrid(2, 3, 4).token_count == 24


def test_media_sequence_validates_batch_and_token_axes():
    sequence = MediaSequence(
        embeddings=torch.zeros(2, 3, 8),
        attention_mask=torch.ones(2, 3, dtype=torch.bool),
        modality=MediaModality.AUDIO,
        sources=(
            MediaSource(0, 0, "a0"),
            MediaSource(1, 0, "a1"),
        ),
        timestamps=torch.tensor(
            [[0.0, 0.08, 0.16], [0.0, 0.08, 0.16]]
        ),
    )

    sequence.validate()


def test_media_sequence_rejects_non_monotonic_valid_timestamps():
    sequence = make_audio_sequence(
        timestamps=torch.tensor([[0.0, 0.16, 0.08]])
    )

    with pytest.raises(ValueError, match="monotonic"):
        sequence.validate()


def test_media_sequence_ignores_timestamp_values_at_masked_slots():
    sequence = make_audio_sequence(
        attention_mask=torch.tensor([[1, 0, 1]]),
        timestamps=torch.tensor([[0.08, float("nan"), 0.16]]),
    )

    sequence.validate()


@pytest.mark.parametrize(
    ("embeddings", "message"),
    [
        (torch.zeros(2, 3), "embeddings"),
        (torch.zeros(1, 2, 3, 4), "embeddings"),
    ],
)
def test_media_sequence_rejects_non_three_dimensional_embeddings(
    embeddings, message
):
    sequence = MediaSequence(
        embeddings=embeddings,
        attention_mask=torch.ones(
            embeddings.shape[:2], dtype=torch.bool
        ),
        modality=MediaModality.AUDIO,
        sources=(MediaSource(0, 0, "audio-0"),),
    )

    with pytest.raises(ValueError, match=message):
        sequence.validate()


def test_media_sequence_rejects_mask_shape_mismatch():
    sequence = MediaSequence(
        embeddings=torch.zeros(1, 3, 8),
        attention_mask=torch.ones(1, 2, dtype=torch.bool),
        modality=MediaModality.AUDIO,
        sources=(MediaSource(0, 0, "audio-0"),),
    )

    with pytest.raises(ValueError, match="attention_mask"):
        sequence.validate()


@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.float32, torch.float64, torch.complex64],
)
def test_media_sequence_rejects_non_boolean_non_integer_masks(dtype):
    sequence = make_audio_sequence(
        attention_mask=torch.ones(1, 3, dtype=dtype)
    )

    with pytest.raises(TypeError, match="attention_mask"):
        sequence.validate()


@pytest.mark.parametrize(
    "dtype",
    [
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ],
)
def test_media_sequence_accepts_boolean_and_integer_masks(dtype):
    make_audio_sequence(
        attention_mask=torch.ones(1, 3, dtype=dtype)
    ).validate()


@pytest.mark.parametrize(
    "attention_mask",
    [
        torch.tensor([[1, 2, 0]]),
        torch.tensor([[1, -1, 0]]),
    ],
)
def test_media_sequence_rejects_non_binary_integer_masks(attention_mask):
    sequence = make_audio_sequence(attention_mask=attention_mask)

    with pytest.raises(ValueError, match="0 or 1"):
        sequence.validate()


@pytest.mark.parametrize(
    "dtype", [torch.bool, torch.int64, torch.complex64]
)
def test_media_sequence_requires_floating_non_complex_embeddings(dtype):
    sequence = MediaSequence(
        embeddings=torch.zeros(1, 3, 8, dtype=dtype),
        attention_mask=torch.ones(1, 3, dtype=torch.bool),
        modality=MediaModality.AUDIO,
        sources=(MediaSource(0, 0, "audio-0"),),
    )

    with pytest.raises(TypeError, match="embeddings"):
        sequence.validate()


@pytest.mark.parametrize(
    "mismatched_tensor",
    ["embeddings", "attention_mask", "timestamps"],
)
def test_media_sequence_requires_all_tensors_on_one_device(
    mismatched_tensor,
):
    values = {
        "embeddings": torch.zeros(1, 3, 8),
        "attention_mask": torch.ones(1, 3, dtype=torch.bool),
        "timestamps": torch.tensor([[0.0, 0.08, 0.16]]),
    }
    if mismatched_tensor == "embeddings":
        values[mismatched_tensor] = torch.empty(
            1, 3, 8, device="meta"
        )
    elif mismatched_tensor == "attention_mask":
        values[mismatched_tensor] = torch.empty(
            1, 3, dtype=torch.bool, device="meta"
        )
    else:
        values[mismatched_tensor] = torch.empty(
            1, 3, device="meta"
        )
    sequence = MediaSequence(
        **values,
        modality=MediaModality.AUDIO,
        sources=(MediaSource(0, 0, "audio-0"),),
    )

    with pytest.raises(ValueError, match="device"):
        sequence.validate()


def test_media_sequence_requires_one_source_per_batch_row():
    sequence = MediaSequence(
        embeddings=torch.zeros(2, 3, 8),
        attention_mask=torch.ones(2, 3, dtype=torch.bool),
        modality=MediaModality.AUDIO,
        sources=(MediaSource(0, 0, "audio-0"),),
    )

    with pytest.raises(ValueError, match="MediaSource"):
        sequence.validate()


def test_media_sequence_requires_grid_tuple_to_match_batch_size():
    sequence = make_audio_sequence(grid=(None, None))

    with pytest.raises(ValueError, match="grid"):
        sequence.validate()


@pytest.mark.parametrize(
    "modality", [MediaModality.IMAGE, MediaModality.VIDEO]
)
def test_visual_media_sequence_requires_complete_grid_per_row(modality):
    sequence = MediaSequence(
        embeddings=torch.zeros(1, 2, 8),
        attention_mask=torch.ones(1, 2, dtype=torch.bool),
        modality=modality,
        sources=(MediaSource(0, 0, f"{modality.value}-0"),),
        grid=(None,),
    )

    with pytest.raises(ValueError, match="grid"):
        sequence.validate()


@pytest.mark.parametrize(
    "modality", [MediaModality.IMAGE, MediaModality.VIDEO]
)
def test_visual_grid_token_count_must_equal_valid_row_tokens(modality):
    sequence = MediaSequence(
        embeddings=torch.zeros(1, 3, 8),
        attention_mask=torch.tensor([[1, 1, 0]]),
        modality=modality,
        sources=(MediaSource(0, 0, f"{modality.value}-0"),),
        grid=(MediaGrid(1, 1, 3),),
    )

    with pytest.raises(ValueError, match="token count"):
        sequence.validate()


def test_visual_grid_counts_only_valid_tokens():
    sequence = MediaSequence(
        embeddings=torch.zeros(1, 3, 8),
        attention_mask=torch.tensor([[1, 1, 0]]),
        modality=MediaModality.IMAGE,
        sources=(MediaSource(0, 0, "image-0"),),
        grid=(MediaGrid(1, 1, 2),),
    )

    sequence.validate()


def test_media_sequence_requires_seconds_per_grid_to_match_batch_size():
    sequence = make_audio_sequence(seconds_per_grid=(0.08, None))

    with pytest.raises(ValueError, match="seconds_per_grid"):
        sequence.validate()


@pytest.mark.parametrize("value", [True, "0.08", 0.08 + 0j])
def test_video_sequence_seconds_per_grid_type_errors_are_type_errors(value):
    sequence = MediaSequence(
        embeddings=torch.zeros(1, 2, 8),
        attention_mask=torch.ones(1, 2, dtype=torch.bool),
        modality=MediaModality.VIDEO,
        sources=(MediaSource(0, 0, "video-0"),),
        grid=(MediaGrid(1, 1, 2),),
        seconds_per_grid=(value,),
    )

    with pytest.raises(TypeError, match="seconds_per_grid"):
        sequence.validate()


@pytest.mark.parametrize(
    "value", [0.0, -0.08, float("nan"), float("inf")]
)
def test_video_sequence_seconds_per_grid_value_errors_are_value_errors(
    value,
):
    sequence = MediaSequence(
        embeddings=torch.zeros(1, 2, 8),
        attention_mask=torch.ones(1, 2, dtype=torch.bool),
        modality=MediaModality.VIDEO,
        sources=(MediaSource(0, 0, "video-0"),),
        grid=(MediaGrid(1, 1, 2),),
        seconds_per_grid=(value,),
    )

    with pytest.raises(ValueError, match="seconds_per_grid"):
        sequence.validate()


@pytest.mark.parametrize(
    "modality", [MediaModality.AUDIO, MediaModality.IMAGE]
)
def test_only_video_sequences_accept_present_seconds_per_grid(modality):
    sequence = MediaSequence(
        embeddings=torch.zeros(1, 2, 8),
        attention_mask=torch.ones(1, 2, dtype=torch.bool),
        modality=modality,
        sources=(MediaSource(0, 0, f"{modality.value}-0"),),
        grid=(
            (MediaGrid(1, 1, 2),)
            if modality is MediaModality.IMAGE
            else None
        ),
        seconds_per_grid=(0.08,),
    )

    with pytest.raises(ValueError, match="video"):
        sequence.validate()


def test_video_sequence_accepts_positive_seconds_per_grid():
    MediaSequence(
        embeddings=torch.zeros(1, 2, 8),
        attention_mask=torch.ones(1, 2, dtype=torch.bool),
        modality=MediaModality.VIDEO,
        sources=(MediaSource(0, 0, "video-0"),),
        grid=(MediaGrid(1, 1, 2),),
        seconds_per_grid=(0.08,),
    ).validate()


@pytest.mark.parametrize(
    "modality", [MediaModality.AUDIO, MediaModality.IMAGE]
)
def test_non_video_sequences_allow_absent_seconds_per_grid(modality):
    MediaSequence(
        embeddings=torch.zeros(1, 2, 8),
        attention_mask=torch.ones(1, 2, dtype=torch.bool),
        modality=modality,
        sources=(MediaSource(0, 0, f"{modality.value}-0"),),
        grid=(
            (MediaGrid(1, 1, 2),)
            if modality is MediaModality.IMAGE
            else None
        ),
        seconds_per_grid=(None,),
    ).validate()


def test_media_sequence_requires_timestamp_shape_to_match_tokens():
    sequence = make_audio_sequence(timestamps=torch.zeros(1, 2))

    with pytest.raises(ValueError, match="timestamps"):
        sequence.validate()


@pytest.mark.parametrize(
    "timestamps",
    [
        torch.tensor([[0.0, float("nan"), 0.16]]),
        torch.tensor([[0.0, float("inf"), 0.16]]),
        torch.tensor([[0.0, -0.08, 0.16]]),
    ],
)
def test_media_sequence_rejects_invalid_valid_timestamps(timestamps):
    sequence = make_audio_sequence(timestamps=timestamps)

    with pytest.raises(ValueError, match="timestamps"):
        sequence.validate()


def test_media_sequence_rejects_duplicate_sample_item_keys():
    sequence = make_audio_sequence(
        attention_mask=torch.ones(2, 3, dtype=torch.bool),
        sources=(
            MediaSource(0, 0, "audio-a"),
            MediaSource(0, 0, "audio-b"),
        ),
    )

    with pytest.raises(ValueError, match="sample_index, item_index"):
        sequence.validate()


def test_media_sequence_rejects_duplicate_source_ids_within_sample():
    sequence = make_audio_sequence(
        attention_mask=torch.ones(2, 3, dtype=torch.bool),
        sources=(
            MediaSource(0, 0, "same"),
            MediaSource(0, 1, "same"),
        ),
    )

    with pytest.raises(ValueError, match="sample_index, source_id"):
        sequence.validate()


def test_media_sequence_allows_source_ids_to_repeat_across_samples():
    sequence = make_audio_sequence(
        attention_mask=torch.ones(2, 3, dtype=torch.bool),
        sources=(
            MediaSource(0, 0, "sample-local"),
            MediaSource(1, 0, "sample-local"),
        ),
    )

    sequence.validate()


def test_sequence_span_kind_values_are_stable():
    assert [kind.value for kind in SequenceSpanKind] == [
        "text",
        "media",
        "timestamp",
    ]


def test_text_span_forbids_all_media_metadata():
    metadata = {
        "modality": MediaModality.AUDIO,
        "grid": MediaGrid(1, 1, 1),
        "timestamps": torch.tensor([0.0]),
        "seconds_per_grid": 0.08,
        "source": MediaSource(0, 0, "audio-0"),
        "source_token_indices": (0,),
    }

    for field, value in metadata.items():
        span = make_span(**{field: value})
        with pytest.raises(ValueError, match="text"):
            span.validate()


def test_timestamp_span_requires_canonical_source_and_modality():
    make_span(
        kind=SequenceSpanKind.TIMESTAMP,
        source=MediaSource(0, 0, "audio-0"),
        modality=MediaModality.AUDIO,
    ).validate()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source", None),
        ("modality", None),
        ("grid", MediaGrid(1, 1, 1)),
        ("timestamps", torch.tensor([0.0])),
        ("seconds_per_grid", 0.08),
        ("source_token_indices", (0,)),
    ],
)
def test_timestamp_span_rejects_missing_or_noncanonical_metadata(
    field,
    value,
):
    values = {
        "kind": SequenceSpanKind.TIMESTAMP,
        "source": MediaSource(0, 0, "audio-0"),
        "modality": MediaModality.AUDIO,
    }
    values[field] = value

    with pytest.raises(ValueError, match=field):
        make_span(
            **values,
        ).validate()


def test_non_video_spans_reject_seconds_per_grid():
    with pytest.raises(ValueError, match="seconds_per_grid"):
        make_media_span(
            modality=MediaModality.AUDIO,
            grid=None,
            seconds_per_grid=0.08,
        ).validate()

    with pytest.raises(ValueError, match="seconds_per_grid"):
        make_span(
            kind=SequenceSpanKind.TIMESTAMP,
            source=MediaSource(0, 0, "video-0"),
            modality=MediaModality.VIDEO,
            seconds_per_grid=0.08,
        ).validate()


@pytest.mark.parametrize("value", [True, "0.08", 0.08 + 0j])
def test_video_span_seconds_per_grid_type_errors_are_type_errors(value):
    with pytest.raises(TypeError, match="seconds_per_grid"):
        make_media_span(
            modality=MediaModality.VIDEO,
            seconds_per_grid=value,
        ).validate()


@pytest.mark.parametrize(
    "value", [0.0, -0.08, float("nan"), float("inf")]
)
def test_video_span_seconds_per_grid_value_errors_are_value_errors(value):
    with pytest.raises(ValueError, match="seconds_per_grid"):
        make_media_span(
            modality=MediaModality.VIDEO,
            seconds_per_grid=value,
        ).validate()


def test_video_span_preserves_positive_seconds_per_grid():
    make_media_span(
        modality=MediaModality.VIDEO,
        seconds_per_grid=0.08,
    ).validate()


def test_media_span_requires_source_and_modality():
    with pytest.raises(ValueError, match="source"):
        make_span(
            kind=SequenceSpanKind.MEDIA,
            modality=MediaModality.AUDIO,
            source_token_indices=(0,),
        ).validate()

    with pytest.raises(ValueError, match="modality"):
        make_span(
            kind=SequenceSpanKind.MEDIA,
            source=MediaSource(0, 0, "audio-0"),
            source_token_indices=(0,),
        ).validate()


@pytest.mark.parametrize(
    "modality", [MediaModality.IMAGE, MediaModality.VIDEO]
)
def test_visual_media_span_requires_complete_source_grid(modality):
    with pytest.raises(ValueError, match="grid"):
        make_span(
            kind=SequenceSpanKind.MEDIA,
            modality=modality,
            grid=None,
            source=MediaSource(0, 0, f"{modality.value}-0"),
            source_token_indices=(0,),
        ).validate()


def test_media_span_requires_one_source_index_per_output_token():
    with pytest.raises(ValueError, match="source_token_indices"):
        make_media_span(source_token_indices=(0,)).validate()


@pytest.mark.parametrize(
    "indices",
    [
        (-1, 0),
        (False, 1),
        (0.0, 1),
        (1, 0),
        (0, 0),
    ],
)
def test_media_span_requires_non_negative_strictly_increasing_indices(
    indices,
):
    with pytest.raises((TypeError, ValueError), match="source_token_indices"):
        make_media_span(source_token_indices=indices).validate()


def test_media_fragment_indices_are_bounded_by_complete_source_grid():
    span = make_media_span(
        start=0,
        end=2,
        grid=MediaGrid(1, 2, 2),
        source_token_indices=(2, 4),
    )

    with pytest.raises(ValueError, match="grid"):
        span.validate()


def test_fragment_span_keeps_complete_grid_not_fragment_grid():
    source = MediaSource(0, 0, "image-0")
    complete_grid = MediaGrid(1, 2, 2)
    spans = (
        make_media_span(
            start=0,
            end=2,
            grid=complete_grid,
            source=source,
            source_token_indices=(0, 1),
        ),
        make_media_span(
            start=2,
            end=4,
            grid=complete_grid,
            source=source,
            source_token_indices=(2, 3),
        ),
    )

    make_assembled(
        attention_mask=torch.ones(1, 4, dtype=torch.bool),
        spans=spans,
    ).validate()


def test_assembled_sequence_allows_cross_source_interleaving():
    source_a = MediaSource(0, 0, "image-a")
    source_b = MediaSource(0, 1, "image-b")
    complete_grid = MediaGrid(1, 1, 2)
    spans = (
        make_media_span(
            start=0,
            end=1,
            grid=complete_grid,
            source=source_a,
            source_token_indices=(0,),
        ),
        make_media_span(
            start=1,
            end=2,
            grid=complete_grid,
            source=source_b,
            source_token_indices=(0,),
        ),
        make_media_span(
            start=2,
            end=3,
            grid=complete_grid,
            source=source_a,
            source_token_indices=(1,),
        ),
        make_media_span(
            start=3,
            end=4,
            grid=complete_grid,
            source=source_b,
            source_token_indices=(1,),
        ),
    )

    make_assembled(
        attention_mask=torch.ones(1, 4, dtype=torch.bool),
        spans=spans,
    ).validate()


@pytest.mark.parametrize(
    "fragments",
    [
        ((0, 1), (1, 2)),
        ((0, 1), (3,)),
        ((2, 3), (0, 1)),
    ],
)
def test_assembled_sequence_rejects_incomplete_or_reordered_source_indices(
    fragments,
):
    source = MediaSource(0, 0, "image-0")
    complete_grid = MediaGrid(1, 2, 2)
    first, second = fragments
    split = len(first)
    spans = (
        make_media_span(
            start=0,
            end=split,
            grid=complete_grid,
            source=source,
            source_token_indices=first,
        ),
        make_media_span(
            start=split,
            end=split + len(second),
            grid=complete_grid,
            source=source,
            source_token_indices=second,
        ),
    )

    with pytest.raises(ValueError, match="row-major"):
        make_assembled(
            attention_mask=torch.ones(
                1, split + len(second), dtype=torch.bool
            ),
            spans=spans,
        ).validate()


def test_assembled_sequence_rejects_conflicting_complete_source_grids():
    source = MediaSource(0, 0, "image-0")
    spans = (
        make_media_span(
            start=0,
            end=1,
            grid=MediaGrid(1, 1, 2),
            source=source,
            source_token_indices=(0,),
        ),
        make_media_span(
            start=1,
            end=2,
            grid=MediaGrid(1, 1, 3),
            source=source,
            source_token_indices=(1,),
        ),
    )

    with pytest.raises(ValueError, match="complete source grid"):
        make_assembled(spans=spans).validate()


def test_assembled_sequence_validates_core_shapes_and_optional_labels():
    make_assembled(labels=torch.zeros(1, 2, dtype=torch.long)).validate()
    make_assembled(labels=None).validate()


@pytest.mark.parametrize(
    "inputs_embeds",
    [
        torch.zeros(1, 2, 4, dtype=torch.bool),
        torch.zeros(1, 2, 4, dtype=torch.int64),
        torch.zeros(1, 2, 4, dtype=torch.complex64),
    ],
)
def test_assembled_sequence_requires_floating_non_complex_embeddings(
    inputs_embeds,
):
    with pytest.raises(TypeError, match="inputs_embeds"):
        make_assembled(inputs_embeds=inputs_embeds).validate()


@pytest.mark.parametrize("dtype", [torch.int32, torch.float32])
def test_assembled_sequence_requires_long_labels(dtype):
    with pytest.raises(TypeError, match="labels"):
        make_assembled(labels=torch.zeros(1, 2, dtype=dtype)).validate()


def test_assembled_sequence_rejects_negative_expanded_input_ids():
    with pytest.raises(ValueError, match="non-negative"):
        make_assembled(
            expanded_input_ids=torch.tensor([[0, -1]])
        ).validate()


def test_assembled_sequence_rejects_non_binary_integer_masks():
    with pytest.raises(ValueError, match="0 or 1"):
        make_assembled(
            attention_mask=torch.tensor([[1, 2]])
        ).validate()


@pytest.mark.parametrize(
    "mismatched_tensor",
    [
        "expanded_input_ids",
        "inputs_embeds",
        "attention_mask",
        "labels",
        "span_timestamps",
    ],
)
def test_assembled_sequence_requires_all_tensors_on_one_device(
    mismatched_tensor,
):
    values = {
        "expanded_input_ids": torch.zeros(1, 2, dtype=torch.long),
        "inputs_embeds": torch.zeros(1, 2, 4),
        "attention_mask": torch.ones(1, 2, dtype=torch.bool),
        "labels": torch.zeros(1, 2, dtype=torch.long),
    }
    timestamps = torch.tensor([0.0, 0.08])
    if mismatched_tensor == "span_timestamps":
        timestamps = torch.empty(2, device="meta")
    else:
        tensor = values[mismatched_tensor]
        values[mismatched_tensor] = torch.empty(
            tensor.shape, dtype=tensor.dtype, device="meta"
        )
    span = make_media_span(
        modality=MediaModality.AUDIO,
        grid=None,
        timestamps=timestamps,
        source_token_indices=(0, 1),
    )
    assembled = make_assembled(
        **values,
        spans=(span,),
    )

    with pytest.raises(ValueError, match="device"):
        assembled.validate()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "expanded_input_ids",
            torch.zeros(2, dtype=torch.long),
            "expanded_input_ids",
        ),
        (
            "expanded_input_ids",
            torch.zeros(1, 2, dtype=torch.int32),
            "expanded_input_ids",
        ),
        (
            "inputs_embeds",
            torch.zeros(1, 2),
            "inputs_embeds",
        ),
        (
            "inputs_embeds",
            torch.zeros(1, 3, 4),
            "inputs_embeds",
        ),
        (
            "attention_mask",
            torch.ones(1, 2, dtype=torch.float32),
            "attention_mask",
        ),
        (
            "labels",
            torch.zeros(1, 3, dtype=torch.long),
            "labels",
        ),
    ],
)
def test_assembled_sequence_rejects_invalid_core_contracts(
    field, value, message
):
    assembled = make_assembled(**{field: value})

    with pytest.raises((TypeError, ValueError), match=message):
        assembled.validate()


def test_assembled_sequence_rejects_attention_mask_shape_mismatch():
    assembled = make_assembled(
        attention_mask=torch.ones(1, 3, dtype=torch.bool),
        expanded_input_ids=torch.zeros(1, 2, dtype=torch.long),
        inputs_embeds=torch.zeros(1, 2, 4),
        spans=(make_span(start=0, end=2),),
    )

    with pytest.raises(ValueError, match="attention_mask"):
        assembled.validate()


def test_assembled_sequence_rejects_overlapping_spans():
    spans = (
        make_span(start=0, end=2),
        make_span(start=1, end=2),
    )

    with pytest.raises(ValueError, match="overlap"):
        make_assembled(spans=spans).validate()


def test_assembled_sequence_requires_exact_valid_token_coverage():
    with pytest.raises(ValueError, match="coverage"):
        make_assembled(
            attention_mask=torch.tensor([[1, 1, 0]]),
            spans=(make_span(start=0, end=1),),
        ).validate()

    with pytest.raises(ValueError, match="masked"):
        make_assembled(
            attention_mask=torch.tensor([[1, 0]]),
            spans=(make_span(start=0, end=2),),
        ).validate()


def test_assembled_sequence_allows_same_offsets_in_different_samples():
    spans = (
        make_span(sample_index=0, start=0, end=2),
        make_span(sample_index=1, start=0, end=2),
    )

    make_assembled(
        attention_mask=torch.ones(2, 2, dtype=torch.bool),
        spans=spans,
    ).validate()


def test_assembled_sequence_requires_spans_in_canonical_offset_order():
    spans = (
        make_span(start=1, end=2),
        make_span(start=0, end=1),
    )

    with pytest.raises(ValueError, match="canonical"):
        make_assembled(spans=spans).validate()


def test_assembled_sequence_requires_spans_in_canonical_sample_order():
    spans = (
        make_span(sample_index=1, start=0, end=2),
        make_span(sample_index=0, start=0, end=2),
    )

    with pytest.raises(ValueError, match="canonical"):
        make_assembled(
            attention_mask=torch.ones(2, 2, dtype=torch.bool),
            spans=spans,
        ).validate()


@pytest.mark.parametrize(
    "span",
    [
        make_span(sample_index=1, start=0, end=1),
        make_span(sample_index=0, start=-1, end=1),
        make_span(sample_index=0, start=1, end=1),
        make_span(sample_index=0, start=0, end=3),
    ],
)
def test_assembled_sequence_rejects_out_of_bounds_spans(span):
    with pytest.raises(ValueError, match="span"):
        make_assembled(spans=(span,)).validate()


def test_media_span_source_must_belong_to_the_same_sample():
    span = make_media_span(
        source=MediaSource(1, 0, "image-0"),
    )

    with pytest.raises(ValueError, match="sample_index"):
        span.validate()


def test_position_batch_accepts_fractional_non_monotonic_axes_and_negative_delta():
    attention_mask = torch.tensor([[1, 1, 1, 1, 0]])
    position_ids = torch.tensor(
        [
            [[0.0, 0.5, 0.5, 1.0, 0.0]],
            [[0.0, 0.0, 1.0, 1.0, 0.0]],
            [[0.0, 1.0, 0.0, 1.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    batch = PositionBatch(
        position_ids=position_ids,
        rope_deltas=torch.tensor([[-2.0]], dtype=torch.float32),
        axis_names=("temporal", "height", "width"),
    )

    batch.validate(attention_mask)


def test_position_batch_accepts_integer_positions():
    PositionBatch(
        position_ids=torch.tensor([[[0, 1, 2]]]),
        rope_deltas=torch.tensor([[0]]),
        axis_names=("sequence",),
    ).validate(torch.ones(1, 3, dtype=torch.bool))


def test_position_batch_rejects_non_binary_integer_attention_mask():
    batch = PositionBatch(
        position_ids=torch.tensor([[[0, 1]]]),
        rope_deltas=torch.tensor([[0]]),
        axis_names=("sequence",),
    )

    with pytest.raises(ValueError, match="0 or 1"):
        batch.validate(torch.tensor([[1, 2]]))


@pytest.mark.parametrize(
    "mismatched_tensor",
    ["position_ids", "rope_deltas", "attention_mask"],
)
def test_position_batch_requires_all_tensors_on_one_device(
    mismatched_tensor,
):
    values = {
        "position_ids": torch.zeros(1, 1, 2),
        "rope_deltas": torch.zeros(1, 1),
        "attention_mask": torch.ones(1, 2, dtype=torch.bool),
    }
    tensor = values[mismatched_tensor]
    values[mismatched_tensor] = torch.empty(
        tensor.shape, dtype=tensor.dtype, device="meta"
    )
    batch = PositionBatch(
        position_ids=values["position_ids"],
        rope_deltas=values["rope_deltas"],
        axis_names=("sequence",),
    )

    with pytest.raises(ValueError, match="device"):
        batch.validate(values["attention_mask"])


@pytest.mark.parametrize(
    ("position_ids", "rope_deltas", "axis_names", "mask", "message"),
    [
        (
            torch.zeros(1, 2),
            torch.zeros(1, 1),
            ("sequence",),
            torch.ones(1, 2, dtype=torch.bool),
            "position_ids",
        ),
        (
            torch.zeros(1, 1, 2),
            torch.zeros(1, 1),
            ("sequence",),
            torch.ones(1, 3, dtype=torch.bool),
            "attention_mask",
        ),
        (
            torch.zeros(1, 1, 2),
            torch.zeros(1),
            ("sequence",),
            torch.ones(1, 2, dtype=torch.bool),
            "rope_deltas",
        ),
        (
            torch.zeros(1, 1, 2),
            torch.zeros(2, 1),
            ("sequence",),
            torch.ones(1, 2, dtype=torch.bool),
            "rope_deltas",
        ),
        (
            torch.zeros(2, 1, 2),
            torch.zeros(1, 1),
            ("temporal",),
            torch.ones(1, 2, dtype=torch.bool),
            "axis_names",
        ),
        (
            torch.zeros(2, 1, 2),
            torch.zeros(1, 1),
            ("same", "same"),
            torch.ones(1, 2, dtype=torch.bool),
            "axis_names",
        ),
    ],
)
def test_position_batch_rejects_shape_and_axis_contract_violations(
    position_ids, rope_deltas, axis_names, mask, message
):
    batch = PositionBatch(
        position_ids=position_ids,
        rope_deltas=rope_deltas,
        axis_names=axis_names,
    )

    with pytest.raises(ValueError, match=message):
        batch.validate(mask)


@pytest.mark.parametrize(
    ("position_ids", "rope_deltas"),
    [
        (
            torch.zeros(1, 1, 2, dtype=torch.float32),
            torch.zeros(1, 1, dtype=torch.float64),
        ),
        (
            torch.zeros(1, 1, 2, dtype=torch.bool),
            torch.zeros(1, 1, dtype=torch.bool),
        ),
        (
            torch.zeros(1, 1, 2, dtype=torch.complex64),
            torch.zeros(1, 1, dtype=torch.complex64),
        ),
    ],
)
def test_position_batch_requires_matching_numeric_non_complex_dtypes(
    position_ids, rope_deltas
):
    batch = PositionBatch(
        position_ids=position_ids,
        rope_deltas=rope_deltas,
        axis_names=("sequence",),
    )

    with pytest.raises(TypeError, match="dtype"):
        batch.validate(torch.ones(1, 2, dtype=torch.bool))


@pytest.mark.parametrize(
    ("position_ids", "rope_deltas"),
    [
        (
            torch.tensor([[[0.0, float("nan")]]]),
            torch.zeros(1, 1),
        ),
        (
            torch.zeros(1, 1, 2),
            torch.tensor([[float("inf")]]),
        ),
    ],
)
def test_position_batch_requires_finite_values(position_ids, rope_deltas):
    batch = PositionBatch(
        position_ids=position_ids,
        rope_deltas=rope_deltas,
        axis_names=("sequence",),
    )

    with pytest.raises(ValueError, match="finite"):
        batch.validate(torch.ones(1, 2, dtype=torch.bool))


def test_position_batch_rejects_negative_valid_positions():
    batch = PositionBatch(
        position_ids=torch.tensor([[[0, -1]]]),
        rope_deltas=torch.zeros(1, 1, dtype=torch.long),
        axis_names=("sequence",),
    )

    with pytest.raises(ValueError, match="non-negative"):
        batch.validate(torch.ones(1, 2, dtype=torch.bool))


def test_position_batch_requires_zero_positions_at_masked_slots():
    batch = PositionBatch(
        position_ids=torch.tensor([[[0, 1, 7]]]),
        rope_deltas=torch.zeros(1, 1, dtype=torch.long),
        axis_names=("sequence",),
    )

    with pytest.raises(ValueError, match="masked"):
        batch.validate(torch.tensor([[1, 1, 0]]))
