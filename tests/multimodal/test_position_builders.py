from __future__ import annotations

import pytest
import torch

from qwen3_omni_pretrain.multimodal.modalities import MediaModality
from qwen3_omni_pretrain.multimodal.positions import (
    LEGACY_AXIS_NAMES,
    THREE_AXIS_NAMES,
    LegacyPositionBuilder,
    Qwen3DisjointPositionBuilder,
    Qwen3DisjointPositionConfig,
    TMRoPEConfig,
    TMRoPEPositionBuilder,
)
from qwen3_omni_pretrain.multimodal.sequence_assembler import (
    SequenceAssembler,
    TimestampInterleaveExpansion,
)
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


def text_span(sample_index: int, start: int, end: int) -> SequenceSpan:
    return SequenceSpan(
        sample_index=sample_index,
        start=start,
        end=end,
        kind=SequenceSpanKind.TEXT,
        modality=None,
        grid=None,
        timestamps=None,
        seconds_per_grid=None,
        source=None,
        source_token_indices=None,
    )


def media_span(
    *,
    sample_index: int,
    start: int,
    end: int,
    modality: MediaModality,
    source: MediaSource,
    indices: tuple[int, ...],
    grid: MediaGrid | None = None,
    timestamps: torch.Tensor | None = None,
    seconds_per_grid: float | None = None,
) -> SequenceSpan:
    return SequenceSpan(
        sample_index=sample_index,
        start=start,
        end=end,
        kind=SequenceSpanKind.MEDIA,
        modality=modality,
        grid=grid,
        timestamps=timestamps,
        seconds_per_grid=seconds_per_grid,
        source=source,
        source_token_indices=indices,
    )


def timestamp_span(
    *,
    sample_index: int,
    start: int,
    end: int,
    modality: MediaModality,
    source: MediaSource,
) -> SequenceSpan:
    return SequenceSpan(
        sample_index=sample_index,
        start=start,
        end=end,
        kind=SequenceSpanKind.TIMESTAMP,
        modality=modality,
        grid=None,
        timestamps=None,
        seconds_per_grid=None,
        source=source,
        source_token_indices=None,
    )


def assembled(
    *,
    masks: list[list[int]],
    spans: tuple[SequenceSpan, ...],
    device: torch.device | str = "cpu",
) -> AssembledSequence:
    mask = torch.tensor(masks, dtype=torch.bool, device=device)
    batch_size, sequence_length = mask.shape
    result = AssembledSequence(
        expanded_input_ids=torch.zeros(
            batch_size,
            sequence_length,
            dtype=torch.long,
            device=device,
        ),
        inputs_embeds=torch.zeros(
            batch_size,
            sequence_length,
            4,
            dtype=torch.float32,
            device=device,
        ),
        attention_mask=mask,
        labels=None,
        spans=spans,
    )
    result.validate()
    return result


def text_only(*masks: list[int]) -> AssembledSequence:
    spans = tuple(
        text_span(row, 0, sum(mask))
        for row, mask in enumerate(masks)
    )
    return assembled(masks=list(masks), spans=spans)


def official_image_example() -> AssembledSequence:
    source = MediaSource(0, 0, "image")
    return assembled(
        masks=[[1] * 8],
        spans=(
            text_span(0, 0, 2),
            media_span(
                sample_index=0,
                start=2,
                end=6,
                modality=MediaModality.IMAGE,
                source=source,
                indices=(0, 1, 2, 3),
                grid=MediaGrid(1, 2, 2),
            ),
            text_span(0, 6, 8),
        ),
    )


def official_video_example(
    *,
    temporal: int = 2,
    seconds_per_grid: float | None = 1.0,
) -> AssembledSequence:
    source = MediaSource(0, 0, "video")
    grid = MediaGrid(temporal, 1 if temporal == 8 else 2, 1 if temporal == 8 else 2)
    count = grid.token_count
    return assembled(
        masks=[[1] * (count + 2)],
        spans=(
            text_span(0, 0, 1),
            media_span(
                sample_index=0,
                start=1,
                end=1 + count,
                modality=MediaModality.VIDEO,
                source=source,
                indices=tuple(range(count)),
                grid=grid,
                timestamps=torch.arange(count, dtype=torch.float32),
                seconds_per_grid=seconds_per_grid,
            ),
            text_span(0, 1 + count, count + 2),
        ),
    )


def test_legacy_positions_are_long_prefix_arange_with_zero_delta():
    source = MediaSource(1, 0, "audio")
    value = assembled(
        masks=[[1, 0, 0], [1, 1, 1]],
        spans=(
            text_span(0, 0, 1),
            text_span(1, 0, 1),
            media_span(
                sample_index=1,
                start=1,
                end=3,
                modality=MediaModality.AUDIO,
                source=source,
                indices=(0, 1),
                timestamps=torch.tensor([0.0, 0.1]),
            ),
        ),
    )

    positions = LegacyPositionBuilder().build(value)

    assert positions.position_ids.dtype is torch.long
    assert positions.position_ids.tolist() == [[[0, 0, 0], [0, 1, 2]]]
    assert positions.rope_deltas.dtype is torch.long
    assert positions.rope_deltas.tolist() == [[0], [0]]
    assert positions.axis_names == LEGACY_AXIS_NAMES == ("sequence",)
    positions.validate(value.attention_mask)


def test_qwen3_official_image_uses_pinned_three_axis_vector():
    value = official_image_example()
    positions = Qwen3DisjointPositionBuilder(
        Qwen3DisjointPositionConfig()
    ).build(value)

    assert positions.position_ids.dtype is torch.float32
    assert positions.axis_names == THREE_AXIS_NAMES
    assert positions.position_ids[:, 0].tolist() == [
        [0.0, 1.0, 2.0, 2.0, 2.0, 2.0, 4.0, 5.0],
        [0.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 5.0],
        [0.0, 1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 5.0],
    ]
    assert positions.rope_deltas.tolist() == [[-2.0]]


def test_qwen3_fractional_video_preserves_official_float32_order():
    value = official_video_example(
        temporal=8,
        seconds_per_grid=0.08,
    )
    positions = Qwen3DisjointPositionBuilder(
        Qwen3DisjointPositionConfig()
    ).build(value)

    assert positions.position_ids[0, 0].tolist() == [
        0.0,
        1.0,
        2.0399999618530273,
        3.0799999237060547,
        4.119999885559082,
        5.159999847412109,
        6.199999809265137,
        7.239999771118164,
        8.280000686645508,
        9.280000686645508,
    ]
    assert positions.rope_deltas.tolist() == [[0.2800006866455078]]


def test_experimental_tm_rope_aligns_audio_to_80ms_grid():
    source = MediaSource(0, 0, "audio")
    value = assembled(
        masks=[[1, 1, 1]],
        spans=(
            media_span(
                sample_index=0,
                start=0,
                end=3,
                modality=MediaModality.AUDIO,
                source=source,
                indices=(0, 1, 2),
                timestamps=torch.tensor([0.0, 0.08, 0.16]),
            ),
        ),
    )
    positions = TMRoPEPositionBuilder(
        TMRoPEConfig(
            temporal_seconds_per_id=0.08,
            rotary_sections=(24, 20, 20),
        )
    ).build(value)

    audio_ids = positions.position_ids[:, 0]
    assert torch.equal(audio_ids[0], audio_ids[1])
    assert torch.equal(audio_ids[1], audio_ids[2])
    assert torch.diff(audio_ids[0]).tolist() == [1.0, 1.0]
    assert positions.rope_deltas.tolist() == [[0.0]]


@pytest.mark.parametrize(
    "config",
    [
        Qwen3DisjointPositionConfig(),
        TMRoPEConfig(0.08, (24, 20, 20)),
    ],
)
def test_three_axis_text_padding_is_zero(config):
    value = text_only([1, 0, 0])
    builder = (
        Qwen3DisjointPositionBuilder(config)
        if isinstance(config, Qwen3DisjointPositionConfig)
        else TMRoPEPositionBuilder(config)
    )
    positions = builder.build(value)
    assert torch.count_nonzero(positions.position_ids[:, 0, 1:]) == 0
    positions.validate(value.attention_mask)


@pytest.mark.parametrize(
    "builder",
    [
        LegacyPositionBuilder(),
        Qwen3DisjointPositionBuilder(Qwen3DisjointPositionConfig()),
        TMRoPEPositionBuilder(TMRoPEConfig(0.08, (24, 20, 20))),
    ],
)
def test_every_builder_rejects_non_prefix_or_empty_rows(builder):
    holey = assembled(
        masks=[[1, 0, 1]],
        spans=(text_span(0, 0, 1), text_span(0, 2, 3)),
    )
    all_masked = assembled(masks=[[0, 0]], spans=())

    with pytest.raises(ValueError, match="prefix"):
        builder.build(holey)
    with pytest.raises(ValueError, match="prefix"):
        builder.build(all_masked)
    with pytest.raises(TypeError, match="AssembledSequence"):
        builder.build(object())


@pytest.mark.parametrize(
    "builder",
    [
        LegacyPositionBuilder(),
        Qwen3DisjointPositionBuilder(Qwen3DisjointPositionConfig()),
        TMRoPEPositionBuilder(TMRoPEConfig(0.08, (24, 20, 20))),
    ],
)
def test_every_builder_rejects_empty_shapes_and_calls_input_validation(builder):
    empty_batch = AssembledSequence(
        expanded_input_ids=torch.empty(0, 2, dtype=torch.long),
        inputs_embeds=torch.empty(0, 2, 4),
        attention_mask=torch.empty(0, 2, dtype=torch.bool),
        labels=None,
        spans=(),
    )
    empty_sequence = AssembledSequence(
        expanded_input_ids=torch.empty(1, 0, dtype=torch.long),
        inputs_embeds=torch.empty(1, 0, 4),
        attention_mask=torch.empty(1, 0, dtype=torch.bool),
        labels=None,
        spans=(),
    )
    missing_coverage = AssembledSequence(
        expanded_input_ids=torch.zeros(1, 1, dtype=torch.long),
        inputs_embeds=torch.zeros(1, 1, 4),
        attention_mask=torch.ones(1, 1, dtype=torch.bool),
        labels=None,
        spans=(),
    )

    with pytest.raises(ValueError, match="non-empty"):
        builder.build(empty_batch)
    with pytest.raises(ValueError, match="non-empty"):
        builder.build(empty_sequence)
    with pytest.raises(ValueError, match="coverage"):
        builder.build(missing_coverage)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize(
    "builder",
    [
        LegacyPositionBuilder(),
        Qwen3DisjointPositionBuilder(Qwen3DisjointPositionConfig()),
        TMRoPEPositionBuilder(TMRoPEConfig(0.08, (24, 20, 20))),
    ],
)
def test_position_builders_preserve_cuda_device(builder):
    value = assembled(
        masks=[[1, 0]],
        spans=(text_span(0, 0, 1),),
        device="cuda",
    )
    result = builder.build(value)
    assert result.position_ids.device.type == "cuda"
    assert result.rope_deltas.device.type == "cuda"


@pytest.mark.parametrize(
    ("config_type", "kwargs", "error"),
    [
        (
            Qwen3DisjointPositionConfig,
            {"position_id_per_seconds": True},
            TypeError,
        ),
        (
            Qwen3DisjointPositionConfig,
            {"position_id_per_seconds": 0.0},
            ValueError,
        ),
        (
            Qwen3DisjointPositionConfig,
            {"position_id_per_seconds": float("nan")},
            ValueError,
        ),
        (
            Qwen3DisjointPositionConfig,
            {"position_id_per_seconds": float("inf")},
            ValueError,
        ),
        (
            Qwen3DisjointPositionConfig,
            {"position_id_per_seconds": "13"},
            TypeError,
        ),
        (
            Qwen3DisjointPositionConfig,
            {"rotary_sections": [24, 20, 20]},
            TypeError,
        ),
        (
            Qwen3DisjointPositionConfig,
            {"rotary_sections": (24, 0, 20)},
            ValueError,
        ),
        (
            Qwen3DisjointPositionConfig,
            {"rotary_sections": (24, 20)},
            ValueError,
        ),
        (
            TMRoPEConfig,
            {
                "temporal_seconds_per_id": True,
                "rotary_sections": (24, 20, 20),
            },
            TypeError,
        ),
        (
            TMRoPEConfig,
            {
                "temporal_seconds_per_id": -0.08,
                "rotary_sections": (24, 20, 20),
            },
            ValueError,
        ),
        (
            TMRoPEConfig,
            {
                "temporal_seconds_per_id": float("inf"),
                "rotary_sections": (24, 20, 20),
            },
            ValueError,
        ),
        (
            TMRoPEConfig,
            {
                "temporal_seconds_per_id": 0.08,
                "rotary_sections": (24, True, 20),
            },
            TypeError,
        ),
        (
            TMRoPEConfig,
            {
                "temporal_seconds_per_id": 0.08,
                "rotary_sections": (24, 20, 20),
                "interleaved": 1,
            },
            TypeError,
        ),
    ],
)
def test_position_configs_reject_ambiguous_values(
    config_type,
    kwargs,
    error,
):
    with pytest.raises(error):
        config_type(**kwargs)


def test_rotary_sections_and_interleaving_are_preserved_as_metadata():
    qwen = Qwen3DisjointPositionConfig()
    tm_true = TMRoPEConfig(0.08, (24, 20, 20), interleaved=True)
    tm_false = TMRoPEConfig(0.08, (24, 20, 20), interleaved=False)
    assert qwen.rotary_sections == (24, 20, 20)
    assert tm_true.rotary_sections == (24, 20, 20)
    assert tm_true.interleaved is True
    assert tm_false.interleaved is False


def test_qwen3_pure_text_padding_uses_official_filler_delta_quirk():
    builder = Qwen3DisjointPositionBuilder(
        Qwen3DisjointPositionConfig()
    )
    one_valid = builder.build(text_only([1, 0, 0]))
    two_valid = builder.build(text_only([1, 1, 0]))
    no_padding = builder.build(text_only([1, 1, 1]))

    assert one_valid.position_ids[:, 0].tolist() == [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
    assert one_valid.rope_deltas.tolist() == [[1.0]]
    assert two_valid.rope_deltas.tolist() == [[0.0]]
    assert no_padding.rope_deltas.tolist() == [[0.0]]


def test_qwen3_media_branch_selection_is_batch_global():
    image = MediaSource(0, 0, "image")
    value = assembled(
        masks=[[1] * 8, [1, 0, 0, 0, 0, 0, 0, 0]],
        spans=(
            text_span(0, 0, 2),
            media_span(
                sample_index=0,
                start=2,
                end=6,
                modality=MediaModality.IMAGE,
                source=image,
                indices=(0, 1, 2, 3),
                grid=MediaGrid(1, 2, 2),
            ),
            text_span(0, 6, 8),
            text_span(1, 0, 1),
        ),
    )

    positions = Qwen3DisjointPositionBuilder(
        Qwen3DisjointPositionConfig()
    ).build(value)

    assert positions.rope_deltas[:, 0].tolist() == [-2.0, 0.0]
    assert torch.count_nonzero(positions.position_ids[:, 1, 1:]) == 0


def test_qwen3_integer_video_and_audio_match_pinned_vectors():
    video = Qwen3DisjointPositionBuilder(
        Qwen3DisjointPositionConfig()
    ).build(official_video_example())
    assert video.position_ids[:, 0].tolist() == [
        [0.0, 1.0, 1.0, 1.0, 1.0, 14.0, 14.0, 14.0, 14.0, 15.0],
        [0.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 15.0],
        [0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 15.0],
    ]
    assert video.rope_deltas.tolist() == [[6.0]]

    source = MediaSource(0, 0, "audio")
    audio_value = assembled(
        masks=[[1, 1, 1, 1]],
        spans=(
            text_span(0, 0, 1),
            media_span(
                sample_index=0,
                start=1,
                end=3,
                modality=MediaModality.AUDIO,
                source=source,
                indices=(0, 1),
                timestamps=torch.tensor([0.0, 0.08]),
            ),
            text_span(0, 3, 4),
        ),
    )
    audio = Qwen3DisjointPositionBuilder(
        Qwen3DisjointPositionConfig()
    ).build(audio_value)
    assert audio.position_ids[:, 0].tolist() == [
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
    ]
    assert audio.rope_deltas.tolist() == [[0.0]]


def test_qwen3_continuation_uses_all_axes_across_disjoint_media_blocks():
    image = MediaSource(0, 0, "image")
    audio = MediaSource(0, 1, "audio")
    value = assembled(
        masks=[[1, 1, 1, 1, 1, 1, 1]],
        spans=(
            text_span(0, 0, 1),
            media_span(
                sample_index=0,
                start=1,
                end=5,
                modality=MediaModality.IMAGE,
                source=image,
                indices=(0, 1, 2, 3),
                grid=MediaGrid(1, 2, 2),
            ),
            media_span(
                sample_index=0,
                start=5,
                end=7,
                modality=MediaModality.AUDIO,
                source=audio,
                indices=(0, 1),
                timestamps=torch.tensor([0.0, 0.08]),
            ),
        ),
    )
    positions = Qwen3DisjointPositionBuilder(
        Qwen3DisjointPositionConfig()
    ).build(value)

    assert positions.position_ids[:, 0, 1:5].tolist() == [
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 2.0, 2.0],
        [1.0, 2.0, 1.0, 2.0],
    ]
    assert positions.position_ids[:, 0, 5:7].tolist() == [
        [3.0, 4.0],
        [3.0, 4.0],
        [3.0, 4.0],
    ]
    assert positions.rope_deltas.tolist() == [[-2.0]]


def test_qwen3_video_cadence_and_source_fragment_contracts():
    builder = Qwen3DisjointPositionBuilder(
        Qwen3DisjointPositionConfig()
    )
    with pytest.raises(ValueError, match="seconds_per_grid"):
        builder.build(
            official_video_example(
                temporal=2,
                seconds_per_grid=None,
            )
        )

    temporal_one = official_video_example(
        temporal=1,
        seconds_per_grid=None,
    )
    assert builder.build(temporal_one).position_ids.shape == (3, 1, 6)

    source = MediaSource(0, 0, "fragmented")
    adjacent = assembled(
        masks=[[1, 1]],
        spans=(
            media_span(
                sample_index=0,
                start=0,
                end=1,
                modality=MediaModality.AUDIO,
                source=source,
                indices=(0,),
                timestamps=torch.tensor([0.0]),
            ),
            media_span(
                sample_index=0,
                start=1,
                end=2,
                modality=MediaModality.AUDIO,
                source=source,
                indices=(1,),
                timestamps=torch.tensor([0.1]),
            ),
        ),
    )
    with pytest.raises(ValueError, match="one complete MEDIA span"):
        builder.build(adjacent)

    other = MediaSource(0, 1, "other")
    aba = assembled(
        masks=[[1, 1, 1]],
        spans=(
            media_span(
                sample_index=0,
                start=0,
                end=1,
                modality=MediaModality.AUDIO,
                source=source,
                indices=(0,),
                timestamps=torch.tensor([0.0]),
            ),
            media_span(
                sample_index=0,
                start=1,
                end=2,
                modality=MediaModality.AUDIO,
                source=other,
                indices=(0,),
                timestamps=torch.tensor([0.05]),
            ),
            media_span(
                sample_index=0,
                start=2,
                end=3,
                modality=MediaModality.AUDIO,
                source=source,
                indices=(1,),
                timestamps=torch.tensor([0.1]),
            ),
        ),
    )
    with pytest.raises(ValueError, match="one complete MEDIA span"):
        builder.build(aba)


def test_tm_rope_keeps_80ms_and_160ms_grids_distinct():
    source = MediaSource(0, 0, "audio")
    value = assembled(
        masks=[[1, 1, 1]],
        spans=(
            media_span(
                sample_index=0,
                start=0,
                end=3,
                modality=MediaModality.AUDIO,
                source=source,
                indices=(0, 1, 2),
                timestamps=torch.tensor([0.0, 0.08, 0.16]),
            ),
        ),
    )
    ids_80 = TMRoPEPositionBuilder(
        TMRoPEConfig(0.08, (24, 20, 20))
    ).build(value).position_ids[0, 0]
    ids_160 = TMRoPEPositionBuilder(
        TMRoPEConfig(0.16, (24, 20, 20))
    ).build(value).position_ids[0, 0]

    assert ids_80.tolist() == [0.0, 1.0, 2.0]
    assert ids_160.tolist() == [0.0, 1.0, 1.0]


def test_tm_rope_audio_and_video_share_global_time_not_spatial_axes():
    audio_source = MediaSource(0, 0, "audio")
    video_source = MediaSource(0, 1, "video")
    value = assembled(
        masks=[[1, 1, 1]],
        spans=(
            media_span(
                sample_index=0,
                start=0,
                end=1,
                modality=MediaModality.AUDIO,
                source=audio_source,
                indices=(0,),
                timestamps=torch.tensor([0.32]),
            ),
            media_span(
                sample_index=0,
                start=1,
                end=3,
                modality=MediaModality.VIDEO,
                source=video_source,
                indices=(0, 1),
                grid=MediaGrid(1, 1, 2),
                timestamps=torch.tensor([0.32, 0.32]),
                seconds_per_grid=0.5,
            ),
        ),
    )
    positions = TMRoPEPositionBuilder(
        TMRoPEConfig(0.16, (24, 20, 20))
    ).build(value)

    assert positions.position_ids[0, 0].tolist() == [2.0, 2.0, 2.0]
    assert positions.position_ids[1, 0].tolist() == [2.0, 0.0, 0.0]
    assert positions.position_ids[2, 0].tolist() == [2.0, 0.0, 1.0]
    assert positions.rope_deltas.tolist() == [[0.0]]


def test_tm_rope_multiframe_video_quantizes_once_per_frame():
    source = MediaSource(0, 0, "video")
    value = assembled(
        masks=[[1, 1, 1, 1]],
        spans=(
            media_span(
                sample_index=0,
                start=0,
                end=4,
                modality=MediaModality.VIDEO,
                source=source,
                indices=(0, 1, 2, 3),
                grid=MediaGrid(2, 1, 2),
                timestamps=torch.tensor([0.0, 0.0, 0.16, 0.16]),
                seconds_per_grid=0.16,
            ),
        ),
    )
    positions = TMRoPEPositionBuilder(
        TMRoPEConfig(0.08, (24, 20, 20))
    ).build(value)

    assert positions.position_ids[:, 0].tolist() == [
        [0.0, 0.0, 2.0, 2.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
    ]
    assert positions.rope_deltas.tolist() == [[-1.0]]


def test_tm_rope_text_and_markers_advance_without_moving_timeline_anchor():
    source = MediaSource(0, 0, "audio")
    value = assembled(
        masks=[[1, 1, 1, 1]],
        spans=(
            media_span(
                sample_index=0,
                start=0,
                end=1,
                modality=MediaModality.AUDIO,
                source=source,
                indices=(0,),
                timestamps=torch.tensor([0.0]),
            ),
            text_span(0, 1, 2),
            timestamp_span(
                sample_index=0,
                start=2,
                end=3,
                modality=MediaModality.AUDIO,
                source=source,
            ),
            media_span(
                sample_index=0,
                start=3,
                end=4,
                modality=MediaModality.AUDIO,
                source=source,
                indices=(1,),
                timestamps=torch.tensor([0.0]),
            ),
        ),
    )
    positions = TMRoPEPositionBuilder(
        TMRoPEConfig(0.08, (24, 20, 20))
    ).build(value)

    assert positions.position_ids[:, 0].tolist() == [
        [0.0, 1.0, 2.0, 0.0],
        [0.0, 1.0, 2.0, 0.0],
        [0.0, 1.0, 2.0, 0.0],
    ]
    assert positions.rope_deltas.tolist() == [[-1.0]]


def test_tm_rope_consecutive_images_and_first_av_use_safe_continuation():
    first = MediaSource(0, 0, "first-image")
    second = MediaSource(0, 1, "second-image")
    images = assembled(
        masks=[[1, 1]],
        spans=(
            media_span(
                sample_index=0,
                start=0,
                end=1,
                modality=MediaModality.IMAGE,
                source=first,
                indices=(0,),
                grid=MediaGrid(1, 1, 1),
            ),
            media_span(
                sample_index=0,
                start=1,
                end=2,
                modality=MediaModality.IMAGE,
                source=second,
                indices=(0,),
                grid=MediaGrid(1, 1, 1),
            ),
        ),
    )
    builder = TMRoPEPositionBuilder(
        TMRoPEConfig(0.08, (24, 20, 20))
    )
    assert builder.build(images).position_ids[:, 0].tolist() == [
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ]

    image = MediaSource(0, 0, "image")
    audio = MediaSource(0, 1, "audio")
    image_then_audio = assembled(
        masks=[[1, 1, 1, 1, 1]],
        spans=(
            media_span(
                sample_index=0,
                start=0,
                end=4,
                modality=MediaModality.IMAGE,
                source=image,
                indices=(0, 1, 2, 3),
                grid=MediaGrid(1, 2, 2),
            ),
            media_span(
                sample_index=0,
                start=4,
                end=5,
                modality=MediaModality.AUDIO,
                source=audio,
                indices=(0,),
                timestamps=torch.tensor([0.0]),
            ),
        ),
    )
    positions = builder.build(image_then_audio)
    assert positions.position_ids[:, 0, :4].tolist() == [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
    ]
    assert positions.position_ids[:, 0, 4].tolist() == [2.0, 2.0, 2.0]


def test_tm_rope_accepts_aba_av_fragments_with_original_coordinates():
    video = MediaSource(0, 0, "video")
    audio = MediaSource(0, 1, "audio")
    value = assembled(
        masks=[[1, 1, 1]],
        spans=(
            media_span(
                sample_index=0,
                start=0,
                end=1,
                modality=MediaModality.VIDEO,
                source=video,
                indices=(0,),
                grid=MediaGrid(1, 1, 2),
                timestamps=torch.tensor([0.0]),
            ),
            media_span(
                sample_index=0,
                start=1,
                end=2,
                modality=MediaModality.AUDIO,
                source=audio,
                indices=(0,),
                timestamps=torch.tensor([0.08]),
            ),
            media_span(
                sample_index=0,
                start=2,
                end=3,
                modality=MediaModality.VIDEO,
                source=video,
                indices=(1,),
                grid=MediaGrid(1, 1, 2),
                timestamps=torch.tensor([0.0]),
            ),
        ),
    )
    positions = TMRoPEPositionBuilder(
        TMRoPEConfig(0.08, (24, 20, 20))
    ).build(value)

    assert positions.position_ids[:, 0].tolist() == [
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
    ]
    assert positions.rope_deltas.tolist() == [[-1.0]]


def test_tm_rope_rejects_unsupported_metadata_and_split_images():
    builder = TMRoPEPositionBuilder(
        TMRoPEConfig(0.08, (24, 20, 20))
    )
    audio = MediaSource(0, 0, "audio")
    missing_audio_time = assembled(
        masks=[[1]],
        spans=(
            media_span(
                sample_index=0,
                start=0,
                end=1,
                modality=MediaModality.AUDIO,
                source=audio,
                indices=(0,),
            ),
        ),
    )
    with pytest.raises(ValueError, match="timestamps"):
        builder.build(missing_audio_time)

    video = MediaSource(0, 0, "video")
    missing_video_time = assembled(
        masks=[[1]],
        spans=(
            media_span(
                sample_index=0,
                start=0,
                end=1,
                modality=MediaModality.VIDEO,
                source=video,
                indices=(0,),
                grid=MediaGrid(1, 1, 1),
            ),
        ),
    )
    with pytest.raises(ValueError, match="timestamps"):
        builder.build(missing_video_time)

    image = MediaSource(0, 0, "image")
    timestamped_image = assembled(
        masks=[[1]],
        spans=(
            media_span(
                sample_index=0,
                start=0,
                end=1,
                modality=MediaModality.IMAGE,
                source=image,
                indices=(0,),
                grid=MediaGrid(1, 1, 1),
                timestamps=torch.tensor([0.0]),
            ),
        ),
    )
    with pytest.raises(ValueError, match="untimestamped"):
        builder.build(timestamped_image)

    split_image = assembled(
        masks=[[1, 1]],
        spans=(
            media_span(
                sample_index=0,
                start=0,
                end=1,
                modality=MediaModality.IMAGE,
                source=image,
                indices=(0,),
                grid=MediaGrid(1, 1, 2),
            ),
            media_span(
                sample_index=0,
                start=1,
                end=2,
                modality=MediaModality.IMAGE,
                source=image,
                indices=(1,),
                grid=MediaGrid(1, 1, 2),
            ),
        ),
    )
    with pytest.raises(ValueError, match="split image"):
        builder.build(split_image)


def test_tm_rope_rejects_inconsistent_video_frame_patch_timestamps():
    source = MediaSource(0, 0, "video")
    value = assembled(
        masks=[[1, 1]],
        spans=(
            media_span(
                sample_index=0,
                start=0,
                end=2,
                modality=MediaModality.VIDEO,
                source=source,
                indices=(0, 1),
                grid=MediaGrid(1, 1, 2),
                timestamps=torch.tensor([0.0, 0.08]),
            ),
        ),
    )
    with pytest.raises(ValueError, match="one frame"):
        TMRoPEPositionBuilder(
            TMRoPEConfig(0.08, (24, 20, 20))
        ).build(value)


def test_tm_rope_padding_and_interleaved_metadata_do_not_change_positions():
    source = MediaSource(0, 0, "audio")
    unpadded = assembled(
        masks=[[1, 1]],
        spans=(
            media_span(
                sample_index=0,
                start=0,
                end=2,
                modality=MediaModality.AUDIO,
                source=source,
                indices=(0, 1),
                timestamps=torch.tensor([0.0, 0.08]),
            ),
        ),
    )
    padded = assembled(
        masks=[[1, 1, 0, 0]],
        spans=unpadded.spans,
    )
    true_builder = TMRoPEPositionBuilder(
        TMRoPEConfig(0.08, (24, 20, 20), interleaved=True)
    )
    false_builder = TMRoPEPositionBuilder(
        TMRoPEConfig(0.08, (24, 20, 20), interleaved=False)
    )

    compact = true_builder.build(unpadded)
    with_padding = true_builder.build(padded)
    false_layout = false_builder.build(unpadded)
    assert torch.equal(
        compact.position_ids,
        with_padding.position_ids[:, :, :2],
    )
    assert torch.equal(compact.rope_deltas, with_padding.rope_deltas)
    assert torch.equal(compact.position_ids, false_layout.position_ids)
    assert torch.equal(compact.rope_deltas, false_layout.rope_deltas)
    assert torch.count_nonzero(with_padding.position_ids[:, :, 2:]) == 0


def test_tm_rope_supports_unequal_text_only_batch_rows():
    value = text_only([1, 0, 0], [1, 1, 1])
    positions = TMRoPEPositionBuilder(
        TMRoPEConfig(0.08, (24, 20, 20))
    ).build(value)
    assert positions.position_ids[:, 0].tolist() == [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
    assert positions.position_ids[:, 1].tolist() == [
        [0.0, 1.0, 2.0],
        [0.0, 1.0, 2.0],
        [0.0, 1.0, 2.0],
    ]
    assert positions.rope_deltas.tolist() == [[0.0], [0.0]]


def test_task5_markers_and_tm_positions_share_float32_half_up_boundaries():
    step = torch.tensor(0.08, dtype=torch.float32)
    half = step * torch.tensor(0.5, dtype=torch.float32)
    one_and_half = step * torch.tensor(1.5, dtype=torch.float32)
    negative = torch.tensor(float("-inf"), dtype=torch.float32)
    positive = torch.tensor(float("inf"), dtype=torch.float32)
    timestamps = torch.stack(
        (
            torch.nextafter(half, negative),
            half,
            torch.nextafter(half, positive),
            torch.nextafter(one_and_half, negative),
            one_and_half,
            torch.nextafter(one_and_half, positive),
        )
    )
    expected_buckets = quantize_timestamps_half_up(timestamps, 0.08)
    assert expected_buckets.tolist() == [0, 1, 1, 1, 2, 2]

    source = MediaSource(0, 0, "audio")
    media = MediaSequence(
        embeddings=torch.zeros(1, timestamps.numel(), 4),
        attention_mask=torch.ones(
            1,
            timestamps.numel(),
            dtype=torch.bool,
        ),
        modality=MediaModality.AUDIO,
        sources=(source,),
        timestamps=timestamps.unsqueeze(0),
    )
    media.validate()
    marker_ids = {0: 30, 1: 31, 2: 32}
    assembled_value = SequenceAssembler().assemble(
        input_ids=torch.tensor([[12]]),
        text_embeddings=torch.zeros(1, 1, 4),
        attention_mask=torch.ones(1, 1, dtype=torch.long),
        labels=None,
        media_sequences=(media,),
        tokens=ResolvedMultimodalTokens(
            image_pad=10,
            video_pad=11,
            audio_pad=12,
            vision_start=13,
            vision_end=14,
            audio_start=15,
            audio_end=16,
        ),
        expansion_policy=TimestampInterleaveExpansion(
            seconds_per_bucket=0.08,
            bucket_token_ids=marker_ids,
        ),
        embedding_lookup=lambda token_ids: torch.zeros(
            token_ids.shape[0], 4
        ),
        pad_token_id=0,
        max_assembled_length=32,
    )
    emitted_markers = [
        int(assembled_value.expanded_input_ids[0, span.start])
        for span in assembled_value.spans
        if span.kind is SequenceSpanKind.TIMESTAMP
    ]
    assert emitted_markers == [30, 31, 32]

    positions = TMRoPEPositionBuilder(
        TMRoPEConfig(0.08, (24, 20, 20))
    ).build(assembled_value)
    media_temporal = torch.cat(
        [
            positions.position_ids[
                0,
                0,
                span.start : span.end,
            ]
            for span in assembled_value.spans
            if span.kind is SequenceSpanKind.MEDIA
        ]
    )
    assert torch.equal(media_temporal - 1.0, expected_buckets.float())
