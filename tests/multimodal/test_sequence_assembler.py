from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch

from qwen3_omni_pretrain.multimodal.modalities import MediaModality
from qwen3_omni_pretrain.multimodal.sequence_assembler import (
    AssembledLengthError,
    ExpandedMediaRow,
    ExpandedMediaSample,
    ExpansionToken,
    IdentityMediaExpansion,
    MediaTokenRef,
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
    MediaGrid,
    MediaSequence,
    MediaSource,
    SequenceSpanKind,
)


def resolved_tokens(
    *,
    image_pad: int = 10,
    video_pad: int = 11,
    audio_pad: int = 12,
) -> ResolvedMultimodalTokens:
    return ResolvedMultimodalTokens(
        image_pad=image_pad,
        video_pad=video_pad,
        audio_pad=audio_pad,
        vision_start=13,
        vision_end=14,
        audio_start=15,
        audio_end=16,
    )


def make_image_sequence(
    *,
    embeddings: torch.Tensor,
    source: MediaSource,
) -> MediaSequence:
    sequence = MediaSequence(
        embeddings=embeddings,
        attention_mask=torch.ones(
            embeddings.shape[:2],
            dtype=torch.bool,
        ),
        modality=MediaModality.IMAGE,
        sources=(source,),
        grid=(MediaGrid(1, 1, embeddings.shape[1]),),
    )
    sequence.validate()
    return sequence


def make_audio_sequence(
    *,
    values: list[float],
    source: MediaSource,
    timestamps: list[float] | None = None,
    attention_mask: list[int] | None = None,
    hidden_size: int = 3,
    dtype: torch.dtype = torch.float32,
) -> MediaSequence:
    count = len(values)
    embeddings = torch.tensor(
        [[[value] * hidden_size for value in values]],
        dtype=dtype,
    )
    mask_values = attention_mask or [1] * count
    sequence = MediaSequence(
        embeddings=embeddings,
        attention_mask=torch.tensor(
            [mask_values],
            dtype=torch.bool,
        ),
        modality=MediaModality.AUDIO,
        sources=(source,),
        grid=None,
        timestamps=(
            None
            if timestamps is None
            else torch.tensor([timestamps], dtype=torch.float32)
        ),
        seconds_per_grid=None,
    )
    sequence.validate()
    return sequence


def make_video_sequence(
    *,
    values: list[float],
    source: MediaSource,
    timestamps: list[float] | None,
    grid: MediaGrid | None = None,
    seconds_per_grid: float | None = None,
    attention_mask: list[int] | None = None,
    hidden_size: int = 3,
) -> MediaSequence:
    mask_values = attention_mask or [1] * len(values)
    valid_count = sum(mask_values)
    sequence = MediaSequence(
        embeddings=torch.tensor(
            [[[value] * hidden_size for value in values]],
            dtype=torch.float32,
        ),
        attention_mask=torch.tensor(
            [mask_values],
            dtype=torch.bool,
        ),
        modality=MediaModality.VIDEO,
        sources=(source,),
        grid=(grid or MediaGrid(1, 1, valid_count),),
        timestamps=(
            None
            if timestamps is None
            else torch.tensor([timestamps], dtype=torch.float32)
        ),
        seconds_per_grid=(seconds_per_grid,),
    )
    sequence.validate()
    return sequence


def text_embeddings(
    input_ids: torch.Tensor,
    *,
    hidden_size: int = 3,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return torch.arange(
        input_ids.numel() * hidden_size,
        dtype=dtype,
        device=input_ids.device,
    ).reshape(*input_ids.shape, hidden_size)


def assemble(
    input_ids: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    media_sequences=(),
    tokens: ResolvedMultimodalTokens | None = None,
    expansion_policy=None,
    embedding_lookup=None,
    pad_token_id: int = 0,
    joint_separator_token_ids: frozenset[int] = frozenset(),
    max_assembled_length: int = 64,
    embeddings: torch.Tensor | None = None,
):
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    if embeddings is None:
        embeddings = text_embeddings(input_ids)
    return SequenceAssembler().assemble(
        input_ids=input_ids,
        text_embeddings=embeddings,
        attention_mask=attention_mask,
        labels=labels,
        media_sequences=media_sequences,
        tokens=tokens or resolved_tokens(),
        expansion_policy=expansion_policy,
        embedding_lookup=embedding_lookup,
        pad_token_id=pad_token_id,
        joint_separator_token_ids=joint_separator_token_ids,
        max_assembled_length=max_assembled_length,
    )


class RecordingIdentityPolicy:
    def __init__(self):
        self.calls = []
        self.delegate = IdentityMediaExpansion()

    def expand_sample(self, *, sample_index, groups):
        self.calls.append((sample_index, groups))
        return self.delegate.expand_sample(
            sample_index=sample_index,
            groups=groups,
        )


class RecordingLookup:
    def __init__(self, *, hidden_size=3, dtype=torch.float32):
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.calls = []

    def __call__(self, token_ids):
        self.calls.append(token_ids.clone())
        return token_ids.to(dtype=self.dtype).unsqueeze(1).repeat(
            1,
            self.hidden_size,
        )


class TwoTimestampPolicy:
    def expand_sample(self, *, sample_index, groups):
        identity = IdentityMediaExpansion().expand_sample(
            sample_index=sample_index,
            groups=groups,
        )
        replacements = {}
        for group in groups:
            for placeholder in group.placeholders:
                row = identity.replacements[placeholder.text_position]
                replacements[placeholder.text_position] = ExpandedMediaRow(
                    (
                        ExpansionToken(
                            30,
                            SequenceSpanKind.TIMESTAMP,
                            source=placeholder.source,
                        ),
                        ExpansionToken(
                            31,
                            SequenceSpanKind.TIMESTAMP,
                            source=placeholder.source,
                        ),
                        *row.tokens,
                    )
                )
        return ExpandedMediaSample(replacements)


class FunctionalPolicy:
    def __init__(self, function):
        self.function = function

    def expand_sample(self, *, sample_index, groups):
        return self.function(sample_index, groups)


def strict_kwargs(policy=None):
    ids = torch.tensor([[5, 0]])
    return {
        "input_ids": ids,
        "text_embeddings": text_embeddings(ids),
        "attention_mask": torch.tensor([[1, 0]]),
        "labels": torch.tensor([[5, -100]]),
        "media_sequences": (),
        "tokens": resolved_tokens(),
        "expansion_policy": policy,
        "pad_token_id": 0,
        "max_assembled_length": 16,
    }


def test_assembler_replaces_sentinel_and_masks_media_labels():
    input_ids = torch.tensor([[5, 10, 6, 0]])
    text_embeds = torch.arange(4 * 3).view(1, 4, 3).float()
    media = make_image_sequence(
        embeddings=torch.tensor(
            [[[100.0] * 3, [200.0] * 3]]
        ),
        source=MediaSource(0, 0, "image-0"),
    )

    result = SequenceAssembler().assemble(
        input_ids=input_ids,
        text_embeddings=text_embeds,
        attention_mask=torch.tensor([[1, 1, 1, 0]]),
        labels=torch.tensor([[-100, -100, 6, -100]]),
        media_sequences=(media,),
        tokens=resolved_tokens(),
        pad_token_id=0,
        max_assembled_length=16,
    )

    assert result.inputs_embeds.shape == (1, 4, 3)
    assert result.expanded_input_ids.tolist() == [[5, 10, 10, 6]]
    assert result.inputs_embeds[0, 1:3, 0].tolist() == [100.0, 200.0]
    assert result.labels.tolist() == [[-100, -100, -100, 6]]
    assert result.attention_mask.tolist() == [[1, 1, 1, 1]]


def test_half_up_quantizer_preserves_shape_device_and_downcasts_float64():
    values = torch.tensor(
        [[0.49, 0.5], [1.49, 1.5]],
        dtype=torch.float64,
    )

    buckets = quantize_timestamps_half_up(values, 1.0)

    expected = torch.floor(
        values.float()
        / torch.tensor(1.0, dtype=torch.float32)
        + torch.tensor(0.5, dtype=torch.float32)
    ).long()
    assert torch.equal(buckets, expected)
    assert buckets.tolist() == [[0, 1], [1, 2]]
    assert buckets.dtype is torch.long
    assert buckets.shape == values.shape
    assert buckets.device == values.device


@pytest.mark.parametrize(
    "timestamps",
    [
        [0.0],
        torch.tensor([0], dtype=torch.long),
        torch.tensor([1 + 0j]),
        torch.tensor([-0.1]),
        torch.tensor([float("nan")]),
        torch.tensor([float("inf")]),
        torch.tensor([1e40], dtype=torch.float64),
    ],
)
def test_half_up_quantizer_rejects_invalid_or_float32_overflow_values(
    timestamps,
):
    with pytest.raises((TypeError, ValueError)):
        quantize_timestamps_half_up(timestamps, 1.0)


@pytest.mark.parametrize(
    "step",
    [
        True,
        "1.0",
        0.0,
        -1.0,
        float("nan"),
        float("inf"),
        1e40,
        1e-50,
    ],
)
def test_half_up_quantizer_rejects_invalid_or_float32_bad_steps(step):
    with pytest.raises((TypeError, ValueError)):
        quantize_timestamps_half_up(torch.tensor([0.0]), step)


def test_half_up_quantizer_rejects_bucket_result_overflow():
    with pytest.raises(ValueError, match="bucket values"):
        quantize_timestamps_half_up(
            torch.tensor([3e38], dtype=torch.float32),
            1e-38,
        )


def test_transport_order_does_not_change_image_text_audio_pairing():
    image = make_image_sequence(
        embeddings=torch.tensor(
            [[[101.0] * 3, [102.0] * 3]]
        ),
        source=MediaSource(0, 0, "image-0"),
    )
    audio = make_audio_sequence(
        values=[201.0, 202.0],
        source=MediaSource(0, 1, "audio-1"),
        timestamps=[0.0, 0.1],
    )
    ids = torch.tensor([[10, 5, 12]])
    labels = torch.tensor([[-100, 5, -100]])

    result = assemble(
        ids,
        labels=labels,
        media_sequences=(audio, image),
    )

    assert result.expanded_input_ids.tolist() == [
        [10, 10, 5, 12, 12]
    ]
    assert result.inputs_embeds[0, :, 0].tolist() == [
        101.0,
        102.0,
        text_embeddings(ids)[0, 1, 0].item(),
        201.0,
        202.0,
    ]
    assert result.labels.tolist() == [[-100, -100, 5, -100, -100]]
    assert [
        (
            span.kind,
            span.start,
            span.end,
            span.source.source_id if span.source else None,
            span.source_token_indices,
        )
        for span in result.spans
    ] == [
        (SequenceSpanKind.MEDIA, 0, 2, "image-0", (0, 1)),
        (SequenceSpanKind.TEXT, 2, 3, None, None),
        (SequenceSpanKind.MEDIA, 3, 5, "audio-1", (0, 1)),
    ]


def test_row_order_inside_media_sequence_is_transport_only():
    first = MediaSource(0, 0, "first")
    second = MediaSource(0, 1, "second")

    def batched_sequence(order: tuple[int, int]) -> MediaSequence:
        values = torch.tensor([[[10.0] * 3], [[20.0] * 3]])
        sources = (first, second)
        sequence = MediaSequence(
            embeddings=values[list(order)],
            attention_mask=torch.ones(2, 1, dtype=torch.bool),
            modality=MediaModality.AUDIO,
            sources=tuple(sources[index] for index in order),
        )
        sequence.validate()
        return sequence

    canonical = assemble(
        torch.tensor([[12, 12]]),
        media_sequences=(batched_sequence((0, 1)),),
    )
    shuffled = assemble(
        torch.tensor([[12, 12]]),
        media_sequences=(batched_sequence((1, 0)),),
    )

    assert torch.equal(
        canonical.expanded_input_ids,
        shuffled.expanded_input_ids,
    )
    assert torch.equal(canonical.inputs_embeds, shuffled.inputs_embeds)
    assert torch.equal(canonical.attention_mask, shuffled.attention_mask)
    assert [span.source for span in canonical.spans] == [
        span.source for span in shuffled.spans
    ]


def test_two_same_modality_items_and_image_video_preserve_item_order():
    first = make_image_sequence(
        embeddings=torch.tensor([[[10.0] * 3]]),
        source=MediaSource(0, 0, "first"),
    )
    second = make_image_sequence(
        embeddings=torch.tensor([[[20.0] * 3]]),
        source=MediaSource(0, 1, "second"),
    )
    same = assemble(
        torch.tensor([[10, 10]]),
        media_sequences=(second, first),
    )
    assert same.inputs_embeds[0, :, 0].tolist() == [10.0, 20.0]

    image = make_image_sequence(
        embeddings=torch.tensor([[[30.0] * 3]]),
        source=MediaSource(0, 0, "image"),
    )
    video = make_video_sequence(
        values=[40.0, 41.0],
        source=MediaSource(0, 1, "video"),
        timestamps=[0.0, 0.5],
        grid=MediaGrid(1, 1, 2),
        seconds_per_grid=0.5,
    )
    mixed = assemble(
        torch.tensor([[10, 11]]),
        media_sequences=(video, image),
    )
    assert mixed.expanded_input_ids.tolist() == [[10, 11, 11]]
    assert mixed.inputs_embeds[0, :, 0].tolist() == [30.0, 40.0, 41.0]
    video_span = mixed.spans[-1]
    assert video_span.grid == MediaGrid(1, 1, 2)
    assert video_span.timestamps.tolist() == [0.0, 0.5]
    assert video_span.seconds_per_grid == 0.5


def test_nonzero_output_padding_and_labels_none_are_preserved():
    media = make_image_sequence(
        embeddings=torch.tensor(
            [[[100.0] * 3, [200.0] * 3, [300.0] * 3]]
        ),
        source=MediaSource(1, 0, "image-1"),
    )
    ids = torch.tensor([[5, 6], [10, 7]])
    original_embeddings = text_embeddings(ids)

    result = assemble(
        ids,
        labels=None,
        media_sequences=(media,),
        pad_token_id=99,
        embeddings=original_embeddings,
    )

    assert result.expanded_input_ids.tolist() == [
        [5, 6, 99, 99],
        [10, 10, 10, 7],
    ]
    assert result.attention_mask.tolist() == [
        [True, True, False, False],
        [True, True, True, True],
    ]
    assert result.labels is None
    assert torch.equal(
        result.inputs_embeds[0, :2],
        original_embeddings[0],
    )
    assert torch.count_nonzero(result.inputs_embeds[0, 2:]) == 0


def test_valid_text_token_may_equal_pad_token_id():
    ids = torch.tensor([[99]])
    result = assemble(ids, pad_token_id=99)
    assert result.expanded_input_ids.tolist() == [[99]]
    assert result.attention_mask.tolist() == [[True]]


def test_no_media_row_calls_policy_once_with_empty_groups():
    policy = RecordingIdentityPolicy()
    result = assemble(
        torch.tensor([[5, 6]]),
        expansion_policy=policy,
    )
    assert result.expanded_input_ids.tolist() == [[5, 6]]
    assert len(policy.calls) == 1
    assert policy.calls[0] == (0, ())


def test_mixed_batch_calls_policy_once_for_text_only_and_media_rows():
    policy = RecordingIdentityPolicy()
    media = make_audio_sequence(
        values=[7.0],
        source=MediaSource(1, 0, "audio-1"),
    )
    ids = torch.tensor([[5, 0], [12, 0]])

    result = assemble(
        ids,
        attention_mask=torch.tensor([[1, 0], [1, 0]]),
        media_sequences=(media,),
        expansion_policy=policy,
    )

    assert [(sample, len(groups)) for sample, groups in policy.calls] == [
        (0, 0),
        (1, 1),
    ]
    assert result.expanded_input_ids.tolist() == [[5], [12]]


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        (
            "input_ids",
            torch.tensor([[5.0, 0.0]]),
            TypeError,
        ),
        (
            "input_ids",
            torch.tensor([[5, -1]]),
            ValueError,
        ),
        (
            "input_ids",
            torch.tensor([5, 0]),
            ValueError,
        ),
        (
            "text_embeddings",
            torch.zeros(1, 2, 3, dtype=torch.long),
            TypeError,
        ),
        (
            "text_embeddings",
            torch.zeros(1, 2, 3, dtype=torch.complex64),
            TypeError,
        ),
        (
            "text_embeddings",
            torch.zeros(1, 3, 3),
            ValueError,
        ),
        (
            "attention_mask",
            torch.tensor([[1.0, 0.0]]),
            TypeError,
        ),
        (
            "attention_mask",
            torch.tensor([[1, 2]]),
            ValueError,
        ),
        (
            "attention_mask",
            torch.tensor([[0, 1]]),
            ValueError,
        ),
        (
            "attention_mask",
            torch.tensor([[0, 0]]),
            ValueError,
        ),
        (
            "labels",
            torch.tensor([[5.0, -100.0]]),
            TypeError,
        ),
        (
            "labels",
            torch.tensor([[5, -1]]),
            ValueError,
        ),
        (
            "labels",
            torch.tensor([[5, -2]]),
            ValueError,
        ),
        (
            "labels",
            torch.tensor([[5, 7]]),
            ValueError,
        ),
    ],
)
def test_strict_text_boundary_rejects_before_policy(
    field,
    value,
    error,
):
    policy = RecordingIdentityPolicy()
    kwargs = strict_kwargs(policy)
    kwargs[field] = value
    if field == "input_ids" and value.ndim == 2:
        kwargs["text_embeddings"] = torch.zeros(
            *value.shape,
            3,
            dtype=torch.float32,
        )

    with pytest.raises(error):
        SequenceAssembler().assemble(**kwargs)

    assert policy.calls == []


def test_masked_input_id_must_equal_explicit_pad():
    kwargs = strict_kwargs()
    kwargs["input_ids"] = torch.tensor([[5, 9]])
    with pytest.raises(ValueError, match="pad_token_id"):
        SequenceAssembler().assemble(**kwargs)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pad_token_id", True),
        ("pad_token_id", 10),
        ("max_assembled_length", True),
        ("max_assembled_length", 0),
        ("joint_separator_token_ids", {13}),
        ("joint_separator_token_ids", frozenset({True})),
        ("joint_separator_token_ids", frozenset({0})),
        ("joint_separator_token_ids", frozenset({10})),
    ],
)
def test_scalar_and_separator_contracts_are_strict(field, value):
    kwargs = strict_kwargs()
    kwargs[field] = value
    with pytest.raises((TypeError, ValueError)):
        SequenceAssembler().assemble(**kwargs)


@pytest.mark.parametrize(
    "tokens",
    [
        ResolvedMultimodalTokens(
            True,
            11,
            12,
            13,
            14,
            15,
            16,
        ),
        ResolvedMultimodalTokens(
            -1,
            11,
            12,
            13,
            14,
            15,
            16,
        ),
        ResolvedMultimodalTokens(
            10,
            10,
            12,
            13,
            14,
            15,
            16,
        ),
        ResolvedMultimodalTokens(
            None,
            11,
            12,
            13,
            14,
            15,
            16,
        ),
    ],
)
def test_resolved_token_ids_reject_bool_negative_collision_or_missing(
    tokens,
):
    kwargs = strict_kwargs()
    kwargs["tokens"] = tokens
    with pytest.raises((TypeError, ValueError)):
        SequenceAssembler().assemble(**kwargs)


def test_holey_or_empty_media_masks_fail_before_policy():
    policy = RecordingIdentityPolicy()
    for mask in (
        torch.tensor([[1, 0, 1]], dtype=torch.bool),
        torch.tensor([[0, 0, 0]], dtype=torch.bool),
    ):
        sequence = MediaSequence(
            embeddings=torch.zeros(1, 3, 3),
            attention_mask=mask,
            modality=MediaModality.AUDIO,
            sources=(MediaSource(0, 0, "audio-0"),),
            timestamps=torch.zeros(1, 3),
        )
        with pytest.raises(ValueError, match="prefix"):
            assemble(
                torch.tensor([[12]]),
                media_sequences=(sequence,),
                expansion_policy=policy,
            )
    assert policy.calls == []


def test_media_prefix_gathers_physical_columns_zero_and_one_only():
    sequence = make_audio_sequence(
        values=[10.0, 20.0, 999.0],
        source=MediaSource(0, 0, "audio-0"),
        timestamps=[0.0, 0.1, 0.0],
        attention_mask=[1, 1, 0],
    )
    result = assemble(
        torch.tensor([[12]]),
        media_sequences=(sequence,),
    )
    assert result.inputs_embeds[0, :, 0].tolist() == [10.0, 20.0]
    assert result.spans[0].source_token_indices == (0, 1)


@pytest.mark.parametrize(
    ("sequence", "error"),
    [
        (
            make_audio_sequence(
                values=[1.0],
                source=MediaSource(0, 0, "audio"),
                hidden_size=4,
            ),
            "hidden",
        ),
        (
            make_audio_sequence(
                values=[1.0],
                source=MediaSource(0, 0, "audio"),
                dtype=torch.bfloat16,
            ),
            "dtype",
        ),
    ],
)
def test_media_hidden_size_and_dtype_must_match_text(sequence, error):
    with pytest.raises((TypeError, ValueError), match=error):
        assemble(
            torch.tensor([[12]]),
            media_sequences=(sequence,),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_media_and_text_devices_must_match_exactly():
    sequence = MediaSequence(
        embeddings=torch.ones(1, 1, 3, device="cuda"),
        attention_mask=torch.ones(1, 1, dtype=torch.bool, device="cuda"),
        modality=MediaModality.AUDIO,
        sources=(MediaSource(0, 0, "audio"),),
    )
    sequence.validate()

    with pytest.raises(ValueError, match="devices"):
        assemble(
            torch.tensor([[12]]),
            media_sequences=(sequence,),
        )


def test_sentinel_item_count_and_modality_must_match_exactly():
    image = make_image_sequence(
        embeddings=torch.zeros(1, 1, 3),
        source=MediaSource(0, 0, "image"),
    )
    with pytest.raises(ValueError, match="no matching"):
        assemble(torch.tensor([[10]]))
    with pytest.raises(ValueError, match="extra media"):
        assemble(torch.tensor([[5]]), media_sequences=(image,))
    with pytest.raises(ValueError, match="modality"):
        assemble(torch.tensor([[12]]), media_sequences=(image,))


def test_global_source_uniqueness_item_gaps_and_batch_bounds():
    duplicate_item_a = make_audio_sequence(
        values=[1.0],
        source=MediaSource(0, 0, "a"),
    )
    duplicate_item_b = make_audio_sequence(
        values=[2.0],
        source=MediaSource(0, 0, "b"),
    )
    with pytest.raises(ValueError, match="item_index"):
        assemble(
            torch.tensor([[12, 12]]),
            media_sequences=(duplicate_item_a, duplicate_item_b),
        )

    duplicate_id = make_audio_sequence(
        values=[2.0],
        source=MediaSource(0, 1, "a"),
    )
    with pytest.raises(ValueError, match="source_id"):
        assemble(
            torch.tensor([[12, 12]]),
            media_sequences=(duplicate_item_a, duplicate_id),
        )

    gap = make_audio_sequence(
        values=[1.0],
        source=MediaSource(0, 1, "gap"),
    )
    with pytest.raises(ValueError, match="0..N-1"):
        assemble(
            torch.tensor([[12]]),
            media_sequences=(gap,),
        )

    outside = make_audio_sequence(
        values=[1.0],
        source=MediaSource(1, 0, "outside"),
    )
    with pytest.raises(ValueError, match="outside text batch"):
        assemble(
            torch.tensor([[12]]),
            media_sequences=(outside,),
        )


def test_media_sequences_container_and_rows_are_type_checked():
    with pytest.raises(TypeError, match="media_sequences"):
        assemble(
            torch.tensor([[5]]),
            media_sequences={"not": "a sequence"},
        )
    with pytest.raises(TypeError, match="MediaSequence"):
        assemble(
            torch.tensor([[5]]),
            media_sequences=(object(),),
        )


def test_groups_are_maximal_for_adjacent_and_whitelisted_intervals():
    audio = make_audio_sequence(
        values=[1.0],
        source=MediaSource(0, 0, "audio"),
        timestamps=[0.0],
    )
    video = make_video_sequence(
        values=[2.0],
        source=MediaSource(0, 1, "video"),
        timestamps=[0.1],
    )
    adjacent_policy = RecordingIdentityPolicy()
    assemble(
        torch.tensor([[12, 11]]),
        media_sequences=(audio, video),
        expansion_policy=adjacent_policy,
    )
    assert len(adjacent_policy.calls[0][1]) == 1
    assert len(adjacent_policy.calls[0][1][0].placeholders) == 2

    wrapper_policy = RecordingIdentityPolicy()
    assemble(
        torch.tensor([[12, 13, 11]]),
        media_sequences=(audio, video),
        expansion_policy=wrapper_policy,
        joint_separator_token_ids=frozenset({13}),
    )
    group = wrapper_policy.calls[0][1][0]
    assert (group.first_text_position, group.last_text_position) == (0, 2)
    assert len(group.placeholders) == 2

    natural_policy = RecordingIdentityPolicy()
    assemble(
        torch.tensor([[12, 5, 11]]),
        labels=torch.tensor([[-100, -100, -100]]),
        media_sequences=(audio, video),
        expansion_policy=natural_policy,
    )
    assert len(natural_policy.calls[0][1]) == 2


def test_timestamp_interleave_joint_order_markers_and_lookup_contract():
    audio = make_audio_sequence(
        values=[10.0, 11.0],
        source=MediaSource(0, 0, "audio"),
        timestamps=[0.00, 0.16],
    )
    video = make_video_sequence(
        values=[20.0, 21.0],
        source=MediaSource(0, 1, "video"),
        timestamps=[0.08, 0.24],
        grid=MediaGrid(1, 1, 2),
        seconds_per_grid=0.08,
    )
    policy = TimestampInterleaveExpansion(
        seconds_per_bucket=0.08,
        bucket_token_ids={0: 20, 1: 21, 2: 22, 3: 23},
    )
    lookup = RecordingLookup()

    result = assemble(
        torch.tensor([[12, 11]]),
        media_sequences=(video, audio),
        expansion_policy=policy,
        embedding_lookup=lookup,
    )

    assert result.expanded_input_ids.tolist() == [
        [20, 12, 21, 11, 22, 12, 23, 11]
    ]
    assert len(lookup.calls) == 1
    assert lookup.calls[0].dtype is torch.long
    assert lookup.calls[0].tolist() == [20, 21, 22, 23]
    media_spans = [
        span
        for span in result.spans
        if span.kind is SequenceSpanKind.MEDIA
    ]
    assert [span.source.source_id for span in media_spans] == [
        "audio",
        "video",
        "audio",
        "video",
    ]
    video_spans = [
        span
        for span in media_spans
        if span.modality is MediaModality.VIDEO
    ]
    assert [span.source_token_indices for span in video_spans] == [
        (0,),
        (1,),
    ]
    assert [span.timestamps.tolist() for span in video_spans] == [
        [pytest.approx(0.08)],
        [pytest.approx(0.24)],
    ]
    assert all(
        span.grid == MediaGrid(1, 1, 2)
        and span.seconds_per_grid == 0.08
        for span in video_spans
    )
    assert [
        span.source.source_id
        for span in result.spans
        if span.kind is SequenceSpanKind.TIMESTAMP
    ] == ["audio", "video", "audio", "video"]
    assert all(
        span.grid is None
        and span.timestamps is None
        and span.seconds_per_grid is None
        and span.source_token_indices is None
        for span in result.spans
        if span.kind is SequenceSpanKind.TIMESTAMP
    )


def test_timestamp_policy_emits_first_marker_and_one_per_repeated_bucket():
    audio = make_audio_sequence(
        values=[1.0, 2.0],
        source=MediaSource(0, 0, "audio"),
        timestamps=[0.0, 0.2],
    )
    policy = TimestampInterleaveExpansion(
        seconds_per_bucket=1.0,
        bucket_token_ids={0: 20},
    )
    lookup = RecordingLookup()

    result = assemble(
        torch.tensor([[12]]),
        media_sequences=(audio,),
        expansion_policy=policy,
        embedding_lookup=lookup,
    )

    assert result.expanded_input_ids.tolist() == [[20, 12, 12]]
    assert lookup.calls[0].tolist() == [20]
    timestamp_span = result.spans[0]
    assert timestamp_span.kind is SequenceSpanKind.TIMESTAMP
    assert timestamp_span.source == MediaSource(0, 0, "audio")


@pytest.mark.parametrize(
    "mapping",
    [
        [],
        {True: 20},
        {-1: 20},
        {0: True},
        {0: -1},
    ],
)
def test_timestamp_policy_rejects_invalid_bucket_mappings(mapping):
    with pytest.raises((TypeError, ValueError)):
        TimestampInterleaveExpansion(
            seconds_per_bucket=1.0,
            bucket_token_ids=mapping,
        )


@pytest.mark.parametrize(
    "step",
    [True, 0.0, -1.0, float("nan"), float("inf")],
)
def test_timestamp_policy_rejects_invalid_steps(step):
    with pytest.raises((TypeError, ValueError)):
        TimestampInterleaveExpansion(
            seconds_per_bucket=step,
            bucket_token_ids={},
        )


def test_timestamp_policy_snapshots_mapping_and_empty_map_is_image_only():
    mapping = {0: 20}
    policy = TimestampInterleaveExpansion(
        seconds_per_bucket=1.0,
        bucket_token_ids=mapping,
    )
    mapping[0] = 99
    mapping[1] = 21
    audio = make_audio_sequence(
        values=[1.0],
        source=MediaSource(0, 0, "audio"),
        timestamps=[0.0],
    )
    lookup = RecordingLookup()
    result = assemble(
        torch.tensor([[12]]),
        media_sequences=(audio,),
        expansion_policy=policy,
        embedding_lookup=lookup,
    )
    assert result.expanded_input_ids[0, 0].item() == 20

    empty = TimestampInterleaveExpansion(
        seconds_per_bucket=1.0,
        bucket_token_ids={},
    )
    image = make_image_sequence(
        embeddings=torch.ones(1, 1, 3),
        source=MediaSource(0, 0, "image"),
    )
    image_result = assemble(
        torch.tensor([[10]]),
        media_sequences=(image,),
        expansion_policy=empty,
    )
    assert image_result.expanded_input_ids.tolist() == [[10]]
    with pytest.raises(ValueError, match="no mapped token"):
        assemble(
            torch.tensor([[12]]),
            media_sequences=(audio,),
            expansion_policy=empty,
            embedding_lookup=lookup,
        )


def test_missing_av_timestamps_and_unmapped_bucket_fail_explicitly():
    missing = make_video_sequence(
        values=[1.0],
        source=MediaSource(0, 0, "video"),
        timestamps=None,
    )
    policy = TimestampInterleaveExpansion(
        seconds_per_bucket=1.0,
        bucket_token_ids={0: 20},
    )
    with pytest.raises(ValueError, match="requires AV timestamps"):
        assemble(
            torch.tensor([[11]]),
            media_sequences=(missing,),
            expansion_policy=policy,
            embedding_lookup=RecordingLookup(),
        )

    audio = make_audio_sequence(
        values=[1.0],
        source=MediaSource(0, 0, "audio"),
        timestamps=[2.0],
    )
    with pytest.raises(ValueError, match="no mapped token"):
        assemble(
            torch.tensor([[12]]),
            media_sequences=(audio,),
            expansion_policy=policy,
            embedding_lookup=RecordingLookup(),
        )


def test_image_and_av_can_use_separate_groups_but_not_one_mixed_group():
    image = make_image_sequence(
        embeddings=torch.ones(1, 1, 3),
        source=MediaSource(0, 0, "image"),
    )
    audio = make_audio_sequence(
        values=[2.0],
        source=MediaSource(0, 1, "audio"),
        timestamps=[0.0],
    )
    policy = TimestampInterleaveExpansion(
        seconds_per_bucket=1.0,
        bucket_token_ids={0: 20},
    )
    lookup = RecordingLookup()

    separate = assemble(
        torch.tensor([[10, 5, 12]]),
        media_sequences=(audio, image),
        expansion_policy=policy,
        embedding_lookup=lookup,
    )
    assert separate.expanded_input_ids.tolist() == [[10, 5, 20, 12]]

    with pytest.raises(ValueError, match="mix image with AV"):
        assemble(
            torch.tensor([[10, 12]]),
            media_sequences=(image, audio),
            expansion_policy=policy,
            embedding_lookup=lookup,
        )


def test_custom_timestamp_policy_uses_one_lookup_and_maximal_marker_span():
    audio = make_audio_sequence(
        values=[1.0, 2.0],
        source=MediaSource(0, 0, "audio"),
        timestamps=[0.0, 0.5],
    )
    lookup = RecordingLookup()
    result = assemble(
        torch.tensor([[12]]),
        media_sequences=(audio,),
        expansion_policy=TwoTimestampPolicy(),
        embedding_lookup=lookup,
    )
    assert lookup.calls[0].tolist() == [30, 31]
    assert result.expanded_input_ids.tolist() == [[30, 31, 12, 12]]
    assert result.spans[0].kind is SequenceSpanKind.TIMESTAMP
    assert (result.spans[0].start, result.spans[0].end) == (0, 2)


def test_timestamp_expansion_requires_lookup_but_image_never_calls_it():
    audio = make_audio_sequence(
        values=[1.0],
        source=MediaSource(0, 0, "audio"),
        timestamps=[0.0],
    )
    policy = TimestampInterleaveExpansion(
        seconds_per_bucket=1.0,
        bucket_token_ids={0: 20},
    )
    with pytest.raises(ValueError, match="embedding_lookup"):
        assemble(
            torch.tensor([[12]]),
            media_sequences=(audio,),
            expansion_policy=policy,
        )

    image = make_image_sequence(
        embeddings=torch.ones(1, 1, 3),
        source=MediaSource(0, 0, "image"),
    )
    lookup = RecordingLookup()
    assemble(
        torch.tensor([[10]]),
        media_sequences=(image,),
        expansion_policy=policy,
        embedding_lookup=lookup,
    )
    assert lookup.calls == []


def test_timestamp_lookup_is_called_once_per_timestamp_bearing_batch_row():
    first = make_audio_sequence(
        values=[1.0],
        source=MediaSource(0, 0, "first"),
        timestamps=[0.0],
    )
    second = make_audio_sequence(
        values=[2.0],
        source=MediaSource(1, 0, "second"),
        timestamps=[1.0],
    )
    policy = TimestampInterleaveExpansion(
        seconds_per_bucket=1.0,
        bucket_token_ids={0: 20, 1: 21},
    )
    lookup = RecordingLookup()

    assemble(
        torch.tensor([[12], [12]]),
        media_sequences=(second, first),
        expansion_policy=policy,
        embedding_lookup=lookup,
    )

    assert [call.tolist() for call in lookup.calls] == [[20], [21]]


@pytest.mark.parametrize(
    "lookup",
    [
        lambda token_ids: "not a tensor",
        lambda token_ids: torch.zeros(1),
        lambda token_ids: torch.zeros(token_ids.shape[0], 4),
        lambda token_ids: torch.zeros(
            token_ids.shape[0],
            3,
            dtype=torch.bfloat16,
        ),
        lambda token_ids: torch.zeros(
            token_ids.shape[0],
            3,
            device="meta",
        ),
    ],
)
def test_timestamp_lookup_result_contract_is_exact(lookup):
    audio = make_audio_sequence(
        values=[1.0],
        source=MediaSource(0, 0, "audio"),
        timestamps=[0.0],
    )
    policy = TimestampInterleaveExpansion(
        seconds_per_bucket=1.0,
        bucket_token_ids={0: 20},
    )
    with pytest.raises((TypeError, ValueError)):
        assemble(
            torch.tensor([[12]]),
            media_sequences=(audio,),
            expansion_policy=policy,
            embedding_lookup=lookup,
        )


def test_hard_limit_passes_at_equality_and_exposes_all_error_fields():
    audio = make_audio_sequence(
        values=[1.0, 2.0],
        source=MediaSource(0, 0, "audio"),
    )
    equal = assemble(
        torch.tensor([[5, 12]]),
        media_sequences=(audio,),
        max_assembled_length=3,
    )
    assert equal.attention_mask.sum().item() == 3

    with pytest.raises(AssembledLengthError) as captured:
        assemble(
            torch.tensor([[5, 12]]),
            media_sequences=(audio,),
            max_assembled_length=2,
        )
    error = captured.value
    assert error.sample_index == 0
    assert error.assembled_length == 3
    assert error.max_assembled_length == 2
    assert error.retained_text_tokens == 1
    assert error.inserted_expansion_tokens == 2
    for value in (
        "sample_index=0",
        "assembled_length=3",
        "max_assembled_length=2",
        "retained_text_tokens=1",
        "inserted_expansion_tokens=2",
    ):
        assert value in str(error)


def test_timestamp_tokens_count_toward_limit_before_lookup():
    audio = make_audio_sequence(
        values=[1.0, 2.0],
        source=MediaSource(0, 0, "audio"),
        timestamps=[0.0, 0.1],
    )
    policy = TimestampInterleaveExpansion(
        seconds_per_bucket=1.0,
        bucket_token_ids={0: 20},
    )
    lookup = RecordingLookup()

    with pytest.raises(AssembledLengthError) as captured:
        assemble(
            torch.tensor([[12]]),
            media_sequences=(audio,),
            expansion_policy=policy,
            embedding_lookup=lookup,
            max_assembled_length=2,
        )

    assert captured.value.retained_text_tokens == 0
    assert captured.value.inserted_expansion_tokens == 3
    assert lookup.calls == []


def test_policy_result_container_and_exact_keys_are_validated():
    audio = make_audio_sequence(
        values=[1.0],
        source=MediaSource(0, 0, "audio"),
    )
    ids = torch.tensor([[12]])
    policies = [
        FunctionalPolicy(lambda sample, groups: object()),
        FunctionalPolicy(
            lambda sample, groups: ExpandedMediaSample({})
        ),
        FunctionalPolicy(
            lambda sample, groups: ExpandedMediaSample(
                {99: ExpandedMediaRow(())}
            )
        ),
        FunctionalPolicy(
            lambda sample, groups: ExpandedMediaSample(
                {0: object()}
            )
        ),
    ]
    malformed = object.__new__(ExpandedMediaSample)
    object.__setattr__(malformed, "replacements", [])
    policies.append(
        FunctionalPolicy(lambda sample, groups: malformed)
    )

    for policy in policies:
        with pytest.raises((TypeError, ValueError)):
            assemble(
                ids,
                media_sequences=(audio,),
                expansion_policy=policy,
            )


def _one_source_policy_tokens(groups, tokens):
    position = groups[0].first_text_position
    return ExpandedMediaSample(
        {position: ExpandedMediaRow(tuple(tokens(groups[0])))}
    )


@pytest.mark.parametrize(
    "token_builder",
    [
        lambda source: (
            ExpansionToken(
                12,
                SequenceSpanKind.MEDIA,
                MediaTokenRef(source, 0),
            ),
        ),
        lambda source: (
            ExpansionToken(
                12,
                SequenceSpanKind.MEDIA,
                MediaTokenRef(source, 0),
            ),
            ExpansionToken(
                12,
                SequenceSpanKind.MEDIA,
                MediaTokenRef(source, 0),
            ),
        ),
        lambda source: (
            ExpansionToken(
                12,
                SequenceSpanKind.MEDIA,
                MediaTokenRef(source, 2),
            ),
            ExpansionToken(
                12,
                SequenceSpanKind.MEDIA,
                MediaTokenRef(source, 1),
            ),
        ),
        lambda source: (
            ExpansionToken(
                12,
                SequenceSpanKind.MEDIA,
                MediaTokenRef(source, 1),
            ),
            ExpansionToken(
                12,
                SequenceSpanKind.MEDIA,
                MediaTokenRef(source, 0),
            ),
        ),
        lambda source: (
            ExpansionToken(
                10,
                SequenceSpanKind.MEDIA,
                MediaTokenRef(source, 0),
            ),
            ExpansionToken(
                10,
                SequenceSpanKind.MEDIA,
                MediaTokenRef(source, 1),
            ),
        ),
        lambda source: (
            ExpansionToken(30, SequenceSpanKind.TEXT),
        ),
        lambda source: (
            ExpansionToken(
                0,
                SequenceSpanKind.TIMESTAMP,
                source=source,
            ),
            ExpansionToken(
                12,
                SequenceSpanKind.MEDIA,
                MediaTokenRef(source, 0),
            ),
            ExpansionToken(
                12,
                SequenceSpanKind.MEDIA,
                MediaTokenRef(source, 1),
            ),
        ),
        lambda source: (
            ExpansionToken(
                30,
                SequenceSpanKind.TIMESTAMP,
                media_ref=MediaTokenRef(source, 0),
                source=source,
            ),
            ExpansionToken(
                12,
                SequenceSpanKind.MEDIA,
                MediaTokenRef(source, 0),
            ),
            ExpansionToken(
                12,
                SequenceSpanKind.MEDIA,
                MediaTokenRef(source, 1),
            ),
        ),
    ],
)
def test_policy_rejects_missing_duplicate_out_of_order_or_bad_tokens(
    token_builder,
):
    source = MediaSource(0, 0, "audio")
    audio = make_audio_sequence(
        values=[1.0, 2.0],
        source=source,
    )
    policy = FunctionalPolicy(
        lambda sample, groups: _one_source_policy_tokens(
            groups,
            lambda group: token_builder(source),
        )
    )
    with pytest.raises((TypeError, ValueError)):
        assemble(
            torch.tensor([[12]]),
            media_sequences=(audio,),
            expansion_policy=policy,
            embedding_lookup=RecordingLookup(),
        )


def test_negative_and_arbitrary_media_token_ids_are_rejected():
    source = MediaSource(0, 0, "audio")
    audio = make_audio_sequence(values=[1.0], source=source)
    for token_id in (-1, 99, 10):
        token = ExpansionToken(
            12,
            SequenceSpanKind.MEDIA,
            MediaTokenRef(source, 0),
        )
        object.__setattr__(token, "token_id", token_id)
        policy = FunctionalPolicy(
            lambda sample, groups, token=token: ExpandedMediaSample(
                {0: ExpandedMediaRow((token,))}
            )
        )
        with pytest.raises((TypeError, ValueError)):
            assemble(
                torch.tensor([[12]]),
                media_sequences=(audio,),
                expansion_policy=policy,
            )


def test_policy_sources_must_match_refs_and_the_approved_group():
    audio_source = MediaSource(0, 0, "audio")
    video_source = MediaSource(0, 1, "video")
    foreign_source = MediaSource(0, 2, "foreign")
    audio = make_audio_sequence(values=[1.0], source=audio_source)
    video = make_video_sequence(
        values=[2.0],
        source=video_source,
        timestamps=[0.0],
    )
    correct_video = ExpansionToken(
        11,
        SequenceSpanKind.MEDIA,
        MediaTokenRef(video_source, 0),
    )
    bad_rows = (
        (
            ExpansionToken(
                12,
                SequenceSpanKind.MEDIA,
                MediaTokenRef(audio_source, 0),
                source=video_source,
            ),
            correct_video,
        ),
        (
            ExpansionToken(
                30,
                SequenceSpanKind.TIMESTAMP,
                source=foreign_source,
            ),
            ExpansionToken(
                12,
                SequenceSpanKind.MEDIA,
                MediaTokenRef(audio_source, 0),
            ),
            correct_video,
        ),
    )

    for row_tokens in bad_rows:
        policy = FunctionalPolicy(
            lambda sample, groups, row_tokens=row_tokens: (
                ExpandedMediaSample(
                    {
                        0: ExpandedMediaRow(row_tokens),
                        1: ExpandedMediaRow(()),
                    }
                )
            )
        )
        with pytest.raises(ValueError):
            assemble(
                torch.tensor([[12, 11]]),
                media_sequences=(audio, video),
                expansion_policy=policy,
                embedding_lookup=RecordingLookup(),
            )


def test_joint_refs_cannot_move_to_later_placeholder_or_split():
    audio_source = MediaSource(0, 0, "audio")
    video_source = MediaSource(0, 1, "video")
    audio = make_audio_sequence(
        values=[1.0],
        source=audio_source,
    )
    video = make_video_sequence(
        values=[2.0, 3.0],
        source=video_source,
        timestamps=[0.0, 0.1],
        grid=MediaGrid(1, 1, 2),
    )
    audio_token = ExpansionToken(
        12,
        SequenceSpanKind.MEDIA,
        MediaTokenRef(audio_source, 0),
    )
    video_tokens = tuple(
        ExpansionToken(
            11,
            SequenceSpanKind.MEDIA,
            MediaTokenRef(video_source, index),
        )
        for index in range(2)
    )
    policies = [
        FunctionalPolicy(
            lambda sample, groups: ExpandedMediaSample(
                {
                    0: ExpandedMediaRow(()),
                    1: ExpandedMediaRow((audio_token, *video_tokens)),
                }
            )
        ),
        FunctionalPolicy(
            lambda sample, groups: ExpandedMediaSample(
                {
                    0: ExpandedMediaRow(
                        (audio_token, video_tokens[0])
                    ),
                    1: ExpandedMediaRow((video_tokens[1],)),
                }
            )
        ),
    ]
    for policy in policies:
        with pytest.raises(ValueError, match="placeholder"):
            assemble(
                torch.tensor([[12, 11]]),
                media_sequences=(audio, video),
                expansion_policy=policy,
            )


def test_foreign_refs_cannot_cross_natural_text_group_boundary():
    audio_source = MediaSource(0, 0, "audio")
    video_source = MediaSource(0, 1, "video")
    audio = make_audio_sequence(values=[1.0], source=audio_source)
    video = make_video_sequence(
        values=[2.0],
        source=video_source,
        timestamps=[0.0],
    )
    policy = FunctionalPolicy(
        lambda sample, groups: ExpandedMediaSample(
            {
                0: ExpandedMediaRow(
                    (
                        ExpansionToken(
                            11,
                            SequenceSpanKind.MEDIA,
                            MediaTokenRef(video_source, 0),
                        ),
                    )
                ),
                2: ExpandedMediaRow(
                    (
                        ExpansionToken(
                            12,
                            SequenceSpanKind.MEDIA,
                            MediaTokenRef(audio_source, 0),
                        ),
                    )
                ),
            }
        )
    )
    with pytest.raises(ValueError, match="approved group"):
        assemble(
            torch.tensor([[12, 5, 11]]),
            media_sequences=(audio, video),
            expansion_policy=policy,
        )


def test_no_media_policy_result_must_be_empty():
    policy = FunctionalPolicy(
        lambda sample, groups: ExpandedMediaSample(
            {0: ExpandedMediaRow(())}
        )
    )
    with pytest.raises(ValueError, match="replacement keys"):
        assemble(
            torch.tensor([[5]]),
            expansion_policy=policy,
        )
