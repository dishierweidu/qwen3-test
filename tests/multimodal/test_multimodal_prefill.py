from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest
import torch
from torch import nn

from qwen3_omni_pretrain.data.collators import MediaLoadError
from qwen3_omni_pretrain.data.profile_collator import ProfileStage2Collator
from qwen3_omni_pretrain.multimodal import (
    MultimodalPrefillOutput,
    MultimodalPrefillPipeline,
)
from qwen3_omni_pretrain.multimodal.encoders import (
    AudioWindowEncoder,
    PatchVisionEncoder,
    TemporalVideoEncoder,
)
from qwen3_omni_pretrain.multimodal.io import (
    DecodedMedia,
    MediaRequest,
)
from qwen3_omni_pretrain.multimodal.modalities import MediaModality
from qwen3_omni_pretrain.multimodal.positions import (
    Qwen3DisjointPositionBuilder,
    Qwen3DisjointPositionConfig,
    TMRoPEConfig,
    TMRoPEPositionBuilder,
)
from qwen3_omni_pretrain.multimodal.sequence_assembler import (
    AssembledLengthError,
    IdentityMediaExpansion,
    SequenceAssembler,
    TimestampInterleaveExpansion,
)
from qwen3_omni_pretrain.multimodal.tokenization.schema import (
    MultimodalTokenSchema,
    ResolvedMultimodalTokens,
)
from qwen3_omni_pretrain.multimodal.types import (
    AssembledSequence,
    MediaSource,
    PositionBatch,
    SequenceSpanKind,
)


TOKENS = ResolvedMultimodalTokens(
    image_pad=10,
    video_pad=11,
    audio_pad=12,
    vision_start=13,
    vision_end=14,
    audio_start=15,
    audio_end=16,
)


class RecordingEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int = 64, hidden_size: int = 4):
        super().__init__(num_embeddings, hidden_size)
        self.calls: list[torch.Tensor] = []

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        self.calls.append(token_ids.detach().clone())
        return super().forward(token_ids)


def request(
    modality: MediaModality,
    *,
    item_index: int,
    source_id: str,
    sample_index: int = 0,
) -> MediaRequest:
    return MediaRequest(
        sample_id=f"sample-{sample_index}",
        sample_index=sample_index,
        item_index=item_index,
        source_id=source_id,
        modality=modality,
        path=f"/{source_id}.bin",
        original_sample_index=sample_index,
    )


def image_item(
    *,
    item_index: int,
    source_id: str = "image",
    sample_index: int = 0,
    value: float = 1.0,
) -> DecodedMedia:
    return DecodedMedia(
        request=request(
            MediaModality.IMAGE,
            item_index=item_index,
            source_id=source_id,
            sample_index=sample_index,
        ),
        tensor=torch.full((3, 2, 2), value),
        length=1,
        timestamps=None,
        seconds_per_grid=None,
        metadata={"original_width": 2, "original_height": 2},
    )


def audio_item(
    *,
    item_index: int,
    source_id: str = "audio",
    sample_index: int = 0,
    value: float = 1.0,
) -> DecodedMedia:
    return DecodedMedia(
        request=request(
            MediaModality.AUDIO,
            item_index=item_index,
            source_id=source_id,
            sample_index=sample_index,
        ),
        tensor=torch.full((1, 4), value),
        length=4,
        timestamps=None,
        seconds_per_grid=None,
        metadata={"sample_rate": 8},
    )


def video_item(
    *,
    item_index: int,
    source_id: str = "video",
    sample_index: int = 0,
    value: float = 1.0,
) -> DecodedMedia:
    return DecodedMedia(
        request=request(
            MediaModality.VIDEO,
            item_index=item_index,
            source_id=source_id,
            sample_index=sample_index,
        ),
        tensor=torch.full((2, 3, 2, 2), value),
        length=2,
        timestamps=torch.tensor([0.0, 0.5]),
        seconds_per_grid=0.5,
        metadata={"width": 2, "height": 2},
    )


def tiny_pipeline(
    *,
    tokens: ResolvedMultimodalTokens = TOKENS,
    image_encoder: PatchVisionEncoder | None = None,
    video_encoder: TemporalVideoEncoder | None = None,
    audio_encoder: AudioWindowEncoder | None = None,
    assembler: SequenceAssembler | None = None,
    expansion_policy: object | None = None,
    position_builder: object | None = None,
    pad_token_id: int = 0,
    joint_separator_token_ids: frozenset[int] = frozenset(),
    max_assembled_length: int = 64,
) -> MultimodalPrefillPipeline:
    return MultimodalPrefillPipeline(
        tokens=tokens,
        image_encoder=(
            PatchVisionEncoder(
                in_channels=3,
                hidden_size=4,
                patch_size=2,
            )
            if image_encoder is None
            else image_encoder
        ),
        video_encoder=(
            TemporalVideoEncoder(hidden_size=4)
            if video_encoder is None
            else video_encoder
        ),
        audio_encoder=(
            AudioWindowEncoder(
                hidden_size=4,
                window_size=4,
                hop_size=2,
                sample_rate=8,
            )
            if audio_encoder is None
            else audio_encoder
        ),
        assembler=(
            SequenceAssembler() if assembler is None else assembler
        ),
        expansion_policy=(
            IdentityMediaExpansion()
            if expansion_policy is None
            else expansion_policy
        ),
        position_builder=(
            Qwen3DisjointPositionBuilder(
                Qwen3DisjointPositionConfig()
            )
            if position_builder is None
            else position_builder
        ),
        pad_token_id=pad_token_id,
        joint_separator_token_ids=joint_separator_token_ids,
        max_assembled_length=max_assembled_length,
    )


def test_public_prefill_keeps_image_and_audio_in_prompt_order():
    pipeline = tiny_pipeline()
    embedding = RecordingEmbedding()
    output = pipeline.encode_and_assemble(
        input_ids=torch.tensor([[5, 10, 6, 12, 7]]),
        attention_mask=torch.ones(1, 5, dtype=torch.long),
        labels=torch.tensor([[-100, -100, -100, -100, 7]]),
        decoded_media=(
            image_item(item_index=0),
            audio_item(item_index=1),
        ),
        text_embedding=embedding,
    )

    assert isinstance(output, MultimodalPrefillOutput)
    assert [sequence.modality for sequence in output.media_sequences] == [
        MediaModality.IMAGE,
        MediaModality.AUDIO,
    ]
    assert [
        span.modality
        for span in output.assembled.spans
        if span.kind is SequenceSpanKind.MEDIA
    ] == [MediaModality.IMAGE, MediaModality.AUDIO]
    assert output.positions.position_ids.shape[:2] == (3, 1)
    assert output.assembled.labels is not None
    for span in output.assembled.spans:
        if span.kind is SequenceSpanKind.MEDIA:
            assert output.assembled.labels[
                span.sample_index, span.start : span.end
            ].tolist() == [-100] * (span.end - span.start)
    assert len(embedding.calls) == 1
    assert embedding.calls[0].ndim == 2


def test_transport_shuffle_preserves_complete_public_result():
    pipeline = tiny_pipeline()
    embedding = RecordingEmbedding()
    ids = torch.tensor([[10, 11, 12]])
    media = (
        image_item(item_index=0, value=1.0),
        video_item(item_index=1, value=2.0),
        audio_item(item_index=2, value=3.0),
    )
    first = pipeline(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        labels=None,
        decoded_media=media,
        text_embedding=embedding,
    )
    second = pipeline(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        labels=None,
        decoded_media=tuple(reversed(media)),
        text_embedding=embedding,
    )

    assert [sequence.sources for sequence in first.media_sequences] == [
        sequence.sources for sequence in second.media_sequences
    ]
    for left, right in zip(first.media_sequences, second.media_sequences):
        assert torch.equal(left.embeddings, right.embeddings)
        assert torch.equal(left.attention_mask, right.attention_mask)
    assert torch.equal(
        first.assembled.expanded_input_ids,
        second.assembled.expanded_input_ids,
    )
    assert torch.equal(
        first.assembled.inputs_embeds,
        second.assembled.inputs_embeds,
    )
    assert torch.equal(
        first.positions.position_ids,
        second.positions.position_ids,
    )
    assert torch.equal(
        first.positions.rope_deltas,
        second.positions.rope_deltas,
    )


def test_text_only_path_calls_no_encoder_and_still_builds_positions():
    pipeline = tiny_pipeline()
    embedding = RecordingEmbedding()
    calls = {"image": 0, "video": 0, "audio": 0}
    hooks = [
        pipeline.image_encoder.register_forward_hook(
            lambda *args: calls.__setitem__("image", calls["image"] + 1)
        ),
        pipeline.video_encoder.register_forward_hook(
            lambda *args: calls.__setitem__("video", calls["video"] + 1)
        ),
        pipeline.audio_encoder.register_forward_hook(
            lambda *args: calls.__setitem__("audio", calls["audio"] + 1)
        ),
    ]
    try:
        output = pipeline.encode_and_assemble(
            input_ids=torch.tensor([[5, 6]]),
            attention_mask=torch.ones(1, 2, dtype=torch.long),
            labels=None,
            decoded_media=(),
            text_embedding=embedding,
        )
    finally:
        for hook in hooks:
            hook.remove()

    assert calls == {"image": 0, "video": 0, "audio": 0}
    assert output.media_sequences == ()
    assert all(
        span.kind is SequenceSpanKind.TEXT
        for span in output.assembled.spans
    )
    assert len(embedding.calls) == 1


def test_pipeline_registers_only_media_encoders_not_external_embedding():
    pipeline = tiny_pipeline()
    before = tuple(pipeline.state_dict())
    assert any(key.startswith("image_encoder.patch_embed") for key in before)
    assert any(key.startswith("video_encoder.temporal_proj") for key in before)
    assert any(key.startswith("audio_encoder.proj") for key in before)
    assert not any("video_encoder.patch" in key for key in before)

    embedding = RecordingEmbedding()
    pipeline.encode_and_assemble(
        input_ids=torch.tensor([[5]]),
        attention_mask=torch.ones(1, 1, dtype=torch.long),
        labels=None,
        decoded_media=(),
        text_embedding=embedding,
    )
    assert tuple(pipeline.state_dict()) == before
    parameter_ids = {id(parameter) for parameter in pipeline.parameters()}
    assert id(embedding.weight) not in parameter_ids


def test_named_api_uses_module_call_hooks():
    pipeline = tiny_pipeline()
    embedding = RecordingEmbedding()
    calls = []
    handle = pipeline.register_forward_hook(
        lambda module, args, output: calls.append(output)
    )
    try:
        output = pipeline.encode_and_assemble(
            input_ids=torch.tensor([[5]]),
            attention_mask=torch.ones(1, 1, dtype=torch.long),
            labels=None,
            decoded_media=(),
            text_embedding=embedding,
        )
    finally:
        handle.remove()
    assert calls == [output]


def test_constructor_rejects_mutable_separators_and_bool_scalars():
    kwargs = dict(
        tokens=TOKENS,
        image_encoder=PatchVisionEncoder(
            in_channels=3,
            hidden_size=4,
            patch_size=2,
        ),
        video_encoder=TemporalVideoEncoder(hidden_size=4),
        audio_encoder=AudioWindowEncoder(
            hidden_size=4,
            window_size=4,
            hop_size=2,
            sample_rate=8,
        ),
        assembler=SequenceAssembler(),
        expansion_policy=IdentityMediaExpansion(),
        position_builder=Qwen3DisjointPositionBuilder(
            Qwen3DisjointPositionConfig()
        ),
        pad_token_id=0,
        joint_separator_token_ids=frozenset(),
        max_assembled_length=64,
    )
    for field, value in (
        ("joint_separator_token_ids", {13}),
        ("pad_token_id", True),
        ("max_assembled_length", True),
    ):
        invalid = dict(kwargs)
        invalid[field] = value
        with pytest.raises((TypeError, ValueError)):
            MultimodalPrefillPipeline(**invalid)


class CountingAssembler(SequenceAssembler):
    def __init__(self) -> None:
        self.calls = 0

    def assemble(self, **kwargs):
        self.calls += 1
        return super().assemble(**kwargs)


class CountingPolicy:
    def __init__(self) -> None:
        self.calls = 0
        self.delegate = IdentityMediaExpansion()

    def expand_sample(self, *, sample_index, groups):
        self.calls += 1
        return self.delegate.expand_sample(
            sample_index=sample_index,
            groups=groups,
        )


class CountingBuilder:
    def __init__(self, delegate=None) -> None:
        self.calls = 0
        self.delegate = delegate or Qwen3DisjointPositionBuilder(
            Qwen3DisjointPositionConfig()
        )

    def build(self, assembled):
        self.calls += 1
        return self.delegate.build(assembled)


def test_constructor_enforces_concrete_components_and_hidden_size():
    with pytest.raises(TypeError, match="image_encoder"):
        tiny_pipeline(image_encoder=nn.Linear(4, 4))
    with pytest.raises(TypeError, match="video_encoder"):
        tiny_pipeline(video_encoder=nn.Linear(4, 4))
    with pytest.raises(TypeError, match="audio_encoder"):
        tiny_pipeline(audio_encoder=nn.Linear(4, 4))
    with pytest.raises(TypeError, match="assembler"):
        MultimodalPrefillPipeline(
            tokens=TOKENS,
            image_encoder=PatchVisionEncoder(
                in_channels=3,
                hidden_size=4,
                patch_size=2,
            ),
            video_encoder=TemporalVideoEncoder(hidden_size=4),
            audio_encoder=AudioWindowEncoder(
                hidden_size=4,
                window_size=4,
                hop_size=2,
                sample_rate=8,
            ),
            assembler=object(),
            expansion_policy=IdentityMediaExpansion(),
            position_builder=Qwen3DisjointPositionBuilder(
                Qwen3DisjointPositionConfig()
            ),
            pad_token_id=0,
            joint_separator_token_ids=frozenset(),
            max_assembled_length=64,
        )
    with pytest.raises(ValueError, match="hidden sizes"):
        tiny_pipeline(video_encoder=TemporalVideoEncoder(hidden_size=5))


def test_constructor_rejects_module_policy_or_builder_and_missing_methods():
    class ModulePolicy(nn.Module):
        def expand_sample(self, *, sample_index, groups):
            raise AssertionError("must not run")

    class ModuleBuilder(nn.Module):
        def build(self, assembled):
            raise AssertionError("must not run")

    with pytest.raises(TypeError, match="non-nn.Module"):
        tiny_pipeline(expansion_policy=ModulePolicy())
    with pytest.raises(TypeError, match="non-nn.Module"):
        tiny_pipeline(position_builder=ModuleBuilder())
    with pytest.raises(TypeError, match="expand_sample"):
        tiny_pipeline(expansion_policy=object())
    with pytest.raises(TypeError, match="define callable build"):
        tiny_pipeline(position_builder=object())


@pytest.mark.parametrize(
    "tokens",
    [
        ResolvedMultimodalTokens(True, 11, 12, 13, 14, 15, 16),
        ResolvedMultimodalTokens(-1, 11, 12, 13, 14, 15, 16),
        ResolvedMultimodalTokens(10, 10, 12, 13, 14, 15, 16),
        ResolvedMultimodalTokens(None, 11, 12, 13, 14, 15, 16),
        ResolvedMultimodalTokens(10, 11, 12, 13, 14, "15", 16),
    ],
)
def test_constructor_rejects_malformed_resolved_tokens(tokens):
    with pytest.raises((TypeError, ValueError)):
        tiny_pipeline(tokens=tokens)


def test_constructor_allows_explicit_wrapper_separator_ids():
    pipeline = tiny_pipeline(
        joint_separator_token_ids=frozenset(
            {TOKENS.vision_start, TOKENS.vision_end}
        )
    )
    assert pipeline.joint_separator_token_ids == frozenset({13, 14})


@pytest.mark.parametrize(
    "overrides",
    [
        {"input_ids": torch.tensor([5]), "attention_mask": torch.tensor([1])},
        {
            "input_ids": torch.empty(1, 0, dtype=torch.long),
            "attention_mask": torch.empty(1, 0, dtype=torch.long),
        },
        {"input_ids": torch.tensor([[5.0]])},
        {"input_ids": torch.tensor([[-1]])},
        {"attention_mask": torch.tensor([[1, 1]])},
        {"attention_mask": torch.tensor([[1.0]])},
        {"attention_mask": torch.tensor([[2]])},
        {
            "input_ids": torch.tensor([[5, 0, 6]]),
            "attention_mask": torch.tensor([[1, 0, 1]]),
        },
        {"attention_mask": torch.tensor([[0]])},
        {
            "input_ids": torch.tensor([[5, 9]]),
            "attention_mask": torch.tensor([[1, 0]]),
        },
        {"labels": torch.tensor([[5, 6]])},
        {"labels": torch.tensor([[5.0]])},
        {"labels": torch.tensor([[-1]])},
        {
            "input_ids": torch.tensor([[5, 0]]),
            "attention_mask": torch.tensor([[1, 0]]),
            "labels": torch.tensor([[5, 0]]),
        },
    ],
)
def test_text_preflight_fails_before_every_downstream_component(overrides):
    assembler = CountingAssembler()
    policy = CountingPolicy()
    builder = CountingBuilder()
    pipeline = tiny_pipeline(
        assembler=assembler,
        expansion_policy=policy,
        position_builder=builder,
    )
    embedding = RecordingEmbedding()
    kwargs = {
        "input_ids": torch.tensor([[5]]),
        "attention_mask": torch.tensor([[1]]),
        "labels": None,
        "decoded_media": (),
        "text_embedding": embedding,
    }
    kwargs.update(overrides)

    with pytest.raises((TypeError, ValueError)):
        pipeline(**kwargs)

    assert embedding.calls == []
    assert assembler.calls == 0
    assert policy.calls == 0
    assert builder.calls == 0


@pytest.mark.parametrize(
    "media",
    [
        (
            image_item(item_index=0, source_id="image"),
            audio_item(item_index=0, source_id="audio"),
        ),
        (
            image_item(item_index=0, source_id="shared"),
            audio_item(item_index=1, source_id="shared"),
        ),
        (image_item(item_index=0, sample_index=1),),
        (
            image_item(item_index=0, source_id="image"),
            audio_item(item_index=2, source_id="audio"),
        ),
    ],
)
def test_global_media_key_validation_precedes_lookup_and_encoding(media):
    pipeline = tiny_pipeline()
    embedding = RecordingEmbedding()
    calls = []
    handles = [
        encoder.register_forward_hook(
            lambda *args, name=name: calls.append(name)
        )
        for name, encoder in (
            ("image", pipeline.image_encoder),
            ("video", pipeline.video_encoder),
            ("audio", pipeline.audio_encoder),
        )
    ]
    try:
        with pytest.raises(ValueError):
            pipeline(
                input_ids=torch.tensor([[5]]),
                attention_mask=torch.tensor([[1]]),
                labels=None,
                decoded_media=media,
                text_embedding=embedding,
            )
    finally:
        for handle in handles:
            handle.remove()
    assert embedding.calls == []
    assert calls == []


@pytest.mark.parametrize(
    ("lookup", "error"),
    [
        (lambda ids: object(), TypeError),
        (lambda ids: torch.zeros(*ids.shape), ValueError),
        (
            lambda ids: torch.zeros(*ids.shape, 4, dtype=torch.long),
            TypeError,
        ),
        (lambda ids: torch.zeros(*ids.shape, 3), ValueError),
    ],
)
def test_text_embedding_result_is_validated_at_its_boundary(lookup, error):
    pipeline = tiny_pipeline()
    with pytest.raises(error):
        pipeline(
            input_ids=torch.tensor([[5]]),
            attention_mask=torch.tensor([[1]]),
            labels=None,
            decoded_media=(),
            text_embedding=lookup,
        )


def test_omitted_encoder_placement_mismatch_fails_before_orchestration():
    assembler = CountingAssembler()
    policy = CountingPolicy()
    builder = CountingBuilder()
    pipeline = tiny_pipeline(
        assembler=assembler,
        expansion_policy=policy,
        position_builder=builder,
    )
    pipeline.audio_encoder.to(dtype=torch.bfloat16)
    embedding = RecordingEmbedding()

    with pytest.raises(ValueError, match="one device and dtype"):
        pipeline(
            input_ids=torch.tensor([[5]]),
            attention_mask=torch.tensor([[1]]),
            labels=None,
            decoded_media=(),
            text_embedding=embedding,
        )

    assert len(embedding.calls) == 1
    assert assembler.calls == 0
    assert policy.calls == 0
    assert builder.calls == 0


def test_bfloat16_text_only_requires_and_preserves_exact_dtype():
    pipeline = tiny_pipeline().to(dtype=torch.bfloat16)
    matched = RecordingEmbedding().to(dtype=torch.bfloat16)
    output = pipeline(
        input_ids=torch.tensor([[5]]),
        attention_mask=torch.tensor([[1]]),
        labels=None,
        decoded_media=(),
        text_embedding=matched,
    )
    assert output.assembled.inputs_embeds.dtype is torch.bfloat16

    with pytest.raises(ValueError, match="share one device and dtype"):
        pipeline(
            input_ids=torch.tensor([[5]]),
            attention_mask=torch.tensor([[1]]),
            labels=None,
            decoded_media=(),
            text_embedding=RecordingEmbedding(),
        )


class RecordingVideoEncoder(TemporalVideoEncoder):
    def __init__(self, *, hidden_size: int) -> None:
        super().__init__(hidden_size=hidden_size)
        self.patch_encoder_id = None

    def forward(self, items, *, patch_encoder):
        self.patch_encoder_id = id(patch_encoder)
        return super().forward(items, patch_encoder=patch_encoder)


def test_video_receives_exact_registered_image_encoder_without_ownership():
    video_encoder = RecordingVideoEncoder(hidden_size=4)
    pipeline = tiny_pipeline(video_encoder=video_encoder)
    output = pipeline(
        input_ids=torch.tensor([[11]]),
        attention_mask=torch.tensor([[1]]),
        labels=None,
        decoded_media=(video_item(item_index=0),),
        text_embedding=RecordingEmbedding(),
    )
    assert output.media_sequences[0].modality is MediaModality.VIDEO
    assert video_encoder.patch_encoder_id == id(pipeline.image_encoder)
    assert not any(
        key.startswith("video_encoder.image_encoder")
        or key.startswith("video_encoder.patch_encoder")
        for key in pipeline.state_dict()
    )


class BadAssembler(SequenceAssembler):
    def __init__(self) -> None:
        self.calls = 0

    def assemble(self, **kwargs):
        self.calls += 1
        return object()


class BadBuilder:
    def __init__(self) -> None:
        self.calls = 0

    def build(self, assembled):
        self.calls += 1
        return object()


def test_malformed_assembler_and_builder_returns_fail_at_declared_boundary():
    bad_assembler = BadAssembler()
    untouched_builder = CountingBuilder()
    pipeline = tiny_pipeline(
        assembler=bad_assembler,
        position_builder=untouched_builder,
    )
    with pytest.raises(TypeError, match="AssembledSequence"):
        pipeline(
            input_ids=torch.tensor([[5]]),
            attention_mask=torch.tensor([[1]]),
            labels=None,
            decoded_media=(),
            text_embedding=RecordingEmbedding(),
        )
    assert bad_assembler.calls == 1
    assert untouched_builder.calls == 0

    assembler = CountingAssembler()
    bad_builder = BadBuilder()
    pipeline = tiny_pipeline(
        assembler=assembler,
        position_builder=bad_builder,
    )
    with pytest.raises(TypeError, match="PositionBatch"):
        pipeline(
            input_ids=torch.tensor([[5]]),
            attention_mask=torch.tensor([[1]]),
            labels=None,
            decoded_media=(),
            text_embedding=RecordingEmbedding(),
        )
    assert assembler.calls == 1
    assert bad_builder.calls == 1


def test_hard_limit_occurs_after_encoding_but_before_timestamp_lookup():
    policy = TimestampInterleaveExpansion(
        seconds_per_bucket=0.16,
        bucket_token_ids={0: 20},
    )
    builder = CountingBuilder(
        TMRoPEPositionBuilder(
            TMRoPEConfig(
                temporal_seconds_per_id=0.16,
                rotary_sections=(24, 20, 20),
            )
        )
    )
    pipeline = tiny_pipeline(
        expansion_policy=policy,
        position_builder=builder,
        max_assembled_length=1,
    )
    embedding = RecordingEmbedding()
    audio_calls = []
    handle = pipeline.audio_encoder.register_forward_hook(
        lambda *args: audio_calls.append(True)
    )
    try:
        with pytest.raises(AssembledLengthError) as captured:
            pipeline(
                input_ids=torch.tensor([[12]]),
                attention_mask=torch.tensor([[1]]),
                labels=None,
                decoded_media=(audio_item(item_index=0),),
                text_embedding=embedding,
            )
    finally:
        handle.remove()

    assert len(embedding.calls) == 1
    assert embedding.calls[0].ndim == 2
    assert audio_calls == [True]
    assert builder.calls == 0
    error = captured.value
    assert error.sample_index == 0
    assert error.assembled_length == 2
    assert error.max_assembled_length == 1
    assert error.retained_text_tokens == 0
    assert error.inserted_expansion_tokens == 2


@pytest.mark.parametrize(
    "modalities",
    [
        (MediaModality.IMAGE,),
        (MediaModality.VIDEO,),
        (MediaModality.AUDIO,),
        (MediaModality.IMAGE, MediaModality.VIDEO),
        (MediaModality.IMAGE, MediaModality.AUDIO),
        (MediaModality.VIDEO, MediaModality.AUDIO),
        (
            MediaModality.IMAGE,
            MediaModality.VIDEO,
            MediaModality.AUDIO,
        ),
    ],
)
def test_every_nonempty_modality_subset_calls_only_present_encoders(modalities):
    pipeline = tiny_pipeline()
    counts = {modality: 0 for modality in MediaModality}
    handles = [
        encoder.register_forward_hook(
            lambda *args, modality=modality: counts.__setitem__(
                modality,
                counts[modality] + 1,
            )
        )
        for modality, encoder in (
            (MediaModality.IMAGE, pipeline.image_encoder),
            (MediaModality.VIDEO, pipeline.video_encoder),
            (MediaModality.AUDIO, pipeline.audio_encoder),
        )
    ]
    factories = {
        MediaModality.IMAGE: image_item,
        MediaModality.VIDEO: video_item,
        MediaModality.AUDIO: audio_item,
    }
    token_ids = {
        MediaModality.IMAGE: TOKENS.image_pad,
        MediaModality.VIDEO: TOKENS.video_pad,
        MediaModality.AUDIO: TOKENS.audio_pad,
    }
    media = tuple(
        factories[modality](
            item_index=item_index,
            source_id=f"{modality.value}-{item_index}",
        )
        for item_index, modality in enumerate(modalities)
    )
    try:
        output = pipeline(
            input_ids=torch.tensor(
                [[token_ids[modality] for modality in modalities]]
            ),
            attention_mask=torch.ones(1, len(modalities), dtype=torch.long),
            labels=None,
            decoded_media=tuple(reversed(media)),
            text_embedding=RecordingEmbedding(),
        )
    finally:
        for handle in handles:
            handle.remove()

    assert counts == {
        modality: int(modality in modalities) for modality in MediaModality
    }
    canonical_modalities = tuple(
        modality
        for modality in (
            MediaModality.IMAGE,
            MediaModality.VIDEO,
            MediaModality.AUDIO,
        )
        if modality in modalities
    )
    assert tuple(
        sequence.modality for sequence in output.media_sequences
    ) == canonical_modalities
    assert {
        span.source.source_id
        for span in output.assembled.spans
        if span.kind is SequenceSpanKind.MEDIA
        and span.source is not None
    } == {item.request.source_id for item in media}


@pytest.mark.parametrize(
    "builder",
    [
        Qwen3DisjointPositionBuilder(Qwen3DisjointPositionConfig()),
        TMRoPEPositionBuilder(
            TMRoPEConfig(
                temporal_seconds_per_id=0.16,
                rotary_sections=(24, 20, 20),
            )
        ),
    ],
)
def test_text_only_is_valid_for_both_three_axis_position_builders(builder):
    pipeline = tiny_pipeline(position_builder=builder)
    embedding = RecordingEmbedding()
    output = pipeline(
        input_ids=torch.tensor([[5, 6]]),
        attention_mask=torch.tensor([[1, 1]]),
        labels=None,
        decoded_media=(),
        text_embedding=embedding,
    )
    assert output.media_sequences == ()
    assert len(embedding.calls) == 1
    assert embedding.calls[0].ndim == 2
    assert output.positions.axis_names == ("temporal", "height", "width")
    assert output.positions.position_ids[:, 0].tolist() == [
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ]
    assert output.positions.rope_deltas.tolist() == [[0.0]]


def test_mixed_text_only_and_media_rows_create_no_fake_media():
    pipeline = tiny_pipeline()
    output = pipeline(
        input_ids=torch.tensor([[10, 5], [6, 0]]),
        attention_mask=torch.tensor([[1, 1], [1, 0]]),
        labels=torch.tensor([[-100, 5], [6, -100]]),
        decoded_media=(image_item(item_index=0),),
        text_embedding=RecordingEmbedding(),
    )
    media_spans = [
        span
        for span in output.assembled.spans
        if span.kind is SequenceSpanKind.MEDIA
    ]
    assert {span.sample_index for span in media_spans} == {0}
    assert all(
        span.kind is SequenceSpanKind.TEXT
        for span in output.assembled.spans
        if span.sample_index == 1
    )
    assert output.assembled.labels is not None
    assert output.assembled.labels[0, 0].item() == -100


def test_gradients_reach_present_encoders_and_external_embedding():
    pipeline = tiny_pipeline()
    embedding = RecordingEmbedding()
    output = pipeline(
        input_ids=torch.tensor([[5, 10, 11, 12]]),
        attention_mask=torch.ones(1, 4, dtype=torch.long),
        labels=None,
        decoded_media=(
            image_item(item_index=0),
            video_item(item_index=1),
            audio_item(item_index=2),
        ),
        text_embedding=embedding,
    )
    weights = torch.arange(
        output.assembled.inputs_embeds.numel(),
        dtype=output.assembled.inputs_embeds.dtype,
    ).reshape_as(output.assembled.inputs_embeds)
    (output.assembled.inputs_embeds * weights).sum().backward()

    assert embedding.weight.grad is not None
    for encoder in (
        pipeline.image_encoder,
        pipeline.video_encoder,
        pipeline.audio_encoder,
    ):
        assert any(
            parameter.grad is not None for parameter in encoder.parameters()
        )
    video_sequence = next(
        sequence
        for sequence in output.media_sequences
        if sequence.modality is MediaModality.VIDEO
    )
    assert video_sequence.timestamps is not None
    assert video_sequence.timestamps.dtype is torch.float32
    assert video_sequence.timestamps.device == output.assembled.inputs_embeds.device


def test_omitted_encoder_parameters_receive_no_fabricated_gradient():
    pipeline = tiny_pipeline()
    embedding = RecordingEmbedding()
    output = pipeline(
        input_ids=torch.tensor([[5, 10]]),
        attention_mask=torch.ones(1, 2, dtype=torch.long),
        labels=None,
        decoded_media=(image_item(item_index=0),),
        text_embedding=embedding,
    )
    weights = torch.arange(
        output.assembled.inputs_embeds.numel(),
        dtype=output.assembled.inputs_embeds.dtype,
    ).reshape_as(output.assembled.inputs_embeds)
    (output.assembled.inputs_embeds * weights).sum().backward()
    assert any(
        parameter.grad is not None
        for parameter in pipeline.image_encoder.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in pipeline.video_encoder.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in pipeline.audio_encoder.parameters()
    )


def test_qwen_video_positions_use_literal_thirteen_ids_per_second():
    pipeline = tiny_pipeline(
        position_builder=Qwen3DisjointPositionBuilder(
            Qwen3DisjointPositionConfig(
                position_id_per_seconds=13.0,
                rotary_sections=(24, 20, 20),
            )
        )
    )
    output = pipeline(
        input_ids=torch.tensor([[11]]),
        attention_mask=torch.tensor([[1]]),
        labels=None,
        decoded_media=(video_item(item_index=0),),
        text_embedding=RecordingEmbedding(),
    )
    assert output.positions.position_ids[0, 0].tolist() == [0.0, 6.5]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pad_token_id", "0"),
        ("pad_token_id", -1),
        ("pad_token_id", 10),
        ("max_assembled_length", "64"),
        ("max_assembled_length", 0),
        ("max_assembled_length", -1),
        ("joint_separator_token_ids", frozenset({True})),
        ("joint_separator_token_ids", frozenset({-1})),
        ("joint_separator_token_ids", frozenset({0})),
        ("joint_separator_token_ids", frozenset({10})),
    ],
)
def test_constructor_rejects_all_invalid_scalar_and_separator_forms(
    field,
    value,
):
    kwargs = {field: value}
    with pytest.raises((TypeError, ValueError)):
        tiny_pipeline(**kwargs)


def test_module_assembler_is_rejected_even_when_it_subclasses_assembler():
    class ModuleAssembler(SequenceAssembler, nn.Module):
        def __init__(self) -> None:
            nn.Module.__init__(self)

    with pytest.raises(TypeError, match="non-nn.Module"):
        tiny_pipeline(assembler=ModuleAssembler())


def test_valid_nonzero_pad_id_and_none_labels_are_preserved():
    pipeline = tiny_pipeline(pad_token_id=9)
    output = pipeline(
        input_ids=torch.tensor([[5, 9]]),
        attention_mask=torch.tensor([[1, 0]]),
        labels=None,
        decoded_media=(),
        text_embedding=RecordingEmbedding(),
    )
    assert output.assembled.expanded_input_ids.tolist() == [[5]]
    assert output.assembled.labels is None


def test_timestamped_media_under_identity_adds_no_rank1_lookup():
    pipeline = tiny_pipeline()
    embedding = RecordingEmbedding()
    pipeline(
        input_ids=torch.tensor([[11, 12]]),
        attention_mask=torch.tensor([[1, 1]]),
        labels=None,
        decoded_media=(
            video_item(item_index=0),
            audio_item(item_index=1),
        ),
        text_embedding=embedding,
    )
    assert len(embedding.calls) == 1
    assert embedding.calls[0].ndim == 2


@pytest.mark.parametrize("timestamp_rows", [1, 2])
def test_timestamp_marker_lookup_occurs_once_per_emitting_row(timestamp_rows):
    policy = TimestampInterleaveExpansion(
        seconds_per_bucket=0.16,
        bucket_token_ids={0: 20},
    )
    pipeline = tiny_pipeline(
        expansion_policy=policy,
        position_builder=TMRoPEPositionBuilder(
            TMRoPEConfig(
                temporal_seconds_per_id=0.16,
                rotary_sections=(24, 20, 20),
            )
        ),
    )
    embedding = RecordingEmbedding()
    input_rows = [[12], [12 if timestamp_rows == 2 else 5]]
    media = tuple(
        audio_item(
            item_index=0,
            source_id=f"audio-{sample_index}",
            sample_index=sample_index,
        )
        for sample_index in range(timestamp_rows)
    )
    pipeline(
        input_ids=torch.tensor(input_rows),
        attention_mask=torch.ones(2, 1, dtype=torch.long),
        labels=None,
        decoded_media=media,
        text_embedding=embedding,
    )
    assert len(embedding.calls) == 1 + timestamp_rows
    assert embedding.calls[0].shape == (2, 1)
    assert all(call.ndim == 1 for call in embedding.calls[1:])
    assert all(call.tolist() == [20] for call in embedding.calls[1:])


@pytest.mark.parametrize("modality", list(MediaModality))
def test_each_encoder_result_is_type_checked(modality, monkeypatch):
    pipeline = tiny_pipeline()
    if modality is MediaModality.IMAGE:
        encoder = pipeline.image_encoder
        media = (image_item(item_index=0),)
        token_id = TOKENS.image_pad
        monkeypatch.setattr(encoder, "forward", lambda items: object())
    elif modality is MediaModality.VIDEO:
        encoder = pipeline.video_encoder
        media = (video_item(item_index=0),)
        token_id = TOKENS.video_pad
        monkeypatch.setattr(
            encoder,
            "forward",
            lambda items, *, patch_encoder: object(),
        )
    else:
        encoder = pipeline.audio_encoder
        media = (audio_item(item_index=0),)
        token_id = TOKENS.audio_pad
        monkeypatch.setattr(encoder, "forward", lambda items: object())

    with pytest.raises(TypeError, match="MediaSequence"):
        pipeline(
            input_ids=torch.tensor([[token_id]]),
            attention_mask=torch.tensor([[1]]),
            labels=None,
            decoded_media=media,
            text_embedding=RecordingEmbedding(),
        )


def test_encoder_sources_must_equal_canonical_requests(monkeypatch):
    pipeline = tiny_pipeline()
    original_forward = pipeline.image_encoder.forward

    def wrong_source(items):
        result = original_forward(items)
        return replace(
            result,
            sources=(MediaSource(0, 0, "different-source"),),
        )

    monkeypatch.setattr(pipeline.image_encoder, "forward", wrong_source)
    with pytest.raises(ValueError, match="canonical requests"):
        pipeline(
            input_ids=torch.tensor([[10]]),
            attention_mask=torch.tensor([[1]]),
            labels=None,
            decoded_media=(image_item(item_index=0),),
            text_embedding=RecordingEmbedding(),
        )


class InvalidPositionBuilder:
    def __init__(self) -> None:
        self.calls = 0

    def build(self, assembled: AssembledSequence) -> PositionBatch:
        self.calls += 1
        batch_size, sequence_length = assembled.attention_mask.shape
        return PositionBatch(
            position_ids=torch.zeros(batch_size, sequence_length),
            rope_deltas=torch.zeros(batch_size, 1),
            axis_names=("sequence",),
        )


def test_position_batch_contract_is_checked_after_builder_return():
    builder = InvalidPositionBuilder()
    pipeline = tiny_pipeline(position_builder=builder)
    with pytest.raises(ValueError, match="axes"):
        pipeline(
            input_ids=torch.tensor([[5]]),
            attention_mask=torch.tensor([[1]]),
            labels=None,
            decoded_media=(),
            text_embedding=RecordingEmbedding(),
        )
    assert builder.calls == 1


def test_fp32_decoded_image_can_feed_bfloat16_pipeline_with_gradients():
    pipeline = tiny_pipeline().to(dtype=torch.bfloat16)
    embedding = RecordingEmbedding().to(dtype=torch.bfloat16)
    decoded = image_item(item_index=0)
    assert decoded.tensor.dtype is torch.float32
    output = pipeline(
        input_ids=torch.tensor([[5, 10]]),
        attention_mask=torch.tensor([[1, 1]]),
        labels=None,
        decoded_media=(decoded,),
        text_embedding=embedding,
    )
    assert output.assembled.inputs_embeds.dtype is torch.bfloat16
    weights = torch.arange(
        output.assembled.inputs_embeds.numel(),
        dtype=torch.bfloat16,
    ).reshape_as(output.assembled.inputs_embeds)
    (output.assembled.inputs_embeds * weights).sum().backward()
    gradients = [
        parameter.grad
        for parameter in pipeline.image_encoder.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(bool(torch.isfinite(gradient).all().item()) for gradient in gradients)
    assert embedding.weight.grad is not None
    assert bool(torch.isfinite(embedding.weight.grad).all().item())


def timed_audio_item(*, item_index: int) -> DecodedMedia:
    return DecodedMedia(
        request=request(
            MediaModality.AUDIO,
            item_index=item_index,
            source_id="timed-audio",
        ),
        tensor=torch.tensor(
            [[0.0, 1.0, 0.5, -0.5, 1.0, -1.0, 0.25, 0.75]]
        ),
        length=8,
        timestamps=None,
        seconds_per_grid=None,
        metadata={"sample_rate": 25},
    )


def timed_video_item(*, item_index: int) -> DecodedMedia:
    return DecodedMedia(
        request=request(
            MediaModality.VIDEO,
            item_index=item_index,
            source_id="timed-video",
        ),
        tensor=torch.stack(
            (
                torch.arange(12, dtype=torch.float32).reshape(3, 2, 2),
                torch.arange(12, 24, dtype=torch.float32).reshape(3, 2, 2),
            )
        ),
        length=2,
        timestamps=torch.tensor([0.08, 0.24]),
        seconds_per_grid=0.16,
        metadata={"width": 2, "height": 2},
    )


def test_literal_160ms_pair_interleaves_audio_and_video_timeline():
    pipeline = tiny_pipeline(
        audio_encoder=AudioWindowEncoder(
            hidden_size=4,
            window_size=4,
            hop_size=4,
            sample_rate=25,
        ),
        expansion_policy=TimestampInterleaveExpansion(
            seconds_per_bucket=0.16,
            bucket_token_ids={0: 20, 1: 21, 2: 22},
        ),
        position_builder=TMRoPEPositionBuilder(
            TMRoPEConfig(
                temporal_seconds_per_id=0.16,
                rotary_sections=(24, 20, 20),
            )
        ),
    )
    embedding = RecordingEmbedding()
    output = pipeline(
        input_ids=torch.tensor([[12, 11]]),
        attention_mask=torch.tensor([[1, 1]]),
        labels=None,
        decoded_media=(
            timed_audio_item(item_index=0),
            timed_video_item(item_index=1),
        ),
        text_embedding=embedding,
    )
    assert output.assembled.expanded_input_ids.tolist() == [
        [20, 12, 21, 11, 12, 22, 11]
    ]
    assert [
        span.source.source_id
        for span in output.assembled.spans
        if span.kind is SequenceSpanKind.MEDIA and span.source is not None
    ] == ["timed-audio", "timed-video", "timed-audio", "timed-video"]
    assert output.positions.position_ids[0, 0].tolist() == [
        0.0,
        1.0,
        2.0,
        2.0,
        2.0,
        3.0,
        3.0,
    ]
    assert [call.ndim for call in embedding.calls] == [2, 1]
    assert embedding.calls[1].tolist() == [20, 21, 22]


class PipelineTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    special_ids = {
        "<|image_pad|>": 10,
        "<|video_pad|>": 11,
        "<|audio_pad|>": 12,
        "<|vision_start|>": 13,
        "<|vision_end|>": 14,
        "<|audio_start|>": 15,
        "<|audio_end|>": 16,
    }

    def get_vocab(self):
        return dict(self.special_ids)

    def __call__(self, text, *, add_special_tokens=False):
        del add_special_tokens
        ids = []
        position = 0
        while position < len(text):
            token = next(
                (
                    candidate
                    for candidate in self.special_ids
                    if text.startswith(candidate, position)
                ),
                None,
            )
            if token is not None:
                ids.append(self.special_ids[token])
                position += len(token)
            else:
                ids.append(2 + ord(text[position]) % 8)
                position += 1
        return {"input_ids": ids}


class PipelineControlledLoader:
    def __init__(self, *, bad_paths=()) -> None:
        self.bad_paths = frozenset(bad_paths)

    def load(self, media_request):
        if media_request.path in self.bad_paths:
            cause = ValueError("corrupt media fixture")
            raise MediaLoadError(
                modality=media_request.modality.value,
                path=media_request.path,
                sample_id=media_request.sample_id,
                cause=cause,
            ) from cause
        if media_request.modality is MediaModality.IMAGE:
            return DecodedMedia(
                request=media_request,
                tensor=torch.arange(12, dtype=torch.float32).reshape(3, 2, 2),
                length=1,
                timestamps=None,
                seconds_per_grid=None,
                metadata={"original_width": 2, "original_height": 2},
            )
        if media_request.modality is MediaModality.VIDEO:
            return DecodedMedia(
                request=media_request,
                tensor=torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2),
                length=2,
                timestamps=torch.tensor([0.0, 0.5]),
                seconds_per_grid=0.5,
                metadata={"width": 2, "height": 2},
            )
        return DecodedMedia(
            request=media_request,
            tensor=torch.tensor([[0.0, 1.0, -1.0, 0.5]]),
            length=4,
            timestamps=None,
            seconds_per_grid=None,
            metadata={"sample_rate": 8},
        )


def profile_collator(*, bad_paths=()) -> ProfileStage2Collator:
    return ProfileStage2Collator(
        PipelineTokenizer(),
        max_seq_length=32,
        token_schema=MultimodalTokenSchema.qwen3(),
        media_loader=PipelineControlledLoader(bad_paths=bad_paths),
        quarantine_bad_samples=True,
    )


def profile_media(source_id, modality, path):
    return {
        "id": source_id,
        "modality": modality.value,
        "path": path,
    }


def test_quarantined_first_row_feeds_dense_final_requests_without_mutation():
    batch = profile_collator(bad_paths={"bad-image"})(
        [
            {
                "id": "bad-first",
                "input_text": "<|image_pad|>",
                "target_text": "x",
                "media": [
                    profile_media(
                        "bad-source",
                        MediaModality.IMAGE,
                        "bad-image",
                    )
                ],
            },
            {
                "id": "retained-second",
                "input_text": (
                    "<|audio_pad|><|video_pad|><|image_pad|>"
                ),
                "target_text": "y",
                "media": [
                    profile_media("audio", MediaModality.AUDIO, "good-audio"),
                    profile_media("video", MediaModality.VIDEO, "good-video"),
                    profile_media("image", MediaModality.IMAGE, "good-image"),
                ],
            },
        ]
    )
    snapshot = batch["decoded_media"]
    snapshot_ids = tuple(id(item) for item in snapshot)
    assert batch["_sample_ids"] == ["retained-second"]
    assert len(batch["_media_errors"]) == 1
    assert batch["_media_errors"][0]["sample_id"] == "bad-first"
    assert [item.request.sample_index for item in snapshot] == [0, 0, 0]
    assert [item.request.item_index for item in snapshot] == [0, 1, 2]
    assert [
        item.request.original_sample_index for item in snapshot
    ] == [1, 1, 1]

    pipeline = tiny_pipeline()
    embedding = RecordingEmbedding()
    first = pipeline(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        decoded_media=snapshot,
        text_embedding=embedding,
    )
    second = pipeline(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        decoded_media=tuple(reversed(snapshot)),
        text_embedding=embedding,
    )

    assert tuple(id(item) for item in batch["decoded_media"]) == snapshot_ids
    assert not hasattr(first, "media_errors")
    assert [sequence.sources for sequence in first.media_sequences] == [
        sequence.sources for sequence in second.media_sequences
    ]
    assert torch.equal(
        first.assembled.expanded_input_ids,
        second.assembled.expanded_input_ids,
    )
    assert torch.equal(
        first.assembled.inputs_embeds,
        second.assembled.inputs_embeds,
    )
    assert torch.equal(
        first.positions.position_ids,
        second.positions.position_ids,
    )
    def span_record(span):
        return (
            span.sample_index,
            span.start,
            span.end,
            span.kind,
            span.modality,
            span.grid,
            None if span.timestamps is None else span.timestamps.tolist(),
            span.seconds_per_grid,
            span.source,
            span.source_token_indices,
        )

    assert [span_record(span) for span in first.assembled.spans] == [
        span_record(span) for span in second.assembled.spans
    ]


def test_encoder_failure_after_collation_propagates_without_quarantine(
    monkeypatch,
):
    batch = profile_collator()(
        [
            {
                "id": "valid",
                "input_text": "<|image_pad|>",
                "target_text": "x",
                "media": [
                    profile_media("image", MediaModality.IMAGE, "good-image")
                ],
            }
        ]
    )
    pipeline = tiny_pipeline()

    def fail_encoder(items):
        raise RuntimeError("encoder failure")

    monkeypatch.setattr(pipeline.image_encoder, "forward", fail_encoder)
    with pytest.raises(RuntimeError, match="encoder failure"):
        pipeline(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            decoded_media=batch["decoded_media"],
            text_embedding=RecordingEmbedding(),
        )
    assert batch["_media_errors"] == []


def test_all_quarantined_batch_fails_before_pipeline_hook_runs():
    pipeline = tiny_pipeline()
    calls = []
    handle = pipeline.register_forward_hook(
        lambda *args: calls.append(True)
    )
    try:
        with pytest.raises(MediaLoadError, match="all-quarantined"):
            batch = profile_collator(bad_paths={"bad-image"})(
                [
                    {
                        "id": "bad-only",
                        "input_text": "<|image_pad|>",
                        "target_text": "x",
                        "media": [
                            profile_media(
                                "image",
                                MediaModality.IMAGE,
                                "bad-image",
                            )
                        ],
                    }
                ]
            )
            pipeline(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                decoded_media=batch["decoded_media"],
                text_embedding=RecordingEmbedding(),
            )
    finally:
        handle.remove()
    assert calls == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_placement_is_strict_for_text_only_path():
    device = torch.device("cuda", torch.cuda.current_device())
    pipeline = tiny_pipeline().to(device=device)
    embedding = RecordingEmbedding().to(device=device)
    output = pipeline(
        input_ids=torch.tensor([[5]], device=device),
        attention_mask=torch.tensor([[1]], device=device),
        labels=None,
        decoded_media=(),
        text_embedding=embedding,
    )
    assert output.assembled.inputs_embeds.device == device

    with pytest.raises(ValueError, match="input_ids.device"):
        pipeline(
            input_ids=torch.tensor([[5]], device=device),
            attention_mask=torch.tensor([[1]], device=device),
            labels=None,
            decoded_media=(),
            text_embedding=lambda ids: torch.zeros(*ids.shape, 4),
        )


def test_output_is_frozen_and_does_not_accept_collator_diagnostics():
    pipeline = tiny_pipeline()
    output = pipeline(
        input_ids=torch.tensor([[5]]),
        attention_mask=torch.tensor([[1]]),
        labels=None,
        decoded_media=(),
        text_embedding=RecordingEmbedding(),
    )
    with pytest.raises(FrozenInstanceError):
        output.media_sequences = ()
    assert not hasattr(output, "media_errors")
    with pytest.raises(TypeError):
        pipeline.encode_and_assemble(
            input_ids=torch.tensor([[5]]),
            attention_mask=torch.tensor([[1]]),
            labels=None,
            decoded_media=(),
            text_embedding=RecordingEmbedding(),
            _media_errors=(),
        )
