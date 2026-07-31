from __future__ import annotations

import re

import torch
from torch import nn

from qwen3_omni_pretrain.multimodal.modalities import MediaModality
from qwen3_omni_pretrain.multimodal.sequence_assembler import (
    MediaExpansionGroup,
    MediaPlaceholder,
    SequenceAssembler,
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
from qwen3_omni_pretrain.profiles.qwen35_omni_inspired.timestamp_alignment import (
    Qwen35TimestampExpansionPolicy,
    build_qwen35_position_builder,
)


class FakeTimestampTokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool):
        assert add_special_tokens is False
        match = re.search(r"([0-9]+\.[0-9]+)", text)
        assert match is not None
        centiseconds = round(float(match.group(1)) * 100)
        prefix = 40 if text.startswith("[") else 80
        return {"input_ids": [prefix + centiseconds]}


def _audio(
    timestamps: tuple[float, ...],
    *,
    sample: int = 0,
    item: int = 0,
    hidden: int = 8,
) -> MediaSequence:
    count = len(timestamps)
    result = MediaSequence(
        embeddings=torch.randn(1, count, hidden),
        attention_mask=torch.ones(1, count, dtype=torch.bool),
        modality=MediaModality.AUDIO,
        sources=(MediaSource(sample, item, f"audio-{sample}-{item}"),),
        timestamps=torch.tensor([timestamps], dtype=torch.float32),
    )
    result.validate()
    return result


def _video(
    timestamps: tuple[float, ...],
    *,
    sample: int = 0,
    item: int = 1,
    hidden: int = 8,
) -> MediaSequence:
    count = len(timestamps)
    result = MediaSequence(
        embeddings=torch.randn(1, count, hidden),
        attention_mask=torch.ones(1, count, dtype=torch.bool),
        modality=MediaModality.VIDEO,
        sources=(MediaSource(sample, item, f"video-{sample}-{item}"),),
        grid=(MediaGrid(count, 1, 1),),
        timestamps=torch.tensor([timestamps], dtype=torch.float32),
        seconds_per_grid=None,
    )
    result.validate()
    return result


def _tokens() -> ResolvedMultimodalTokens:
    return ResolvedMultimodalTokens(
        image_pad=11,
        video_pad=12,
        audio_pad=13,
        vision_start=14,
        vision_end=15,
        audio_start=16,
        audio_end=17,
    )


def test_each_temporal_audio_unit_gets_a_timestamp_prefix():
    sequence = _audio((0.00, 0.16, 0.32))
    placeholder = MediaPlaceholder(
        text_position=1,
        sentinel_token_id=13,
        modality=MediaModality.AUDIO,
        sequence=sequence,
        sequence_row=0,
    )
    policy = Qwen35TimestampExpansionPolicy(
        tokenizer=FakeTimestampTokenizer(),
        timestamp_format="[{seconds:.2f}s]",
    )
    expanded = policy.expand_sample(
        sample_index=0,
        groups=(MediaExpansionGroup((placeholder,), 1, 1),),
    ).replacements[1]
    assert [token.token_id for token in expanded.tokens] == [
        40,
        13,
        56,
        13,
        72,
        13,
    ]
    assert [token.kind for token in expanded.tokens] == [
        SequenceSpanKind.TIMESTAMP,
        SequenceSpanKind.MEDIA,
    ] * 3


def test_audio_and_video_are_interleaved_per_sample_timeline():
    torch.manual_seed(2)
    embedding = nn.Embedding(128, 8)
    prompt = torch.tensor([[5, 13, 9, 12, 6]])
    assembled = SequenceAssembler().assemble(
        input_ids=prompt,
        text_embeddings=embedding(prompt),
        attention_mask=torch.ones_like(prompt, dtype=torch.bool),
        labels=None,
        media_sequences=(
            _video((0.08, 0.24), item=1),
            _audio((0.00, 0.16), item=0),
        ),
        tokens=_tokens(),
        expansion_policy=Qwen35TimestampExpansionPolicy(
            tokenizer=FakeTimestampTokenizer()
        ),
        embedding_lookup=embedding,
        pad_token_id=0,
        joint_separator_token_ids=frozenset({9}),
        max_assembled_length=64,
    )
    media_modalities = [
        span.modality.value
        for span in assembled.spans
        if span.kind is SequenceSpanKind.MEDIA
    ]
    assert media_modalities == ["audio", "video", "audio", "video"]
    media_times = [
        float(span.timestamps[0].item())
        for span in assembled.spans
        if span.kind is SequenceSpanKind.MEDIA
    ]
    torch.testing.assert_close(
        torch.tensor(media_times),
        torch.tensor([0.0, 0.08, 0.16, 0.24]),
    )


def test_equal_audio_video_time_uses_one_160ms_temporal_bucket():
    torch.manual_seed(4)
    embedding = nn.Embedding(128, 8)
    prompt = torch.tensor([[5, 13, 9, 12, 6]])
    assembled = SequenceAssembler().assemble(
        input_ids=prompt,
        text_embeddings=embedding(prompt),
        attention_mask=torch.ones_like(prompt, dtype=torch.bool),
        labels=None,
        media_sequences=(
            _audio((0.32,), item=0),
            _video((0.32,), item=1),
        ),
        tokens=_tokens(),
        expansion_policy=Qwen35TimestampExpansionPolicy(
            tokenizer=FakeTimestampTokenizer()
        ),
        embedding_lookup=embedding,
        pad_token_id=0,
        joint_separator_token_ids=frozenset({9}),
        max_assembled_length=64,
    )
    positions = build_qwen35_position_builder().build(assembled)
    media_spans = [
        span for span in assembled.spans if span.kind is SequenceSpanKind.MEDIA
    ]
    temporal = [
        float(positions.position_ids[0, 0, span.start].item())
        for span in media_spans
    ]
    assert temporal[0] == temporal[1]
    assert positions.axis_names == ("temporal", "height", "width")
    assert build_qwen35_position_builder().config.temporal_seconds_per_id == 0.16


def test_timestamp_format_changes_ordinary_token_ids():
    sequence = _audio((0.16,))
    placeholder = MediaPlaceholder(
        text_position=0,
        sentinel_token_id=13,
        modality=MediaModality.AUDIO,
        sequence=sequence,
        sequence_row=0,
    )
    group = MediaExpansionGroup((placeholder,), 0, 0)
    bracket = Qwen35TimestampExpansionPolicy(
        tokenizer=FakeTimestampTokenizer(),
        timestamp_format="[{seconds:.2f}s]",
    ).expand_sample(sample_index=0, groups=(group,))
    angle = Qwen35TimestampExpansionPolicy(
        tokenizer=FakeTimestampTokenizer(),
        timestamp_format="<time:{seconds:.2f}>",
    ).expand_sample(sample_index=0, groups=(group,))
    assert bracket.replacements[0].tokens[0].token_id == 56
    assert angle.replacements[0].tokens[0].token_id == 96
