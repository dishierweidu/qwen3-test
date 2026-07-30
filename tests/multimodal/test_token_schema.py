from dataclasses import FrozenInstanceError

import pytest

from qwen3_omni_pretrain.architecture.profiles import (
    ArchitectureProfile,
)
from qwen3_omni_pretrain.multimodal.modalities import MediaModality
from qwen3_omni_pretrain.multimodal.tokenization.schema import (
    MultimodalTokenSchema,
    ResolvedMultimodalTokens,
    resolve_token_schema,
    schema_for_profile,
)


class FakeTokenizer:
    def __init__(self, vocab):
        self._vocab = dict(vocab)

    def get_vocab(self):
        return dict(self._vocab)


QWEN_VOCAB = {
    "<|image_pad|>": 1,
    "<|video_pad|>": 2,
    "<|audio_pad|>": 3,
    "<|vision_start|>": 4,
    "<|vision_end|>": 5,
    "<|audio_start|>": 6,
    "<|audio_end|>": 7,
}


def test_qwen_and_mimo_audio_strings_cannot_be_interchanged():
    qwen = schema_for_profile(
        ArchitectureProfile.QWEN3_OMNI_REFERENCE
    )
    mimo = schema_for_profile(
        ArchitectureProfile.MIMO_V25_EXPERIMENTAL
    )

    assert qwen.audio_pad == "<|audio_pad|>"
    assert mimo.audio_start == "<|mimo_audio_start|>"
    assert qwen.audio_start != mimo.audio_start


def test_qwen35_has_an_explicit_schema_entry():
    qwen = schema_for_profile(
        ArchitectureProfile.QWEN3_OMNI_REFERENCE
    )
    qwen35 = schema_for_profile(
        ArchitectureProfile.QWEN35_OMNI_INSPIRED
    )

    assert qwen35 is not qwen
    assert qwen35.audio_start == "<|audio_start|>"
    assert qwen35.image_pad == "<|image_pad|>"


def test_legacy_schema_keeps_new_vision_wrappers_optional():
    legacy = schema_for_profile(
        ArchitectureProfile.LEGACY_PROTOTYPE
    )

    assert legacy.vision_start is None
    assert legacy.vision_end is None
    assert legacy.audio_start == "<|audio_start|>"
    assert legacy.audio_end == "<|audio_end|>"


def test_resolution_collects_every_missing_required_field():
    tokenizer = FakeTokenizer(
        {
            "<|image_pad|>": 10,
            "<|video_pad|>": 11,
            "<|audio_pad|>": 12,
        }
    )

    with pytest.raises(ValueError) as captured:
        resolve_token_schema(
            tokenizer,
            MultimodalTokenSchema.qwen3(),
            vocab_size=32,
        )

    message = str(captured.value)
    for field in (
        "vision_start",
        "vision_end",
        "audio_start",
        "audio_end",
    ):
        assert field in message


def test_resolution_returns_immutable_ids_and_modality_sentinels():
    resolved = resolve_token_schema(
        FakeTokenizer(QWEN_VOCAB),
        MultimodalTokenSchema.qwen3(),
        vocab_size=8,
    )

    assert resolved == ResolvedMultimodalTokens(
        image_pad=1,
        video_pad=2,
        audio_pad=3,
        vision_start=4,
        vision_end=5,
        audio_start=6,
        audio_end=7,
    )
    assert resolved.sentinel_for(MediaModality.IMAGE) == 1
    assert resolved.sentinel_for(MediaModality.VIDEO) == 2
    assert resolved.sentinel_for(MediaModality.AUDIO) == 3
    with pytest.raises(FrozenInstanceError):
        resolved.image_pad = 9


def test_optional_wrapper_tokens_resolve_to_none():
    schema = MultimodalTokenSchema(
        image_pad="<image>",
        video_pad="<video>",
        audio_pad="<audio>",
    )

    resolved = resolve_token_schema(
        FakeTokenizer(
            {
                "<image>": 1,
                "<video>": 2,
                "<audio>": 3,
            }
        ),
        schema,
        vocab_size=4,
    )

    assert resolved.vision_start is None
    assert resolved.vision_end is None
    assert resolved.audio_start is None
    assert resolved.audio_end is None


def test_resolution_rejects_duplicate_ids():
    vocab = dict(QWEN_VOCAB)
    vocab["<|audio_end|>"] = vocab["<|audio_start|>"]

    with pytest.raises(ValueError, match="distinct IDs"):
        resolve_token_schema(
            FakeTokenizer(vocab),
            MultimodalTokenSchema.qwen3(),
            vocab_size=8,
        )


@pytest.mark.parametrize("bad_id", [-1, 8])
def test_resolution_rejects_ids_outside_model_vocab(bad_id):
    vocab = dict(QWEN_VOCAB)
    vocab["<|audio_end|>"] = bad_id

    with pytest.raises(ValueError, match="outside model vocab_size"):
        resolve_token_schema(
            FakeTokenizer(vocab),
            MultimodalTokenSchema.qwen3(),
            vocab_size=8,
        )
