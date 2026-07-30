from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from qwen3_omni_pretrain.architecture.profiles import (
    ArchitectureProfile,
)
from qwen3_omni_pretrain.multimodal.modalities import MediaModality


@dataclass(frozen=True)
class MultimodalTokenSchema:
    image_pad: str
    video_pad: str
    audio_pad: str
    vision_start: str | None = None
    vision_end: str | None = None
    audio_start: str | None = None
    audio_end: str | None = None

    @classmethod
    def legacy(cls) -> MultimodalTokenSchema:
        return cls(
            image_pad="<|image_pad|>",
            video_pad="<|video_pad|>",
            audio_pad="<|audio_pad|>",
            audio_start="<|audio_start|>",
            audio_end="<|audio_end|>",
        )

    @classmethod
    def qwen3(cls) -> MultimodalTokenSchema:
        return cls(
            image_pad="<|image_pad|>",
            video_pad="<|video_pad|>",
            audio_pad="<|audio_pad|>",
            vision_start="<|vision_start|>",
            vision_end="<|vision_end|>",
            audio_start="<|audio_start|>",
            audio_end="<|audio_end|>",
        )

    @classmethod
    def qwen35(cls) -> MultimodalTokenSchema:
        return cls(
            image_pad="<|image_pad|>",
            video_pad="<|video_pad|>",
            audio_pad="<|audio_pad|>",
            vision_start="<|vision_start|>",
            vision_end="<|vision_end|>",
            audio_start="<|audio_start|>",
            audio_end="<|audio_end|>",
        )

    @classmethod
    def mimo_v25(cls) -> MultimodalTokenSchema:
        return cls(
            image_pad="<|mimo_image_pad|>",
            video_pad="<|mimo_video_pad|>",
            audio_pad="<|mimo_audio_pad|>",
            vision_start="<|mimo_vision_start|>",
            vision_end="<|mimo_vision_end|>",
            audio_start="<|mimo_audio_start|>",
            audio_end="<|mimo_audio_end|>",
        )


@dataclass(frozen=True)
class ResolvedMultimodalTokens:
    image_pad: int
    video_pad: int
    audio_pad: int
    vision_start: int | None
    vision_end: int | None
    audio_start: int | None
    audio_end: int | None

    def sentinel_for(self, modality: MediaModality) -> int:
        return {
            MediaModality.IMAGE: self.image_pad,
            MediaModality.VIDEO: self.video_pad,
            MediaModality.AUDIO: self.audio_pad,
        }[modality]


def schema_for_profile(
    profile: ArchitectureProfile,
) -> MultimodalTokenSchema:
    if not isinstance(profile, ArchitectureProfile):
        raise TypeError("profile must be ArchitectureProfile")
    if profile is ArchitectureProfile.LEGACY_PROTOTYPE:
        return MultimodalTokenSchema.legacy()
    if profile is ArchitectureProfile.QWEN3_OMNI_REFERENCE:
        return MultimodalTokenSchema.qwen3()
    if profile is ArchitectureProfile.QWEN35_OMNI_INSPIRED:
        return MultimodalTokenSchema.qwen35()
    if profile is ArchitectureProfile.MIMO_V25_EXPERIMENTAL:
        return MultimodalTokenSchema.mimo_v25()
    raise ValueError(f"unsupported architecture profile: {profile.value}")


def resolve_token_schema(
    tokenizer: Any,
    schema: MultimodalTokenSchema,
    vocab_size: int,
) -> ResolvedMultimodalTokens:
    vocab = tokenizer.get_vocab()
    schema_values = {
        field.name: getattr(schema, field.name)
        for field in fields(MultimodalTokenSchema)
    }
    required = {
        name: token
        for name, token in schema_values.items()
        if token is not None
    }
    missing = [
        name for name, token in required.items() if token not in vocab
    ]
    if missing:
        raise ValueError(
            "tokenizer is missing required multimodal token fields: "
            + ", ".join(missing)
        )

    resolved: dict[str, int | None] = {
        name: (
            None
            if token is None
            else int(vocab[token])
        )
        for name, token in schema_values.items()
    }
    present_ids = [
        token_id
        for token_id in resolved.values()
        if token_id is not None
    ]
    if len(set(present_ids)) != len(present_ids):
        raise ValueError(
            "multimodal special tokens must resolve to distinct IDs"
        )

    model_vocab_size = int(vocab_size)
    invalid = {
        name: token_id
        for name, token_id in resolved.items()
        if token_id is not None
        and not 0 <= token_id < model_vocab_size
    }
    if invalid:
        name, token_id = next(iter(invalid.items()))
        raise ValueError(
            f"{name}={token_id} is outside "
            f"model vocab_size={model_vocab_size}"
        )

    return ResolvedMultimodalTokens(**resolved)


__all__ = [
    "MultimodalTokenSchema",
    "ResolvedMultimodalTokens",
    "resolve_token_schema",
    "schema_for_profile",
]
