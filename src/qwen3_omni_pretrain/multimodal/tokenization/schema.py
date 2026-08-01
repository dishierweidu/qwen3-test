from __future__ import annotations

from collections.abc import Mapping
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
        if not isinstance(modality, MediaModality):
            raise TypeError("modality must be MediaModality")
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
    if not isinstance(schema, MultimodalTokenSchema):
        raise TypeError("schema must be MultimodalTokenSchema")
    if type(vocab_size) is not int:
        raise TypeError("vocab_size must be an integer")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    vocab = tokenizer.get_vocab()
    if not isinstance(vocab, Mapping):
        raise TypeError("tokenizer.get_vocab() must return a mapping")
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

    resolved: dict[str, int | None] = {}
    for name, token in schema_values.items():
        if token is None:
            resolved[name] = None
            continue
        token_id = vocab[token]
        if type(token_id) is not int:
            raise TypeError(
                f"tokenizer ID for {name} must be an integer"
            )
        resolved[name] = token_id
    present_ids = [
        token_id
        for token_id in resolved.values()
        if token_id is not None
    ]
    if len(set(present_ids)) != len(present_ids):
        raise ValueError(
            "multimodal special tokens must resolve to distinct IDs"
        )

    invalid = {
        name: token_id
        for name, token_id in resolved.items()
        if token_id is not None
        and not 0 <= token_id < vocab_size
    }
    if invalid:
        name, token_id = next(iter(invalid.items()))
        raise ValueError(
            f"{name}={token_id} is outside "
            f"model vocab_size={vocab_size}"
        )

    return ResolvedMultimodalTokens(**resolved)


__all__ = [
    "MultimodalTokenSchema",
    "ResolvedMultimodalTokens",
    "resolve_token_schema",
    "schema_for_profile",
]
