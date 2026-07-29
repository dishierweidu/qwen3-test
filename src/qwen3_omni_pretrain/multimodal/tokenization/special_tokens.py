from __future__ import annotations

from types import MappingProxyType
from typing import Any


MULTIMODAL_SPECIAL_TOKENS = MappingProxyType(
    {
        "image_token_id": "<|image_pad|>",
        "video_token_id": "<|video_pad|>",
        "audio_token_id": "<|audio_pad|>",
        "audio_start_token_id": "<|audio_start|>",
        "audio_end_token_id": "<|audio_end|>",
    }
)


def reconcile_multimodal_token_ids(
    config: Any,
    tokenizer: Any,
) -> dict[str, int]:
    vocab = tokenizer.get_vocab()
    resolved = {
        field: int(vocab[token])
        for field, token in MULTIMODAL_SPECIAL_TOKENS.items()
    }
    for field, token_id in resolved.items():
        setattr(config, field, token_id)
    return resolved
