from __future__ import annotations

from types import MappingProxyType
from typing import Any
import warnings


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
    missing = [
        token
        for token in MULTIMODAL_SPECIAL_TOKENS.values()
        if token not in vocab
    ]
    if missing:
        raise ValueError(
            f"tokenizer is missing required multimodal token {missing[0]!r}"
        )

    resolved = {
        field: int(vocab[token])
        for field, token in MULTIMODAL_SPECIAL_TOKENS.items()
    }
    if len(set(resolved.values())) != len(resolved):
        raise ValueError(
            "multimodal special tokens must resolve to distinct IDs"
        )

    vocab_size = int(config.vocab_size)
    if len(tokenizer) > vocab_size:
        raise ValueError(
            f"tokenizer length {len(tokenizer)} exceeds "
            f"model vocab_size {vocab_size}"
        )
    invalid = {
        field: token_id
        for field, token_id in resolved.items()
        if not 0 <= token_id < vocab_size
    }
    if invalid:
        field, token_id = next(iter(invalid.items()))
        raise ValueError(
            f"{field}={token_id} is outside model vocab_size={vocab_size}"
        )

    for field, token_id in resolved.items():
        current = getattr(config, field, None)
        if current is not None and int(current) != token_id:
            warnings.warn(
                f"{field}={current} disagrees with tokenizer ID "
                f"{token_id}; using tokenizer value in memory",
                RuntimeWarning,
                stacklevel=2,
            )
    for field, token_id in resolved.items():
        setattr(config, field, token_id)
    return resolved
