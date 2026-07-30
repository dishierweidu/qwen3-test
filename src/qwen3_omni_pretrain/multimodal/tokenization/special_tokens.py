from __future__ import annotations

from types import MappingProxyType
from typing import Any
import warnings

from qwen3_omni_pretrain.architecture.profiles import (
    ArchitectureProfile,
)
from qwen3_omni_pretrain.multimodal.tokenization.schema import (
    resolve_token_schema,
    schema_for_profile,
)


_LEGACY_CONFIG_FIELDS = MappingProxyType(
    {
        "image_token_id": "image_pad",
        "video_token_id": "video_pad",
        "audio_token_id": "audio_pad",
        "audio_start_token_id": "audio_start",
        "audio_end_token_id": "audio_end",
    }
)
_LEGACY_SCHEMA = schema_for_profile(
    ArchitectureProfile.LEGACY_PROTOTYPE
)
MULTIMODAL_SPECIAL_TOKENS = MappingProxyType(
    {
        config_field: getattr(_LEGACY_SCHEMA, schema_field)
        for config_field, schema_field in _LEGACY_CONFIG_FIELDS.items()
    }
)


def reconcile_multimodal_token_ids(
    config: Any,
    tokenizer: Any,
) -> dict[str, int]:
    vocab_size = int(config.vocab_size)
    schema_ids = resolve_token_schema(
        tokenizer,
        _LEGACY_SCHEMA,
        vocab_size,
    )
    if len(tokenizer) > vocab_size:
        raise ValueError(
            f"tokenizer length {len(tokenizer)} exceeds "
            f"model vocab_size {vocab_size}"
        )
    resolved = {
        config_field: int(getattr(schema_ids, schema_field))
        for config_field, schema_field in _LEGACY_CONFIG_FIELDS.items()
    }

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
