"""Pinned official Qwen3-Omni reference profile."""

from typing import Any

from .pins import (
    QWEN3_OMNI_MODEL_ID,
    QWEN3_OMNI_REVISION,
    QWEN3_TRANSFORMERS_VERSION,
    REFERENCE_DISTRIBUTION_VERSIONS,
)


def qwen3_reference_manifest():
    from .oracle import qwen3_reference_manifest as implementation

    return implementation()


def load_reference_config(
    source: str = QWEN3_OMNI_MODEL_ID,
    *,
    local_files_only: bool = True,
) -> Any:
    from .oracle import load_reference_config as implementation

    return implementation(source, local_files_only=local_files_only)


def load_reference_processor(
    source: str = QWEN3_OMNI_MODEL_ID,
    *,
    local_files_only: bool = True,
) -> Any:
    from .oracle import load_reference_processor as implementation

    return implementation(source, local_files_only=local_files_only)


def load_reference_model(
    source: str = QWEN3_OMNI_MODEL_ID,
    *,
    torch_dtype: object,
    device_map: object,
    local_files_only: bool = True,
) -> Any:
    from .oracle import load_reference_model as implementation

    return implementation(
        source,
        torch_dtype=torch_dtype,
        device_map=device_map,
        local_files_only=local_files_only,
    )


__all__ = [
    "QWEN3_OMNI_MODEL_ID",
    "QWEN3_OMNI_REVISION",
    "QWEN3_TRANSFORMERS_VERSION",
    "REFERENCE_DISTRIBUTION_VERSIONS",
    "load_reference_config",
    "load_reference_model",
    "load_reference_processor",
    "qwen3_reference_manifest",
]
