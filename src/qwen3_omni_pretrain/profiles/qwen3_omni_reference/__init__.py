"""Pinned official Qwen3-Omni reference profile."""

from .oracle import (
    QWEN3_OMNI_MODEL_ID,
    QWEN3_OMNI_REVISION,
    QWEN3_TRANSFORMERS_VERSION,
    REFERENCE_DISTRIBUTION_VERSIONS,
    load_reference_config,
    load_reference_model,
    load_reference_processor,
    qwen3_reference_manifest,
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
