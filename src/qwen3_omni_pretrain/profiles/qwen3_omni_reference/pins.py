"""Lightweight immutable pins for the official Qwen3-Omni reference."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping


QWEN3_OMNI_MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
QWEN3_OMNI_REVISION = "26291f793822fb6be9555850f06dfe95f2d7e695"
QWEN3_TRANSFORMERS_VERSION = "5.2.0"

REFERENCE_DISTRIBUTION_VERSIONS: Mapping[str, str] = MappingProxyType(
    {
        "torch": "2.10.0",
        "torchvision": "0.25.0",
        "torchaudio": "2.10.0",
        "transformers": QWEN3_TRANSFORMERS_VERSION,
        "qwen-omni-utils": "0.0.9",
    }
)
TORCH_DISTRIBUTIONS = frozenset({"torch", "torchvision", "torchaudio"})


def distribution_version_matches(
    distribution: str,
    actual: str,
    expected: str,
) -> bool:
    """Compare exact pins while permitting local CUDA tags for torch wheels."""

    comparable = (
        actual.partition("+")[0]
        if distribution in TORCH_DISTRIBUTIONS
        else actual
    )
    return comparable == expected


__all__ = [
    "QWEN3_OMNI_MODEL_ID",
    "QWEN3_OMNI_REVISION",
    "QWEN3_TRANSFORMERS_VERSION",
    "REFERENCE_DISTRIBUTION_VERSIONS",
    "TORCH_DISTRIBUTIONS",
    "distribution_version_matches",
]
