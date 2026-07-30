from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import shutil
from types import MappingProxyType
from typing import Mapping

import transformers
from transformers import (
    Qwen3OmniMoeConfig,
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoeProcessor,
)

from qwen3_omni_pretrain.architecture.manifest import (
    ProfileManifest,
    SourceRevision,
)
from qwen3_omni_pretrain.architecture.profiles import (
    ArchitectureProfile,
    CompatibilityLevel,
)


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
_TORCH_DISTRIBUTIONS = frozenset({"torch", "torchvision", "torchaudio"})

_EXPECTED_CONFIG_SIGNATURE: Mapping[str, object] = MappingProxyType(
    {
        "model_type": "qwen3_omni_moe",
        "thinker_config.text_config.hidden_size": 2048,
        "thinker_config.text_config.num_hidden_layers": 48,
        "thinker_config.text_config.num_attention_heads": 32,
        "thinker_config.text_config.num_key_value_heads": 4,
        "thinker_config.text_config.num_experts": 128,
        "thinker_config.text_config.num_experts_per_tok": 8,
        "thinker_config.text_config.moe_intermediate_size": 768,
        "thinker_config.text_config.vocab_size": 152_064,
        "thinker_config.audio_config.encoder_layers": 32,
        "thinker_config.audio_config.encoder_attention_heads": 20,
        "thinker_config.audio_config.d_model": 1280,
        "thinker_config.audio_config.output_dim": 2048,
        "thinker_config.vision_config.depth": 27,
        "thinker_config.vision_config.hidden_size": 1152,
        "thinker_config.vision_config.out_hidden_size": 2048,
        "talker_config.text_config.num_hidden_layers": 20,
        "talker_config.code_predictor_config.num_hidden_layers": 5,
        "code2wav_config.codebook_size": 2048,
        "code2wav_config.num_quantizers": 16,
    }
)


def qwen3_reference_manifest() -> ProfileManifest:
    return ProfileManifest(
        architecture_profile=ArchitectureProfile.QWEN3_OMNI_REFERENCE,
        compatibility_level=CompatibilityLevel.STRUCTURE_ALIGNED,
        sources={
            "model": SourceRevision(
                name=QWEN3_OMNI_MODEL_ID,
                revision=QWEN3_OMNI_REVISION,
            ),
            "transformers": SourceRevision(
                name="transformers",
                revision=QWEN3_TRANSFORMERS_VERSION,
            ),
            "qwen_omni_utils": SourceRevision(
                name="qwen-omni-utils",
                revision=REFERENCE_DISTRIBUTION_VERSIONS["qwen-omni-utils"],
            ),
        },
        assumptions=(
            "config and processor structure captured; model state unverified",
            "numerical, cache, offline-text, and offline-speech evidence pending",
        ),
        exact_official_checkpoint_compatible=False,
        validated_context_length=0,
    )


def _require_reference_environment() -> None:
    if transformers.__version__ != QWEN3_TRANSFORMERS_VERSION:
        raise RuntimeError(
            "qwen3_omni_reference requires transformers==5.2.0"
        )

    mismatches: list[str] = []
    for distribution, expected in REFERENCE_DISTRIBUTION_VERSIONS.items():
        try:
            actual = version(distribution)
        except PackageNotFoundError:
            mismatches.append(f"{distribution} is not installed")
            continue
        comparable = (
            actual.partition("+")[0]
            if distribution in _TORCH_DISTRIBUTIONS
            else actual
        )
        if comparable != expected:
            mismatches.append(
                f"{distribution}=={actual}, expected {distribution}=={expected}"
            )
    if mismatches:
        raise RuntimeError(
            "qwen3_omni_reference environment mismatch: "
            + "; ".join(mismatches)
        )
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("qwen3_omni_reference requires ffmpeg in PATH")


def _read_config_path(config: object, path: str) -> object:
    value = config
    for component in path.split("."):
        if not hasattr(value, component):
            raise ValueError(f"reference config is missing {path}")
        value = getattr(value, component)
    return value


def _validate_reference_config(config: object) -> None:
    mismatches = []
    for path, expected in _EXPECTED_CONFIG_SIGNATURE.items():
        try:
            actual = _read_config_path(config, path)
        except ValueError:
            mismatches.append(f"{path}=<missing>, expected {expected!r}")
            continue
        if actual != expected:
            mismatches.append(f"{path}={actual!r}, expected {expected!r}")
    if mismatches:
        raise ValueError(
            "pinned Qwen3-Omni config contradiction: "
            + "; ".join(mismatches)
        )


def load_reference_config(
    source: str = QWEN3_OMNI_MODEL_ID,
    *,
    local_files_only: bool = True,
) -> Qwen3OmniMoeConfig:
    _require_reference_environment()
    config = Qwen3OmniMoeConfig.from_pretrained(
        source,
        revision=QWEN3_OMNI_REVISION,
        local_files_only=local_files_only,
    )
    _validate_reference_config(config)
    return config


def load_reference_processor(
    source: str = QWEN3_OMNI_MODEL_ID,
    *,
    local_files_only: bool = True,
) -> Qwen3OmniMoeProcessor:
    _require_reference_environment()
    return Qwen3OmniMoeProcessor.from_pretrained(
        source,
        revision=QWEN3_OMNI_REVISION,
        local_files_only=local_files_only,
    )


def load_reference_model(
    source: str = QWEN3_OMNI_MODEL_ID,
    *,
    torch_dtype: object,
    device_map: object,
    local_files_only: bool = True,
) -> Qwen3OmniMoeForConditionalGeneration:
    _require_reference_environment()
    config = load_reference_config(
        source,
        local_files_only=local_files_only,
    )
    _validate_reference_config(config)
    return Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        source,
        revision=QWEN3_OMNI_REVISION,
        local_files_only=local_files_only,
        torch_dtype=torch_dtype,
        device_map=device_map,
        config=config,
    )
