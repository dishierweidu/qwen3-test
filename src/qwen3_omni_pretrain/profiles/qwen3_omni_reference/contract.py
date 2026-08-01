"""Immutable Task 6 configuration and metadata provenance contract."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from .pins import (
    QWEN3_OMNI_MODEL_ID,
    QWEN3_OMNI_REVISION,
    QWEN3_TRANSFORMERS_VERSION,
)


QWEN3_OMNI_MODEL_TYPE = "qwen3_omni_moe"


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            key: _thaw(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


QWEN3_OMNI_CONFIG_CONTRACT: Mapping[str, object] = _freeze(
    {
        "audio_encoder": {
            "conv_hidden_size": 480,
            "conv_kernel_size": 3,
            "conv_layers": 3,
            "conv_stride": 2,
            "hidden_size": 1280,
            "input_sample_rate_hz": 16_000,
            "intermediate_size": 5120,
            "num_attention_heads": 20,
            "num_hidden_layers": 32,
            "num_mel_bins": 128,
            "projector_dimensions": [1280, 1280, 2048],
        },
        "code2wav": {
            "codebook_size": 2048,
            "hidden_size": 1024,
            "num_attention_heads": 16,
            "num_hidden_layers": 8,
            "num_quantizers": 16,
            "num_semantic_quantizers": 1,
            "output_sample_rate_hz": 24_000,
            "samples_per_code": 1920,
            "sliding_window": 72,
            "upsample_rates": [8, 5, 4, 3],
            "upsampling_ratios": [2, 2],
        },
        "code_predictor": {
            "hidden_size": 1024,
            "intermediate_size": 3072,
            "num_attention_heads": 16,
            "num_code_groups": 16,
            "num_hidden_layers": 5,
            "num_key_value_heads": 8,
            "vocab_size": 2048,
        },
        "schema_version": 1,
        "source": {
            "artifacts": {
                "README.md": {
                    "revision": QWEN3_OMNI_REVISION,
                    "sha256": (
                        "0e44065c4c4a27071f7239afd5b5a33a"
                        "f5bc2e437dd7ea9950e51aafabfde3df"
                    ),
                },
                "config.json": {
                    "revision": QWEN3_OMNI_REVISION,
                    "sha256": (
                        "eab5093d47807aaf894119506b238b2b"
                        "1cee70d08456e894fee9a012d88f2e0d"
                    ),
                },
                "preprocessor_config.json": {
                    "revision": QWEN3_OMNI_REVISION,
                    "sha256": (
                        "b10e27fd4542cf89ec7145942b87f3e6"
                        "5408d4e9f9d031a29acdd293c15fb3fc"
                    ),
                },
                "vocab.json": {
                    "revision": QWEN3_OMNI_REVISION,
                    "sha256": (
                        "ca10d7e9fb3ed18575dd1e277a2579c1"
                        "6d108e32f27439684afa0e10b1440910"
                    ),
                },
            },
            "derived_fields": {
                "audio_conv_kernel_and_stride": {
                    "artifact": (
                        "src/transformers/models/qwen3_omni_moe/"
                        "modeling_qwen3_omni_moe.py"
                    ),
                    "extraction": (
                        "AST of Qwen3OmniMoeAudioEncoder.__init__ "
                        "conv2d1/2/3 assignments"
                    ),
                    "revision": (
                        "7d9754a05193eb79b1d86aa744b622b8068008cd"
                    ),
                    "sha256": (
                        "0b6e9a6e9d88814de3e25b1ca65c49d"
                        "0677be7331b8118b6a0cd022c6c1dd270"
                    ),
                },
                "output_sample_rate_hz": {
                    "artifact": "README.md",
                    "extraction": (
                        "samplerate used to serialize audio returned by "
                        "model.generate"
                    ),
                    "locator": (
                        "lines 257-277; samplerate=24000 at line 277"
                    ),
                    "revision": QWEN3_OMNI_REVISION,
                    "sha256": (
                        "0e44065c4c4a27071f7239afd5b5a33a"
                        "f5bc2e437dd7ea9950e51aafabfde3df"
                    ),
                    "url": (
                        "https://huggingface.co/"
                        f"{QWEN3_OMNI_MODEL_ID}/resolve/"
                        f"{QWEN3_OMNI_REVISION}/README.md"
                    ),
                },
                "regular_vocab_size": {
                    "artifact": "vocab.json",
                    "extraction": (
                        "len(top-level token-to-id mapping)"
                    ),
                    "revision": QWEN3_OMNI_REVISION,
                    "sha256": (
                        "ca10d7e9fb3ed18575dd1e277a2579c1"
                        "6d108e32f27439684afa0e10b1440910"
                    ),
                    "url": (
                        "https://huggingface.co/"
                        f"{QWEN3_OMNI_MODEL_ID}/resolve/"
                        f"{QWEN3_OMNI_REVISION}/vocab.json"
                    ),
                    "validation": (
                        "IDs are contiguous from 0 through 151642"
                    ),
                },
            },
            "model_id": QWEN3_OMNI_MODEL_ID,
            "revision": QWEN3_OMNI_REVISION,
            "transformers_version": QWEN3_TRANSFORMERS_VERSION,
        },
        "talker": {
            "hidden_size": 1024,
            "moe_intermediate_size": 384,
            "num_attention_heads": 16,
            "num_experts": 128,
            "num_experts_per_tok": 6,
            "num_hidden_layers": 20,
            "num_key_value_heads": 2,
            "shared_expert_intermediate_size": 768,
        },
        "thinker": {
            "hidden_size": 2048,
            "moe_intermediate_size": 768,
            "num_attention_heads": 32,
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "num_hidden_layers": 48,
            "num_key_value_heads": 4,
        },
        "tm_rope": {
            "interleaved": True,
            "mrope_section": [24, 20, 20],
            "rope_theta": 1_000_000,
        },
        "vision_encoder": {
            "deepstack_visual_indexes": [8, 16, 24],
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "merger_dimensions": [4608, 4608, 2048],
            "num_attention_heads": 16,
            "num_hidden_layers": 27,
            "patch_kernel": [2, 16, 16],
            "spatial_merge_size": 2,
        },
        "vocabulary": {
            "embedding_vocab_size": 152_064,
            "regular_vocab_size": 151_643,
        },
    }
)

QWEN3_OMNI_METADATA_SHA256: Mapping[str, str] = MappingProxyType(
    {
        "README.md": (
            "0e44065c4c4a27071f7239afd5b5a33a"
            "f5bc2e437dd7ea9950e51aafabfde3df"
        ),
        "chat_template.json": (
            "90c1b81f29e41b7642b0cc02c877a10"
            "c8bf6751a8d8fa1d16ac9a718cf1c3d86"
        ),
        "config.json": (
            "eab5093d47807aaf894119506b238b2b"
            "1cee70d08456e894fee9a012d88f2e0d"
        ),
        "merges.txt": (
            "599bab54075088774b1733fde865d5bd7"
            "47cbcc7a547c5bc12610e874e26f5e3"
        ),
        "preprocessor_config.json": (
            "b10e27fd4542cf89ec7145942b87f3e6"
            "5408d4e9f9d031a29acdd293c15fb3fc"
        ),
        "tokenizer_config.json": (
            "dc3c31c3bdaedd5016382bb3cbe07323"
            "026775ad51f5a4fb564505992ae4a670"
        ),
        "vocab.json": (
            "ca10d7e9fb3ed18575dd1e277a2579c1"
            "6d108e32f27439684afa0e10b1440910"
        ),
    }
)


def config_contract_to_dict(
    contract: Mapping[str, object] = QWEN3_OMNI_CONFIG_CONTRACT,
) -> dict[str, object]:
    """Return a mutable JSON-shaped copy for serialization or parity checks."""

    thawed = _thaw(contract)
    if not isinstance(thawed, dict):
        raise TypeError("Qwen3 config contract must thaw to a dictionary")
    return thawed


__all__ = [
    "QWEN3_OMNI_CONFIG_CONTRACT",
    "QWEN3_OMNI_METADATA_SHA256",
    "QWEN3_OMNI_MODEL_TYPE",
    "config_contract_to_dict",
]
