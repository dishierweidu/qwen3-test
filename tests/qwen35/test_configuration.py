from __future__ import annotations

import pytest

from qwen3_omni_pretrain.architecture.manifest import ProfileManifest
from qwen3_omni_pretrain.profiles.qwen35_omni_inspired.configuration import (
    PREDECESSOR_CODEC_MODEL_ID,
    PREDECESSOR_CODEC_PROVENANCE,
    PREDECESSOR_CODEC_REVISION,
    QWEN35_BACKBONE_REVISION,
    Qwen35InspiredConfig,
)


def tiny_qwen35_config_dict() -> dict[str, object]:
    return {
        "backbone_config": {
            "model_type": "qwen3_5_moe_text",
            "vocab_size": 64,
            "hidden_size": 32,
            "num_hidden_layers": 4,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
        },
        "audio_config": {
            "hidden_size": 32,
            "encoder_layers": 1,
            "attention_heads": 4,
            "intermediate_size": 64,
        },
        "aria_config": {},
        "codec_proxy": {
            "source_model": PREDECESSOR_CODEC_MODEL_ID,
            "source_revision": PREDECESSOR_CODEC_REVISION,
            "provenance_label": PREDECESSOR_CODEC_PROVENANCE,
        },
        "source_revision": QWEN35_BACKBONE_REVISION,
    }


def test_qwen35_profile_cannot_claim_official_omni_compatibility():
    config = Qwen35InspiredConfig(**tiny_qwen35_config_dict())
    assert config.model_type == "qwen35_omni_inspired"
    assert config.profile_manifest.compatibility_level.value == "paper-inspired"
    assert not config.profile_manifest.exact_official_checkpoint_compatible
    raw = config.to_dict()
    manifest = dict(raw["profile_manifest"])
    manifest["exact_official_checkpoint_compatible"] = True
    raw["profile_manifest"] = manifest
    with pytest.raises((ValueError, TypeError), match="checkpoint|exact"):
        Qwen35InspiredConfig(**raw)


def test_predecessor_codec_proxy_is_mandatory():
    raw = tiny_qwen35_config_dict()
    raw["codec_proxy"] = {}
    with pytest.raises(ValueError, match="predecessor"):
        Qwen35InspiredConfig(**raw)


def test_config_and_manifest_round_trip_is_canonical():
    config = Qwen35InspiredConfig(**tiny_qwen35_config_dict())
    restored = Qwen35InspiredConfig(**config.to_dict())
    assert restored.profile_manifest == config.profile_manifest
    assert ProfileManifest.from_dict(config.profile_manifest.to_dict()) == (
        config.profile_manifest
    )
    assert restored.profile_manifest.canonical_sha256() == (
        config.profile_manifest.canonical_sha256()
    )


def test_timestamp_format_changes_manifest_identity():
    raw = tiny_qwen35_config_dict()
    first = Qwen35InspiredConfig(**raw)
    raw["timestamp_format"] = "<time:{seconds:.3f}>"
    second = Qwen35InspiredConfig(**raw)
    assert first.profile_manifest.canonical_sha256() != (
        second.profile_manifest.canonical_sha256()
    )
