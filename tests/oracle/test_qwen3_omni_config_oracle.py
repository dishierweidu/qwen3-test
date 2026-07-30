from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen3_omni_pretrain.architecture.profiles import (
    ArchitectureProfile,
    CompatibilityLevel,
)
from qwen3_omni_pretrain.profiles.qwen3_omni_reference import oracle
from qwen3_omni_pretrain.profiles.qwen3_omni_reference.oracle import (
    QWEN3_OMNI_MODEL_ID,
    QWEN3_OMNI_REVISION,
    QWEN3_TRANSFORMERS_VERSION,
    load_reference_config,
    load_reference_model,
    load_reference_processor,
    qwen3_reference_manifest,
)
from tests.oracle.extract_qwen3_omni_config_contract import extract_contract


CONTRACT_PATH = Path("tests/fixtures/qwen3_omni/config_contract.json")


def _pinned_config() -> SimpleNamespace:
    return SimpleNamespace(
        model_type="qwen3_omni_moe",
        thinker_config=SimpleNamespace(
            audio_config=SimpleNamespace(
                num_mel_bins=128,
                downsample_hidden_size=480,
                encoder_layers=32,
                encoder_attention_heads=20,
                d_model=1280,
                encoder_ffn_dim=5120,
                output_dim=2048,
            ),
            vision_config=SimpleNamespace(
                temporal_patch_size=2,
                patch_size=16,
                spatial_merge_size=2,
                deepstack_visual_indexes=[8, 16, 24],
                depth=27,
                num_heads=16,
                hidden_size=1152,
                intermediate_size=4304,
                out_hidden_size=2048,
            ),
            text_config=SimpleNamespace(
                hidden_size=2048,
                num_hidden_layers=48,
                num_attention_heads=32,
                num_key_value_heads=4,
                num_experts=128,
                num_experts_per_tok=8,
                moe_intermediate_size=768,
                rope_scaling={
                    "interleaved": True,
                    "mrope_section": [24, 20, 20],
                },
                rope_theta=1_000_000,
                vocab_size=152_064,
            ),
        ),
        talker_config=SimpleNamespace(
            text_config=SimpleNamespace(
                hidden_size=1024,
                num_hidden_layers=20,
                num_attention_heads=16,
                num_key_value_heads=2,
                num_experts=128,
                num_experts_per_tok=6,
                moe_intermediate_size=384,
                shared_expert_intermediate_size=768,
            ),
            code_predictor_config=SimpleNamespace(
                hidden_size=1024,
                num_hidden_layers=5,
                num_attention_heads=16,
                num_key_value_heads=8,
                intermediate_size=3072,
                num_code_groups=16,
                vocab_size=2048,
            ),
        ),
        code2wav_config=SimpleNamespace(
            codebook_size=2048,
            num_quantizers=16,
            num_semantic_quantizers=1,
            num_hidden_layers=8,
            hidden_size=1024,
            num_attention_heads=16,
            sliding_window=72,
            upsample_rates=[8, 5, 4, 3],
            upsampling_ratios=[2, 2],
        ),
    )


def test_qwen3_oracle_pins_and_evidence_are_immutable():
    assert QWEN3_OMNI_MODEL_ID == "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    assert QWEN3_OMNI_REVISION == (
        "26291f793822fb6be9555850f06dfe95f2d7e695"
    )
    assert QWEN3_TRANSFORMERS_VERSION == "5.2.0"
    assert dict(oracle.REFERENCE_DISTRIBUTION_VERSIONS) == {
        "torch": "2.10.0",
        "torchvision": "0.25.0",
        "torchaudio": "2.10.0",
        "transformers": "5.2.0",
        "qwen-omni-utils": "0.0.9",
    }

    manifest = qwen3_reference_manifest()
    assert manifest.architecture_profile is ArchitectureProfile.QWEN3_OMNI_REFERENCE
    assert manifest.compatibility_level is CompatibilityLevel.STRUCTURE_ALIGNED
    assert not manifest.exact_official_checkpoint_compatible
    assert manifest.validated_context_length == 0
    assert manifest.to_dict()["sources"] == {
        "model": {
            "name": QWEN3_OMNI_MODEL_ID,
            "revision": QWEN3_OMNI_REVISION,
        },
        "qwen_omni_utils": {
            "name": "qwen-omni-utils",
            "revision": "0.0.9",
        },
        "transformers": {
            "name": "transformers",
            "revision": "5.2.0",
        },
    }
    assert manifest.assumptions == (
        "config and processor structure captured; model state unverified",
        "numerical, cache, offline-text, and offline-speech evidence pending",
    )


def test_checked_in_contract_is_a_small_pinned_provenance_extract():
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

    assert set(contract) == {
        "schema_version",
        "source",
        "vocabulary",
        "audio_encoder",
        "vision_encoder",
        "tm_rope",
        "thinker",
        "talker",
        "code_predictor",
        "code2wav",
    }
    assert contract["schema_version"] == 1
    assert contract["source"] == {
        "model_id": QWEN3_OMNI_MODEL_ID,
        "revision": QWEN3_OMNI_REVISION,
        "transformers_version": "5.2.0",
        "config_files": ["config.json", "preprocessor_config.json"],
        "derived_fields": {
            "audio_conv_kernel_and_stride": "transformers==5.2.0",
            "output_sample_rate_hz": "Qwen3-Omni Technical Report",
            "regular_vocab_size": "Qwen3-Omni Technical Report",
        },
    }
    assert contract["vocabulary"] == {
        "regular_vocab_size": 151_643,
        "embedding_vocab_size": 152_064,
    }
    assert contract["thinker"] == {
        "hidden_size": 2048,
        "num_hidden_layers": 48,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 768,
    }
    assert contract["talker"]["num_hidden_layers"] == 20
    assert contract["code_predictor"]["num_hidden_layers"] == 5
    assert contract["code2wav"]["output_sample_rate_hz"] == 24_000
    assert CONTRACT_PATH.stat().st_size < 8_192


def test_contract_extractor_selects_only_the_pinned_architecture_fields():
    extracted = extract_contract(
        _pinned_config(),
        preprocessor_config={"sampling_rate": 16_000},
    )

    assert extracted["audio_encoder"] == {
        "input_sample_rate_hz": 16_000,
        "num_mel_bins": 128,
        "conv_layers": 3,
        "conv_hidden_size": 480,
        "conv_kernel_size": 3,
        "conv_stride": 2,
        "num_hidden_layers": 32,
        "num_attention_heads": 20,
        "hidden_size": 1280,
        "intermediate_size": 5120,
        "projector_dimensions": [1280, 1280, 2048],
    }
    assert extracted["vision_encoder"] == {
        "patch_kernel": [2, 16, 16],
        "spatial_merge_size": 2,
        "deepstack_visual_indexes": [8, 16, 24],
        "num_hidden_layers": 27,
        "num_attention_heads": 16,
        "hidden_size": 1152,
        "intermediate_size": 4304,
        "merger_dimensions": [4608, 4608, 2048],
    }
    assert extracted["tm_rope"] == {
        "mrope_section": [24, 20, 20],
        "rope_theta": 1_000_000,
        "interleaved": True,
    }
    assert extracted["code2wav"]["samples_per_code"] == 1920
    assert extracted["code2wav"]["output_sample_rate_hz"] == 24_000


def test_load_reference_config_uses_the_pinned_revision_and_local_only(
    monkeypatch,
):
    calls = []
    expected = _pinned_config()

    class FakeConfigLoader:
        @classmethod
        def from_pretrained(cls, source, **kwargs):
            calls.append((source, kwargs))
            return expected

    monkeypatch.setattr(oracle, "_require_reference_environment", lambda: None)
    monkeypatch.setattr(oracle, "Qwen3OmniMoeConfig", FakeConfigLoader)

    loaded = load_reference_config()

    assert loaded is expected
    assert calls == [
        (
            QWEN3_OMNI_MODEL_ID,
            {
                "revision": QWEN3_OMNI_REVISION,
                "local_files_only": True,
            },
        )
    ]


def test_load_reference_processor_uses_the_pinned_revision_and_local_only(
    monkeypatch,
):
    calls = []
    expected = object()

    class FakeProcessorLoader:
        @classmethod
        def from_pretrained(cls, source, **kwargs):
            calls.append((source, kwargs))
            return expected

    monkeypatch.setattr(oracle, "_require_reference_environment", lambda: None)
    monkeypatch.setattr(oracle, "Qwen3OmniMoeProcessor", FakeProcessorLoader)

    loaded = load_reference_processor()

    assert loaded is expected
    assert calls == [
        (
            QWEN3_OMNI_MODEL_ID,
            {
                "revision": QWEN3_OMNI_REVISION,
                "local_files_only": True,
            },
        )
    ]


def test_load_reference_model_requires_explicit_hardware_and_validates_first(
    monkeypatch,
):
    with pytest.raises(TypeError):
        load_reference_model()

    config = _pinned_config()
    events = []
    expected = object()

    class FakeModelLoader:
        @classmethod
        def from_pretrained(cls, source, **kwargs):
            events.append(("model", source, kwargs))
            return expected

    def fake_config_loader(source, *, local_files_only):
        events.append(("config", source, local_files_only))
        return config

    monkeypatch.setattr(oracle, "_require_reference_environment", lambda: None)
    monkeypatch.setattr(oracle, "load_reference_config", fake_config_loader)
    monkeypatch.setattr(
        oracle,
        "Qwen3OmniMoeForConditionalGeneration",
        FakeModelLoader,
    )

    loaded = load_reference_model(
        torch_dtype="bfloat16",
        device_map={"": "cpu"},
    )

    assert loaded is expected
    assert events == [
        ("config", QWEN3_OMNI_MODEL_ID, True),
        (
            "model",
            QWEN3_OMNI_MODEL_ID,
            {
                "revision": QWEN3_OMNI_REVISION,
                "local_files_only": True,
                "torch_dtype": "bfloat16",
                "device_map": {"": "cpu"},
                "config": config,
            },
        ),
    ]


def test_model_loader_rejects_config_contradiction_before_construction(
    monkeypatch,
):
    incompatible = SimpleNamespace(model_type="qwen3_omni_prototype")
    model_called = False

    class ForbiddenModelLoader:
        @classmethod
        def from_pretrained(cls, source, **kwargs):
            nonlocal model_called
            model_called = True
            raise AssertionError("model construction must not be reached")

    monkeypatch.setattr(oracle, "_require_reference_environment", lambda: None)
    monkeypatch.setattr(
        oracle,
        "load_reference_config",
        lambda source, *, local_files_only: incompatible,
    )
    monkeypatch.setattr(
        oracle,
        "Qwen3OmniMoeForConditionalGeneration",
        ForbiddenModelLoader,
    )

    with pytest.raises(ValueError, match="model_type"):
        load_reference_model(
            torch_dtype="bfloat16",
            device_map={"": "cpu"},
        )

    assert not model_called


def test_environment_mismatch_fails_before_external_loading(monkeypatch):
    external_called = False

    class ForbiddenConfigLoader:
        @classmethod
        def from_pretrained(cls, source, **kwargs):
            nonlocal external_called
            external_called = True
            raise AssertionError("external loading must not be reached")

    monkeypatch.setattr(oracle.transformers, "__version__", "4.57.6")
    monkeypatch.setattr(oracle, "Qwen3OmniMoeConfig", ForbiddenConfigLoader)

    with pytest.raises(RuntimeError, match="transformers==5.2.0"):
        load_reference_config()

    assert not external_called


def test_reference_environment_accepts_pinned_cuda_build_suffixes(monkeypatch):
    installed = {
        "torch": "2.10.0+cu128",
        "torchvision": "0.25.0+cu128",
        "torchaudio": "2.10.0+cu128",
        "transformers": "5.2.0",
        "qwen-omni-utils": "0.0.9",
    }
    monkeypatch.setattr(
        oracle,
        "version",
        lambda distribution: installed[distribution],
    )
    monkeypatch.setattr(oracle.transformers, "__version__", "5.2.0")
    monkeypatch.setattr(oracle.shutil, "which", lambda command: "/usr/bin/ffmpeg")

    oracle._require_reference_environment()


def test_reference_environment_rejects_non_torch_local_build_suffixes(
    monkeypatch,
):
    installed = {
        "torch": "2.10.0+cu128",
        "torchvision": "0.25.0+cu128",
        "torchaudio": "2.10.0+cu128",
        "transformers": "5.2.0",
        "qwen-omni-utils": "0.0.9+local",
    }
    monkeypatch.setattr(
        oracle,
        "version",
        lambda distribution: installed[distribution],
    )
    monkeypatch.setattr(oracle.transformers, "__version__", "5.2.0")
    monkeypatch.setattr(oracle.shutil, "which", lambda command: "/usr/bin/ffmpeg")

    with pytest.raises(RuntimeError, match="qwen-omni-utils"):
        oracle._require_reference_environment()
