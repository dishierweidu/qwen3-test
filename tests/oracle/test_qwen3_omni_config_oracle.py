from __future__ import annotations

from collections.abc import Mapping
import copy
import importlib
import json
from pathlib import Path
import re
from types import SimpleNamespace

import pytest
import torch

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


class _PinnedTokenizerVocab(Mapping[str, int]):
    def __len__(self):
        return 151_643

    def __iter__(self):
        return (str(index) for index in range(len(self)))

    def __getitem__(self, key):
        index = int(key)
        if index < 0 or index >= len(self):
            raise KeyError(key)
        return index

    def values(self):
        return range(len(self))


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
                    "rope_theta": 1_000_000,
                },
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
    source = contract["source"]
    assert set(source) == {
        "model_id",
        "revision",
        "transformers_version",
        "artifacts",
        "derived_fields",
    }
    assert {
        key: source[key]
        for key in ("model_id", "revision", "transformers_version")
    } == {
        "model_id": QWEN3_OMNI_MODEL_ID,
        "revision": QWEN3_OMNI_REVISION,
        "transformers_version": "5.2.0",
    }
    assert source["artifacts"] == {
        "README.md": {
            "revision": QWEN3_OMNI_REVISION,
            "sha256": (
                "0e44065c4c4a27071f7239afd5b5a33af5bc2e437dd7ea9950e51aafabfde3df"
            ),
        },
        "config.json": {
            "revision": QWEN3_OMNI_REVISION,
            "sha256": (
                "eab5093d47807aaf894119506b238b2b1cee70d08456e894fee9a012d88f2e0d"
            ),
        },
        "preprocessor_config.json": {
            "revision": QWEN3_OMNI_REVISION,
            "sha256": (
                "b10e27fd4542cf89ec7145942b87f3e65408d4e9f9d031a29acdd293c15fb3fc"
            ),
        },
        "vocab.json": {
            "revision": QWEN3_OMNI_REVISION,
            "sha256": (
                "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"
            ),
        },
    }
    assert source["derived_fields"]["audio_conv_kernel_and_stride"] == {
        "artifact": (
            "src/transformers/models/qwen3_omni_moe/"
            "modeling_qwen3_omni_moe.py"
        ),
        "extraction": (
            "AST of Qwen3OmniMoeAudioEncoder.__init__ "
            "conv2d1/2/3 assignments"
        ),
        "revision": "7d9754a05193eb79b1d86aa744b622b8068008cd",
        "sha256": (
            "0b6e9a6e9d88814de3e25b1ca65c49d0677be7331b8118b6a0cd022c6c1dd270"
        ),
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


def test_production_contract_exactly_matches_checked_fixture():
    from qwen3_omni_pretrain.profiles.qwen3_omni_reference.contract import (
        config_contract_to_dict,
    )

    expected = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

    assert config_contract_to_dict() == expected


def test_production_metadata_hashes_are_config_and_processor_union():
    from qwen3_omni_pretrain.profiles.qwen3_omni_reference.contract import (
        QWEN3_OMNI_METADATA_SHA256,
    )

    config_contract = json.loads(
        CONTRACT_PATH.read_text(encoding="utf-8")
    )
    processor_contract = json.loads(
        Path(
            "tests/fixtures/qwen3_omni/processor_contract.json"
        ).read_text(encoding="utf-8")
    )
    expected = dict(
        processor_contract["source"]["artifact_sha256"]
    )
    expected.update(
        {
            filename: provenance["sha256"]
            for filename, provenance in config_contract["source"][
                "artifacts"
            ].items()
        }
    )

    assert dict(QWEN3_OMNI_METADATA_SHA256) == expected
    assert set(expected) == {
        "README.md",
        "chat_template.json",
        "config.json",
        "merges.txt",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "vocab.json",
    }


def test_extractor_parity_guard_rejects_contract_drift():
    from qwen3_omni_pretrain.profiles.qwen3_omni_reference.contract import (
        config_contract_to_dict,
    )
    from tests.oracle.extract_qwen3_omni_config_contract import (
        validate_contract_parity,
    )

    extracted = config_contract_to_dict()
    validate_contract_parity(extracted)
    extracted["vocabulary"]["regular_vocab_size"] = 1

    with pytest.raises(ValueError, match="production pinned contract"):
        validate_contract_parity(extracted)


def test_extractor_reproduces_production_and_checked_contract():
    from qwen3_omni_pretrain.profiles.qwen3_omni_reference.contract import (
        config_contract_to_dict,
    )
    from tests.oracle.extract_qwen3_omni_config_contract import (
        validate_contract_parity,
    )

    extracted = extract_contract(
        _pinned_config(),
        preprocessor_config={"sampling_rate": 16_000},
        tokenizer_vocab=_PinnedTokenizerVocab(),
        model_readme="samplerate=24_000",
        audio_implementation_contract={
            "audio_encoder.conv_layers": 3,
            "audio_encoder.conv_kernel_size": 3,
            "audio_encoder.conv_stride": 2,
        },
    )
    expected = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

    validate_contract_parity(extracted)
    assert extracted == expected == config_contract_to_dict()


def test_contract_extractor_selects_only_the_pinned_architecture_fields():
    extracted = extract_contract(
        _pinned_config(),
        preprocessor_config={"sampling_rate": 16_000},
        tokenizer_vocab={"a": 0, "b": 1, "c": 2},
        model_readme="sf.write('audio.wav', audio, samplerate=24_000)",
        audio_implementation_contract={
            "audio_encoder.conv_layers": 3,
            "audio_encoder.conv_kernel_size": 3,
            "audio_encoder.conv_stride": 2,
        },
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
    monkeypatch.setattr(
        oracle,
        "_reference_config_class",
        lambda: FakeConfigLoader,
    )

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
    monkeypatch.setattr(
        oracle,
        "_reference_processor_class",
        lambda: FakeProcessorLoader,
    )

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


def test_local_processor_loader_scopes_and_restores_hub_offline_mode(
    monkeypatch,
):
    import huggingface_hub.constants as hub_constants

    observed_offline_modes = []
    expected = object()

    class FakeProcessorLoader:
        @classmethod
        def from_pretrained(cls, source, **kwargs):
            observed_offline_modes.append(hub_constants.HF_HUB_OFFLINE)
            return expected

    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setattr(oracle, "_require_reference_environment", lambda: None)
    monkeypatch.setattr(
        oracle,
        "_reference_processor_class",
        lambda: FakeProcessorLoader,
    )

    loaded = load_reference_processor(local_files_only=True)

    assert loaded is expected
    assert observed_offline_modes == [True]
    assert hub_constants.HF_HUB_OFFLINE is False


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
        "_reference_model_class",
        lambda: FakeModelLoader,
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
        "_reference_model_class",
        lambda: ForbiddenModelLoader,
    )

    with pytest.raises(ValueError, match="model_type"):
        load_reference_model(
            torch_dtype="bfloat16",
            device_map={"": "cpu"},
        )

    assert not model_called


def _set_config_path(config, path, value):
    target = config
    components = path.split(".")
    for component in components[:-1]:
        target = (
            target[component]
            if isinstance(target, dict)
            else getattr(target, component)
        )
    if isinstance(target, dict):
        target[components[-1]] = value
    else:
        setattr(target, components[-1], value)


@pytest.mark.parametrize(
    ("config_path", "contract_path", "contradiction"),
    [
        (
            "thinker_config.text_config.rope_scaling.rope_theta",
            "tm_rope.rope_theta",
            10_000,
        ),
        (
            "thinker_config.audio_config.num_mel_bins",
            "audio_encoder.num_mel_bins",
            64,
        ),
        (
            "thinker_config.vision_config.patch_size",
            "vision_encoder.patch_kernel",
            14,
        ),
        (
            "talker_config.text_config.shared_expert_intermediate_size",
            "talker.shared_expert_intermediate_size",
            1_024,
        ),
        (
            "talker_config.code_predictor_config.num_code_groups",
            "code_predictor.num_code_groups",
            8,
        ),
        (
            "code2wav_config.upsample_rates",
            "code2wav.upsample_rates",
            [8, 5, 4, 2],
        ),
    ],
    ids=["tm-rope", "audio", "vision", "talker", "predictor", "code2wav"],
)
def test_model_loader_rejects_each_major_contract_contradiction(
    monkeypatch,
    config_path,
    contract_path,
    contradiction,
):
    incompatible = copy.deepcopy(_pinned_config())
    _set_config_path(incompatible, config_path, contradiction)
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
        "_reference_model_class",
        lambda: ForbiddenModelLoader,
    )

    with pytest.raises(ValueError, match=re.escape(contract_path)):
        load_reference_model(
            torch_dtype="bfloat16",
            device_map={"": "cpu"},
        )

    assert not model_called


def test_model_loader_rejects_audio_convolution_contract_before_construction(
    monkeypatch,
):
    class ContradictoryAudioEncoder:
        def __init__(self, config):
            self.conv2d1 = torch.nn.Conv2d(1, 480, 5, 2, padding=1)
            self.conv2d2 = torch.nn.Conv2d(480, 480, 5, 2, padding=1)
            self.conv2d3 = torch.nn.Conv2d(480, 480, 5, 2, padding=1)

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
        lambda source, *, local_files_only: _pinned_config(),
    )
    monkeypatch.setattr(
        oracle,
        "_reference_audio_encoder_class",
        lambda: ContradictoryAudioEncoder,
    )
    monkeypatch.setattr(
        oracle,
        "_reference_model_class",
        lambda: ForbiddenModelLoader,
    )

    with pytest.raises(ValueError, match="audio_encoder.conv_kernel_size"):
        load_reference_model(
            torch_dtype="bfloat16",
            device_map={"": "cpu"},
        )

    assert not model_called


def test_environment_mismatch_fails_before_external_loading(monkeypatch):
    external_called = False

    def forbidden_config_class():
        nonlocal external_called
        external_called = True
        raise AssertionError("external loading must not be reached")

    monkeypatch.setattr(oracle.transformers, "__version__", "4.57.6")
    monkeypatch.setattr(
        oracle,
        "_reference_config_class",
        forbidden_config_class,
    )

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


def test_oracle_import_does_not_bind_reference_only_transformers_classes():
    eager_reference_types = {
        "Qwen3OmniMoeConfig",
        "Qwen3OmniMoeForConditionalGeneration",
        "Qwen3OmniMoeProcessor",
    }

    assert eager_reference_types.isdisjoint(vars(oracle))


def test_collection_and_runtime_consume_the_same_lightweight_pins():
    pins = importlib.import_module(
        "qwen3_omni_pretrain.profiles.qwen3_omni_reference.pins"
    )
    conftest = importlib.import_module("conftest")

    assert oracle.REFERENCE_DISTRIBUTION_VERSIONS is (
        pins.REFERENCE_DISTRIBUTION_VERSIONS
    )
    assert conftest.REFERENCE_DISTRIBUTION_VERSIONS is (
        pins.REFERENCE_DISTRIBUTION_VERSIONS
    )
    assert "transformers" not in vars(pins)


def test_contract_provenance_is_immutable_and_machine_reproducible():
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    provenance = contract["source"]["derived_fields"]

    regular_vocab = provenance["regular_vocab_size"]
    assert regular_vocab["artifact"] == "vocab.json"
    assert regular_vocab["revision"] == QWEN3_OMNI_REVISION
    assert regular_vocab["extraction"] == "len(top-level token-to-id mapping)"
    assert regular_vocab["sha256"] == (
        "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"
    )

    output_rate = provenance["output_sample_rate_hz"]
    assert output_rate["artifact"] == "README.md"
    assert output_rate["revision"] == QWEN3_OMNI_REVISION
    assert output_rate["locator"] == (
        "lines 257-277; samplerate=24000 at line 277"
    )
    assert output_rate["extraction"] == (
        "samplerate used to serialize audio returned by model.generate"
    )
    assert output_rate["sha256"] == (
        "0e44065c4c4a27071f7239afd5b5a33af5bc2e437dd7ea9950e51aafabfde3df"
    )


def test_contract_extractor_derives_vocab_and_sample_rate_from_sources():
    config = _pinned_config()

    extracted = extract_contract(
        config,
        preprocessor_config={"sampling_rate": 16_000},
        tokenizer_vocab={"a": 0, "b": 1, "c": 2},
        model_readme="sf.write('audio.wav', audio, samplerate=12_345)",
        audio_implementation_contract={
            "audio_encoder.conv_layers": 3,
            "audio_encoder.conv_kernel_size": 3,
            "audio_encoder.conv_stride": 2,
        },
    )

    assert extracted["vocabulary"]["regular_vocab_size"] == 3
    assert extracted["code2wav"]["output_sample_rate_hz"] == 12_345
