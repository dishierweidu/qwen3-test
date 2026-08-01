from __future__ import annotations

from collections.abc import Mapping
import hashlib
import importlib
import json
from pathlib import Path

import pytest
import torch
import yaml

from qwen3_omni_pretrain.architecture.profiles import (
    ArchitectureProfile,
    CompatibilityLevel,
)
from qwen3_omni_pretrain.profiles.qwen3_omni_reference.pins import (
    QWEN3_OMNI_MODEL_ID,
    QWEN3_OMNI_REVISION,
)
from qwen3_omni_pretrain.profiles.registry import (
    ProfileBuildRequest,
    get_profile_factory,
    parse_profile,
    register_profile_factory,
)


def _write_tiny_legacy_config(path: Path) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "model_type": "qwen3_omni_prototype",
                "architecture_profile": "legacy_prototype",
                "vocab_size": 32,
                "thinker_config": {
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "max_position_embeddings": 16,
                    "use_moe": False,
                    "routing_kind": "dense",
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _write_reference_snapshot(path: Path) -> dict[str, str]:
    artifact_payloads = {
        "README.md": b"samplerate=24000",
        "chat_template.json": b'{"chat_template":"pinned"}',
        "config.json": b'{"model_type":"qwen3_omni_moe"}',
        "merges.txt": b"#version: 0.2\\na b\\n",
        "preprocessor_config.json": b'{"sampling_rate":16000}',
        "tokenizer_config.json": b'{"model_max_length":32768}',
        "vocab.json": b'{"a":0}',
    }
    for filename, payload in artifact_payloads.items():
        (path / filename).write_bytes(payload)
    return {
        filename: hashlib.sha256(payload).hexdigest()
        for filename, payload in artifact_payloads.items()
    }


def _to_json_value(value):
    if isinstance(value, Mapping):
        return {
            key: _to_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return [_to_json_value(item) for item in value]
    return value


def test_registry_does_not_import_reference_backend_for_legacy(
    monkeypatch,
):
    imported: list[str] = []
    real_import = importlib.import_module

    def capture(name, package=None):
        imported.append(name)
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", capture)

    factory = get_profile_factory(
        ArchitectureProfile.LEGACY_PROTOTYPE
    )

    assert factory.profile is ArchitectureProfile.LEGACY_PROTOTYPE
    assert type(factory).__name__ == "LegacyPrototypeFactory"
    assert not any(
        "qwen3_omni_reference" in name for name in imported
    )


def test_legacy_factory_validates_and_builds_a_real_local_model(tmp_path):
    config_path = _write_tiny_legacy_config(tmp_path / "legacy.yaml")
    request = ProfileBuildRequest(
        profile=ArchitectureProfile.LEGACY_PROTOTYPE,
        config_or_checkpoint=str(config_path),
        device="cpu",
        requested_capabilities=(
            "incremental_decode_state",
            "streaming_generation",
        ),
    )
    factory = get_profile_factory(request.profile)

    validated_manifest = factory.validate(request)
    manifest = factory.manifest(request)
    result = factory.build(request)

    assert validated_manifest == manifest
    assert manifest.architecture_profile is (
        ArchitectureProfile.LEGACY_PROTOTYPE
    )
    assert manifest.compatibility_level is (
        CompatibilityLevel.LEGACY_PROTOTYPE
    )
    assert not manifest.exact_official_checkpoint_compatible
    assert isinstance(result.artifact, torch.nn.Module)
    assert result.artifact.config.to_dict()["model_type"] == (
        "qwen3_omni_prototype"
    )
    assert result.artifact.config.thinker_config.hidden_size == 8
    assert result.manifest == manifest
    assert result.architecture_summary.profile == "legacy_prototype"
    assert result.architecture_summary.model_type == (
        "qwen3_omni_prototype"
    )
    assert result.architecture_summary.total_parameters > 0
    assert len(result.architecture_summary.layers) == 1
    assert result.architecture_summary.unsupported_capabilities == (
        "streaming_generation",
    )
    assert result.architecture_summary.capabilities[
        "incremental_decode_state"
    ] is True


@pytest.mark.parametrize(
    ("name", "payload", "message"),
    [
        ("empty.json", {}, "model configuration is empty"),
        ("nonmapping.json", [], "model configuration must be a mapping"),
    ],
)
def test_legacy_factory_rejects_malformed_json_config(
    tmp_path,
    name,
    payload,
    message,
):
    config_path = tmp_path / name
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    request = ProfileBuildRequest(
        profile=ArchitectureProfile.LEGACY_PROTOTYPE,
        config_or_checkpoint=str(config_path),
    )
    factory = get_profile_factory(request.profile)

    with pytest.raises((TypeError, ValueError), match=message):
        factory.validate(request)


def test_legacy_factory_builds_old_checkpoint_directory_with_one_warning(
    tmp_path,
):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_omni_moe",
                "vocab_size": 32,
                "thinker_config": {
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "max_position_embeddings": 16,
                    "use_moe": False,
                },
            }
        ),
        encoding="utf-8",
    )
    request = ProfileBuildRequest(
        profile=ArchitectureProfile.LEGACY_PROTOTYPE,
        config_or_checkpoint=str(checkpoint),
        device="meta",
    )
    factory = get_profile_factory(request.profile)

    with pytest.warns(DeprecationWarning) as warnings:
        result = factory.build(request)

    assert len(warnings) == 1
    assert result.architecture_summary.model_type == (
        "qwen3_omni_prototype"
    )


def test_reference_factory_builds_offline_oracle_artifact_without_weights(
    tmp_path,
    monkeypatch,
):
    factory = get_profile_factory(
        ArchitectureProfile.QWEN3_OMNI_REFERENCE
    )
    factory_module = importlib.import_module(type(factory).__module__)
    artifact_sha256 = _write_reference_snapshot(tmp_path)
    loader_calls: list[tuple[str, str, bool]] = []
    download_calls: list[tuple[str, str, str, bool]] = []

    def download(*, repo_id, filename, revision, local_files_only):
        download_calls.append(
            (repo_id, filename, revision, local_files_only)
        )
        return str(tmp_path / filename)

    def load_config(source, *, local_files_only):
        loader_calls.append(("config", source, local_files_only))
        return "config"

    def load_processor(source, *, local_files_only):
        loader_calls.append(("processor", source, local_files_only))
        return "processor"

    monkeypatch.setattr(
        factory_module,
        "load_reference_config",
        load_config,
    )
    monkeypatch.setattr(
        factory_module,
        "load_reference_processor",
        load_processor,
    )
    monkeypatch.setattr(
        factory_module,
        "_QWEN3_LOCAL_ARTIFACT_SHA256",
        artifact_sha256,
    )
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    request = ProfileBuildRequest(
        profile=ArchitectureProfile.QWEN3_OMNI_REFERENCE,
        config_or_checkpoint=QWEN3_OMNI_MODEL_ID,
        local_files_only=True,
        requested_capabilities=("streaming_generation",),
    )

    factory.validate(request)
    result = factory.build(request)

    assert loader_calls == []
    assert download_calls == []
    assert type(result.artifact).__name__ == "Qwen3OracleArtifact"
    expected_contract = json.loads(
        Path(
            "tests/fixtures/qwen3_omni/config_contract.json"
        ).read_text(encoding="utf-8")
    )
    assert _to_json_value(result.artifact.config_contract) == (
        expected_contract
    )
    assert result.artifact.config_contract["audio_encoder"][
        "input_sample_rate_hz"
    ] == 16_000
    assert result.artifact.config_contract["code2wav"][
        "output_sample_rate_hz"
    ] == 24_000
    assert result.artifact.config_contract["vocabulary"][
        "regular_vocab_size"
    ] == 151_643
    assert "README.md" in result.artifact.config_contract["source"][
        "artifacts"
    ]
    assert result.artifact.load_config() == "config"
    assert result.artifact.load_processor() == "processor"
    assert loader_calls == [
        ("config", str(tmp_path), True),
        ("processor", str(tmp_path), True),
    ]
    expected_downloads = [
        (
            QWEN3_OMNI_MODEL_ID,
            filename,
            QWEN3_OMNI_REVISION,
            True,
        )
        for filename in artifact_sha256
    ]
    assert download_calls == expected_downloads * 2
    assert result.manifest.architecture_profile is (
        ArchitectureProfile.QWEN3_OMNI_REFERENCE
    )
    assert result.manifest.compatibility_level is (
        CompatibilityLevel.STRUCTURE_ALIGNED
    )
    assert not result.manifest.exact_official_checkpoint_compatible
    assert result.architecture_summary.profile == (
        "qwen3_omni_reference"
    )
    assert result.architecture_summary.model_type == "qwen3_omni_moe"
    assert result.architecture_summary.tokenizer_vocab_size == 0
    assert result.architecture_summary.embedding_vocab_size == 152_064
    assert result.architecture_summary.total_parameters == 0
    assert result.architecture_summary.layers == ()
    assert (
        result.architecture_summary.capabilities["config_oracle"]
        is True
    )
    assert (
        result.architecture_summary.capabilities["inference_runtime"]
        is False
    )
    assert (
        result.architecture_summary.capabilities[
            "tokenizer_vocab_size"
        ]
        is False
    )
    assert "streaming_generation" in (
        result.architecture_summary.unsupported_capabilities
    )
    assert not hasattr(result.artifact.load_config, "keywords")
    with pytest.raises(TypeError):
        result.artifact.config_contract["source"]["revision"] = "changed"
    with pytest.raises(TypeError):
        result.artifact.config_contract["audio_encoder"][
            "projector_dimensions"
        ][0] = 0


def test_reference_factory_rejects_unpinned_remote_source():
    factory = get_profile_factory(
        ArchitectureProfile.QWEN3_OMNI_REFERENCE
    )
    request = ProfileBuildRequest(
        profile=ArchitectureProfile.QWEN3_OMNI_REFERENCE,
        config_or_checkpoint="someone/definitely-not-the-pinned-model",
    )

    with pytest.raises(ValueError, match="pinned model ID"):
        factory.validate(request)


def test_reference_factory_rejects_incomplete_actual_snapshot(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    factory = get_profile_factory(
        ArchitectureProfile.QWEN3_OMNI_REFERENCE
    )
    request = ProfileBuildRequest(
        profile=ArchitectureProfile.QWEN3_OMNI_REFERENCE,
        config_or_checkpoint=str(tmp_path),
    )

    with pytest.raises(ValueError, match="missing pinned artifacts"):
        factory.validate(request)


def test_reference_factory_accepts_matching_local_snapshot(
    tmp_path,
    monkeypatch,
):
    artifact_sha256 = _write_reference_snapshot(tmp_path)
    factory = get_profile_factory(
        ArchitectureProfile.QWEN3_OMNI_REFERENCE
    )
    factory_module = importlib.import_module(type(factory).__module__)
    monkeypatch.setattr(
        factory_module,
        "_QWEN3_LOCAL_ARTIFACT_SHA256",
        artifact_sha256,
    )
    processor_sources = []

    def load_processor(source, *, local_files_only):
        processor_sources.append((source, local_files_only))
        return "processor"

    monkeypatch.setattr(
        factory_module,
        "load_reference_processor",
        load_processor,
    )
    request = ProfileBuildRequest(
        profile=ArchitectureProfile.QWEN3_OMNI_REFERENCE,
        config_or_checkpoint=str(tmp_path),
    )

    factory.validate(request)
    result = factory.build(request)

    assert result.artifact.load_processor() == "processor"
    assert processor_sources == [(str(tmp_path), True)]


@pytest.mark.parametrize(
    ("filename", "loader_name"),
    [
        ("config.json", "load_config"),
        ("vocab.json", "load_processor"),
    ],
)
def test_reference_lazy_loaders_revalidate_local_snapshot(
    tmp_path,
    monkeypatch,
    filename,
    loader_name,
):
    artifact_sha256 = _write_reference_snapshot(tmp_path)
    factory = get_profile_factory(
        ArchitectureProfile.QWEN3_OMNI_REFERENCE
    )
    factory_module = importlib.import_module(type(factory).__module__)
    monkeypatch.setattr(
        factory_module,
        "_QWEN3_LOCAL_ARTIFACT_SHA256",
        artifact_sha256,
    )
    delegated = []

    def delegate(source, *, local_files_only):
        delegated.append((source, local_files_only))
        return "loaded"

    monkeypatch.setattr(factory_module, "load_reference_config", delegate)
    monkeypatch.setattr(
        factory_module,
        "load_reference_processor",
        delegate,
    )
    request = ProfileBuildRequest(
        profile=ArchitectureProfile.QWEN3_OMNI_REFERENCE,
        config_or_checkpoint=str(tmp_path),
    )
    result = factory.build(request)
    (tmp_path / filename).write_bytes(b"modified after build")

    with pytest.raises(ValueError, match="do not match pinned"):
        getattr(result.artifact, loader_name)()
    assert delegated == []


@pytest.mark.parametrize(
    ("filename", "loader_name"),
    [
        ("config.json", "load_config"),
        ("README.md", "load_processor"),
    ],
)
def test_reference_lazy_loaders_revalidate_resolved_cached_metadata(
    tmp_path,
    monkeypatch,
    filename,
    loader_name,
):
    artifact_sha256 = _write_reference_snapshot(tmp_path)
    factory = get_profile_factory(
        ArchitectureProfile.QWEN3_OMNI_REFERENCE
    )
    factory_module = importlib.import_module(type(factory).__module__)
    monkeypatch.setattr(
        factory_module,
        "_QWEN3_LOCAL_ARTIFACT_SHA256",
        artifact_sha256,
    )
    download_calls = []
    delegated = []

    def download(*, repo_id, filename, revision, local_files_only):
        download_calls.append(filename)
        return str(tmp_path / filename)

    def delegate(source, *, local_files_only):
        delegated.append((source, local_files_only))
        return "loaded"

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(factory_module, "load_reference_config", delegate)
    monkeypatch.setattr(
        factory_module,
        "load_reference_processor",
        delegate,
    )
    request = ProfileBuildRequest(
        profile=ArchitectureProfile.QWEN3_OMNI_REFERENCE,
        config_or_checkpoint=QWEN3_OMNI_MODEL_ID,
    )
    result = factory.build(request)
    assert download_calls == []
    (tmp_path / filename).write_bytes(b"modified after build")

    with pytest.raises(ValueError, match="do not match pinned"):
        getattr(result.artifact, loader_name)()
    assert filename in download_calls
    assert delegated == []


@pytest.mark.parametrize(
    ("loader_name", "delegate_name"),
    [
        ("load_config", "config"),
        ("load_processor", "processor"),
    ],
)
@pytest.mark.parametrize("local_files_only", [True, False])
def test_reference_lazy_loaders_resolve_all_pinned_metadata_at_revision(
    tmp_path,
    monkeypatch,
    loader_name,
    delegate_name,
    local_files_only,
):
    artifact_sha256 = _write_reference_snapshot(tmp_path)
    factory = get_profile_factory(
        ArchitectureProfile.QWEN3_OMNI_REFERENCE
    )
    factory_module = importlib.import_module(type(factory).__module__)
    monkeypatch.setattr(
        factory_module,
        "_QWEN3_LOCAL_ARTIFACT_SHA256",
        artifact_sha256,
    )
    download_calls = []
    delegated = []

    def download(*, repo_id, filename, revision, local_files_only):
        download_calls.append(
            (repo_id, filename, revision, local_files_only)
        )
        return str(tmp_path / filename)

    def config_delegate(source, *, local_files_only):
        delegated.append(("config", source, local_files_only))
        return "config"

    def processor_delegate(source, *, local_files_only):
        delegated.append(("processor", source, local_files_only))
        return "processor"

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(
        factory_module,
        "load_reference_config",
        config_delegate,
    )
    monkeypatch.setattr(
        factory_module,
        "load_reference_processor",
        processor_delegate,
    )
    request = ProfileBuildRequest(
        profile=ArchitectureProfile.QWEN3_OMNI_REFERENCE,
        config_or_checkpoint=QWEN3_OMNI_MODEL_ID,
        local_files_only=local_files_only,
    )
    result = factory.build(request)

    assert getattr(result.artifact, loader_name)() == delegate_name
    assert download_calls == [
        (
            QWEN3_OMNI_MODEL_ID,
            filename,
            QWEN3_OMNI_REVISION,
            local_files_only,
        )
        for filename in artifact_sha256
    ]
    assert delegated == [
        (delegate_name, str(tmp_path), True)
    ]
    assert all(
        not filename.endswith((".bin", ".safetensors"))
        for _, filename, _, _ in download_calls
    )


def test_reference_factory_rejects_incomplete_local_snapshot(
    tmp_path,
    monkeypatch,
):
    config_bytes = b'{"model_type":"qwen3_omni_moe"}'
    (tmp_path / "config.json").write_bytes(config_bytes)
    factory = get_profile_factory(
        ArchitectureProfile.QWEN3_OMNI_REFERENCE
    )
    factory_module = importlib.import_module(type(factory).__module__)
    monkeypatch.setattr(
        factory_module,
        "_QWEN3_LOCAL_ARTIFACT_SHA256",
        {
            "config.json": hashlib.sha256(config_bytes).hexdigest(),
            "preprocessor_config.json": "0" * 64,
        },
    )
    request = ProfileBuildRequest(
        profile=ArchitectureProfile.QWEN3_OMNI_REFERENCE,
        config_or_checkpoint=str(tmp_path),
    )

    with pytest.raises(ValueError, match="missing pinned artifacts"):
        factory.validate(request)


def test_reference_factory_rejects_modified_local_processor_artifact(
    tmp_path,
    monkeypatch,
):
    artifact_payloads = {
        "config.json": b"pinned config",
        "preprocessor_config.json": b"modified processor",
    }
    for filename, payload in artifact_payloads.items():
        (tmp_path / filename).write_bytes(payload)
    factory = get_profile_factory(
        ArchitectureProfile.QWEN3_OMNI_REFERENCE
    )
    factory_module = importlib.import_module(type(factory).__module__)
    monkeypatch.setattr(
        factory_module,
        "_QWEN3_LOCAL_ARTIFACT_SHA256",
        {
            "config.json": hashlib.sha256(b"pinned config").hexdigest(),
            "preprocessor_config.json": hashlib.sha256(
                b"pinned processor"
            ).hexdigest(),
        },
    )
    request = ProfileBuildRequest(
        profile=ArchitectureProfile.QWEN3_OMNI_REFERENCE,
        config_or_checkpoint=str(tmp_path),
    )

    with pytest.raises(ValueError, match="do not match pinned"):
        factory.validate(request)


def test_reference_factory_rejects_custom_processor_source():
    factory = get_profile_factory(
        ArchitectureProfile.QWEN3_OMNI_REFERENCE
    )
    request = ProfileBuildRequest(
        profile=ArchitectureProfile.QWEN3_OMNI_REFERENCE,
        config_or_checkpoint=QWEN3_OMNI_MODEL_ID,
        tokenizer="someone/other-tokenizer",
    )

    with pytest.raises(ValueError, match="tokenizer overrides"):
        factory.validate(request)


@pytest.mark.parametrize(
    ("profile", "factory_name"),
    [
        (
            ArchitectureProfile.QWEN35_OMNI_INSPIRED,
            "Qwen35InspiredFactory",
        ),
        (
            ArchitectureProfile.MIMO_V25_EXPERIMENTAL,
            "MimoV25ExperimentalFactory",
        ),
    ],
)
def test_completed_profiles_are_registered_lazily(profile, factory_name):
    assert type(get_profile_factory(profile)).__name__ == factory_name


def test_unknown_profile_fails_before_model_construction():
    with pytest.raises(ValueError, match="unknown architecture profile"):
        parse_profile("qwen3.5-omni-plus")


def test_duplicate_factory_registration_is_rejected():
    with pytest.raises(ValueError, match="already registered"):
        register_profile_factory(
            ArchitectureProfile.LEGACY_PROTOTYPE,
            "example.factories",
            "ReplacementFactory",
        )


@pytest.mark.parametrize(
    ("updates", "exception", "message"),
    [
        (
            {"profile": "legacy_prototype"},
            TypeError,
            "profile must be ArchitectureProfile",
        ),
        (
            {"config_or_checkpoint": ""},
            ValueError,
            "config_or_checkpoint must be non-empty",
        ),
        (
            {"requested_capabilities": ["streaming"]},
            TypeError,
            "requested_capabilities must be a tuple",
        ),
    ],
)
def test_build_request_rejects_ambiguous_or_mutable_contracts(
    updates,
    exception,
    message,
):
    values = {
        "profile": ArchitectureProfile.LEGACY_PROTOTYPE,
        "config_or_checkpoint": "config.yaml",
    }
    values.update(updates)

    with pytest.raises(exception, match=message):
        ProfileBuildRequest(**values)
