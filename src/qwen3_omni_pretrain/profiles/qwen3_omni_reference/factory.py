from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping

from qwen3_omni_pretrain.architecture.manifest import ProfileManifest
from qwen3_omni_pretrain.architecture.profiles import (
    ArchitectureProfile,
)
from qwen3_omni_pretrain.architecture.summary import ArchitectureSummary
from qwen3_omni_pretrain.profiles.qwen3_omni_reference import (
    load_reference_config,
    load_reference_processor,
    qwen3_reference_manifest,
)
from qwen3_omni_pretrain.profiles.qwen3_omni_reference.pins import (
    QWEN3_OMNI_MODEL_ID,
    QWEN3_OMNI_REVISION,
    QWEN3_TRANSFORMERS_VERSION,
)
from qwen3_omni_pretrain.profiles.registry import (
    ProfileBuildRequest,
    ProfileBuildResult,
)

_QWEN3_LOCAL_ARTIFACT_SHA256: Mapping[str, str] = MappingProxyType(
    {
        "chat_template.json": (
            "90c1b81f29e41b7642b0cc02c877a10c8bf6751a8d8fa1d16ac9a718cf1c3d86"
        ),
        "config.json": (
            "eab5093d47807aaf894119506b238b2b1cee70d08456e894fee9a012d88f2e0d"
        ),
        "merges.txt": (
            "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3"
        ),
        "preprocessor_config.json": (
            "b10e27fd4542cf89ec7145942b87f3e65408d4e9f9d031a29acdd293c15fb3fc"
        ),
        "tokenizer_config.json": (
            "dc3c31c3bdaedd5016382bb3cbe07323026775ad51f5a4fb564505992ae4a670"
        ),
        "vocab.json": (
            "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"
        ),
    }
)


@dataclass(frozen=True)
class Qwen3OracleArtifact:
    config_contract: Mapping[str, object]
    load_config: Callable[[], object]
    load_processor: Callable[[], object]

    def __post_init__(self) -> None:
        if not isinstance(self.config_contract, Mapping):
            raise TypeError("config_contract must be a mapping")
        if not callable(self.load_config) or not callable(
            self.load_processor
        ):
            raise TypeError("oracle loaders must be callable")
        object.__setattr__(
            self,
            "config_contract",
            MappingProxyType(dict(self.config_contract)),
        )


def _pinned_config_contract() -> Mapping[str, object]:
    # The oracle owns these Task 6 literals. Reading its immutable in-process
    # contract avoids either duplicating pins or depending on test fixtures.
    from qwen3_omni_pretrain.profiles.qwen3_omni_reference import oracle

    fields = MappingProxyType(
        {
            name: field.expected
            for name, field in sorted(
                oracle._CONFIG_CONTRACT_FIELDS.items()
            )
        }
    )
    implementation_fields = MappingProxyType(
        dict(sorted(oracle._AUDIO_IMPLEMENTATION_CONTRACT.items()))
    )
    source = MappingProxyType(
        {
            "model_id": QWEN3_OMNI_MODEL_ID,
            "revision": QWEN3_OMNI_REVISION,
            "transformers_version": QWEN3_TRANSFORMERS_VERSION,
            "artifact_sha256": _QWEN3_LOCAL_ARTIFACT_SHA256,
        }
    )
    return MappingProxyType(
        {
            "source": source,
            "fields": fields,
            "implementation_fields": implementation_fields,
        }
    )


def _validate_reference_source(source: str) -> None:
    if source == QWEN3_OMNI_MODEL_ID:
        return

    source_path = Path(source)
    if not source_path.is_dir():
        raise ValueError(
            "Qwen3 reference source must be the pinned model ID "
            f"{QWEN3_OMNI_MODEL_ID!r} or a pinned local snapshot"
        )
    missing = [
        filename
        for filename in _QWEN3_LOCAL_ARTIFACT_SHA256
        if not (source_path / filename).is_file()
    ]
    if missing:
        raise ValueError(
            "Qwen3 reference local snapshot is missing pinned artifacts: "
            + ", ".join(missing)
        )
    mismatched = [
        filename
        for filename, expected_sha256 in (
            _QWEN3_LOCAL_ARTIFACT_SHA256.items()
        )
        if hashlib.sha256(
            (source_path / filename).read_bytes()
        ).hexdigest()
        != expected_sha256
    ]
    if mismatched:
        raise ValueError(
            "Qwen3 reference local snapshot artifacts do not match pinned "
            "revision: "
            + ", ".join(mismatched)
        )


def _lazy_config_loader(
    source: str,
    *,
    local_files_only: bool,
) -> Callable[[], object]:
    loader = load_reference_config

    def load() -> object:
        _validate_reference_source(source)
        return loader(
            source,
            local_files_only=local_files_only,
        )

    return load


def _lazy_processor_loader(
    source: str,
    *,
    local_files_only: bool,
) -> Callable[[], object]:
    loader = load_reference_processor

    def load() -> object:
        _validate_reference_source(source)
        return loader(
            source,
            local_files_only=local_files_only,
        )

    return load


class Qwen3ReferenceFactory:
    profile = ArchitectureProfile.QWEN3_OMNI_REFERENCE

    @staticmethod
    def _validate_request(request: ProfileBuildRequest) -> None:
        if not isinstance(request, ProfileBuildRequest):
            raise TypeError("request must be ProfileBuildRequest")
        if request.profile is not ArchitectureProfile.QWEN3_OMNI_REFERENCE:
            raise ValueError(
                "Qwen3 reference factory requires profile "
                "qwen3_omni_reference"
            )
        _validate_reference_source(request.config_or_checkpoint)
        if request.tokenizer is not None:
            raise ValueError(
                "Qwen3 reference processor must use the pinned model "
                "source; tokenizer overrides are unsupported"
            )
        if request.dtype is not None or request.device is not None:
            raise ValueError(
                "Qwen3 reference oracle has no weight runtime; dtype and "
                "device are unsupported"
            )

    def manifest(
        self,
        request: ProfileBuildRequest,
    ) -> ProfileManifest:
        self._validate_request(request)
        manifest = qwen3_reference_manifest()
        manifest.validate()
        return manifest

    def validate(self, request: ProfileBuildRequest) -> None:
        self.manifest(request)
        _pinned_config_contract()

    def build(
        self,
        request: ProfileBuildRequest,
    ) -> ProfileBuildResult:
        self._validate_request(request)
        manifest = self.manifest(request)
        contract = _pinned_config_contract()
        artifact = Qwen3OracleArtifact(
            config_contract=contract,
            load_config=_lazy_config_loader(
                request.config_or_checkpoint,
                local_files_only=request.local_files_only,
            ),
            load_processor=_lazy_processor_loader(
                request.config_or_checkpoint,
                local_files_only=request.local_files_only,
            ),
        )

        capabilities = {
            "config_oracle": True,
            "lazy_config_loader": True,
            "lazy_processor_loader": True,
            "inference_runtime": False,
            "tokenizer_vocab_size": False,
            "verified_layer_layout": False,
            "weight_parameter_counts": False,
        }
        unsupported = [
            capability
            for capability, available in capabilities.items()
            if not available
        ]
        unsupported.extend(
            capability
            for capability in request.requested_capabilities
            if not capabilities.get(capability, False)
        )
        fields = contract["fields"]
        if not isinstance(fields, Mapping):
            raise TypeError("pinned config fields must be a mapping")
        embedding_vocab_size = fields[
            "vocabulary.embedding_vocab_size"
        ]
        model_type = fields["model_type"]
        if (
            type(embedding_vocab_size) is not int
            or not isinstance(model_type, str)
        ):
            raise TypeError("pinned Qwen3 oracle identity is malformed")
        summary = ArchitectureSummary(
            profile=manifest.architecture_profile.value,
            compatibility_level=manifest.compatibility_level.value,
            model_type=model_type,
            tokenizer_vocab_size=0,
            embedding_vocab_size=embedding_vocab_size,
            total_parameters=0,
            active_parameters_per_token=0,
            routed_parameters=0,
            shared_parameters=0,
            dense_parameters=0,
            capabilities=capabilities,
            layers=(),
            unsupported_capabilities=tuple(
                dict.fromkeys(unsupported)
            ),
        )
        return ProfileBuildResult(
            artifact=artifact,
            manifest=manifest,
            architecture_summary=summary,
        )


__all__ = ["Qwen3OracleArtifact", "Qwen3ReferenceFactory"]
