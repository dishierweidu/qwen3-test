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
from qwen3_omni_pretrain.profiles.qwen3_omni_reference.contract import (
    QWEN3_OMNI_CONFIG_CONTRACT,
    QWEN3_OMNI_METADATA_SHA256,
    QWEN3_OMNI_MODEL_TYPE,
)
from qwen3_omni_pretrain.profiles.qwen3_omni_reference.pins import (
    QWEN3_OMNI_MODEL_ID,
    QWEN3_OMNI_REVISION,
)
from qwen3_omni_pretrain.profiles.registry import (
    ProfileBuildRequest,
    ProfileBuildResult,
)

_QWEN3_LOCAL_ARTIFACT_SHA256 = QWEN3_OMNI_METADATA_SHA256


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
    return QWEN3_OMNI_CONFIG_CONTRACT


def _verify_reference_artifacts(
    artifact_paths: Mapping[str, Path],
) -> Path:
    missing = [
        filename
        for filename in _QWEN3_LOCAL_ARTIFACT_SHA256
        if filename not in artifact_paths
        or not artifact_paths[filename].is_file()
    ]
    if missing:
        raise ValueError(
            "Qwen3 reference local snapshot is missing pinned artifacts: "
            + ", ".join(missing)
        )

    snapshot_roots = {
        artifact_paths[filename].parent
        for filename in _QWEN3_LOCAL_ARTIFACT_SHA256
    }
    if len(snapshot_roots) != 1:
        raise ValueError(
            "Qwen3 reference metadata did not resolve to one pinned "
            "snapshot"
        )

    mismatched = [
        filename
        for filename, expected_sha256 in (
            _QWEN3_LOCAL_ARTIFACT_SHA256.items()
        )
        if hashlib.sha256(
            artifact_paths[filename].read_bytes()
        ).hexdigest()
        != expected_sha256
    ]
    if mismatched:
        raise ValueError(
            "Qwen3 reference local snapshot artifacts do not match pinned "
            "revision: "
            + ", ".join(mismatched)
        )
    return snapshot_roots.pop()


def _verify_local_reference_snapshot(source_path: Path) -> Path:
    if not source_path.is_dir():
        raise ValueError(
            "Qwen3 reference source must be the pinned model ID "
            f"{QWEN3_OMNI_MODEL_ID!r} or a pinned local snapshot"
        )
    return _verify_reference_artifacts(
        {
            filename: source_path / filename
            for filename in _QWEN3_LOCAL_ARTIFACT_SHA256
        }
    )


def _validate_reference_source(source: str) -> None:
    if source == QWEN3_OMNI_MODEL_ID:
        return
    _verify_local_reference_snapshot(Path(source))


def _resolve_reference_source(
    source: str,
    *,
    local_files_only: bool,
) -> Path:
    if source != QWEN3_OMNI_MODEL_ID:
        return _verify_local_reference_snapshot(Path(source))

    from huggingface_hub import hf_hub_download

    artifact_paths = {}
    for filename in _QWEN3_LOCAL_ARTIFACT_SHA256:
        resolved = Path(
            hf_hub_download(
                repo_id=QWEN3_OMNI_MODEL_ID,
                filename=filename,
                revision=QWEN3_OMNI_REVISION,
                local_files_only=local_files_only,
            )
        )
        if resolved.name != filename:
            raise ValueError(
                "Qwen3 reference metadata resolved an unexpected "
                f"artifact path for {filename}"
            )
        artifact_paths[filename] = resolved
    return _verify_reference_artifacts(artifact_paths)


def _lazy_config_loader(
    source: str,
    *,
    local_files_only: bool,
) -> Callable[[], object]:
    loader = load_reference_config

    def load() -> object:
        verified_source = _resolve_reference_source(
            source,
            local_files_only=local_files_only,
        )
        return loader(
            str(verified_source),
            local_files_only=True,
        )

    return load


def _lazy_processor_loader(
    source: str,
    *,
    local_files_only: bool,
) -> Callable[[], object]:
    loader = load_reference_processor

    def load() -> object:
        verified_source = _resolve_reference_source(
            source,
            local_files_only=local_files_only,
        )
        return loader(
            str(verified_source),
            local_files_only=True,
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

    def validate(
        self,
        request: ProfileBuildRequest,
    ) -> ProfileManifest:
        manifest = self.manifest(request)
        _pinned_config_contract()
        return manifest

    def build(
        self,
        request: ProfileBuildRequest,
    ) -> ProfileBuildResult:
        manifest = self.validate(request)
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
        vocabulary = contract["vocabulary"]
        if not isinstance(vocabulary, Mapping):
            raise TypeError("pinned config vocabulary must be a mapping")
        embedding_vocab_size = vocabulary["embedding_vocab_size"]
        if type(embedding_vocab_size) is not int:
            raise TypeError("pinned Qwen3 oracle identity is malformed")
        summary = ArchitectureSummary(
            profile=manifest.architecture_profile.value,
            compatibility_level=manifest.compatibility_level.value,
            model_type=QWEN3_OMNI_MODEL_TYPE,
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
