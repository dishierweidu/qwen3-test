from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping
import tempfile
import warnings

import torch

from qwen3_omni_pretrain.architecture.manifest import (
    ProfileManifest,
)
from qwen3_omni_pretrain.architecture.profiles import (
    ArchitectureProfile,
    CompatibilityLevel,
)
from qwen3_omni_pretrain.architecture.summary import ArchitectureSummary
from qwen3_omni_pretrain.profiles.legacy_prototype.config_adapter import (
    LEGACY_MODEL_TYPE,
)


_METADATA_FILENAME = "architecture.json"
_SCHEMA_VERSION = 2
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


class CheckpointArtifactKind(str, Enum):
    """The training graph whose tensors are stored in a checkpoint."""

    STAGE1_TRAINING = "stage1-training"
    STAGE2_TRAINING = "stage2-training"


@dataclass(frozen=True)
class ModelTopology:
    """Allocation-free, deterministic evidence for the complete wrapper graph."""

    model_class: str
    component_types: Mapping[str, str]
    state_schema_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.model_class, str) or not self.model_class:
            raise ValueError("model_class must be a non-empty string")
        if not isinstance(self.component_types, Mapping):
            raise TypeError("component_types must be a mapping")
        components: dict[str, str] = {}
        for name, type_name in self.component_types.items():
            if not isinstance(name, str) or not isinstance(type_name, str):
                raise TypeError(
                    "component_types keys and values must be strings"
                )
            if not type_name:
                raise ValueError(
                    "component_types values must be non-empty"
                )
            components[name] = type_name
        if _SHA256_PATTERN.fullmatch(self.state_schema_sha256) is None:
            raise ValueError(
                "state_schema_sha256 must be a 64-character lowercase "
                "hex digest"
            )
        object.__setattr__(
            self,
            "component_types",
            MappingProxyType(dict(sorted(components.items()))),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "model_class": self.model_class,
            "component_types": dict(self.component_types),
            "state_schema_sha256": self.state_schema_sha256,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "ModelTopology":
        if not isinstance(raw, Mapping):
            raise TypeError("checkpoint topology must be a mapping")
        expected = {
            "model_class",
            "component_types",
            "state_schema_sha256",
        }
        missing = expected - set(raw)
        unknown = set(raw) - expected
        if missing or unknown:
            raise ValueError(
                "invalid checkpoint topology keys: "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        components = raw["component_types"]
        if not isinstance(components, Mapping):
            raise TypeError("component_types must be a mapping")
        return cls(
            model_class=raw["model_class"],  # type: ignore[arg-type]
            component_types=components,  # type: ignore[arg-type]
            state_schema_sha256=raw["state_schema_sha256"],  # type: ignore[arg-type]
        )


def _qualified_type(value: object) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _logical_shape(tensor: torch.Tensor) -> list[int]:
    logical = getattr(tensor, "ds_shape", None)
    shape = logical if logical is not None else tensor.shape
    return [int(dimension) for dimension in shape]


def describe_model_topology(model: torch.nn.Module) -> ModelTopology:
    """Describe classes and state schema without reading tensor contents."""
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    graph = model.module if hasattr(model, "module") else model

    try:
        named_modules = graph.named_modules(remove_duplicate=False)
    except TypeError:  # pragma: no cover - compatibility with old torch
        named_modules = graph.named_modules()
    component_types = {
        name: _qualified_type(module)
        for name, module in named_modules
    }

    aliases: dict[int, int] = {}
    state_schema: list[dict[str, object]] = []

    def add_tensor(
        name: str,
        kind: str,
        tensor: torch.Tensor,
    ) -> None:
        identity = id(tensor)
        if identity not in aliases:
            aliases[identity] = len(aliases)
        state_schema.append(
            {
                "name": name,
                "kind": kind,
                "shape": _logical_shape(tensor),
                "requires_grad": bool(
                    getattr(tensor, "requires_grad", False)
                ),
                "alias": aliases[identity],
            }
        )

    try:
        parameters = graph.named_parameters(remove_duplicate=False)
    except TypeError:  # pragma: no cover - compatibility with old torch
        parameters = graph.named_parameters()
    for name, parameter in parameters:
        add_tensor(name, "parameter", parameter)

    try:
        buffers = graph.named_buffers(remove_duplicate=False)
    except TypeError:  # pragma: no cover - compatibility with old torch
        buffers = graph.named_buffers()
    for name, buffer in buffers:
        add_tensor(name, "buffer", buffer)

    serialized = json.dumps(
        state_schema,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return ModelTopology(
        model_class=_qualified_type(graph),
        component_types=component_types,
        state_schema_sha256=hashlib.sha256(serialized).hexdigest(),
    )


def tokenizer_identity_sha256(tokenizer: object) -> str:
    """Hash complete tokenizer serialization and special-token roles."""
    special_token_roles: dict[str, int | None | list[int]] = {}
    for name in (
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "unk_token_id",
        "sep_token_id",
        "cls_token_id",
        "mask_token_id",
    ):
        value = getattr(tokenizer, name, None)
        if value is not None and type(value) is not int:
            raise ValueError(
                "tokenizer special-token evidence is malformed"
            )
        special_token_roles[name] = value
    additional_ids = getattr(
        tokenizer,
        "additional_special_tokens_ids",
        [],
    )
    if not isinstance(additional_ids, (list, tuple)) or any(
        type(index) is not int for index in additional_ids
    ):
        raise ValueError("tokenizer special-token evidence is malformed")
    special_token_roles["additional_special_tokens_ids"] = list(
        additional_ids
    )

    serialization: dict[str, object] | None = None
    backend = getattr(tokenizer, "backend_tokenizer", None)
    backend_to_str = getattr(backend, "to_str", None)
    if callable(backend_to_str):
        try:
            serialized_backend = backend_to_str()
            if not isinstance(serialized_backend, str):
                raise TypeError("backend serialization must be text")
            backend_payload = json.loads(serialized_backend)
            if not isinstance(backend_payload, dict):
                raise TypeError(
                    "backend serialization must be an object"
                )
        except Exception:
            pass
        else:
            serialization = {
                "kind": "backend_tokenizer",
                "payload": backend_payload,
            }

    if serialization is None:
        save_pretrained = getattr(tokenizer, "save_pretrained", None)
        if callable(save_pretrained):
            try:
                with tempfile.TemporaryDirectory() as directory:
                    save_pretrained(directory)
                    root = Path(directory)
                    artifacts = [
                        {
                            "path": artifact.relative_to(root).as_posix(),
                            "sha256": hashlib.sha256(
                                artifact.read_bytes()
                            ).hexdigest(),
                        }
                        for artifact in sorted(root.rglob("*"))
                        if artifact.is_file()
                    ]
            except Exception:
                artifacts = []
            if artifacts:
                serialization = {
                    "kind": "saved_artifacts",
                    "artifacts": artifacts,
                }

    if serialization is None:
        raise ValueError(
            "tokenizer identity evidence is unavailable: full "
            "serialization is required"
        )

    payload = json.dumps(
        {
            "serialization": serialization,
            "special_token_roles": special_token_roles,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class CheckpointMetadata:
    manifest: ProfileManifest
    architecture: ArchitectureSummary
    artifact_kind: CheckpointArtifactKind
    topology: ModelTopology
    tokenizer_sha256: str
    implementation_commit: str

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, ProfileManifest):
            raise TypeError("manifest must be a ProfileManifest")
        if not isinstance(self.architecture, ArchitectureSummary):
            raise TypeError("architecture must be an ArchitectureSummary")
        if not isinstance(self.artifact_kind, CheckpointArtifactKind):
            raise TypeError(
                "artifact_kind must be a CheckpointArtifactKind"
            )
        if not isinstance(self.topology, ModelTopology):
            raise TypeError("topology must be a ModelTopology")
        self.manifest.validate()
        if (
            not isinstance(self.tokenizer_sha256, str)
            or _SHA256_PATTERN.fullmatch(self.tokenizer_sha256) is None
        ):
            raise ValueError(
                "tokenizer_sha256 must be a 64-character lowercase hex digest"
            )
        if (
            not isinstance(self.implementation_commit, str)
            or _COMMIT_PATTERN.fullmatch(self.implementation_commit) is None
        ):
            raise ValueError(
                "implementation_commit must be a 40-character lowercase "
                "hex commit"
            )
        if (
            self.architecture.profile
            != self.manifest.architecture_profile.value
        ):
            raise ValueError(
                "architecture profile contradicts checkpoint manifest"
            )
        if (
            self.architecture.compatibility_level
            != self.manifest.compatibility_level.value
        ):
            raise ValueError(
                "architecture compatibility level contradicts checkpoint "
                "manifest"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "manifest": self.manifest.to_dict(),
            "architecture": self.architecture.to_dict(),
            "artifact_kind": self.artifact_kind.value,
            "topology": self.topology.to_dict(),
            "tokenizer_sha256": self.tokenizer_sha256,
            "implementation_commit": self.implementation_commit,
        }

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, object],
    ) -> "CheckpointMetadata":
        if not isinstance(raw, Mapping):
            raise TypeError("checkpoint metadata must be a mapping")
        expected = {
            "schema_version",
            "manifest",
            "architecture",
            "artifact_kind",
            "topology",
            "tokenizer_sha256",
            "implementation_commit",
        }
        missing = expected - set(raw)
        unknown = set(raw) - expected
        if missing or unknown:
            raise ValueError(
                "invalid checkpoint metadata keys: "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        if (
            type(raw["schema_version"]) is not int
            or raw["schema_version"] != _SCHEMA_VERSION
        ):
            raise ValueError(
                "unsupported checkpoint metadata schema_version: "
                f"{raw['schema_version']!r}"
            )
        raw_manifest = raw["manifest"]
        raw_architecture = raw["architecture"]
        raw_topology = raw["topology"]
        if not isinstance(raw_manifest, Mapping):
            raise TypeError("checkpoint manifest must be a mapping")
        if not isinstance(raw_architecture, Mapping):
            raise TypeError("checkpoint architecture must be a mapping")
        if not isinstance(raw_topology, Mapping):
            raise TypeError("checkpoint topology must be a mapping")
        try:
            artifact_kind = CheckpointArtifactKind(raw["artifact_kind"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "unsupported checkpoint artifact_kind: "
                f"{raw['artifact_kind']!r}"
            ) from exc
        return cls(
            manifest=ProfileManifest.from_dict(raw_manifest),
            architecture=ArchitectureSummary.from_dict(raw_architecture),
            artifact_kind=artifact_kind,
            topology=ModelTopology.from_dict(raw_topology),
            tokenizer_sha256=raw["tokenizer_sha256"],  # type: ignore[arg-type]
            implementation_commit=raw["implementation_commit"],  # type: ignore[arg-type]
        )


def write_checkpoint_metadata(
    checkpoint_dir: str | os.PathLike[str],
    metadata: CheckpointMetadata,
) -> None:
    """Atomically persist validated metadata as ``architecture.json``."""
    if not isinstance(metadata, CheckpointMetadata):
        raise TypeError("metadata must be CheckpointMetadata")
    metadata = CheckpointMetadata.from_dict(metadata.to_dict())
    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / _METADATA_FILENAME
    temporary = directory / f"{_METADATA_FILENAME}.tmp"
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                metadata.to_dict(),
                handle,
                sort_keys=True,
                separators=(",", ":"),
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _validate_adapted_legacy_identity(
    legacy_config: object,
) -> None:
    if not isinstance(legacy_config, Mapping):
        raise TypeError("adapted legacy config must be a mapping")
    if legacy_config.get("model_type") != LEGACY_MODEL_TYPE:
        raise ValueError(
            "adapted config is not a legacy prototype model"
        )
    if (
        legacy_config.get("architecture_profile")
        != ArchitectureProfile.LEGACY_PROTOTYPE.value
    ):
        raise ValueError(
            "adapted config is not a legacy prototype profile"
        )


def load_checkpoint_metadata(
    checkpoint_dir: str | os.PathLike[str],
    *,
    expected_profile: ArchitectureProfile | None = None,
    expected_compatibility: CompatibilityLevel | None = None,
    expected_architecture: ArchitectureSummary | None = None,
    expected_artifact_kind: CheckpointArtifactKind | None = None,
    expected_topology: ModelTopology | None = None,
    expected_tokenizer_sha256: str | None = None,
    legacy_config: object | None = None,
) -> CheckpointMetadata | None:
    """Load and validate a checkpoint sidecar before any tensor reads.

    A missing sidecar returns ``None`` only when ``expected_profile`` is
    explicitly ``legacy_prototype`` and ``legacy_config`` proves the adapted
    legacy identity. No architecture, tokenizer digest, or implementation
    commit is synthesized for such old checkpoints.
    """
    if expected_profile is not None and not isinstance(
        expected_profile,
        ArchitectureProfile,
    ):
        raise TypeError("expected_profile must be ArchitectureProfile")
    if expected_compatibility is not None and not isinstance(
        expected_compatibility,
        CompatibilityLevel,
    ):
        raise TypeError(
            "expected_compatibility must be CompatibilityLevel"
        )
    if expected_architecture is not None and not isinstance(
        expected_architecture,
        ArchitectureSummary,
    ):
        raise TypeError(
            "expected_architecture must be ArchitectureSummary"
        )
    if expected_artifact_kind is not None and not isinstance(
        expected_artifact_kind,
        CheckpointArtifactKind,
    ):
        raise TypeError(
            "expected_artifact_kind must be CheckpointArtifactKind"
        )
    if expected_topology is not None and not isinstance(
        expected_topology,
        ModelTopology,
    ):
        raise TypeError("expected_topology must be ModelTopology")
    if (
        expected_tokenizer_sha256 is not None
        and (
            not isinstance(expected_tokenizer_sha256, str)
            or _SHA256_PATTERN.fullmatch(expected_tokenizer_sha256) is None
        )
    ):
        raise ValueError(
            "expected_tokenizer_sha256 must be a 64-character lowercase "
            "hex digest"
        )

    metadata_path = Path(checkpoint_dir) / _METADATA_FILENAME
    if not metadata_path.exists():
        if expected_profile is not ArchitectureProfile.LEGACY_PROTOTYPE:
            raise ValueError(
                f"checkpoint is missing required {_METADATA_FILENAME}"
            )
        if legacy_config is None:
            raise ValueError(
                "missing architecture.json requires an adapted legacy config"
            )
        _validate_adapted_legacy_identity(legacy_config)
        if (
            expected_compatibility is not None
            and expected_compatibility
            is not CompatibilityLevel.LEGACY_PROTOTYPE
        ):
            raise ValueError(
                "adapted legacy config contradicts expected compatibility"
            )
        warnings.warn(
            "checkpoint has no architecture.json; loading an old "
            "legacy_prototype checkpoint is deprecated",
            DeprecationWarning,
            stacklevel=2,
        )
        return None

    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"invalid {_METADATA_FILENAME}: {exc}"
        ) from exc
    metadata = CheckpointMetadata.from_dict(raw)
    if (
        expected_profile is not None
        and metadata.manifest.architecture_profile is not expected_profile
    ):
        raise ValueError(
            "checkpoint metadata contradicts expected profile: "
            f"{metadata.manifest.architecture_profile.value!r} != "
            f"{expected_profile.value!r}"
        )
    if (
        expected_compatibility is not None
        and metadata.manifest.compatibility_level
        is not expected_compatibility
    ):
        raise ValueError(
            "checkpoint metadata contradicts expected compatibility: "
            f"{metadata.manifest.compatibility_level.value!r} != "
            f"{expected_compatibility.value!r}"
        )
    if (
        expected_architecture is not None
        and metadata.architecture != expected_architecture
    ):
        raise ValueError(
            "checkpoint metadata contradicts expected architecture"
        )
    if (
        expected_artifact_kind is not None
        and metadata.artifact_kind is not expected_artifact_kind
    ):
        raise ValueError(
            "checkpoint metadata contradicts expected artifact kind: "
            f"{metadata.artifact_kind.value!r} != "
            f"{expected_artifact_kind.value!r}"
        )
    if (
        expected_topology is not None
        and metadata.topology != expected_topology
    ):
        raise ValueError(
            "checkpoint metadata contradicts expected topology"
        )
    if (
        expected_tokenizer_sha256 is not None
        and metadata.tokenizer_sha256 != expected_tokenizer_sha256
    ):
        raise ValueError(
            "checkpoint metadata contradicts expected tokenizer identity"
        )
    return metadata
