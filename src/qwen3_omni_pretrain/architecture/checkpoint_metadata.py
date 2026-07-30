from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Mapping
import warnings

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
    adapt_legacy_config_dict,
)


_METADATA_FILENAME = "architecture.json"
_SCHEMA_VERSION = 1
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class CheckpointMetadata:
    manifest: ProfileManifest
    architecture: ArchitectureSummary
    tokenizer_sha256: str
    implementation_commit: str

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, ProfileManifest):
            raise TypeError("manifest must be a ProfileManifest")
        if not isinstance(self.architecture, ArchitectureSummary):
            raise TypeError("architecture must be an ArchitectureSummary")
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
        if not isinstance(raw_manifest, Mapping):
            raise TypeError("checkpoint manifest must be a mapping")
        if not isinstance(raw_architecture, Mapping):
            raise TypeError("checkpoint architecture must be a mapping")
        return cls(
            manifest=ProfileManifest.from_dict(raw_manifest),
            architecture=ArchitectureSummary.from_dict(raw_architecture),
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


def _legacy_manifest_from_adapted_config(
    legacy_config: object,
) -> ProfileManifest:
    profile_manifest = getattr(legacy_config, "profile_manifest", None)
    if profile_manifest is not None:
        if not isinstance(profile_manifest, ProfileManifest):
            raise TypeError(
                "adapted legacy config profile_manifest must be "
                "ProfileManifest"
            )
        profile_manifest.validate()
        if (
            profile_manifest.architecture_profile
            is not ArchitectureProfile.LEGACY_PROTOTYPE
        ):
            raise ValueError(
                "adapted config is not a legacy prototype profile"
            )
        return profile_manifest

    if not isinstance(legacy_config, Mapping):
        raise TypeError("adapted legacy config must be a mapping or config")
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
    adapt_legacy_config_dict(legacy_config)
    return ProfileManifest(
        architecture_profile=ArchitectureProfile.LEGACY_PROTOTYPE,
        compatibility_level=CompatibilityLevel.LEGACY_PROTOTYPE,
        sources={},
        assumptions=("custom research architecture",),
        exact_official_checkpoint_compatible=False,
    )


def load_checkpoint_metadata(
    checkpoint_dir: str | os.PathLike[str],
    *,
    expected_profile: ArchitectureProfile | None = None,
    expected_compatibility: CompatibilityLevel | None = None,
    expected_architecture: ArchitectureSummary | None = None,
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
        manifest = _legacy_manifest_from_adapted_config(legacy_config)
        if (
            expected_compatibility is not None
            and manifest.compatibility_level is not expected_compatibility
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
        expected_tokenizer_sha256 is not None
        and metadata.tokenizer_sha256 != expected_tokenizer_sha256
    ):
        raise ValueError(
            "checkpoint metadata contradicts expected tokenizer identity"
        )
    return metadata
