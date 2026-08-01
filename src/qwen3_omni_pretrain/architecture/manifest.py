from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Mapping

from .profiles import ArchitectureProfile, CompatibilityLevel


@dataclass(frozen=True)
class SourceRevision:
    name: str
    revision: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not isinstance(self.revision, str):
            raise TypeError("source name and revision must be strings")
        if not self.name.strip() or not self.revision.strip():
            raise ValueError("source name and revision must be non-empty")

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "revision": self.revision}


@dataclass(frozen=True)
class ProfileManifest:
    architecture_profile: ArchitectureProfile
    compatibility_level: CompatibilityLevel
    sources: Mapping[str, SourceRevision]
    assumptions: tuple[str, ...]
    exact_official_checkpoint_compatible: bool
    validated_context_length: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.architecture_profile, ArchitectureProfile):
            raise TypeError("architecture_profile must be ArchitectureProfile")
        if not isinstance(self.compatibility_level, CompatibilityLevel):
            raise TypeError("compatibility_level must be CompatibilityLevel")
        if type(self.exact_official_checkpoint_compatible) is not bool:
            raise TypeError(
                "exact_official_checkpoint_compatible must be boolean"
            )
        if (
            type(self.validated_context_length) is not int
            or self.validated_context_length < 0
        ):
            raise TypeError(
                "validated_context_length must be a non-negative integer"
            )
        if not isinstance(self.sources, Mapping):
            raise TypeError("sources must be a mapping")
        if any(
            not isinstance(source, SourceRevision)
            for source in self.sources.values()
        ):
            raise TypeError("source values must be SourceRevision")
        if not isinstance(self.assumptions, tuple):
            raise TypeError("assumptions must be a tuple")
        object.__setattr__(self, "sources", MappingProxyType(dict(self.sources)))
        self.validate()

    def validate(self) -> None:
        allowed_levels = {
            ArchitectureProfile.LEGACY_PROTOTYPE: {
                CompatibilityLevel.LEGACY_PROTOTYPE,
            },
            ArchitectureProfile.QWEN3_OMNI_REFERENCE: {
                CompatibilityLevel.STRUCTURE_ALIGNED,
                CompatibilityLevel.CHECKPOINT_COMPATIBLE,
            },
            ArchitectureProfile.QWEN35_OMNI_INSPIRED: {
                CompatibilityLevel.PAPER_INSPIRED,
            },
            ArchitectureProfile.MIMO_V25_EXPERIMENTAL: {
                CompatibilityLevel.MIMO_STYLE_EXPERIMENT,
            },
        }
        if self.compatibility_level not in allowed_levels[self.architecture_profile]:
            raise ValueError(
                f"{self.architecture_profile.value} cannot use "
                f"{self.compatibility_level.value}"
            )
        if (
            self.architecture_profile
            is not ArchitectureProfile.QWEN3_OMNI_REFERENCE
            and self.exact_official_checkpoint_compatible
        ):
            raise ValueError(
                "only qwen3_omni_reference may claim exact official "
                "checkpoint compatibility"
            )
        if (
            self.compatibility_level
            is CompatibilityLevel.CHECKPOINT_COMPATIBLE
            and not self.exact_official_checkpoint_compatible
        ):
            raise ValueError(
                "checkpoint-compatible requires "
                "exact_official_checkpoint_compatible=true"
            )
        if (
            self.exact_official_checkpoint_compatible
            and self.compatibility_level
            is not CompatibilityLevel.CHECKPOINT_COMPATIBLE
        ):
            raise ValueError(
                "exact checkpoint compatibility requires the "
                "checkpoint-compatible level"
            )
        if any(not isinstance(key, str) for key in self.sources):
            raise TypeError("source keys must be strings")
        if any(not key.strip() for key in self.sources):
            raise ValueError("source keys must be non-empty")
        if any(not isinstance(item, str) for item in self.assumptions):
            raise TypeError("assumptions must be strings")
        if any(not item.strip() for item in self.assumptions):
            raise ValueError("assumptions must be non-empty strings")

    def __deepcopy__(self, memo: dict[int, object]) -> "ProfileManifest":
        """Immutable manifests are safe to share across config deep copies."""
        memo[id(self)] = self
        return self

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "architecture_profile": self.architecture_profile.value,
            "compatibility_level": self.compatibility_level.value,
            "sources": {
                key: source.to_dict() for key, source in sorted(self.sources.items())
            },
            "assumptions": list(self.assumptions),
            "exact_official_checkpoint_compatible": (
                self.exact_official_checkpoint_compatible
            ),
            "validated_context_length": self.validated_context_length,
        }

    def canonical_sha256(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> ProfileManifest:
        if not isinstance(raw, Mapping):
            raise TypeError("manifest must be a mapping")
        allowed = {
            "architecture_profile",
            "compatibility_level",
            "sources",
            "assumptions",
            "exact_official_checkpoint_compatible",
            "validated_context_length",
        }
        missing = allowed - set(raw)
        unknown = set(raw) - allowed
        if missing or unknown:
            raise ValueError(
                f"invalid manifest keys: missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}"
            )
        raw_sources = raw["sources"]
        if not isinstance(raw_sources, Mapping):
            raise ValueError("sources must be a mapping")
        for key, value in raw_sources.items():
            if not isinstance(key, str) or not isinstance(value, Mapping):
                raise TypeError("source entries must be string-to-mapping")
            if set(value) != {"name", "revision"}:
                raise ValueError(
                    f"source {key!r} requires name and revision only"
                )
            if not isinstance(value["name"], str) or not isinstance(
                value["revision"], str
            ):
                raise TypeError("source name and revision must be strings")
        raw_assumptions = raw["assumptions"]
        if not isinstance(raw_assumptions, (list, tuple)) or any(
            not isinstance(item, str) for item in raw_assumptions
        ):
            raise TypeError("assumptions must be a sequence of strings")
        raw_exact = raw["exact_official_checkpoint_compatible"]
        if type(raw_exact) is not bool:
            raise TypeError(
                "exact_official_checkpoint_compatible must be a boolean"
            )
        raw_context = raw["validated_context_length"]
        if type(raw_context) is not int or raw_context < 0:
            raise TypeError(
                "validated_context_length must be a non-negative integer"
            )
        raw_profile = raw["architecture_profile"]
        raw_compatibility = raw["compatibility_level"]
        if not isinstance(raw_profile, str) or not isinstance(
            raw_compatibility, str
        ):
            raise TypeError("profile and compatibility level must be strings")
        manifest = cls(
            architecture_profile=ArchitectureProfile(raw_profile),
            compatibility_level=CompatibilityLevel(raw_compatibility),
            sources={
                key: SourceRevision(
                    name=value["name"],
                    revision=value["revision"],
                )
                for key, value in raw_sources.items()
            },
            assumptions=tuple(raw_assumptions),
            exact_official_checkpoint_compatible=raw_exact,
            validated_context_length=raw_context,
        )
        manifest.validate()
        return manifest
