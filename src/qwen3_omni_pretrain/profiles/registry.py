from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Protocol, runtime_checkable

from qwen3_omni_pretrain.architecture.manifest import ProfileManifest
from qwen3_omni_pretrain.architecture.profiles import (
    ArchitectureProfile,
)
from qwen3_omni_pretrain.architecture.summary import ArchitectureSummary


@dataclass(frozen=True)
class ProfileBuildRequest:
    profile: ArchitectureProfile
    config_or_checkpoint: str
    tokenizer: str | None = None
    local_files_only: bool = True
    dtype: str | None = None
    device: str | None = None
    requested_capabilities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.profile, ArchitectureProfile):
            raise TypeError("profile must be ArchitectureProfile")
        if (
            not isinstance(self.config_or_checkpoint, str)
            or not self.config_or_checkpoint.strip()
        ):
            raise ValueError(
                "config_or_checkpoint must be non-empty"
            )
        for name, value in (
            ("tokenizer", self.tokenizer),
            ("dtype", self.dtype),
            ("device", self.device),
        ):
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError(f"{name} must be non-empty when provided")
        if type(self.local_files_only) is not bool:
            raise TypeError("local_files_only must be boolean")
        if not isinstance(self.requested_capabilities, tuple):
            raise TypeError("requested_capabilities must be a tuple")
        if any(
            not isinstance(capability, str) or not capability.strip()
            for capability in self.requested_capabilities
        ):
            raise ValueError(
                "requested_capabilities must contain non-empty strings"
            )
        if len(set(self.requested_capabilities)) != len(
            self.requested_capabilities
        ):
            raise ValueError(
                "requested_capabilities must not contain duplicates"
            )


@dataclass(frozen=True)
class ProfileBuildResult:
    artifact: object
    manifest: ProfileManifest
    architecture_summary: ArchitectureSummary

    def __post_init__(self) -> None:
        if self.artifact is None:
            raise TypeError("artifact must not be None")
        if not isinstance(self.manifest, ProfileManifest):
            raise TypeError("manifest must be ProfileManifest")
        if not isinstance(
            self.architecture_summary,
            ArchitectureSummary,
        ):
            raise TypeError(
                "architecture_summary must be ArchitectureSummary"
            )
        self.manifest.validate()
        if (
            self.architecture_summary.profile
            != self.manifest.architecture_profile.value
        ):
            raise ValueError(
                "architecture summary contradicts manifest profile"
            )
        if (
            self.architecture_summary.compatibility_level
            != self.manifest.compatibility_level.value
        ):
            raise ValueError(
                "architecture summary contradicts manifest compatibility"
            )


@runtime_checkable
class ProfileFactory(Protocol):
    profile: ArchitectureProfile

    def manifest(
        self,
        request: ProfileBuildRequest,
    ) -> ProfileManifest:
        ...

    def validate(self, request: ProfileBuildRequest) -> None:
        ...

    def build(
        self,
        request: ProfileBuildRequest,
    ) -> ProfileBuildResult:
        ...


@dataclass(frozen=True)
class _FactoryEntry:
    module_path: str
    class_name: str


_FACTORY_ENTRIES: dict[ArchitectureProfile, _FactoryEntry] = {}


def register_profile_factory(
    profile: ArchitectureProfile,
    module_path: str,
    class_name: str,
) -> None:
    if not isinstance(profile, ArchitectureProfile):
        raise TypeError("profile must be ArchitectureProfile")
    for name, value in (
        ("module_path", module_path),
        ("class_name", class_name),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be non-empty")
    if profile in _FACTORY_ENTRIES:
        raise ValueError(
            f"architecture profile {profile.value!r} is already registered"
        )
    _FACTORY_ENTRIES[profile] = _FactoryEntry(
        module_path=module_path,
        class_name=class_name,
    )


def get_profile_factory(
    profile: ArchitectureProfile,
) -> ProfileFactory:
    if not isinstance(profile, ArchitectureProfile):
        raise TypeError("profile must be ArchitectureProfile")
    entry = _FACTORY_ENTRIES.get(profile)
    if entry is None:
        raise ValueError(
            f"architecture profile {profile.value!r} is not registered"
        )

    module = importlib.import_module(entry.module_path)
    try:
        factory_class = getattr(module, entry.class_name)
    except AttributeError as exc:
        raise RuntimeError(
            f"registered factory class {entry.class_name!r} is missing "
            f"from {entry.module_path!r}"
        ) from exc
    try:
        factory = factory_class()
    except Exception as exc:
        raise RuntimeError(
            f"registered factory {entry.class_name!r} is not instantiable"
        ) from exc
    if not isinstance(factory, ProfileFactory):
        raise TypeError(
            f"registered factory {entry.class_name!r} does not implement "
            "ProfileFactory"
        )
    if factory.profile is not profile:
        raise ValueError(
            f"registered factory {entry.class_name!r} claims "
            f"{factory.profile.value!r}, expected {profile.value!r}"
        )
    return factory


def parse_profile(value: str) -> ArchitectureProfile:
    if not isinstance(value, str):
        raise TypeError("architecture profile must be a string")
    try:
        return ArchitectureProfile(value)
    except ValueError as exc:
        choices = ", ".join(profile.value for profile in ArchitectureProfile)
        raise ValueError(
            f"unknown architecture profile {value!r}; choose from {choices}"
        ) from exc


register_profile_factory(
    ArchitectureProfile.LEGACY_PROTOTYPE,
    "qwen3_omni_pretrain.profiles.legacy_prototype.factory",
    "LegacyPrototypeFactory",
)
register_profile_factory(
    ArchitectureProfile.QWEN3_OMNI_REFERENCE,
    "qwen3_omni_pretrain.profiles.qwen3_omni_reference.factory",
    "Qwen3ReferenceFactory",
)


__all__ = [
    "ProfileBuildRequest",
    "ProfileBuildResult",
    "ProfileFactory",
    "get_profile_factory",
    "parse_profile",
    "register_profile_factory",
]
