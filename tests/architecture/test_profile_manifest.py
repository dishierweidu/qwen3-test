import pytest

from qwen3_omni_pretrain.architecture.manifest import (
    ProfileManifest,
    SourceRevision,
)
from qwen3_omni_pretrain.architecture.profiles import (
    ArchitectureProfile,
    CompatibilityLevel,
)


def valid_manifest() -> ProfileManifest:
    return ProfileManifest(
        architecture_profile=ArchitectureProfile.QWEN35_OMNI_INSPIRED,
        compatibility_level=CompatibilityLevel.PAPER_INSPIRED,
        sources={
            "backbone": SourceRevision(
                name="Qwen3.5-35B-A3B",
                revision="59d61f3ce65a6d9863b86d2e96597125219dc754",
            )
        },
        assumptions=("predecessor codec proxy",),
        exact_official_checkpoint_compatible=False,
    )


def test_manifest_round_trip_preserves_provenance():
    manifest = valid_manifest()
    assert ProfileManifest.from_dict(manifest.to_dict()) == manifest


def test_qwen35_manifest_rejects_checkpoint_compatibility():
    with pytest.raises(ValueError, match="cannot use"):
        ProfileManifest(
            architecture_profile=ArchitectureProfile.QWEN35_OMNI_INSPIRED,
            compatibility_level=CompatibilityLevel.CHECKPOINT_COMPATIBLE,
            sources={},
            assumptions=(),
            exact_official_checkpoint_compatible=True,
        )


def test_manifest_does_not_coerce_boolean_strings():
    raw = valid_manifest().to_dict()
    raw["exact_official_checkpoint_compatible"] = "false"
    with pytest.raises(TypeError, match="boolean"):
        ProfileManifest.from_dict(raw)


def test_manifest_rejects_unknown_fields():
    raw = valid_manifest().to_dict()
    raw["typo"] = 1
    with pytest.raises(ValueError, match="unknown"):
        ProfileManifest.from_dict(raw)


def test_mimo_experiment_cannot_claim_checkpoint_compatibility():
    with pytest.raises(ValueError, match="cannot use"):
        ProfileManifest(
            architecture_profile=ArchitectureProfile.MIMO_V25_EXPERIMENTAL,
            compatibility_level=CompatibilityLevel.CHECKPOINT_COMPATIBLE,
            sources={},
            assumptions=("tiny generic mechanism experiment",),
            exact_official_checkpoint_compatible=True,
        )


def test_structure_aligned_qwen3_cannot_set_exact_flag():
    with pytest.raises(ValueError, match="checkpoint-compatible level"):
        ProfileManifest(
            architecture_profile=ArchitectureProfile.QWEN3_OMNI_REFERENCE,
            compatibility_level=CompatibilityLevel.STRUCTURE_ALIGNED,
            sources={},
            assumptions=(),
            exact_official_checkpoint_compatible=True,
        )


def test_manifest_context_and_digest_are_strict_and_stable():
    manifest = valid_manifest()
    assert manifest.validated_context_length == 0
    assert len(manifest.canonical_sha256()) == 64
    assert (
        ProfileManifest.from_dict(manifest.to_dict()).canonical_sha256()
        == manifest.canonical_sha256()
    )
    raw = manifest.to_dict()
    raw["validated_context_length"] = True
    with pytest.raises(TypeError, match="validated_context_length"):
        ProfileManifest.from_dict(raw)
