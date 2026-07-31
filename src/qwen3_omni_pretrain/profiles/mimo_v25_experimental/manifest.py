"""Pinned provenance for the MiMo-V2.5-inspired mechanism experiments."""

from __future__ import annotations

from qwen3_omni_pretrain.architecture.manifest import (
    ProfileManifest,
    SourceRevision,
)
from qwen3_omni_pretrain.architecture.profiles import (
    ArchitectureProfile,
    CompatibilityLevel,
)


MIMO_V25_MODEL_ID = "XiaomiMiMo/MiMo-V2.5"
MIMO_V25_REVISION = "63651580ca774f8504f676040460aed3e1244ac1"
MIMO_V25_PRO_MODEL_ID = "XiaomiMiMo/MiMo-V2.5-Pro"
MIMO_V25_PRO_REVISION = "21d1ecfecd7bd70f31be25ca49d7edd21f003659"


def mimo_experiment_manifest() -> ProfileManifest:
    """Return the immutable experiment boundary, never a checkpoint claim."""

    return ProfileManifest(
        architecture_profile=ArchitectureProfile.MIMO_V25_EXPERIMENTAL,
        compatibility_level=CompatibilityLevel.MIMO_STYLE_EXPERIMENT,
        sources={
            "mimo-v2.5-architecture-inspiration": SourceRevision(
                name=MIMO_V25_MODEL_ID,
                revision=MIMO_V25_REVISION,
            ),
            "mimo-v2.5-pro-report-inspiration": SourceRevision(
                name=MIMO_V25_PRO_MODEL_ID,
                revision=MIMO_V25_PRO_REVISION,
            ),
        },
        assumptions=(
            "generic Hybrid SWA-MoE experiment; not an official MiMo model class",
            "six decoder layers instead of the reported production depth",
            "eight routed experts and top-2 routing instead of production scale",
            "128-token sliding window and tiny vocabulary for mechanism tests",
            "eager attention sink and local Code-free text-only runtime",
            "no official MiMo checkpoint or state-dict compatibility",
        ),
        exact_official_checkpoint_compatible=False,
        validated_context_length=128,
    )


__all__ = [
    "MIMO_V25_MODEL_ID",
    "MIMO_V25_PRO_MODEL_ID",
    "MIMO_V25_PRO_REVISION",
    "MIMO_V25_REVISION",
    "mimo_experiment_manifest",
]
