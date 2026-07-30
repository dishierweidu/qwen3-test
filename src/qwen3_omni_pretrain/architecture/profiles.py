from enum import Enum


class ArchitectureProfile(str, Enum):
    LEGACY_PROTOTYPE = "legacy_prototype"
    QWEN3_OMNI_REFERENCE = "qwen3_omni_reference"
    QWEN35_OMNI_INSPIRED = "qwen35_omni_inspired"
    MIMO_V25_EXPERIMENTAL = "mimo_v25_experimental"


class CompatibilityLevel(str, Enum):
    LEGACY_PROTOTYPE = "legacy-prototype"
    STRUCTURE_ALIGNED = "structure-aligned"
    CHECKPOINT_COMPATIBLE = "checkpoint-compatible"
    PAPER_INSPIRED = "paper-inspired"
    MIMO_STYLE_EXPERIMENT = "MiMo-style-experiment"
