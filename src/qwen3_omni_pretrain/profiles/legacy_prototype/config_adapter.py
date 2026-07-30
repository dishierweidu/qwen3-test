from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Mapping
import warnings


LEGACY_MODEL_TYPE = "qwen3_omni_prototype"
OLD_COLLIDING_MODEL_TYPE = "qwen3_omni_moe"


def adapt_legacy_config_dict(
    raw: Mapping[str, object],
) -> dict[str, object]:
    migrated = deepcopy(dict(raw))
    model_type = migrated.get("model_type", LEGACY_MODEL_TYPE)
    if model_type == OLD_COLLIDING_MODEL_TYPE:
        warnings.warn(
            "legacy model_type='qwen3_omni_moe' collides with the official "
            "Transformers architecture; loading as qwen3_omni_prototype",
            DeprecationWarning,
            stacklevel=2,
        )
        migrated["model_type"] = LEGACY_MODEL_TYPE
    elif model_type != LEGACY_MODEL_TYPE:
        raise ValueError(f"not a legacy prototype config: {model_type!r}")

    profile = migrated.get("architecture_profile", "legacy_prototype")
    if profile != "legacy_prototype":
        raise ValueError(f"not a legacy prototype profile: {profile!r}")

    thinker = deepcopy(dict(migrated.get("thinker_config", {})))
    indices = str(thinker.get("moe_layer_indices") or "").strip()
    raw_use_moe = thinker.get("use_moe", False)
    if type(raw_use_moe) is not bool:
        raise TypeError("legacy thinker use_moe must be boolean")
    effective_use_moe = raw_use_moe or bool(indices)
    if indices and not raw_use_moe:
        warnings.warn(
            "legacy standard model enabled MoE from moe_layer_indices while "
            "the TP path ignored it; preserving the standard layer graph",
            DeprecationWarning,
            stacklevel=2,
        )
    thinker["use_moe"] = effective_use_moe
    if not effective_use_moe:
        thinker["routing_kind"] = "dense"
    else:
        num_experts = int(thinker.get("num_experts", 8))
        top_k = int(thinker.get("num_experts_per_tok", 2))
        if num_experts <= 0:
            raise ValueError("legacy num_experts must be positive")
        if not 1 <= top_k <= num_experts:
            raise ValueError(
                "legacy num_experts_per_tok must be between 1 and "
                "num_experts"
            )
        if top_k < num_experts:
            thinker["routing_kind"] = "sparse"
        else:
            thinker["routing_kind"] = "dense_ensemble"
    migrated["thinker_config"] = thinker
    migrated["architecture_profile"] = "legacy_prototype"
    return migrated


def load_legacy_config_dict(path: str | Path) -> dict[str, object]:
    config_path = Path(path)
    if config_path.is_dir():
        config_path = config_path / "config.json"
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return adapt_legacy_config_dict(raw)
