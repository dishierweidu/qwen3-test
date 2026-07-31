from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
import json
from pathlib import Path
from typing import Mapping

import torch
from transformers import AutoTokenizer
import yaml

from qwen3_omni_pretrain.architecture.manifest import ProfileManifest
from qwen3_omni_pretrain.architecture.profiles import (
    ArchitectureProfile,
)
from qwen3_omni_pretrain.architecture.summary import summarize_model
from qwen3_omni_pretrain.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeConfig,
)
from qwen3_omni_pretrain.models.qwen3_omni_moe.modeling_thinker_text import (
    Qwen3OmniMoeThinkerTextModel,
)
from qwen3_omni_pretrain.multimodal.tokenization.special_tokens import (
    reconcile_multimodal_token_ids,
)
from qwen3_omni_pretrain.profiles.legacy_prototype.config_adapter import (
    adapt_legacy_config_dict,
)
from qwen3_omni_pretrain.profiles.registry import (
    ProfileBuildRequest,
    ProfileBuildResult,
)


_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _resolve_dtype(name: str | None) -> torch.dtype | None:
    if name is None:
        return None
    normalized = name.removeprefix("torch.")
    try:
        return _DTYPES[normalized]
    except KeyError as exc:
        raise ValueError(
            "legacy dtype must be one of bfloat16, float16, or float32"
        ) from exc


def _load_adapted_mapping(path_value: str) -> dict[str, object]:
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(
            f"legacy config or checkpoint does not exist: {path}"
        )
    if path.is_dir():
        config_path = path / "config.json"
    else:
        config_path = path
    with config_path.open("r", encoding="utf-8") as handle:
        raw = (
            json.load(handle)
            if config_path.suffix.lower() == ".json"
            else yaml.safe_load(handle)
        )
    if raw is None or raw == {}:
        raise ValueError("model configuration is empty")
    if not isinstance(raw, Mapping):
        raise TypeError("model configuration must be a mapping")
    return adapt_legacy_config_dict(raw)


class LegacyPrototypeFactory:
    profile = ArchitectureProfile.LEGACY_PROTOTYPE

    @staticmethod
    def _validate_request(request: ProfileBuildRequest) -> None:
        if not isinstance(request, ProfileBuildRequest):
            raise TypeError("request must be ProfileBuildRequest")
        if request.profile is not ArchitectureProfile.LEGACY_PROTOTYPE:
            raise ValueError(
                "legacy factory requires profile legacy_prototype"
            )
        _resolve_dtype(request.dtype)
        if request.device is not None:
            try:
                torch.device(request.device)
            except (RuntimeError, TypeError) as exc:
                raise ValueError(
                    f"invalid legacy build device: {request.device!r}"
                ) from exc

    def _config(
        self,
        request: ProfileBuildRequest,
    ) -> Qwen3OmniMoeConfig:
        self._validate_request(request)
        config = Qwen3OmniMoeConfig(
            **_load_adapted_mapping(request.config_or_checkpoint)
        )
        config.profile_manifest.validate()
        return config

    def manifest(
        self,
        request: ProfileBuildRequest,
    ) -> ProfileManifest:
        return self._config(request).profile_manifest

    def validate(
        self,
        request: ProfileBuildRequest,
    ) -> ProfileManifest:
        return self._config(request).profile_manifest

    def build(
        self,
        request: ProfileBuildRequest,
    ) -> ProfileBuildResult:
        config = self._config(request)
        if request.tokenizer is not None:
            tokenizer = AutoTokenizer.from_pretrained(
                request.tokenizer,
                use_fast=True,
                local_files_only=request.local_files_only,
            )
            reconcile_multimodal_token_ids(config, tokenizer)
            config._tokenizer_vocab_size = len(tokenizer)

        device_context = (
            torch.device(request.device)
            if request.device is not None
            else nullcontext()
        )
        with device_context:
            model = Qwen3OmniMoeThinkerTextModel(config)
        dtype = _resolve_dtype(request.dtype)
        if dtype is not None:
            model.to(dtype=dtype)

        summary = summarize_model(model, config.profile_manifest)
        if request.requested_capabilities:
            unsupported = tuple(
                dict.fromkeys(
                    summary.unsupported_capabilities
                    + tuple(
                        capability
                        for capability in request.requested_capabilities
                        if summary.capabilities.get(capability) is not True
                    )
                )
            )
            summary = replace(
                summary,
                unsupported_capabilities=unsupported,
            )
        return ProfileBuildResult(
            artifact=model,
            manifest=config.profile_manifest,
            architecture_summary=summary,
        )


__all__ = ["LegacyPrototypeFactory"]
