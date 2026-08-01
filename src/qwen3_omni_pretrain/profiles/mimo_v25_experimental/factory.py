"""Lazy factory for the generic MiMo-style mechanism experiment."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
import json
from pathlib import Path
from typing import Mapping

import torch
import torch.distributed as dist
import yaml

from qwen3_omni_pretrain.architecture.manifest import ProfileManifest
from qwen3_omni_pretrain.architecture.profiles import ArchitectureProfile
from qwen3_omni_pretrain.architecture.summary import summarize_model
from qwen3_omni_pretrain.models.hybrid_swa_moe.configuration_hybrid_swa_moe import (
    HybridSwaMoeConfig,
)
from qwen3_omni_pretrain.models.hybrid_swa_moe.modeling_hybrid_swa_moe import (
    HybridSwaMoeForCausalLM,
)
from qwen3_omni_pretrain.parallel.expert_parallel import (
    ExpertParallelContext,
    ParallelTopology,
    validate_parallel_topology,
)
from qwen3_omni_pretrain.parallel.initialize import (
    get_data_parallel_world_size,
    get_pipeline_model_parallel_world_size,
    get_tensor_model_parallel_world_size,
)
from qwen3_omni_pretrain.profiles.mimo_v25_experimental.manifest import (
    mimo_experiment_manifest,
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


def _resolve_dtype(value: str | None) -> torch.dtype | None:
    if value is None:
        return None
    normalized = value.removeprefix("torch.")
    try:
        return _DTYPES[normalized]
    except KeyError as error:
        raise ValueError(
            "MiMo experiment dtype must be bfloat16, float16, or float32"
        ) from error


def _load_config_mapping(path_value: str) -> dict[str, object]:
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"Hybrid experiment config does not exist: {path}")
    config_path = path / "config.json" if path.is_dir() else path
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Hybrid experiment config file does not exist: {config_path}"
        )
    with config_path.open("r", encoding="utf-8") as handle:
        raw = (
            json.load(handle)
            if config_path.suffix.lower() == ".json"
            else yaml.safe_load(handle)
        )
    if raw is None or raw == {}:
        raise ValueError("Hybrid experiment configuration is empty")
    if not isinstance(raw, Mapping):
        raise TypeError("Hybrid experiment configuration must be a mapping")
    return dict(raw)


class MimoV25ExperimentalFactory:
    profile = ArchitectureProfile.MIMO_V25_EXPERIMENTAL

    def __init__(
        self,
        *,
        topology: ParallelTopology | None = None,
        expert_parallel_context: ExpertParallelContext | None = None,
    ) -> None:
        if topology is not None and not isinstance(topology, ParallelTopology):
            raise TypeError("topology must be ParallelTopology or None")
        if expert_parallel_context is not None and not isinstance(
            expert_parallel_context, ExpertParallelContext
        ):
            raise TypeError(
                "expert_parallel_context must be ExpertParallelContext or None"
            )
        self._topology = topology
        self._expert_parallel_context = expert_parallel_context

    def _resolved_topology(self) -> ParallelTopology:
        if self._topology is not None:
            topology = self._topology
        else:
            # EP is never inferred from ambient world size.  Existing DP/PP/TP
            # behavior is retained with an explicit singleton EP dimension.
            topology = ParallelTopology(
                data_parallel_size=get_data_parallel_world_size(),
                pipeline_parallel_size=get_pipeline_model_parallel_world_size(),
                expert_parallel_size=1,
                tensor_parallel_size=get_tensor_model_parallel_world_size(),
            )
        validated = validate_parallel_topology(
            world_size=topology.world_size,
            data_parallel_size=topology.data_parallel_size,
            pipeline_parallel_size=topology.pipeline_parallel_size,
            expert_parallel_size=topology.expert_parallel_size,
            tensor_parallel_size=topology.tensor_parallel_size,
        )
        if dist.is_available() and dist.is_initialized():
            actual_world_size = dist.get_world_size()
            if actual_world_size != validated.world_size:
                raise ValueError(
                    "parallel topology world size does not match torch.distributed"
                )
        context = self._expert_parallel_context
        if validated.expert_parallel_size == 1:
            if context is not None:
                raise ValueError("EP=1 must not receive an expert parallel context")
        else:
            if context is None:
                raise ValueError("EP=2 requires an initialized expert parallel context")
            if context.world_size != validated.expert_parallel_size:
                raise ValueError("EP topology and expert context world sizes differ")
        return validated

    def _validate_request(self, request: ProfileBuildRequest) -> None:
        if not isinstance(request, ProfileBuildRequest):
            raise TypeError("request must be ProfileBuildRequest")
        if request.profile is not self.profile:
            raise ValueError(
                "MiMo experiment factory requires mimo_v25_experimental"
            )
        if request.tokenizer is not None:
            raise ValueError(
                "MiMo experiment uses the configured generic vocabulary and "
                "does not load an official tokenizer"
            )
        _resolve_dtype(request.dtype)
        if request.device is not None:
            try:
                torch.device(request.device)
            except (RuntimeError, TypeError) as error:
                raise ValueError(
                    f"invalid Hybrid experiment device: {request.device!r}"
                ) from error

    def _config(self, request: ProfileBuildRequest) -> HybridSwaMoeConfig:
        self._validate_request(request)
        config = HybridSwaMoeConfig(
            **_load_config_mapping(request.config_or_checkpoint)
        )
        context = self._expert_parallel_context
        if context is not None and context.num_experts != config.num_experts:
            raise ValueError(
                "expert context range does not cover the configured experts"
            )
        return config

    def manifest(self, request: ProfileBuildRequest) -> ProfileManifest:
        self._resolved_topology()
        self._validate_request(request)
        # Provenance is independent of local tensor allocation and config path.
        return mimo_experiment_manifest()

    def validate(self, request: ProfileBuildRequest) -> ProfileManifest:
        # Topology validation intentionally precedes config IO and model allocation.
        self._resolved_topology()
        return self._config(request).profile_manifest

    def build(self, request: ProfileBuildRequest) -> ProfileBuildResult:
        self._resolved_topology()
        config = self._config(request)
        device_context = (
            torch.device(request.device)
            if request.device is not None
            else nullcontext()
        )
        with device_context:
            model = HybridSwaMoeForCausalLM(
                config,
                expert_parallel_context=self._expert_parallel_context,
            )
        dtype = _resolve_dtype(request.dtype)
        if dtype is not None:
            model.to(dtype=dtype)
        summary = summarize_model(model, config.profile_manifest)
        unsupported = tuple(
            capability
            for capability in request.requested_capabilities
            if summary.capabilities.get(capability) is not True
        )
        if unsupported:
            summary = replace(
                summary,
                unsupported_capabilities=unsupported,
            )
        return ProfileBuildResult(
            artifact=model,
            manifest=config.profile_manifest,
            architecture_summary=summary,
        )


__all__ = ["MimoV25ExperimentalFactory"]
