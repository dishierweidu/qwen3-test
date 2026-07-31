from __future__ import annotations

from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Mapping

import torch

from qwen3_omni_pretrain.architecture.manifest import ProfileManifest
from qwen3_omni_pretrain.utils.model_stats import collect_parameter_stats


def require_exact_typed_keys(
    raw: Mapping[str, object],
    *,
    ints: tuple[str, ...],
    strings: tuple[str, ...],
) -> dict[str, object]:
    if not isinstance(raw, Mapping):
        raise TypeError("serialized value must be a mapping")
    copied = dict(raw)
    expected = set(ints) | set(strings)
    missing = expected - set(copied)
    unknown = set(copied) - expected
    if missing or unknown:
        raise ValueError(
            f"invalid serialized keys: missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    for key in ints:
        value = copied[key]
        if type(value) is not int:
            raise TypeError(f"{key} must be an integer")
        if value < 0:
            raise ValueError(f"{key} must be non-negative")
    for key in strings:
        if not isinstance(copied[key], str):
            raise TypeError(f"{key} must be a string")
    return copied


@dataclass(frozen=True)
class LayerArchitecture:
    index: int
    attention_type: str
    cache_type: str
    ffn_type: str
    routed_experts: int
    experts_per_token: int

    def __post_init__(self) -> None:
        validated = require_exact_typed_keys(
            asdict(self),
            ints=("index", "routed_experts", "experts_per_token"),
            strings=("attention_type", "cache_type", "ffn_type"),
        )
        if not all(validated[key] for key in (
            "attention_type",
            "cache_type",
            "ffn_type",
        )):
            raise ValueError("layer architecture strings must be non-empty")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, object],
    ) -> "LayerArchitecture":
        validated = require_exact_typed_keys(
            raw,
            ints=("index", "routed_experts", "experts_per_token"),
            strings=("attention_type", "cache_type", "ffn_type"),
        )
        return cls(**validated)


_SUMMARY_STRING_FIELDS = (
    "profile",
    "compatibility_level",
    "model_type",
)
_SUMMARY_INTEGER_FIELDS = (
    "tokenizer_vocab_size",
    "embedding_vocab_size",
    "total_parameters",
    "active_parameters_per_token",
    "routed_parameters",
    "shared_parameters",
    "dense_parameters",
)
_SUMMARY_CONTAINER_FIELDS = (
    "capabilities",
    "layers",
    "unsupported_capabilities",
)


def validate_architecture_summary_mapping(
    raw: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(raw, Mapping):
        raise TypeError("architecture summary must be a mapping")
    copied = dict(raw)
    expected = (
        set(_SUMMARY_STRING_FIELDS)
        | set(_SUMMARY_INTEGER_FIELDS)
        | set(_SUMMARY_CONTAINER_FIELDS)
    )
    missing = expected - set(copied)
    unknown = set(copied) - expected
    if missing or unknown:
        raise ValueError(
            "invalid architecture summary keys: "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    for key in _SUMMARY_STRING_FIELDS:
        if not isinstance(copied[key], str):
            raise TypeError(f"{key} must be a string")
        if not copied[key]:
            raise ValueError(f"{key} must be non-empty")
    for key in _SUMMARY_INTEGER_FIELDS:
        value = copied[key]
        if type(value) is not int:
            raise TypeError(f"{key} must be an integer")
        if value < 0:
            raise ValueError(f"{key} must be non-negative")

    raw_capabilities = copied["capabilities"]
    if not isinstance(raw_capabilities, Mapping) or any(
        not isinstance(key, str) or type(value) is not bool
        for key, value in raw_capabilities.items()
    ):
        raise TypeError("capabilities must map strings to booleans")
    copied["capabilities"] = dict(raw_capabilities)

    raw_layers = copied["layers"]
    if not isinstance(raw_layers, list):
        raise TypeError("serialized layers must be a list")
    copied_layers = []
    for layer in raw_layers:
        if not isinstance(layer, Mapping):
            raise TypeError("serialized layers must contain mappings")
        copied_layers.append(dict(layer))
    indices = [
        LayerArchitecture.from_dict(layer).index
        for layer in copied_layers
    ]
    if indices != list(range(len(indices))):
        raise ValueError("layer indices must be contiguous from zero")
    copied["layers"] = copied_layers

    raw_unsupported = copied["unsupported_capabilities"]
    if not isinstance(raw_unsupported, list):
        raise TypeError(
            "serialized unsupported_capabilities must be a list"
        )
    if any(not isinstance(item, str) for item in raw_unsupported):
        raise TypeError("unsupported_capabilities must contain strings")
    copied["unsupported_capabilities"] = list(raw_unsupported)
    return copied


@dataclass(frozen=True)
class ArchitectureSummary:
    profile: str
    compatibility_level: str
    model_type: str
    tokenizer_vocab_size: int
    embedding_vocab_size: int
    total_parameters: int
    active_parameters_per_token: int
    routed_parameters: int
    shared_parameters: int
    dense_parameters: int
    capabilities: Mapping[str, bool]
    layers: tuple[LayerArchitecture, ...]
    unsupported_capabilities: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.capabilities, Mapping) or any(
            not isinstance(key, str) or type(value) is not bool
            for key, value in self.capabilities.items()
        ):
            raise TypeError("capabilities must map strings to booleans")
        if not isinstance(self.layers, tuple):
            raise TypeError("layers must be a tuple")
        if any(
            not isinstance(layer, LayerArchitecture)
            for layer in self.layers
        ):
            raise TypeError("layers must contain LayerArchitecture values")
        if tuple(layer.index for layer in self.layers) != tuple(
            range(len(self.layers))
        ):
            raise ValueError("layer indices must be contiguous from zero")
        if not isinstance(self.unsupported_capabilities, tuple):
            raise TypeError("unsupported_capabilities must be a tuple")
        if any(
            not isinstance(item, str)
            for item in self.unsupported_capabilities
        ):
            raise TypeError("unsupported_capabilities must contain strings")

        for key in _SUMMARY_STRING_FIELDS:
            value = getattr(self, key)
            if not isinstance(value, str):
                raise TypeError(f"{key} must be a string")
            if not value:
                raise ValueError(f"{key} must be non-empty")
        for key in _SUMMARY_INTEGER_FIELDS:
            value = getattr(self, key)
            if type(value) is not int:
                raise TypeError(f"{key} must be an integer")
            if value < 0:
                raise ValueError(f"{key} must be non-negative")
        if self.total_parameters != (
            self.routed_parameters
            + self.shared_parameters
            + self.dense_parameters
        ):
            raise ValueError(
                "parameter categories must sum to total_parameters"
            )
        if self.active_parameters_per_token > self.total_parameters:
            raise ValueError(
                "active_parameters_per_token cannot exceed total_parameters"
            )
        object.__setattr__(
            self,
            "capabilities",
            MappingProxyType(dict(self.capabilities)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "compatibility_level": self.compatibility_level,
            "model_type": self.model_type,
            "tokenizer_vocab_size": self.tokenizer_vocab_size,
            "embedding_vocab_size": self.embedding_vocab_size,
            "total_parameters": self.total_parameters,
            "active_parameters_per_token": self.active_parameters_per_token,
            "routed_parameters": self.routed_parameters,
            "shared_parameters": self.shared_parameters,
            "dense_parameters": self.dense_parameters,
            "capabilities": dict(sorted(self.capabilities.items())),
            "layers": [layer.to_dict() for layer in self.layers],
            "unsupported_capabilities": list(
                self.unsupported_capabilities
            ),
        }

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, object],
    ) -> "ArchitectureSummary":
        validated = validate_architecture_summary_mapping(raw)
        layers = validated.pop("layers")
        unsupported = validated.pop("unsupported_capabilities")
        capabilities = validated.pop("capabilities")
        return cls(
            **validated,
            layers=tuple(
                LayerArchitecture.from_dict(layer)
                for layer in layers
            ),
            capabilities=capabilities,
            unsupported_capabilities=tuple(unsupported),
        )


def _describe_attention(module: torch.nn.Module) -> tuple[str, str]:
    class_name = type(module).__name__
    if class_name == "GatedDeltaNetAttention":
        return "gated-deltanet", "recurrent-state"
    if class_name in {
        "MultiHeadSelfAttention",
        "TensorParallelMultiHeadSelfAttention",
    }:
        return "full-attention", "kv-cache"
    raise TypeError(f"unsupported attention module: {class_name}")


def _describe_layer(
    index: int,
    layer: torch.nn.Module,
) -> LayerArchitecture:
    attention_type, cache_type = _describe_attention(layer.self_attn)
    moe = getattr(layer, "moe_mlp", None)
    if moe is None:
        ffn_type = "dense"
        routed_experts = 0
        experts_per_token = 0
    else:
        routed_experts = int(moe.num_experts)
        experts_per_token = int(moe.num_experts_per_tok)
        ffn_type = (
            "shared-dense-plus-dense-ensemble"
            if experts_per_token == routed_experts
            else "shared-dense-plus-routed-moe"
        )
    return LayerArchitecture(
        index=index,
        attention_type=attention_type,
        cache_type=cache_type,
        ffn_type=ffn_type,
        routed_experts=routed_experts,
        experts_per_token=experts_per_token,
    )


def summarize_model(
    model: torch.nn.Module,
    manifest: ProfileManifest,
    *,
    tokenizer_vocab_size: int | None = None,
) -> ArchitectureSummary:
    if not isinstance(manifest, ProfileManifest):
        raise TypeError("manifest must be a ProfileManifest")
    manifest.validate()
    if not hasattr(model, "layers") or not hasattr(model, "embed_tokens"):
        raise TypeError("model does not expose thinker layers and embeddings")

    stats = collect_parameter_stats(model)
    config = model.config
    embedding_vocab_size = getattr(
        model.embed_tokens,
        "num_embeddings",
        None,
    )
    if type(embedding_vocab_size) is not int:
        raise TypeError("model embedding vocabulary size is unavailable")
    if tokenizer_vocab_size is None:
        tokenizer_vocab_size = getattr(
            config,
            "_tokenizer_vocab_size",
            config.vocab_size,
        )
    if type(tokenizer_vocab_size) is not int or tokenizer_vocab_size < 0:
        raise ValueError(
            "tokenizer_vocab_size must be a non-negative integer"
        )
    cache_support = getattr(model, "cache_support", None)
    capabilities = (
        cache_support.as_dict()
        if cache_support is not None
        and callable(getattr(cache_support, "as_dict", None))
        else {}
    )
    return ArchitectureSummary(
        profile=manifest.architecture_profile.value,
        compatibility_level=manifest.compatibility_level.value,
        model_type=config.model_type,
        tokenizer_vocab_size=tokenizer_vocab_size,
        embedding_vocab_size=embedding_vocab_size,
        total_parameters=stats.total_parameters,
        active_parameters_per_token=(
            stats.estimated_active_parameters_per_token
        ),
        routed_parameters=stats.routed_parameters,
        shared_parameters=stats.shared_parameters,
        dense_parameters=stats.dense_parameters,
        capabilities=capabilities,
        layers=tuple(
            _describe_layer(index, layer)
            for index, layer in enumerate(model.layers)
        ),
        unsupported_capabilities=(),
    )
