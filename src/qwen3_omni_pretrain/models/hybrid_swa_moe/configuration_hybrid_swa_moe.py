"""Configuration for the generic Hybrid SWA routed-MoE experiment."""

from __future__ import annotations

from copy import copy
import math
from collections.abc import Mapping, Sequence

from transformers import PretrainedConfig

from qwen3_omni_pretrain.architecture.manifest import ProfileManifest
from qwen3_omni_pretrain.profiles.mimo_v25_experimental.manifest import (
    mimo_experiment_manifest,
)


def _positive_int(value: object, *, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _finite_positive(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0:
        raise ValueError(f"{field} must be finite and positive")
    return converted


def _finite_non_negative(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0:
        raise ValueError(f"{field} must be finite and non-negative")
    return converted


def _layer_types(
    value: Sequence[str],
    *,
    field: str,
    expected_length: int,
    allowed: set[str],
) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field} must be a sequence")
    copied = list(value)
    if len(copied) != expected_length:
        raise ValueError(f"{field} length must equal num_hidden_layers")
    unknown = {item for item in copied if item not in allowed}
    if unknown:
        raise ValueError(f"{field} contains unsupported types: {sorted(unknown)}")
    return copied


class HybridSwaMoeConfig(PretrainedConfig):
    """Validated tiny mechanism config, isolated from official MiMo classes."""

    model_type = "hybrid_swa_moe_experimental"
    architecture_profile = "mimo_v25_experimental"

    def __init__(
        self,
        *,
        vocab_size: int = 256,
        hidden_size: int = 128,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 8,
        full_num_key_value_heads: int = 2,
        swa_num_key_value_heads: int = 4,
        qk_head_dim: int = 24,
        v_head_dim: int = 16,
        rotary_dim: int = 8,
        attention_layer_types: Sequence[str] = (
            "swa", "swa", "swa", "swa", "swa", "full"
        ),
        ffn_layer_types: Sequence[str] = (
            "dense",
            "routed_moe", "routed_moe", "routed_moe",
            "routed_moe", "routed_moe",
        ),
        swa_window_size: int = 128,
        full_rope_theta: float = 10_000_000.0,
        swa_rope_theta: float = 10_000.0,
        attention_sink: bool = True,
        num_experts: int = 8,
        num_experts_per_token: int = 2,
        expert_intermediate_size: int = 128,
        dense_intermediate_size: int = 512,
        value_scale: float | None = None,
        router_aux_loss_weight: float = 0.01,
        mtp_num_predictors: int = 1,
        mtp_loss_weight: float = 0.1,
        rms_norm_eps: float = 1e-6,
        initializer_range: float = 0.02,
        max_position_embeddings: int = 4096,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
        tie_word_embeddings: bool = False,
        architecture_profile: str = "mimo_v25_experimental",
        profile_manifest: Mapping[str, object] | ProfileManifest | None = None,
        **kwargs: object,
    ) -> None:
        supplied_model_type = kwargs.pop("model_type", self.model_type)
        if supplied_model_type != self.model_type:
            raise ValueError(
                "model_type must be hybrid_swa_moe_experimental; official "
                "MiMo model identities are not accepted"
            )
        if architecture_profile != self.architecture_profile:
            raise ValueError("architecture_profile must be mimo_v25_experimental")
        for identity_field in ("_name_or_path", "name_or_path"):
            identity = kwargs.get(identity_field)
            if isinstance(identity, str) and "mimo" in identity.lower():
                raise ValueError("official MiMo checkpoint identity is not accepted")
        architectures = kwargs.get("architectures")
        if isinstance(architectures, Sequence) and not isinstance(
            architectures, (str, bytes)
        ) and any("mimo" in str(item).lower() for item in architectures):
            raise ValueError("official MiMo architecture names are not accepted")

        self.vocab_size = _positive_int(vocab_size, field="vocab_size")
        self.hidden_size = _positive_int(hidden_size, field="hidden_size")
        self.num_hidden_layers = _positive_int(
            num_hidden_layers, field="num_hidden_layers"
        )
        self.num_attention_heads = _positive_int(
            num_attention_heads, field="num_attention_heads"
        )
        self.full_num_key_value_heads = _positive_int(
            full_num_key_value_heads, field="full_num_key_value_heads"
        )
        self.swa_num_key_value_heads = _positive_int(
            swa_num_key_value_heads, field="swa_num_key_value_heads"
        )
        for field in ("full_num_key_value_heads", "swa_num_key_value_heads"):
            count = getattr(self, field)
            if self.num_attention_heads % count:
                raise ValueError(f"num_attention_heads must be divisible by {field}")
        self.qk_head_dim = _positive_int(qk_head_dim, field="qk_head_dim")
        self.v_head_dim = _positive_int(v_head_dim, field="v_head_dim")
        self.rotary_dim = _positive_int(rotary_dim, field="rotary_dim")
        if self.rotary_dim > self.qk_head_dim or self.rotary_dim % 2:
            raise ValueError("rotary_dim must be even and no larger than qk_head_dim")

        self.attention_layer_types = _layer_types(
            attention_layer_types,
            field="attention_layer_types",
            expected_length=self.num_hidden_layers,
            allowed={"swa", "full"},
        )
        self.ffn_layer_types = _layer_types(
            ffn_layer_types,
            field="ffn_layer_types",
            expected_length=self.num_hidden_layers,
            allowed={"dense", "routed_moe"},
        )
        if self.ffn_layer_types[0] != "dense":
            raise ValueError("layer 0 must use a dense FFN")
        if "routed_moe" not in self.ffn_layer_types:
            raise ValueError("at least one routed_moe layer is required")

        forbidden = sorted(
            key for key in kwargs
            if "deltanet" in key.lower() or key in {"use_linear_attention"}
        )
        if forbidden:
            raise ValueError(
                f"DeltaNet fields are not supported by this profile: {forbidden}"
            )
        self.swa_window_size = _positive_int(
            swa_window_size, field="swa_window_size"
        )
        self.full_rope_theta = _finite_positive(
            full_rope_theta, field="full_rope_theta"
        )
        self.swa_rope_theta = _finite_positive(
            swa_rope_theta, field="swa_rope_theta"
        )
        if type(attention_sink) is not bool:
            raise TypeError("attention_sink must be boolean")
        self.attention_sink = attention_sink
        self.value_scale = (
            None
            if value_scale is None
            else _finite_positive(value_scale, field="value_scale")
        )

        self.num_experts = _positive_int(num_experts, field="num_experts")
        self.num_experts_per_token = _positive_int(
            num_experts_per_token, field="num_experts_per_token"
        )
        # Alias expected by the common routed-parameter protocol.
        self.num_experts_per_tok = self.num_experts_per_token
        if self.num_experts_per_token >= self.num_experts:
            raise ValueError("routed MoE requires top_k < num_experts")
        self.expert_intermediate_size = _positive_int(
            expert_intermediate_size, field="expert_intermediate_size"
        )
        self.dense_intermediate_size = _positive_int(
            dense_intermediate_size, field="dense_intermediate_size"
        )
        self.router_aux_loss_weight = _finite_non_negative(
            router_aux_loss_weight, field="router_aux_loss_weight"
        )
        if type(mtp_num_predictors) is not int or not 0 <= mtp_num_predictors <= 3:
            raise ValueError("mtp_num_predictors must be in [0, 3]")
        self.mtp_num_predictors = mtp_num_predictors
        self.mtp_loss_weight = _finite_non_negative(
            mtp_loss_weight, field="mtp_loss_weight"
        )
        if self.mtp_num_predictors == 0 and self.mtp_loss_weight != 0.0:
            raise ValueError("mtp_loss_weight requires at least one MTP predictor")

        self.rms_norm_eps = _finite_positive(rms_norm_eps, field="rms_norm_eps")
        self.initializer_range = _finite_positive(
            initializer_range, field="initializer_range"
        )
        self.max_position_embeddings = _positive_int(
            max_position_embeddings, field="max_position_embeddings"
        )

        canonical_manifest = mimo_experiment_manifest()
        if profile_manifest is not None:
            supplied_manifest = (
                profile_manifest
                if isinstance(profile_manifest, ProfileManifest)
                else ProfileManifest.from_dict(profile_manifest)
            )
            if supplied_manifest != canonical_manifest:
                raise ValueError(
                    "profile manifest contradicts the pinned MiMo-style "
                    "experimental contract"
                )

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.profile_manifest = canonical_manifest

    def to_dict(self) -> dict[str, object]:
        serialization_copy = copy(self)
        serialization_copy.attention_layer_types = list(self.attention_layer_types)
        serialization_copy.ffn_layer_types = list(self.ffn_layer_types)
        serialization_copy.profile_manifest = self.profile_manifest.to_dict()
        output = PretrainedConfig.to_dict(serialization_copy)
        output.update(
            {
                "model_type": self.model_type,
                "architecture_profile": self.architecture_profile,
                "attention_layer_types": list(self.attention_layer_types),
                "ffn_layer_types": list(self.ffn_layer_types),
                "profile_manifest": self.profile_manifest.to_dict(),
            }
        )
        return output


__all__ = ["HybridSwaMoeConfig"]
