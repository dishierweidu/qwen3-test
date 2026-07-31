from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import torch
import yaml

from qwen3_omni_pretrain.architecture.manifest import ProfileManifest
from qwen3_omni_pretrain.architecture.profiles import ArchitectureProfile
from qwen3_omni_pretrain.architecture.summary import (
    ArchitectureSummary,
    LayerArchitecture,
)
from qwen3_omni_pretrain.multimodal.encoders import (
    PatchVisionEncoder,
    TemporalVideoEncoder,
)
from qwen3_omni_pretrain.multimodal.prefill import MultimodalPrefillPipeline
from qwen3_omni_pretrain.multimodal.sequence_assembler import SequenceAssembler
from qwen3_omni_pretrain.multimodal.tokenization.schema import (
    MultimodalTokenSchema,
    ResolvedMultimodalTokens,
    resolve_token_schema,
)
from qwen3_omni_pretrain.profiles.qwen3_omni_reference.codec_streamer import (
    ReferenceCodecStreamer,
)
from qwen3_omni_pretrain.profiles.qwen35_omni_inspired.aria import AriaScheduler
from qwen3_omni_pretrain.profiles.qwen35_omni_inspired.audio_encoder import (
    Qwen35AudioSequenceAdapter,
    Qwen35AuTEncoder,
)
from qwen3_omni_pretrain.profiles.qwen35_omni_inspired.configuration import (
    QWEN35_BACKBONE_REVISION,
    QWEN35_TRANSFORMERS_VERSION,
    Qwen35InspiredConfig,
)
from qwen3_omni_pretrain.profiles.qwen35_omni_inspired.runtime import (
    Qwen35InspiredRuntime,
)
from qwen3_omni_pretrain.profiles.qwen35_omni_inspired.talker import (
    PredecessorCodecProxy,
    PredecessorMTPProxy,
    PrototypeCode2Wav,
    Qwen35InspiredTalker,
)
from qwen3_omni_pretrain.profiles.qwen35_omni_inspired.thinker import (
    Qwen35InspiredThinker,
)
from qwen3_omni_pretrain.profiles.qwen35_omni_inspired.timestamp_alignment import (
    Qwen35TimestampExpansionPolicy,
    build_qwen35_position_builder,
)
from qwen3_omni_pretrain.profiles.registry import (
    ProfileBuildRequest,
    ProfileBuildResult,
)
from qwen3_omni_pretrain.utils.model_stats import collect_parameter_stats


_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass(frozen=True)
class Qwen35ConfigArtifact:
    config: Qwen35InspiredConfig
    public_backbone_revision: str

    def __post_init__(self) -> None:
        if not isinstance(self.config, Qwen35InspiredConfig):
            raise TypeError("config must be Qwen35InspiredConfig")
        if self.public_backbone_revision != QWEN35_BACKBONE_REVISION:
            raise ValueError("public backbone revision is not pinned")


def _load_mapping(path_value: str) -> dict[str, object]:
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(
            f"Qwen3.5-inspired config does not exist: {path}"
        )
    config_path = path / "config.json" if path.is_dir() else path
    with config_path.open("r", encoding="utf-8") as handle:
        raw = (
            json.load(handle)
            if config_path.suffix.lower() == ".json"
            else yaml.safe_load(handle)
        )
    if raw is None or raw == {}:
        raise ValueError("Qwen3.5-inspired configuration is empty")
    if not isinstance(raw, Mapping):
        raise TypeError("Qwen3.5-inspired configuration must be a mapping")
    return dict(raw)


def _text_config(config: Qwen35InspiredConfig) -> Mapping[str, object]:
    return config.backbone_text_config


def summarize_qwen35_config_contract(
    config: Qwen35InspiredConfig,
    *,
    requested_capabilities: tuple[str, ...] = (),
) -> ArchitectureSummary:
    text = _text_config(config)
    layer_types = text.get("layer_types")
    if layer_types is None:
        layer_count = int(text.get("num_hidden_layers", 0))
        layer_types = [
            "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
            for index in range(layer_count)
        ]
    experts = int(text.get("num_experts", 0))
    top_k = int(text.get("num_experts_per_tok", 0))
    layers = tuple(
        LayerArchitecture(
            index=index,
            attention_type=(
                "gated-delta-net"
                if layer_type == "linear_attention"
                else "full-attention"
            ),
            cache_type=(
                "recurrent-matrix-state"
                if layer_type == "linear_attention"
                else "kv-cache"
            ),
            ffn_type="shared-dense-plus-routed-moe",
            routed_experts=experts,
            experts_per_token=top_k,
        )
        for index, layer_type in enumerate(layer_types)
    )
    capabilities = {
        "model_runtime": True,
        "incremental_decode_state": True,
        "streaming_generation": True,
        "beam_search": False,
        "state_truncate": False,
        "speculative_decode": False,
        "streaming_audio_encoder": False,
        "paper_inspired_aria": True,
        "speech_talker": True,
        "predecessor_codec_proxy": True,
        "allocation_free_inspection": True,
    }
    unsupported = tuple(
        dict.fromkeys(
            name
            for name in requested_capabilities
            if capabilities.get(name) is not True
        )
    )
    vocab_size = int(text.get("vocab_size", 0))
    return ArchitectureSummary(
        profile=config.profile_manifest.architecture_profile.value,
        compatibility_level=config.profile_manifest.compatibility_level.value,
        model_type=config.model_type,
        tokenizer_vocab_size=vocab_size,
        embedding_vocab_size=vocab_size,
        total_parameters=0,
        active_parameters_per_token=0,
        routed_parameters=0,
        shared_parameters=0,
        dense_parameters=0,
        capabilities=capabilities,
        layers=layers,
        unsupported_capabilities=unsupported,
    )


class _DeterministicTimestampTokenizer:
    def __init__(
        self,
        *,
        vocab_size: int,
        forbidden_ids: frozenset[int],
    ) -> None:
        available = tuple(
            value
            for value in range(vocab_size)
            if value not in forbidden_ids
        )
        if not available:
            raise ValueError("no ordinary vocabulary IDs remain for timestamps")
        self.available = available

    def __call__(self, text: str, *, add_special_tokens: bool):
        if add_special_tokens is not False:
            raise ValueError("timestamp tokenizer forbids automatic special tokens")
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        index = int.from_bytes(digest[:8], byteorder="big") % len(self.available)
        return {"input_ids": [self.available[index]]}


def _default_tokens(vocab_size: int, pad_token_id: int) -> ResolvedMultimodalTokens:
    candidates = [value for value in range(vocab_size - 1, -1, -1) if value != pad_token_id]
    if len(candidates) < 7:
        raise ValueError("Qwen3.5-inspired runtime requires at least 8 vocabulary IDs")
    values = candidates[:7]
    return ResolvedMultimodalTokens(
        image_pad=values[0],
        video_pad=values[1],
        audio_pad=values[2],
        vision_start=values[3],
        vision_end=values[4],
        audio_start=values[5],
        audio_end=values[6],
    )


def _resolved_tokens(
    *,
    request: ProfileBuildRequest,
    vocab_size: int,
    pad_token_id: int,
) -> tuple[ResolvedMultimodalTokens, object]:
    if request.tokenizer is None:
        tokens = _default_tokens(vocab_size, pad_token_id)
        forbidden = frozenset(
            value
            for value in (
                pad_token_id,
                tokens.image_pad,
                tokens.video_pad,
                tokens.audio_pad,
                tokens.vision_start,
                tokens.vision_end,
                tokens.audio_start,
                tokens.audio_end,
            )
            if value is not None
        )
        return tokens, _DeterministicTimestampTokenizer(
            vocab_size=vocab_size,
            forbidden_ids=forbidden,
        )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        request.tokenizer,
        use_fast=True,
        local_files_only=request.local_files_only,
    )
    return (
        resolve_token_schema(
            tokenizer,
            MultimodalTokenSchema.qwen35(),
            vocab_size,
        ),
        tokenizer,
    )


def summarize_qwen35_runtime(
    runtime: Qwen35InspiredRuntime | Qwen35InspiredThinker,
    *,
    manifest: ProfileManifest,
    requested_capabilities: tuple[str, ...] = (),
) -> ArchitectureSummary:
    thinker = runtime.thinker if isinstance(runtime, Qwen35InspiredRuntime) else runtime
    stats = collect_parameter_stats(runtime)
    text_config = thinker.backbone.config
    capabilities = thinker.cache_support.as_dict()
    capabilities.update(
        {
            "model_runtime": True,
            "streaming_audio_encoder": False,
            "paper_inspired_aria": isinstance(runtime, Qwen35InspiredRuntime),
            "speech_talker": isinstance(runtime, Qwen35InspiredRuntime)
            and isinstance(runtime.talker, Qwen35InspiredTalker),
            "predecessor_codec_proxy": isinstance(runtime, Qwen35InspiredRuntime)
            and isinstance(runtime.codec_proxy, PredecessorCodecProxy),
        }
    )
    layers = tuple(
        LayerArchitecture(
            index=index,
            attention_type=(
                "gated-delta-net"
                if layer_type == "linear_attention"
                else "full-attention"
            ),
            cache_type=(
                "recurrent-matrix-state"
                if layer_type == "linear_attention"
                else "kv-cache"
            ),
            ffn_type="shared-dense-plus-routed-moe",
            routed_experts=int(text_config.num_experts),
            experts_per_token=int(text_config.num_experts_per_tok),
        )
        for index, layer_type in enumerate(text_config.layer_types)
    )
    unsupported = tuple(
        dict.fromkeys(
            capability
            for capability in requested_capabilities
            if capabilities.get(capability) is not True
        )
    )
    return ArchitectureSummary(
        profile=manifest.architecture_profile.value,
        compatibility_level=manifest.compatibility_level.value,
        model_type=Qwen35InspiredConfig.model_type,
        tokenizer_vocab_size=int(text_config.vocab_size),
        embedding_vocab_size=int(thinker.embed_tokens.num_embeddings),
        total_parameters=stats.total_parameters,
        active_parameters_per_token=stats.estimated_active_parameters_per_token,
        routed_parameters=stats.routed_parameters,
        shared_parameters=stats.shared_parameters,
        dense_parameters=stats.dense_parameters,
        capabilities=capabilities,
        layers=layers,
        unsupported_capabilities=unsupported,
    )


class Qwen35InspiredFactory:
    profile = ArchitectureProfile.QWEN35_OMNI_INSPIRED

    @staticmethod
    def _validate_request(request: ProfileBuildRequest) -> None:
        if not isinstance(request, ProfileBuildRequest):
            raise TypeError("request must be ProfileBuildRequest")
        if request.profile is not ArchitectureProfile.QWEN35_OMNI_INSPIRED:
            raise ValueError(
                "Qwen3.5-inspired factory requires qwen35_omni_inspired"
            )
        if request.dtype is not None and request.dtype.removeprefix("torch.") not in _DTYPES:
            raise ValueError("Qwen3.5 dtype must be float32, float16, or bfloat16")
        if request.device is not None:
            try:
                torch.device(request.device)
            except (RuntimeError, TypeError) as exc:
                raise ValueError("invalid Qwen3.5 build device") from exc

    def _config(self, request: ProfileBuildRequest) -> Qwen35InspiredConfig:
        self._validate_request(request)
        config = Qwen35InspiredConfig(**_load_mapping(request.config_or_checkpoint))
        config.profile_manifest.validate()
        return config

    def manifest(self, request: ProfileBuildRequest) -> ProfileManifest:
        return self._config(request).profile_manifest

    def validate(self, request: ProfileBuildRequest) -> ProfileManifest:
        return self._config(request).profile_manifest

    def build_config_contract(
        self,
        request: ProfileBuildRequest,
    ) -> ProfileBuildResult:
        config = self._config(request)
        artifact = Qwen35ConfigArtifact(
            config=config,
            public_backbone_revision=QWEN35_BACKBONE_REVISION,
        )
        return ProfileBuildResult(
            artifact=artifact,
            manifest=config.profile_manifest,
            architecture_summary=summarize_qwen35_config_contract(
                config,
                requested_capabilities=request.requested_capabilities,
            ),
        )

    def build(self, request: ProfileBuildRequest) -> ProfileBuildResult:
        config = self._config(request)
        import transformers

        if transformers.__version__ != QWEN35_TRANSFORMERS_VERSION:
            raise RuntimeError(
                "qwen35_omni_inspired runtime requires transformers==5.2.0"
            )
        if version("qwen-omni-utils") != "0.0.9":
            raise RuntimeError(
                "qwen35_omni_inspired runtime requires qwen-omni-utils==0.0.9"
            )
        from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

        public_config = Qwen3_5MoeTextConfig(**dict(config.backbone_text_config))
        hidden_size = int(public_config.hidden_size)
        vocab_size = int(public_config.vocab_size)
        pad_token_id = int(public_config.pad_token_id or 0)
        tokens, tokenizer = _resolved_tokens(
            request=request,
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
        )
        device_context = (
            torch.device(request.device)
            if request.device is not None
            else nullcontext()
        )
        with device_context:
            backbone = Qwen3_5MoeForCausalLM(public_config)
            audio_kernel = Qwen35AuTEncoder(
                config.audio_config,
                backbone_hidden_size=hidden_size,
            )
            pipeline = MultimodalPrefillPipeline(
                tokens=tokens,
                image_encoder=PatchVisionEncoder(
                    in_channels=3,
                    hidden_size=hidden_size,
                    patch_size=int(getattr(config, "vision_patch_size", 16)),
                ),
                video_encoder=TemporalVideoEncoder(hidden_size=hidden_size),
                audio_encoder=Qwen35AudioSequenceAdapter(audio_kernel),
                assembler=SequenceAssembler(),
                expansion_policy=Qwen35TimestampExpansionPolicy(
                    tokenizer=tokenizer,
                    timestamp_format=config.timestamp_format,
                ),
                position_builder=build_qwen35_position_builder(),
                pad_token_id=pad_token_id,
                joint_separator_token_ids=frozenset(
                    getattr(config, "joint_separator_token_ids", ())
                ),
                max_assembled_length=int(
                    getattr(config, "max_assembled_length", 4096)
                ),
            )
            thinker = Qwen35InspiredThinker.from_public_model(
                backbone,
                prefill_pipeline=pipeline,
                profile_config=config,
            )
            talker_hidden = int(
                getattr(config, "talker_hidden_size", min(hidden_size, 128))
            )
            talker_heads = int(getattr(config, "talker_attention_heads", 4))
            if talker_hidden % talker_heads:
                raise ValueError("talker_hidden_size must divide talker_attention_heads")
            talker_experts = int(
                getattr(config, "talker_num_experts", min(int(public_config.num_experts), 8))
            )
            talker_top_k = int(
                getattr(config, "talker_num_experts_per_token", min(2, talker_experts - 1))
            )
            talker = Qwen35InspiredTalker(
                thinker_hidden_size=hidden_size,
                text_vocab_size=vocab_size,
                hidden_size=talker_hidden,
                num_hidden_layers=int(getattr(config, "talker_num_hidden_layers", 4)),
                num_attention_heads=talker_heads,
                num_key_value_heads=int(getattr(config, "talker_num_key_value_heads", 2)),
                num_experts=talker_experts,
                num_experts_per_token=talker_top_k,
                expert_intermediate_size=int(
                    getattr(config, "talker_expert_intermediate_size", talker_hidden * 2)
                ),
                shared_intermediate_size=int(
                    getattr(config, "talker_shared_intermediate_size", talker_hidden * 2)
                ),
                manifest=config.profile_manifest,
            )
            mtp = PredecessorMTPProxy(conditioning_size=talker_hidden)
            decoder = PrototypeCode2Wav(samples_per_frame=1920)
            codec = PredecessorCodecProxy(
                mtp_proxy=mtp,
                decoder=decoder,
                codec_streamer=ReferenceCodecStreamer(
                    decoder=decoder,
                    left_context_frames=72,
                    samples_per_frame=1920,
                    sample_rate=24_000,
                ),
            )
            runtime = Qwen35InspiredRuntime(
                thinker=thinker,
                aria_scheduler=AriaScheduler(config.aria_config),
                talker=talker,
                codec_proxy=codec,
            )
        dtype = (
            None
            if request.dtype is None
            else _DTYPES[request.dtype.removeprefix("torch.")]
        )
        if dtype is not None:
            runtime.to(dtype=dtype)
        runtime.named_parameter_groups()
        return ProfileBuildResult(
            artifact=runtime,
            manifest=config.profile_manifest,
            architecture_summary=summarize_qwen35_runtime(
                runtime,
                manifest=config.profile_manifest,
                requested_capabilities=request.requested_capabilities,
            ),
        )


__all__ = [
    "Qwen35ConfigArtifact",
    "Qwen35InspiredFactory",
    "summarize_qwen35_config_contract",
    "summarize_qwen35_runtime",
]
