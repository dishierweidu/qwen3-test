from __future__ import annotations

from copy import copy
from dataclasses import asdict, dataclass
import math
from types import MappingProxyType
from typing import Mapping

from transformers import PretrainedConfig

from qwen3_omni_pretrain.architecture.manifest import (
    ProfileManifest,
    SourceRevision,
)
from qwen3_omni_pretrain.architecture.profiles import (
    ArchitectureProfile,
    CompatibilityLevel,
)


QWEN35_BACKBONE_MODEL_ID = "Qwen/Qwen3.5-35B-A3B"
QWEN35_BACKBONE_REVISION = (
    "59d61f3ce65a6d9863b86d2e96597125219dc754"
)
QWEN35_TRANSFORMERS_VERSION = "5.2.0"
PREDECESSOR_CODEC_MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
PREDECESSOR_CODEC_REVISION = (
    "26291f793822fb6be9555850f06dfe95f2d7e695"
)
PREDECESSOR_CODEC_PROVENANCE = "predecessor-codec-proxy"


def _plain_positive_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be a plain integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _finite_positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


@dataclass(frozen=True)
class Qwen35AuTConfig:
    sample_rate: int = 16_000
    num_mel_bins: int = 128
    window_ms: float = 25.0
    hop_ms: float = 10.0
    temporal_downsample: int = 16
    output_frame_hz: float = 6.25
    hidden_size: int = 512
    encoder_layers: int = 8
    attention_heads: int = 8
    intermediate_size: int = 2048

    def __post_init__(self) -> None:
        for name in (
            "sample_rate",
            "num_mel_bins",
            "temporal_downsample",
            "hidden_size",
            "encoder_layers",
            "attention_heads",
            "intermediate_size",
        ):
            _plain_positive_int(getattr(self, name), name)
        for name in ("window_ms", "hop_ms", "output_frame_hz"):
            _finite_positive(getattr(self, name), name)
        if self.sample_rate != 16_000:
            raise ValueError("Qwen3.5-inspired AuT sample_rate must be 16000")
        if self.num_mel_bins != 128:
            raise ValueError("Qwen3.5-inspired AuT num_mel_bins must be 128")
        if self.window_ms != 25.0 or self.hop_ms != 10.0:
            raise ValueError("Qwen3.5-inspired AuT uses a 25 ms/10 ms frontend")
        if self.temporal_downsample != 16 or self.output_frame_hz != 6.25:
            raise ValueError("Qwen3.5-inspired AuT output rate must be 6.25 Hz")
        if self.hidden_size % self.attention_heads:
            raise ValueError("audio hidden_size must divide attention_heads")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_value(cls, value: Mapping[str, object] | Qwen35AuTConfig) -> Qwen35AuTConfig:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("audio_config must be a mapping")
        return cls(**dict(value))


@dataclass(frozen=True)
class AriaConfig:
    speech_tokens_per_text_num: int = 12
    speech_tokens_per_text_den: int = 5
    text_first: bool = True
    tie_break: str = "text"
    count_eos: bool = False
    count_padding: bool = False

    def __post_init__(self) -> None:
        _plain_positive_int(
            self.speech_tokens_per_text_num,
            "speech_tokens_per_text_num",
        )
        _plain_positive_int(
            self.speech_tokens_per_text_den,
            "speech_tokens_per_text_den",
        )
        if self.text_first is not True:
            raise ValueError("ARIA version one requires text_first=true")
        if self.tie_break not in {"text", "speech"}:
            raise ValueError("ARIA tie_break must be text or speech")
        if type(self.count_eos) is not bool or type(self.count_padding) is not bool:
            raise TypeError("ARIA count flags must be booleans")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_value(cls, value: Mapping[str, object] | AriaConfig) -> AriaConfig:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("aria_config must be a mapping")
        return cls(**dict(value))


@dataclass(frozen=True)
class CodecProxyConfig:
    source_model: str = PREDECESSOR_CODEC_MODEL_ID
    source_revision: str = PREDECESSOR_CODEC_REVISION
    provenance_label: str = PREDECESSOR_CODEC_PROVENANCE

    def __post_init__(self) -> None:
        if self.source_model != PREDECESSOR_CODEC_MODEL_ID:
            raise ValueError("codec proxy must use the Qwen3 predecessor model")
        if self.source_revision != PREDECESSOR_CODEC_REVISION:
            raise ValueError("codec proxy must use the pinned predecessor revision")
        if self.provenance_label != PREDECESSOR_CODEC_PROVENANCE:
            raise ValueError("codec proxy must be labeled predecessor-codec-proxy")

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_value(cls, value: Mapping[str, object] | CodecProxyConfig) -> CodecProxyConfig:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("codec_proxy must be a mapping")
        if not value:
            raise ValueError("predecessor codec proxy configuration is mandatory")
        return cls(**dict(value))


def _backbone_text_config(raw: Mapping[str, object]) -> Mapping[str, object]:
    model_type = raw.get("model_type")
    if model_type == "qwen3_5_moe_text":
        return raw
    if model_type == "qwen3_5_moe":
        text_config = raw.get("text_config")
        if not isinstance(text_config, Mapping):
            raise ValueError("qwen3_5_moe backbone requires text_config")
        return text_config
    raise ValueError("backbone model_type must be Qwen3.5 MoE")


def _validate_layer_pattern(text_config: Mapping[str, object]) -> None:
    layer_types = text_config.get("layer_types")
    if layer_types is None:
        return
    if not isinstance(layer_types, (list, tuple)) or not layer_types:
        raise ValueError("backbone layer_types must be a non-empty sequence")
    allowed = {"linear_attention", "full_attention"}
    if any(layer_type not in allowed for layer_type in layer_types):
        raise ValueError("backbone contains an unsupported layer type")
    expected = tuple(
        "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
        for index in range(len(layer_types))
    )
    if tuple(layer_types) != expected:
        raise ValueError("backbone layer_types must follow the public 3:1 pattern")


def qwen35_inspired_manifest(
    *,
    audio_config: Qwen35AuTConfig,
    aria_config: AriaConfig,
    timestamp_format: str,
) -> ProfileManifest:
    assumptions = (
        f"AuT hidden_size={audio_config.hidden_size}",
        f"AuT encoder_layers={audio_config.encoder_layers}",
        f"AuT attention_heads={audio_config.attention_heads}",
        f"AuT intermediate_size={audio_config.intermediate_size}",
        "AuT Transformer is offline-only; only the Mel/Conv frontend is chunkable",
        f"timestamp_format={timestamp_format}",
        "ARIA speech:text rate is a prototype integer target "
        f"{aria_config.speech_tokens_per_text_num}:"
        f"{aria_config.speech_tokens_per_text_den}",
        PREDECESSOR_CODEC_PROVENANCE,
        "Talker dimensions and training recipe are paper-inspired assumptions",
        "no official Qwen3.5-Omni checkpoint or quality compatibility claim",
    )
    return ProfileManifest(
        architecture_profile=ArchitectureProfile.QWEN35_OMNI_INSPIRED,
        compatibility_level=CompatibilityLevel.PAPER_INSPIRED,
        sources={
            "backbone": SourceRevision(
                name=QWEN35_BACKBONE_MODEL_ID,
                revision=QWEN35_BACKBONE_REVISION,
            ),
            "transformers": SourceRevision(
                name="transformers",
                revision=QWEN35_TRANSFORMERS_VERSION,
            ),
            "codec": SourceRevision(
                name=PREDECESSOR_CODEC_MODEL_ID,
                revision=PREDECESSOR_CODEC_REVISION,
            ),
        },
        assumptions=assumptions,
        exact_official_checkpoint_compatible=False,
        validated_context_length=0,
    )


class Qwen35InspiredConfig(PretrainedConfig):
    model_type = "qwen35_omni_inspired"
    architecture_profile = "qwen35_omni_inspired"

    def __init__(
        self,
        *,
        backbone_config: Mapping[str, object],
        audio_config: Mapping[str, object] | Qwen35AuTConfig,
        aria_config: Mapping[str, object] | AriaConfig,
        codec_proxy: Mapping[str, object] | CodecProxyConfig,
        source_revision: str,
        timestamp_format: str = "[{seconds:.2f}s]",
        profile_manifest: Mapping[str, object] | ProfileManifest | None = None,
        architecture_profile: str = "qwen35_omni_inspired",
        **kwargs: object,
    ) -> None:
        if architecture_profile != self.architecture_profile:
            raise ValueError("architecture_profile must be qwen35_omni_inspired")
        if source_revision != QWEN35_BACKBONE_REVISION:
            raise ValueError("source_revision must match the pinned Qwen3.5 backbone")
        if not isinstance(backbone_config, Mapping):
            raise TypeError("backbone_config must be a mapping")
        copied_backbone = dict(backbone_config)
        text_config = _backbone_text_config(copied_backbone)
        _validate_layer_pattern(text_config)
        if not isinstance(timestamp_format, str) or "{seconds" not in timestamp_format:
            raise ValueError("timestamp_format must contain {seconds")
        try:
            timestamp_format.format(seconds=0.0)
        except (KeyError, ValueError, IndexError) as exc:
            raise ValueError("timestamp_format is invalid") from exc

        audio = Qwen35AuTConfig.from_value(audio_config)
        aria = AriaConfig.from_value(aria_config)
        codec = CodecProxyConfig.from_value(codec_proxy)
        canonical_manifest = qwen35_inspired_manifest(
            audio_config=audio,
            aria_config=aria,
            timestamp_format=timestamp_format,
        )
        if profile_manifest is not None:
            supplied = (
                profile_manifest
                if isinstance(profile_manifest, ProfileManifest)
                else ProfileManifest.from_dict(profile_manifest)
            )
            if supplied != canonical_manifest:
                raise ValueError(
                    "profile checkpoint manifest contradicts the pinned "
                    "Qwen3.5-inspired contract"
                )

        super().__init__(**kwargs)
        self.backbone_config = MappingProxyType(copied_backbone)
        self.audio_config = audio
        self.aria_config = aria
        self.codec_proxy = codec
        self.source_revision = source_revision
        self.timestamp_format = timestamp_format
        self.profile_manifest = canonical_manifest

    @property
    def backbone_text_config(self) -> Mapping[str, object]:
        return MappingProxyType(dict(_backbone_text_config(self.backbone_config)))

    def to_dict(self) -> dict[str, object]:
        serialization_copy = copy(self)
        serialization_copy.backbone_config = dict(self.backbone_config)
        serialization_copy.audio_config = self.audio_config.to_dict()
        serialization_copy.aria_config = self.aria_config.to_dict()
        serialization_copy.codec_proxy = self.codec_proxy.to_dict()
        serialization_copy.profile_manifest = self.profile_manifest.to_dict()
        output = PretrainedConfig.to_dict(serialization_copy)
        output.update(
            {
                "model_type": self.model_type,
                "architecture_profile": self.architecture_profile,
                "backbone_config": dict(self.backbone_config),
                "audio_config": self.audio_config.to_dict(),
                "aria_config": self.aria_config.to_dict(),
                "codec_proxy": self.codec_proxy.to_dict(),
                "source_revision": self.source_revision,
                "timestamp_format": self.timestamp_format,
                "profile_manifest": self.profile_manifest.to_dict(),
            }
        )
        return output


__all__ = [
    "AriaConfig",
    "CodecProxyConfig",
    "PREDECESSOR_CODEC_MODEL_ID",
    "PREDECESSOR_CODEC_PROVENANCE",
    "PREDECESSOR_CODEC_REVISION",
    "QWEN35_BACKBONE_MODEL_ID",
    "QWEN35_BACKBONE_REVISION",
    "QWEN35_TRANSFORMERS_VERSION",
    "Qwen35AuTConfig",
    "Qwen35InspiredConfig",
    "qwen35_inspired_manifest",
]
