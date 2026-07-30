from __future__ import annotations

import ast
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
import inspect
import math
import shutil
import textwrap
from types import MappingProxyType
from typing import Any, Iterator, Mapping

import transformers

from qwen3_omni_pretrain.architecture.manifest import (
    ProfileManifest,
    SourceRevision,
)
from qwen3_omni_pretrain.architecture.profiles import (
    ArchitectureProfile,
    CompatibilityLevel,
)
from qwen3_omni_pretrain.profiles.qwen3_omni_reference.pins import (
    QWEN3_OMNI_MODEL_ID,
    QWEN3_OMNI_REVISION,
    QWEN3_TRANSFORMERS_VERSION,
    REFERENCE_DISTRIBUTION_VERSIONS,
    distribution_version_matches,
)


@dataclass(frozen=True)
class _ContractField:
    config_paths: tuple[str, ...]
    expected: object
    aggregation: str = "scalar"


def _field(config_path: str, expected: object) -> _ContractField:
    return _ContractField((config_path,), expected)


_CONFIG_CONTRACT_FIELDS: Mapping[str, _ContractField] = MappingProxyType(
    {
        "model_type": _field("model_type", "qwen3_omni_moe"),
        "vocabulary.embedding_vocab_size": _field(
            "thinker_config.text_config.vocab_size",
            152_064,
        ),
        "audio_encoder.num_mel_bins": _field(
            "thinker_config.audio_config.num_mel_bins",
            128,
        ),
        "audio_encoder.conv_hidden_size": _field(
            "thinker_config.audio_config.downsample_hidden_size",
            480,
        ),
        "audio_encoder.num_hidden_layers": _field(
            "thinker_config.audio_config.encoder_layers",
            32,
        ),
        "audio_encoder.num_attention_heads": _field(
            "thinker_config.audio_config.encoder_attention_heads",
            20,
        ),
        "audio_encoder.hidden_size": _field(
            "thinker_config.audio_config.d_model",
            1280,
        ),
        "audio_encoder.intermediate_size": _field(
            "thinker_config.audio_config.encoder_ffn_dim",
            5120,
        ),
        "audio_encoder.projector_dimensions[0]": _field(
            "thinker_config.audio_config.d_model",
            1280,
        ),
        "audio_encoder.projector_dimensions[1]": _field(
            "thinker_config.audio_config.d_model",
            1280,
        ),
        "audio_encoder.projector_dimensions[2]": _field(
            "thinker_config.audio_config.output_dim",
            2048,
        ),
        "vision_encoder.patch_kernel[0]": _field(
            "thinker_config.vision_config.temporal_patch_size",
            2,
        ),
        "vision_encoder.patch_kernel[1]": _field(
            "thinker_config.vision_config.patch_size",
            16,
        ),
        "vision_encoder.patch_kernel[2]": _field(
            "thinker_config.vision_config.patch_size",
            16,
        ),
        "vision_encoder.spatial_merge_size": _field(
            "thinker_config.vision_config.spatial_merge_size",
            2,
        ),
        "vision_encoder.deepstack_visual_indexes": _field(
            "thinker_config.vision_config.deepstack_visual_indexes",
            (8, 16, 24),
        ),
        "vision_encoder.num_hidden_layers": _field(
            "thinker_config.vision_config.depth",
            27,
        ),
        "vision_encoder.num_attention_heads": _field(
            "thinker_config.vision_config.num_heads",
            16,
        ),
        "vision_encoder.hidden_size": _field(
            "thinker_config.vision_config.hidden_size",
            1152,
        ),
        "vision_encoder.intermediate_size": _field(
            "thinker_config.vision_config.intermediate_size",
            4304,
        ),
        "vision_encoder.merger_dimensions[0]": _ContractField(
            (
                "thinker_config.vision_config.hidden_size",
                "thinker_config.vision_config.spatial_merge_size",
            ),
            4608,
            "merged_vision_hidden",
        ),
        "vision_encoder.merger_dimensions[1]": _ContractField(
            (
                "thinker_config.vision_config.hidden_size",
                "thinker_config.vision_config.spatial_merge_size",
            ),
            4608,
            "merged_vision_hidden",
        ),
        "vision_encoder.merger_dimensions[2]": _field(
            "thinker_config.vision_config.out_hidden_size",
            2048,
        ),
        "tm_rope.mrope_section": _field(
            "thinker_config.text_config.rope_scaling.mrope_section",
            (24, 20, 20),
        ),
        "tm_rope.rope_theta": _field(
            "thinker_config.text_config.rope_scaling.rope_theta",
            1_000_000,
        ),
        "tm_rope.interleaved": _field(
            "thinker_config.text_config.rope_scaling.interleaved",
            True,
        ),
        "thinker.hidden_size": _field(
            "thinker_config.text_config.hidden_size",
            2048,
        ),
        "thinker.num_hidden_layers": _field(
            "thinker_config.text_config.num_hidden_layers",
            48,
        ),
        "thinker.num_attention_heads": _field(
            "thinker_config.text_config.num_attention_heads",
            32,
        ),
        "thinker.num_key_value_heads": _field(
            "thinker_config.text_config.num_key_value_heads",
            4,
        ),
        "thinker.num_experts": _field(
            "thinker_config.text_config.num_experts",
            128,
        ),
        "thinker.num_experts_per_tok": _field(
            "thinker_config.text_config.num_experts_per_tok",
            8,
        ),
        "thinker.moe_intermediate_size": _field(
            "thinker_config.text_config.moe_intermediate_size",
            768,
        ),
        "talker.hidden_size": _field(
            "talker_config.text_config.hidden_size",
            1024,
        ),
        "talker.num_hidden_layers": _field(
            "talker_config.text_config.num_hidden_layers",
            20,
        ),
        "talker.num_attention_heads": _field(
            "talker_config.text_config.num_attention_heads",
            16,
        ),
        "talker.num_key_value_heads": _field(
            "talker_config.text_config.num_key_value_heads",
            2,
        ),
        "talker.num_experts": _field(
            "talker_config.text_config.num_experts",
            128,
        ),
        "talker.num_experts_per_tok": _field(
            "talker_config.text_config.num_experts_per_tok",
            6,
        ),
        "talker.moe_intermediate_size": _field(
            "talker_config.text_config.moe_intermediate_size",
            384,
        ),
        "talker.shared_expert_intermediate_size": _field(
            "talker_config.text_config.shared_expert_intermediate_size",
            768,
        ),
        "code_predictor.hidden_size": _field(
            "talker_config.code_predictor_config.hidden_size",
            1024,
        ),
        "code_predictor.num_hidden_layers": _field(
            "talker_config.code_predictor_config.num_hidden_layers",
            5,
        ),
        "code_predictor.num_attention_heads": _field(
            "talker_config.code_predictor_config.num_attention_heads",
            16,
        ),
        "code_predictor.num_key_value_heads": _field(
            "talker_config.code_predictor_config.num_key_value_heads",
            8,
        ),
        "code_predictor.intermediate_size": _field(
            "talker_config.code_predictor_config.intermediate_size",
            3072,
        ),
        "code_predictor.num_code_groups": _field(
            "talker_config.code_predictor_config.num_code_groups",
            16,
        ),
        "code_predictor.vocab_size": _field(
            "talker_config.code_predictor_config.vocab_size",
            2048,
        ),
        "code2wav.codebook_size": _field(
            "code2wav_config.codebook_size",
            2048,
        ),
        "code2wav.num_quantizers": _field(
            "code2wav_config.num_quantizers",
            16,
        ),
        "code2wav.num_semantic_quantizers": _field(
            "code2wav_config.num_semantic_quantizers",
            1,
        ),
        "code2wav.num_hidden_layers": _field(
            "code2wav_config.num_hidden_layers",
            8,
        ),
        "code2wav.hidden_size": _field(
            "code2wav_config.hidden_size",
            1024,
        ),
        "code2wav.num_attention_heads": _field(
            "code2wav_config.num_attention_heads",
            16,
        ),
        "code2wav.sliding_window": _field(
            "code2wav_config.sliding_window",
            72,
        ),
        "code2wav.upsample_rates": _field(
            "code2wav_config.upsample_rates",
            (8, 5, 4, 3),
        ),
        "code2wav.upsampling_ratios": _field(
            "code2wav_config.upsampling_ratios",
            (2, 2),
        ),
        "code2wav.samples_per_code": _ContractField(
            (
                "code2wav_config.upsample_rates",
                "code2wav_config.upsampling_ratios",
            ),
            1920,
            "product",
        ),
    }
)
_AUDIO_IMPLEMENTATION_CONTRACT: Mapping[str, int] = MappingProxyType(
    {
        "audio_encoder.conv_layers": 3,
        "audio_encoder.conv_kernel_size": 3,
        "audio_encoder.conv_stride": 2,
    }
)


def qwen3_reference_manifest() -> ProfileManifest:
    return ProfileManifest(
        architecture_profile=ArchitectureProfile.QWEN3_OMNI_REFERENCE,
        compatibility_level=CompatibilityLevel.STRUCTURE_ALIGNED,
        sources={
            "model": SourceRevision(
                name=QWEN3_OMNI_MODEL_ID,
                revision=QWEN3_OMNI_REVISION,
            ),
            "transformers": SourceRevision(
                name="transformers",
                revision=QWEN3_TRANSFORMERS_VERSION,
            ),
            "qwen_omni_utils": SourceRevision(
                name="qwen-omni-utils",
                revision=REFERENCE_DISTRIBUTION_VERSIONS["qwen-omni-utils"],
            ),
        },
        assumptions=(
            "config and processor structure captured; model state unverified",
            "numerical, cache, offline-text, and offline-speech evidence pending",
        ),
        exact_official_checkpoint_compatible=False,
        validated_context_length=0,
    )


def _require_reference_environment() -> None:
    if transformers.__version__ != QWEN3_TRANSFORMERS_VERSION:
        raise RuntimeError(
            "qwen3_omni_reference requires transformers==5.2.0"
        )

    mismatches: list[str] = []
    for distribution, expected in REFERENCE_DISTRIBUTION_VERSIONS.items():
        try:
            actual = version(distribution)
        except PackageNotFoundError:
            mismatches.append(f"{distribution} is not installed")
            continue
        if not distribution_version_matches(distribution, actual, expected):
            mismatches.append(
                f"{distribution}=={actual}, expected {distribution}=={expected}"
            )
    if mismatches:
        raise RuntimeError(
            "qwen3_omni_reference environment mismatch: "
            + "; ".join(mismatches)
        )
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("qwen3_omni_reference requires ffmpeg in PATH")


def _read_config_path(config: object, path: str) -> object:
    value = config
    for component in path.split("."):
        if isinstance(value, Mapping):
            if component not in value:
                raise ValueError(f"reference config is missing {path}")
            value = value[component]
        else:
            if not hasattr(value, component):
                raise ValueError(f"reference config is missing {path}")
            value = getattr(value, component)
    return value


def _read_contract_field(
    config: object,
    field: _ContractField,
) -> object:
    values = tuple(
        _read_config_path(config, path)
        for path in field.config_paths
    )
    if field.aggregation == "scalar":
        actual = values[0]
    elif field.aggregation == "merged_vision_hidden":
        actual = values[0] * values[1] ** 2
    elif field.aggregation == "product":
        actual = math.prod(values[0]) * math.prod(values[1])
    else:
        raise AssertionError(
            f"unknown contract aggregation {field.aggregation!r}"
        )
    if isinstance(field.expected, tuple) and not isinstance(actual, tuple):
        actual = tuple(actual)
    return actual


def _validate_reference_config(config: object) -> None:
    mismatches = []
    for contract_path, field in _CONFIG_CONTRACT_FIELDS.items():
        try:
            actual = _read_contract_field(config, field)
        except ValueError:
            sources = ", ".join(field.config_paths)
            mismatches.append(
                f"{contract_path}=<missing from {sources}>, "
                f"expected {field.expected!r}"
            )
            continue
        if actual != field.expected:
            sources = ", ".join(field.config_paths)
            mismatches.append(
                f"{contract_path} from {sources}={actual!r}, "
                f"expected {field.expected!r}"
            )
    if mismatches:
        raise ValueError(
            "pinned Qwen3-Omni config contradiction: "
            + "; ".join(mismatches)
        )


def _reference_config_class() -> type:
    from transformers import Qwen3OmniMoeConfig

    return Qwen3OmniMoeConfig


def _reference_processor_class() -> type:
    from transformers import Qwen3OmniMoeProcessor

    return Qwen3OmniMoeProcessor


def _reference_model_class() -> type:
    from transformers import Qwen3OmniMoeForConditionalGeneration

    return Qwen3OmniMoeForConditionalGeneration


def _reference_audio_encoder_class() -> type:
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeAudioEncoder,
    )

    return Qwen3OmniMoeAudioEncoder


@contextmanager
def _scoped_hub_offline_mode(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return

    import huggingface_hub.constants as hub_constants

    previous = hub_constants.HF_HUB_OFFLINE
    hub_constants.HF_HUB_OFFLINE = True
    try:
        yield
    finally:
        hub_constants.HF_HUB_OFFLINE = previous


def _literal_call_argument(
    call: ast.Call,
    position: int,
    keyword: str,
) -> object:
    if len(call.args) > position:
        return ast.literal_eval(call.args[position])
    for item in call.keywords:
        if item.arg == keyword:
            return ast.literal_eval(item.value)
    raise ValueError(f"Conv2d is missing {keyword}")


def _extract_audio_implementation_contract(
    audio_encoder_class: type,
) -> dict[str, object]:
    source = textwrap.dedent(
        inspect.getsource(audio_encoder_class.__init__)
    )
    tree = ast.parse(source)
    convolutions: dict[str, tuple[object, object]] = {}
    expected_names = {"conv2d1", "conv2d2", "conv2d3"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        call = node.value
        if (
            not isinstance(target, ast.Attribute)
            or not isinstance(target.value, ast.Name)
            or target.value.id != "self"
            or target.attr not in expected_names
            or not isinstance(call, ast.Call)
            or not isinstance(call.func, ast.Attribute)
            or call.func.attr != "Conv2d"
        ):
            continue
        convolutions[target.attr] = (
            _literal_call_argument(call, 2, "kernel_size"),
            _literal_call_argument(call, 3, "stride"),
        )

    kernels = tuple(
        convolutions[name][0]
        for name in sorted(convolutions)
    )
    strides = tuple(
        convolutions[name][1]
        for name in sorted(convolutions)
    )
    return {
        "audio_encoder.conv_layers": len(convolutions),
        "audio_encoder.conv_kernel_size": (
            kernels[0] if kernels and len(set(kernels)) == 1 else kernels
        ),
        "audio_encoder.conv_stride": (
            strides[0] if strides and len(set(strides)) == 1 else strides
        ),
    }


def _validate_reference_audio_implementation() -> None:
    try:
        actual = _extract_audio_implementation_contract(
            _reference_audio_encoder_class()
        )
    except (OSError, TypeError, SyntaxError, ValueError) as exc:
        raise ValueError(
            "pinned Qwen3-Omni implementation contract is unreadable"
        ) from exc

    mismatches = []
    for contract_path, expected in _AUDIO_IMPLEMENTATION_CONTRACT.items():
        observed = actual.get(contract_path, "<missing>")
        if observed != expected:
            mismatches.append(
                f"{contract_path}={observed!r}, expected {expected!r}"
            )
    if mismatches:
        raise ValueError(
            "pinned Qwen3-Omni implementation contradiction: "
            + "; ".join(mismatches)
        )


def _validate_reference_contract(config: object) -> None:
    _validate_reference_config(config)
    _validate_reference_audio_implementation()


def load_reference_config(
    source: str = QWEN3_OMNI_MODEL_ID,
    *,
    local_files_only: bool = True,
) -> Any:
    _require_reference_environment()
    config = _reference_config_class().from_pretrained(
        source,
        revision=QWEN3_OMNI_REVISION,
        local_files_only=local_files_only,
    )
    _validate_reference_contract(config)
    return config


def load_reference_processor(
    source: str = QWEN3_OMNI_MODEL_ID,
    *,
    local_files_only: bool = True,
) -> Any:
    _require_reference_environment()
    with _scoped_hub_offline_mode(local_files_only):
        return _reference_processor_class().from_pretrained(
            source,
            revision=QWEN3_OMNI_REVISION,
            local_files_only=local_files_only,
        )


def load_reference_model(
    source: str = QWEN3_OMNI_MODEL_ID,
    *,
    torch_dtype: object,
    device_map: object,
    local_files_only: bool = True,
) -> Any:
    _require_reference_environment()
    config = load_reference_config(
        source,
        local_files_only=local_files_only,
    )
    _validate_reference_contract(config)
    return _reference_model_class().from_pretrained(
        source,
        revision=QWEN3_OMNI_REVISION,
        local_files_only=local_files_only,
        torch_dtype=torch_dtype,
        device_map=device_map,
        config=config,
    )
