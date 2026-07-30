from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Mapping

from qwen3_omni_pretrain.profiles.qwen3_omni_reference.oracle import (
    QWEN3_OMNI_MODEL_ID,
    QWEN3_OMNI_REVISION,
    QWEN3_TRANSFORMERS_VERSION,
    _extract_audio_implementation_contract,
    _reference_audio_encoder_class,
    load_reference_config,
)


PINNED_ARTIFACT_SHA256 = {
    "README.md": (
        "0e44065c4c4a27071f7239afd5b5a33af5bc2e437dd7ea9950e51aafabfde3df"
    ),
    "config.json": (
        "eab5093d47807aaf894119506b238b2b1cee70d08456e894fee9a012d88f2e0d"
    ),
    "preprocessor_config.json": (
        "b10e27fd4542cf89ec7145942b87f3e65408d4e9f9d031a29acdd293c15fb3fc"
    ),
    "vocab.json": (
        "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"
    ),
}


def _regular_vocab_size(tokenizer_vocab: Mapping[str, int]) -> int:
    size = len(tokenizer_vocab)
    ids = set(tokenizer_vocab.values())
    if ids != set(range(size)):
        raise ValueError(
            "pinned tokenizer vocabulary IDs must be contiguous from zero"
        )
    return size


def _output_sample_rate_hz(model_readme: str) -> int:
    values = {
        int(match.replace("_", ""))
        for match in re.findall(
            r"\bsamplerate\s*=\s*([0-9][0-9_]*)",
            model_readme,
        )
    }
    if len(values) != 1:
        raise ValueError(
            "pinned README must contain one unambiguous samplerate value"
        )
    return values.pop()


def extract_contract(
    config: object,
    *,
    preprocessor_config: Mapping[str, object],
    tokenizer_vocab: Mapping[str, int],
    model_readme: str,
    audio_implementation_contract: Mapping[str, object],
) -> dict[str, object]:
    audio = config.thinker_config.audio_config
    vision = config.thinker_config.vision_config
    thinker = config.thinker_config.text_config
    talker = config.talker_config.text_config
    code_predictor = config.talker_config.code_predictor_config
    code2wav = config.code2wav_config

    samples_per_code = math.prod(code2wav.upsample_rates) * math.prod(
        code2wav.upsampling_ratios
    )
    merger_hidden = vision.hidden_size * vision.spatial_merge_size**2
    regular_vocab_size = _regular_vocab_size(tokenizer_vocab)
    output_sample_rate_hz = _output_sample_rate_hz(model_readme)

    return {
        "schema_version": 1,
        "source": {
            "model_id": QWEN3_OMNI_MODEL_ID,
            "revision": QWEN3_OMNI_REVISION,
            "transformers_version": QWEN3_TRANSFORMERS_VERSION,
            "artifacts": {
                name: {
                    "revision": QWEN3_OMNI_REVISION,
                    "sha256": digest,
                }
                for name, digest in PINNED_ARTIFACT_SHA256.items()
            },
            "derived_fields": {
                "audio_conv_kernel_and_stride": {
                    "artifact": (
                        "src/transformers/models/qwen3_omni_moe/"
                        "modeling_qwen3_omni_moe.py"
                    ),
                    "revision": (
                        "7d9754a05193eb79b1d86aa744b622b8068008cd"
                    ),
                    "sha256": (
                        "0b6e9a6e9d88814de3e25b1ca65c49d0677be7331b8118b6a0cd022c6c1dd270"
                    ),
                    "extraction": (
                        "AST of Qwen3OmniMoeAudioEncoder.__init__ "
                        "conv2d1/2/3 assignments"
                    ),
                },
                "output_sample_rate_hz": {
                    "artifact": "README.md",
                    "revision": QWEN3_OMNI_REVISION,
                    "sha256": PINNED_ARTIFACT_SHA256["README.md"],
                    "url": (
                        "https://huggingface.co/"
                        f"{QWEN3_OMNI_MODEL_ID}/resolve/"
                        f"{QWEN3_OMNI_REVISION}/README.md"
                    ),
                    "locator": (
                        "lines 257-277; samplerate=24000 at line 277"
                    ),
                    "extraction": (
                        "samplerate used to serialize audio returned by "
                        "model.generate"
                    ),
                },
                "regular_vocab_size": {
                    "artifact": "vocab.json",
                    "revision": QWEN3_OMNI_REVISION,
                    "sha256": PINNED_ARTIFACT_SHA256["vocab.json"],
                    "url": (
                        "https://huggingface.co/"
                        f"{QWEN3_OMNI_MODEL_ID}/resolve/"
                        f"{QWEN3_OMNI_REVISION}/vocab.json"
                    ),
                    "extraction": "len(top-level token-to-id mapping)",
                    "validation": "IDs are contiguous from 0 through 151642",
                },
            },
        },
        "vocabulary": {
            "regular_vocab_size": regular_vocab_size,
            "embedding_vocab_size": thinker.vocab_size,
        },
        "audio_encoder": {
            "input_sample_rate_hz": int(preprocessor_config["sampling_rate"]),
            "num_mel_bins": audio.num_mel_bins,
            "conv_layers": audio_implementation_contract[
                "audio_encoder.conv_layers"
            ],
            "conv_hidden_size": audio.downsample_hidden_size,
            "conv_kernel_size": audio_implementation_contract[
                "audio_encoder.conv_kernel_size"
            ],
            "conv_stride": audio_implementation_contract[
                "audio_encoder.conv_stride"
            ],
            "num_hidden_layers": audio.encoder_layers,
            "num_attention_heads": audio.encoder_attention_heads,
            "hidden_size": audio.d_model,
            "intermediate_size": audio.encoder_ffn_dim,
            "projector_dimensions": [
                audio.d_model,
                audio.d_model,
                audio.output_dim,
            ],
        },
        "vision_encoder": {
            "patch_kernel": [
                vision.temporal_patch_size,
                vision.patch_size,
                vision.patch_size,
            ],
            "spatial_merge_size": vision.spatial_merge_size,
            "deepstack_visual_indexes": list(
                vision.deepstack_visual_indexes
            ),
            "num_hidden_layers": vision.depth,
            "num_attention_heads": vision.num_heads,
            "hidden_size": vision.hidden_size,
            "intermediate_size": vision.intermediate_size,
            "merger_dimensions": [
                merger_hidden,
                merger_hidden,
                vision.out_hidden_size,
            ],
        },
        "tm_rope": {
            "mrope_section": list(thinker.rope_scaling["mrope_section"]),
            "rope_theta": thinker.rope_scaling["rope_theta"],
            "interleaved": thinker.rope_scaling["interleaved"],
        },
        "thinker": {
            "hidden_size": thinker.hidden_size,
            "num_hidden_layers": thinker.num_hidden_layers,
            "num_attention_heads": thinker.num_attention_heads,
            "num_key_value_heads": thinker.num_key_value_heads,
            "num_experts": thinker.num_experts,
            "num_experts_per_tok": thinker.num_experts_per_tok,
            "moe_intermediate_size": thinker.moe_intermediate_size,
        },
        "talker": {
            "hidden_size": talker.hidden_size,
            "num_hidden_layers": talker.num_hidden_layers,
            "num_attention_heads": talker.num_attention_heads,
            "num_key_value_heads": talker.num_key_value_heads,
            "num_experts": talker.num_experts,
            "num_experts_per_tok": talker.num_experts_per_tok,
            "moe_intermediate_size": talker.moe_intermediate_size,
            "shared_expert_intermediate_size": (
                talker.shared_expert_intermediate_size
            ),
        },
        "code_predictor": {
            "hidden_size": code_predictor.hidden_size,
            "num_hidden_layers": code_predictor.num_hidden_layers,
            "num_attention_heads": code_predictor.num_attention_heads,
            "num_key_value_heads": code_predictor.num_key_value_heads,
            "intermediate_size": code_predictor.intermediate_size,
            "num_code_groups": code_predictor.num_code_groups,
            "vocab_size": code_predictor.vocab_size,
        },
        "code2wav": {
            "codebook_size": code2wav.codebook_size,
            "num_quantizers": code2wav.num_quantizers,
            "num_semantic_quantizers": code2wav.num_semantic_quantizers,
            "num_hidden_layers": code2wav.num_hidden_layers,
            "hidden_size": code2wav.hidden_size,
            "num_attention_heads": code2wav.num_attention_heads,
            "sliding_window": code2wav.sliding_window,
            "upsample_rates": list(code2wav.upsample_rates),
            "upsampling_ratios": list(code2wav.upsampling_ratios),
            "samples_per_code": samples_per_code,
            "output_sample_rate_hz": output_sample_rate_hz,
        },
    }


def _load_artifact(
    filename: str,
    *,
    local_files_only: bool,
) -> bytes:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=QWEN3_OMNI_MODEL_ID,
        filename=filename,
        revision=QWEN3_OMNI_REVISION,
        local_files_only=local_files_only,
    )
    payload = Path(path).read_bytes()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    expected_sha256 = PINNED_ARTIFACT_SHA256[filename]
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{filename} sha256={actual_sha256}, expected {expected_sha256}"
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract the pinned, small Qwen3-Omni config contract",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help=(
            "Allow a pinned non-weight metadata fetch; "
            "never downloads model weights"
        ),
    )
    args = parser.parse_args()
    local_files_only = not args.allow_network
    contract = extract_contract(
        load_reference_config(local_files_only=local_files_only),
        preprocessor_config=json.loads(
            _load_artifact(
                "preprocessor_config.json",
                local_files_only=local_files_only,
            )
        ),
        tokenizer_vocab=json.loads(
            _load_artifact(
                "vocab.json",
                local_files_only=local_files_only,
            )
        ),
        model_readme=_load_artifact(
            "README.md",
            local_files_only=local_files_only,
        ).decode("utf-8"),
        audio_implementation_contract=(
            _extract_audio_implementation_contract(
                _reference_audio_encoder_class()
            )
        ),
    )
    _load_artifact(
        "config.json",
        local_files_only=local_files_only,
    )
    payload = json.dumps(
        contract,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    ) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
