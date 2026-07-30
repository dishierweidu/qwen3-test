from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping

from qwen3_omni_pretrain.profiles.qwen3_omni_reference.oracle import (
    QWEN3_OMNI_MODEL_ID,
    QWEN3_OMNI_REVISION,
    QWEN3_TRANSFORMERS_VERSION,
    load_reference_config,
)


REGULAR_VOCAB_SIZE = 151_643
OUTPUT_SAMPLE_RATE_HZ = 24_000


def extract_contract(
    config: object,
    *,
    preprocessor_config: Mapping[str, object],
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

    return {
        "schema_version": 1,
        "source": {
            "model_id": QWEN3_OMNI_MODEL_ID,
            "revision": QWEN3_OMNI_REVISION,
            "transformers_version": QWEN3_TRANSFORMERS_VERSION,
            "config_files": ["config.json", "preprocessor_config.json"],
            "derived_fields": {
                "audio_conv_kernel_and_stride": "transformers==5.2.0",
                "output_sample_rate_hz": "Qwen3-Omni Technical Report",
                "regular_vocab_size": "Qwen3-Omni Technical Report",
            },
        },
        "vocabulary": {
            "regular_vocab_size": REGULAR_VOCAB_SIZE,
            "embedding_vocab_size": thinker.vocab_size,
        },
        "audio_encoder": {
            "input_sample_rate_hz": int(preprocessor_config["sampling_rate"]),
            "num_mel_bins": audio.num_mel_bins,
            "conv_layers": 3,
            "conv_hidden_size": audio.downsample_hidden_size,
            "conv_kernel_size": 3,
            "conv_stride": 2,
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
            "rope_theta": thinker.rope_theta,
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
            "output_sample_rate_hz": OUTPUT_SAMPLE_RATE_HZ,
        },
    }


def _load_preprocessor_config(*, local_files_only: bool) -> dict[str, object]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=QWEN3_OMNI_MODEL_ID,
        filename="preprocessor_config.json",
        revision=QWEN3_OMNI_REVISION,
        local_files_only=local_files_only,
    )
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract the pinned, small Qwen3-Omni config contract",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Allow a pinned config-only fetch; never downloads model weights",
    )
    args = parser.parse_args()
    local_files_only = not args.allow_network
    contract = extract_contract(
        load_reference_config(local_files_only=local_files_only),
        preprocessor_config=_load_preprocessor_config(
            local_files_only=local_files_only,
        ),
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
