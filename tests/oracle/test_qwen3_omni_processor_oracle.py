from __future__ import annotations

import json
from pathlib import Path
import socket

import numpy as np
from PIL import Image
import pytest
import torch

from qwen3_omni_pretrain.profiles.qwen3_omni_reference.oracle import (
    QWEN3_OMNI_MODEL_ID,
    QWEN3_OMNI_REVISION,
    load_reference_processor,
)


pytestmark = pytest.mark.reference

PROCESSOR_CONTRACT_PATH = Path(
    "tests/fixtures/qwen3_omni/processor_contract.json"
)


def _expand_rle(runs):
    return [
        value
        for value, count in runs
        for _ in range(count)
    ]


def test_processor_is_offline_and_preserves_media_sequence_contract(
    monkeypatch,
):
    contract = json.loads(
        PROCESSOR_CONTRACT_PATH.read_text(encoding="utf-8")
    )
    assert contract["source"]["model_id"] == QWEN3_OMNI_MODEL_ID
    assert contract["source"]["revision"] == QWEN3_OMNI_REVISION
    assert contract["source"]["transformers_version"] == "5.2.0"
    assert "local_files_only=True" in contract["source"]["extraction"]
    assert contract["source"]["artifact_sha256"] == {
        "chat_template.json": (
            "90c1b81f29e41b7642b0cc02c877a10c8bf6751a8d8fa1d16ac9a718cf1c3d86"
        ),
        "config.json": (
            "eab5093d47807aaf894119506b238b2b1cee70d08456e894fee9a012d88f2e0d"
        ),
        "merges.txt": (
            "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3"
        ),
        "preprocessor_config.json": (
            "b10e27fd4542cf89ec7145942b87f3e65408d4e9f9d031a29acdd293c15fb3fc"
        ),
        "tokenizer_config.json": (
            "dc3c31c3bdaedd5016382bb3cbe07323026775ad51f5a4fb564505992ae4a670"
        ),
        "vocab.json": (
            "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"
        ),
    }

    def forbid_network(self, address):
        raise AssertionError(f"network access attempted: {address!r}")

    monkeypatch.setattr(socket.socket, "connect", forbid_network)
    try:
        processor = load_reference_processor(local_files_only=True)
    except OSError as exc:
        pytest.skip(f"pinned processor artifacts unavailable locally: {exc}")

    waveform = np.linspace(-0.25, 0.25, 16_000, dtype=np.float32)
    pixels = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
    image = Image.fromarray(pixels, mode="RGB")
    output = processor(
        text=(
            "<|vision_start|><|image_pad|><|vision_end|>"
            "<|audio_start|><|audio_pad|><|audio_end|>"
        ),
        images=image,
        audio=waveform,
        return_tensors="pt",
    )

    input_ids = output["input_ids"]
    attention_mask = output["attention_mask"]
    audio_mask = output["feature_attention_mask"]
    image_grid = output["image_grid_thw"]
    expected = contract["expected"]

    assert input_ids.dtype == getattr(torch, expected["input_ids"]["dtype"])
    assert list(input_ids.shape) == [
        len(expected["input_ids"]["values"]),
        len(expected["input_ids"]["values"][0]),
    ]
    assert input_ids.tolist() == expected["input_ids"]["values"]
    assert list(attention_mask.shape) == expected["attention_mask"]["shape"]
    assert str(attention_mask.dtype).removeprefix("torch.") == (
        expected["attention_mask"]["dtype"]
    )
    assert attention_mask.tolist() == [
        _expand_rle(expected["attention_mask"]["rle"])
    ]
    assert list(audio_mask.shape) == expected["feature_attention_mask"]["shape"]
    assert str(audio_mask.dtype).removeprefix("torch.") == (
        expected["feature_attention_mask"]["dtype"]
    )
    assert audio_mask.tolist() == [
        _expand_rle(expected["feature_attention_mask"]["rle"])
    ]
    assert int(audio_mask.sum()) == expected["feature_attention_mask"]["sum"]
    assert list(output["input_features"].shape) == (
        expected["input_features"]["shape"]
    )
    assert str(output["input_features"].dtype).removeprefix("torch.") == (
        expected["input_features"]["dtype"]
    )
    assert str(image_grid.dtype).removeprefix("torch.") == (
        expected["image_grid_thw"]["dtype"]
    )
    assert image_grid.tolist() == expected["image_grid_thw"]["values"]

    token_values = input_ids[0].tolist()
    for sentinel, details in expected["sentinels"].items():
        positions = [
            index
            for index, token_id in enumerate(token_values)
            if token_id == details["token_id"]
        ]
        assert positions == details["positions"], sentinel
