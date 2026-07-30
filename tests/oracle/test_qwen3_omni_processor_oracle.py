from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import socket
from types import SimpleNamespace

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

EXPECTED_ROPE_IMPLEMENTATION_SHA256 = (
    "4b0de5c1b83a32c7fbb57b16c5fc0c1c6e5a707585cae9188fc87d7283b44ee3"
)
PROCESSOR_CONTRACT_PATH = Path(
    "tests/fixtures/qwen3_omni/processor_contract.json"
)


def _expand_rle(runs):
    return [
        value
        for value, count in runs
        for _ in range(count)
    ]


def _rope_facade():
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoePreTrainedModelForConditionalGeneration,
    )

    base = Qwen3OmniMoePreTrainedModelForConditionalGeneration

    class RopeFacade:
        spatial_merge_size = 2
        config = SimpleNamespace(
            image_token_id=151_655,
            video_token_id=151_656,
            audio_token_id=151_675,
            vision_start_token_id=151_652,
            audio_start_token_id=151_669,
            position_id_per_seconds=13,
        )
        get_llm_pos_ids_for_vision = base.get_llm_pos_ids_for_vision
        get_rope_index = base.get_rope_index

    return RopeFacade(), base


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


def test_text_rope_positions_are_one_axis_monotonic():
    facade, _ = _rope_facade()
    input_ids = torch.tensor([[11, 12, 13, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.long)

    positions, rope_delta = facade.get_rope_index(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    assert positions.shape == (3, 1, 4)
    assert positions.dtype == torch.float32
    assert torch.equal(
        positions[:, 0, :3],
        torch.tensor(
            [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]
        ),
    )
    assert torch.all(torch.diff(positions[:, 0, :3], dim=-1) >= 0)
    assert torch.all(positions >= 0)
    assert rope_delta.tolist() == [[0.0]]


def test_image_rope_positions_reset_height_and_width_exactly():
    facade, _ = _rope_facade()
    input_ids = torch.tensor(
        [[7, 151_652, 151_655, 151_655, 151_655, 151_655, 151_653, 8]],
        dtype=torch.long,
    )

    positions, rope_delta = facade.get_rope_index(
        input_ids=input_ids,
        image_grid_thw=torch.tensor([[1, 4, 4]], dtype=torch.long),
        attention_mask=torch.ones_like(input_ids),
    )

    assert positions.shape == (3, 1, 8)
    assert positions.dtype == torch.float32
    assert positions[:, 0].tolist() == [
        [0.0, 1.0, 2.0, 2.0, 2.0, 2.0, 4.0, 5.0],
        [0.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 5.0],
        [0.0, 1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 5.0],
    ]
    assert torch.all(positions >= 0)
    assert rope_delta.tolist() == [[-2.0]]


def test_video_rope_positions_use_pinned_time_grid_exactly():
    facade, _ = _rope_facade()
    input_ids = torch.tensor(
        [
            [
                151_652,
                151_656,
                151_656,
                151_656,
                151_656,
                151_656,
                151_656,
                151_656,
                151_656,
                151_653,
            ]
        ],
        dtype=torch.long,
    )

    positions, rope_delta = facade.get_rope_index(
        input_ids=input_ids,
        video_grid_thw=torch.tensor([[2, 4, 4]], dtype=torch.long),
        second_per_grids=torch.tensor([1.0]),
        attention_mask=torch.ones_like(input_ids),
    )

    assert positions.shape == (3, 1, 10)
    assert positions[:, 0].tolist() == [
        [0.0, 1.0, 1.0, 1.0, 1.0, 14.0, 14.0, 14.0, 14.0, 15.0],
        [0.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 15.0],
        [0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 15.0],
    ]
    assert torch.all(positions >= 0)
    assert rope_delta.tolist() == [[6.0]]


def test_audio_rope_positions_use_exact_feature_grid():
    facade, _ = _rope_facade()
    input_ids = torch.tensor(
        [[151_669, 151_675, 151_675, 151_670]],
        dtype=torch.long,
    )

    positions, rope_delta = facade.get_rope_index(
        input_ids=input_ids,
        image_grid_thw=torch.empty((0, 3), dtype=torch.long),
        audio_seqlens=torch.tensor([16], dtype=torch.long),
        attention_mask=torch.ones_like(input_ids),
    )

    assert positions.shape == (3, 1, 4)
    assert positions[:, 0].tolist() == [
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
    ]
    assert torch.all(positions >= 0)
    assert rope_delta.tolist() == [[0.0]]


def test_rope_oracle_hashes_the_pinned_transformers_implementation():
    _, base = _rope_facade()
    source = (
        inspect.getsource(base.get_llm_pos_ids_for_vision)
        + inspect.getsource(base.get_rope_index)
    )

    assert hashlib.sha256(source.encode("utf-8")).hexdigest() == (
        EXPECTED_ROPE_IMPLEMENTATION_SHA256
    )
