from __future__ import annotations

import hashlib
import inspect
import socket
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from qwen3_omni_pretrain.profiles.qwen3_omni_reference.oracle import (
    load_reference_processor,
)


pytestmark = pytest.mark.reference

EXPECTED_ROPE_IMPLEMENTATION_SHA256 = (
    "4b0de5c1b83a32c7fbb57b16c5fc0c1c6e5a707585cae9188fc87d7283b44ee3"
)


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
    audio_mask = output["feature_attention_mask"]
    image_grid = output["image_grid_thw"]

    assert input_ids.dtype == torch.long
    assert audio_mask.ndim == 2
    assert audio_mask.shape[0] == 1
    assert 0 < int(audio_mask.sum()) <= audio_mask.shape[1]
    assert output["input_features"].shape[-1] == audio_mask.shape[-1]
    assert image_grid.shape == (1, 3)
    assert torch.all(image_grid > 0)

    image_tokens = int((image_grid[0].prod() // 4).item())
    assert int((input_ids == 151_655).sum()) == image_tokens

    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        _get_feat_extract_output_lengths,
    )

    audio_tokens = int(
        _get_feat_extract_output_lengths(audio_mask.sum(-1))[0].item()
    )
    assert int((input_ids == 151_675).sum()) == audio_tokens


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
