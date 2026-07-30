from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from qwen3_omni_pretrain.multimodal.encoders.audio import (
    AudioWindowEncoder,
)
from qwen3_omni_pretrain.multimodal.encoders.vision import (
    PatchVisionEncoder,
    TemporalVideoEncoder,
)
from qwen3_omni_pretrain.multimodal.io import DecodedMedia, MediaRequest
from qwen3_omni_pretrain.multimodal.modalities import MediaModality
from qwen3_omni_pretrain.multimodal.types import (
    MediaGrid,
    MediaSequence,
    MediaSource,
)


def request(
    modality: MediaModality,
    *,
    sample_index: int = 0,
    item_index: int = 0,
    source_id: str | None = None,
    timeline_offset_seconds: float = 0.0,
) -> MediaRequest:
    return MediaRequest(
        sample_id=f"sample-{sample_index}",
        sample_index=sample_index,
        item_index=item_index,
        source_id=source_id or f"{modality.value}-{item_index}",
        modality=modality,
        path=f"{modality.value}-{sample_index}-{item_index}.media",
        original_sample_index=sample_index,
        timeline_offset_seconds=timeline_offset_seconds,
    )


def image_item(
    pixels: torch.Tensor,
    *,
    sample_index: int = 0,
    item_index: int = 0,
    source_id: str | None = None,
) -> DecodedMedia:
    return DecodedMedia(
        request=request(
            MediaModality.IMAGE,
            sample_index=sample_index,
            item_index=item_index,
            source_id=source_id,
        ),
        tensor=pixels,
        length=1,
        timestamps=None,
        seconds_per_grid=None,
        metadata={
            "original_width": pixels.shape[2],
            "original_height": pixels.shape[1],
        },
    )


def video_item(
    frames: torch.Tensor,
    *,
    timestamps: tuple[float, ...],
    sample_index: int = 0,
    item_index: int = 0,
    source_id: str | None = None,
    offset: float = 0.0,
    seconds_per_grid: float | None = None,
) -> DecodedMedia:
    return DecodedMedia(
        request=request(
            MediaModality.VIDEO,
            sample_index=sample_index,
            item_index=item_index,
            source_id=source_id,
            timeline_offset_seconds=offset,
        ),
        tensor=frames,
        length=frames.shape[0],
        timestamps=torch.tensor(timestamps, dtype=torch.float32),
        seconds_per_grid=seconds_per_grid,
        metadata={
            "width": frames.shape[3],
            "height": frames.shape[2],
        },
    )


def audio_item(
    waveform: torch.Tensor,
    *,
    sample_rate: int = 16,
    sample_index: int = 0,
    item_index: int = 0,
    source_id: str | None = None,
    offset: float = 0.0,
) -> DecodedMedia:
    return DecodedMedia(
        request=request(
            MediaModality.AUDIO,
            sample_index=sample_index,
            item_index=item_index,
            source_id=source_id,
            timeline_offset_seconds=offset,
        ),
        tensor=waveform.reshape(1, -1),
        length=waveform.numel(),
        timestamps=None,
        seconds_per_grid=None,
        metadata={"sample_rate": sample_rate},
    )


def corrupt(item: DecodedMedia, **changes) -> DecodedMedia:
    for name, value in changes.items():
        object.__setattr__(item, name, value)
    return item


def make_sum_patch_encoder(
    *,
    in_channels: int = 1,
    patch_size: int = 2,
) -> PatchVisionEncoder:
    encoder = PatchVisionEncoder(
        in_channels=in_channels,
        hidden_size=1,
        patch_size=patch_size,
    )
    with torch.no_grad():
        encoder.patch_embed.weight.zero_()
        encoder.patch_embed.weight[:, 0].fill_(1.0)
    encoder.norm = nn.Identity()
    return encoder


def make_identity_audio_encoder(
    *,
    window_size: int = 4,
    hop_size: int = 4,
    sample_rate: int = 16,
) -> AudioWindowEncoder:
    encoder = AudioWindowEncoder(
        hidden_size=window_size,
        window_size=window_size,
        hop_size=hop_size,
        sample_rate=sample_rate,
    )
    with torch.no_grad():
        encoder.proj.weight.copy_(torch.eye(window_size))
    encoder.norm = nn.Identity()
    return encoder


def test_patch_encoder_emits_one_token_per_patch():
    encoder = PatchVisionEncoder(
        in_channels=3,
        hidden_size=8,
        patch_size=2,
    )
    image = torch.arange(3 * 4 * 4, dtype=torch.float32).view(
        1,
        3,
        4,
        4,
    )

    sequence = encoder.from_tensor_batch(
        image,
        sources=(MediaSource(0, 0, "image-0"),),
    )

    assert sequence.embeddings.shape == (1, 4, 8)
    assert sequence.grid == (MediaGrid(1, 2, 2),)


def test_audio_window_order_changes_when_waveform_is_reversed():
    encoder = AudioWindowEncoder(
        hidden_size=8,
        window_size=4,
        hop_size=4,
        sample_rate=16,
    )
    sources = (MediaSource(0, 0, "audio-0"),)

    forward = encoder.from_waveforms(
        [torch.arange(16).float()],
        sources=sources,
    )
    reverse = encoder.from_waveforms(
        [torch.arange(16).float().flip(0)],
        sources=sources,
    )

    assert not torch.equal(forward.embeddings, reverse.embeddings)
    assert torch.all(
        forward.timestamps[:, 1:] > forward.timestamps[:, :-1]
    )


def test_patch_encoder_preserves_exact_row_major_patch_order():
    encoder = make_sum_patch_encoder()
    pixels = torch.tensor(
        [
            [
                [
                    [1.0, 1.0, 2.0, 2.0],
                    [1.0, 1.0, 2.0, 2.0],
                    [3.0, 3.0, 4.0, 4.0],
                    [3.0, 3.0, 4.0, 4.0],
                ]
            ]
        ]
    )

    sequence = encoder.from_tensor_batch(
        pixels,
        sources=(MediaSource(0, 0, "image-0"),),
    )

    assert sequence.embeddings[0, :, 0].tolist() == [
        4.0,
        8.0,
        12.0,
        16.0,
    ]
    swapped = pixels.clone()
    swapped[:, :, :2], swapped[:, :, 2:] = (
        pixels[:, :, 2:].clone(),
        pixels[:, :, :2].clone(),
    )
    swapped_sequence = encoder.from_tensor_batch(
        swapped,
        sources=(MediaSource(0, 0, "image-0"),),
    )
    assert swapped_sequence.embeddings[0, :, 0].tolist() == [
        12.0,
        16.0,
        4.0,
        8.0,
    ]


def test_patch_encoder_uses_ceil_grid_without_masking_boundary_patches():
    encoder = make_sum_patch_encoder()
    pixels = torch.ones(1, 1, 3, 5)

    sequence = encoder.from_tensor_batch(
        pixels,
        sources=(MediaSource(0, 0, "image-0"),),
    )

    assert sequence.grid == (MediaGrid(1, 2, 3),)
    assert sequence.attention_mask.dtype is torch.bool
    assert sequence.attention_mask.tolist() == [[True] * 6]
    assert sequence.embeddings[0, :, 0].tolist() == [
        4.0,
        4.0,
        2.0,
        2.0,
        2.0,
        1.0,
    ]


@pytest.mark.parametrize(
    ("height", "width"),
    [(3, 4), (4, 3)],
)
def test_real_boundary_pixel_changes_the_boundary_patch(height, width):
    encoder = make_sum_patch_encoder()
    first = torch.zeros(1, 1, height, width)
    second = first.clone()
    second[0, 0, -1, -1] = 1.0
    source = (MediaSource(0, 0, "image-0"),)

    baseline = encoder.from_tensor_batch(first, sources=source)
    changed = encoder.from_tensor_batch(second, sources=source)

    assert not torch.equal(
        baseline.embeddings[:, -1],
        changed.embeddings[:, -1],
    )
    assert baseline.attention_mask[0, -1]


def test_variable_images_are_encoded_independently_then_token_padded():
    encoder = PatchVisionEncoder(
        in_channels=3,
        hidden_size=4,
        patch_size=2,
    )
    items = [
        image_item(
            torch.randn(3, 3, 3),
            sample_index=0,
            item_index=0,
            source_id="square",
        ),
        image_item(
            torch.randn(3, 2, 5),
            sample_index=1,
            item_index=0,
            source_id="wide",
        ),
    ]

    sequence = encoder(items)

    assert sequence.grid == (
        MediaGrid(1, 2, 2),
        MediaGrid(1, 1, 3),
    )
    assert sequence.attention_mask.tolist() == [
        [True, True, True, True],
        [True, True, True, False],
    ]
    assert torch.count_nonzero(sequence.embeddings[1, 3]) == 0
    assert sequence.sources == (
        MediaSource(0, 0, "square"),
        MediaSource(1, 0, "wide"),
    )


def test_video_uses_frame_major_patch_order_and_global_timestamps():
    patch_encoder = make_sum_patch_encoder(in_channels=3)
    video_encoder = TemporalVideoEncoder(hidden_size=1)
    with torch.no_grad():
        video_encoder.temporal_proj.weight.fill_(1.0)
    video_encoder.temporal_norm = nn.Identity()
    first = torch.ones(3, 4, 4)
    second = torch.full((3, 4, 4), 2.0)
    item = video_item(
        torch.stack([first, second]),
        timestamps=(0.0, 0.25),
        source_id="video-0",
        offset=1.5,
        seconds_per_grid=0.25,
    )

    sequence = video_encoder([item], patch_encoder=patch_encoder)

    assert sequence.grid == (MediaGrid(2, 2, 2),)
    assert sequence.embeddings[0, :, 0].tolist() == (
        [4.0] * 4 + [8.0] * 4
    )
    assert sequence.timestamps.dtype is torch.float32
    assert sequence.timestamps[0].tolist() == (
        [1.5] * 4 + [1.75] * 4
    )
    assert sequence.seconds_per_grid == (0.25,)
    swapped = video_item(
        torch.stack([second, first]),
        timestamps=(0.0, 0.25),
    )
    swapped_sequence = video_encoder(
        [swapped],
        patch_encoder=patch_encoder,
    )
    assert swapped_sequence.embeddings[0, :, 0].tolist() == (
        [8.0] * 4 + [4.0] * 4
    )


def test_variable_videos_only_pad_flattened_token_rows():
    patch_encoder = PatchVisionEncoder(
        in_channels=3,
        hidden_size=4,
        patch_size=2,
    )
    video_encoder = TemporalVideoEncoder(hidden_size=4)
    items = [
        video_item(
            torch.randn(2, 3, 3, 3),
            timestamps=(0.0, 0.5),
            sample_index=0,
            source_id="two-square",
            offset=1.0,
            seconds_per_grid=0.5,
        ),
        video_item(
            torch.randn(1, 3, 2, 5),
            timestamps=(0.2,),
            sample_index=1,
            source_id="one-wide",
            offset=2.0,
        ),
    ]

    sequence = video_encoder(items, patch_encoder=patch_encoder)

    assert sequence.grid == (
        MediaGrid(2, 2, 2),
        MediaGrid(1, 1, 3),
    )
    assert sequence.attention_mask.tolist() == [
        [True] * 8,
        [True, True, True, False, False, False, False, False],
    ]
    assert sequence.timestamps[0].tolist() == (
        [1.0] * 4 + [1.5] * 4
    )
    assert sequence.timestamps[1, :3].tolist() == pytest.approx(
        [2.2, 2.2, 2.2]
    )
    assert torch.count_nonzero(sequence.timestamps[1, 3:]) == 0
    assert torch.count_nonzero(sequence.embeddings[1, 3:]) == 0
    assert sequence.seconds_per_grid == (0.5, None)
    assert sequence.sources == (
        MediaSource(0, 0, "two-square"),
        MediaSource(1, 0, "one-wide"),
    )


def test_audio_windows_preserve_exact_order_and_global_start_times():
    encoder = make_identity_audio_encoder()
    waveform = torch.arange(12, dtype=torch.float32)

    sequence = encoder.from_waveforms(
        [waveform],
        sources=(MediaSource(0, 0, "audio-0"),),
        timeline_offsets_seconds=[2.0],
    )

    assert torch.equal(
        sequence.embeddings[0],
        waveform.reshape(3, 4),
    )
    assert sequence.timestamps[0].tolist() == [2.0, 2.25, 2.5]
    swapped = torch.cat((waveform[4:8], waveform[:4], waveform[8:]))
    swapped_sequence = encoder.from_waveforms(
        [swapped],
        sources=(MediaSource(0, 0, "audio-0"),),
    )
    assert torch.equal(
        swapped_sequence.embeddings[0, 0],
        sequence.embeddings[0, 1],
    )
    assert torch.equal(
        swapped_sequence.embeddings[0, 1],
        sequence.embeddings[0, 0],
    )


@pytest.mark.parametrize(
    ("length", "window_size", "hop_size", "expected_count"),
    [
        (2, 4, 2, 1),
        (4, 4, 2, 1),
        (5, 4, 2, 2),
        (10, 6, 2, 3),
        (11, 6, 2, 4),
    ],
)
def test_audio_window_count_uses_literal_tail_cover_formula(
    length,
    window_size,
    hop_size,
    expected_count,
):
    encoder = AudioWindowEncoder(
        hidden_size=4,
        window_size=window_size,
        hop_size=hop_size,
        sample_rate=16,
    )

    sequence = encoder.from_waveforms(
        [torch.arange(length, dtype=torch.float32)],
        sources=(MediaSource(0, 0, "audio-0"),),
    )

    assert sequence.attention_mask.sum().item() == expected_count
    assert sequence.embeddings.shape[1] == expected_count


def test_unequal_audio_lengths_have_prefix_masks_and_zero_padding():
    encoder = AudioWindowEncoder(
        hidden_size=4,
        window_size=4,
        hop_size=2,
        sample_rate=16,
    )
    sources = (
        MediaSource(0, 0, "short"),
        MediaSource(0, 1, "long"),
    )

    sequence = encoder.from_waveforms(
        [torch.arange(4).float(), torch.arange(9).float()],
        sources=sources,
        timeline_offsets_seconds=(1.0, 2.0),
    )

    assert sequence.attention_mask.tolist() == [
        [True, False, False, False],
        [True, True, True, True],
    ]
    assert sequence.timestamps[0, 0].item() == 1.0
    assert torch.count_nonzero(sequence.timestamps[0, 1:]) == 0
    assert torch.count_nonzero(sequence.embeddings[0, 1:]) == 0
    assert sequence.grid is None
    assert sequence.seconds_per_grid is None
    assert sequence.sources == sources


def test_final_real_audio_tail_changes_the_final_partial_window():
    encoder = make_identity_audio_encoder(window_size=4, hop_size=2)
    first = torch.zeros(5)
    second = first.clone()
    second[-1] = 1.0
    source = (MediaSource(0, 0, "audio-0"),)

    baseline = encoder.from_waveforms([first], sources=source)
    changed = encoder.from_waveforms([second], sources=source)

    assert baseline.attention_mask.tolist() == [[True, True]]
    assert not torch.equal(
        baseline.embeddings[0, -1],
        changed.embeddings[0, -1],
    )


def test_audio_forward_preserves_source_offset_and_checks_sample_rate():
    encoder = AudioWindowEncoder(
        hidden_size=4,
        window_size=4,
        hop_size=4,
        sample_rate=16,
    )
    item = audio_item(
        torch.arange(8).float(),
        sample_index=3,
        item_index=2,
        source_id="audio-source",
        offset=1.25,
    )

    sequence = encoder([item])

    assert sequence.sources == (MediaSource(3, 2, "audio-source"),)
    assert sequence.timestamps[0].tolist() == [1.25, 1.5]
    mismatched = audio_item(
        torch.arange(8).float(),
        sample_rate=8,
    )
    with pytest.raises(ValueError, match="sample_rate"):
        encoder([mismatched])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_patch_encoder_has_finite_outputs_and_gradients(dtype):
    encoder = PatchVisionEncoder(
        in_channels=3,
        hidden_size=8,
        patch_size=2,
    ).to(dtype=dtype)

    sequence = encoder.from_tensor_batch(
        torch.randn(2, 3, 4, 4, dtype=torch.float32),
        sources=(
            MediaSource(0, 0, "image-0"),
            MediaSource(1, 0, "image-1"),
        ),
    )
    sequence.embeddings.float().square().mean().backward()

    assert sequence.embeddings.dtype is dtype
    assert torch.isfinite(sequence.embeddings).all()
    assert all(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        for parameter in encoder.parameters()
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_audio_encoder_has_finite_outputs_and_gradients(dtype):
    encoder = AudioWindowEncoder(
        hidden_size=8,
        window_size=4,
        hop_size=2,
        sample_rate=16,
    ).to(dtype=dtype)

    sequence = encoder.from_waveforms(
        [torch.randn(9, dtype=torch.float32)],
        sources=(MediaSource(0, 0, "audio-0"),),
    )
    sequence.embeddings.float().square().mean().backward()

    assert sequence.embeddings.dtype is dtype
    assert sequence.timestamps.dtype is torch.float32
    assert torch.isfinite(sequence.embeddings).all()
    assert all(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        for parameter in encoder.parameters()
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_video_loss_reaches_shared_patch_and_temporal_parameters(dtype):
    patch_encoder = PatchVisionEncoder(
        in_channels=3,
        hidden_size=8,
        patch_size=2,
    ).to(dtype=dtype)
    video_encoder = TemporalVideoEncoder(hidden_size=8).to(dtype=dtype)
    item = video_item(
        torch.randn(2, 3, 4, 4, dtype=torch.float32),
        timestamps=(0.0, 0.5),
        seconds_per_grid=0.5,
    )

    sequence = video_encoder([item], patch_encoder=patch_encoder)
    sequence.embeddings.float().square().mean().backward()

    assert sequence.embeddings.dtype is dtype
    assert sequence.timestamps.dtype is torch.float32
    for module in (patch_encoder, video_encoder):
        assert all(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            for parameter in module.parameters()
        )


def test_video_encoder_does_not_register_a_patch_alias():
    patch_encoder = PatchVisionEncoder(
        in_channels=3,
        hidden_size=4,
        patch_size=2,
    )
    video_encoder = TemporalVideoEncoder(hidden_size=4)

    class Holder(nn.Module):
        def __init__(self, patch, video):
            super().__init__()
            self.patch = patch
            self.video = video

    holder = Holder(patch_encoder, video_encoder)
    names = tuple(holder.state_dict())

    assert all(not name.startswith("video.patch") for name in names)
    assert sum(name.endswith("patch_embed.weight") for name in names) == 1
    assert tuple(
        name for name, _ in video_encoder.named_parameters()
    ) == (
        "temporal_proj.weight",
        "temporal_norm.weight",
        "temporal_norm.bias",
    )


@pytest.mark.parametrize(
    ("encoder_type", "kwargs"),
    [
        (
            PatchVisionEncoder,
            {"in_channels": True, "hidden_size": 4, "patch_size": 2},
        ),
        (
            PatchVisionEncoder,
            {"in_channels": 3, "hidden_size": 4.5, "patch_size": 2},
        ),
        (
            PatchVisionEncoder,
            {"in_channels": 3, "hidden_size": 4, "patch_size": 0},
        ),
        (TemporalVideoEncoder, {"hidden_size": False}),
        (
            AudioWindowEncoder,
            {
                "hidden_size": 4,
                "window_size": 4,
                "hop_size": 5,
                "sample_rate": 16,
            },
        ),
        (
            AudioWindowEncoder,
            {
                "hidden_size": 4,
                "window_size": 4,
                "hop_size": 2,
                "sample_rate": "16",
            },
        ),
    ],
)
def test_encoder_constructor_contracts_are_strict(encoder_type, kwargs):
    with pytest.raises((TypeError, ValueError)):
        encoder_type(**kwargs)


def test_patch_helper_rejects_invalid_inputs_and_sources():
    encoder = PatchVisionEncoder(
        in_channels=3,
        hidden_size=4,
        patch_size=2,
    )
    source = (MediaSource(0, 0, "image-0"),)
    with pytest.raises(ValueError, match="non-empty"):
        encoder.from_tensor_batch(
            torch.empty(0, 3, 2, 2),
            sources=(),
        )
    with pytest.raises(ValueError, match="\\[B, C, H, W\\]"):
        encoder.from_tensor_batch(torch.zeros(3, 2, 2), sources=source)
    with pytest.raises(TypeError, match="floating"):
        encoder.from_tensor_batch(
            torch.zeros(1, 3, 2, 2, dtype=torch.uint8),
            sources=source,
        )
    with pytest.raises(ValueError, match="channel"):
        encoder.from_tensor_batch(
            torch.zeros(1, 1, 2, 2),
            sources=source,
        )
    with pytest.raises(ValueError, match="source"):
        encoder.from_tensor_batch(
            torch.zeros(1, 3, 2, 2),
            sources=(),
        )
    with pytest.raises(ValueError, match="unique"):
        encoder.from_tensor_batch(
            torch.zeros(2, 3, 2, 2),
            sources=(source[0], source[0]),
        )


@pytest.mark.parametrize(
    "change",
    [
        {"tensor": torch.zeros(3, 2, 2, dtype=torch.uint8)},
        {"tensor": torch.zeros(1, 3, 2, 2)},
        {"length": 2},
    ],
)
def test_patch_forward_revalidates_decoded_items_before_projection(change):
    encoder = PatchVisionEncoder(
        in_channels=3,
        hidden_size=4,
        patch_size=2,
    )
    item = corrupt(image_item(torch.zeros(3, 2, 2)), **change)
    calls = []
    handle = encoder.patch_embed.register_forward_hook(
        lambda *args: calls.append(True)
    )
    try:
        with pytest.raises((TypeError, ValueError)):
            encoder([item])
    finally:
        handle.remove()
    assert calls == []


def test_encoder_forwards_reject_empty_or_wrong_modalities():
    patch_encoder = PatchVisionEncoder(
        in_channels=3,
        hidden_size=4,
        patch_size=2,
    )
    video_encoder = TemporalVideoEncoder(hidden_size=4)
    audio_encoder = AudioWindowEncoder(
        hidden_size=4,
        window_size=4,
        hop_size=2,
        sample_rate=16,
    )
    image = image_item(torch.zeros(3, 2, 2))
    audio = audio_item(torch.zeros(4))
    with pytest.raises(ValueError, match="non-empty"):
        patch_encoder([])
    with pytest.raises(ValueError, match="non-empty"):
        video_encoder([], patch_encoder=patch_encoder)
    with pytest.raises(ValueError, match="non-empty"):
        audio_encoder([])
    with pytest.raises(ValueError, match="modality"):
        patch_encoder([audio])
    with pytest.raises(ValueError, match="modality"):
        video_encoder([image], patch_encoder=patch_encoder)
    with pytest.raises(ValueError, match="modality"):
        audio_encoder([image])


@pytest.mark.parametrize(
    "change",
    [
        {"tensor": torch.zeros(1, 3, 2, 2)},
        {"length": 1},
        {"timestamps": None},
        {"timestamps": torch.tensor([0.0])},
    ],
)
def test_video_forward_revalidates_shape_length_and_timestamps(change):
    patch_encoder = PatchVisionEncoder(
        in_channels=3,
        hidden_size=4,
        patch_size=2,
    )
    video_encoder = TemporalVideoEncoder(hidden_size=4)
    item = corrupt(
        video_item(
            torch.zeros(2, 3, 2, 2),
            timestamps=(0.0, 0.5),
            seconds_per_grid=0.5,
        ),
        **change,
    )
    with pytest.raises((TypeError, ValueError)):
        video_encoder([item], patch_encoder=patch_encoder)


def test_video_encoder_requires_matching_patch_hidden_size():
    patch_encoder = PatchVisionEncoder(
        in_channels=3,
        hidden_size=8,
        patch_size=2,
    )
    video_encoder = TemporalVideoEncoder(hidden_size=4)
    item = video_item(
        torch.zeros(1, 3, 2, 2),
        timestamps=(0.0,),
    )
    with pytest.raises(ValueError, match="hidden"):
        video_encoder([item], patch_encoder=patch_encoder)


def test_video_encoder_normalizes_integral_length_for_strict_grid():
    patch_encoder = PatchVisionEncoder(
        in_channels=3,
        hidden_size=4,
        patch_size=2,
    )
    video_encoder = TemporalVideoEncoder(hidden_size=4)
    item = video_item(
        torch.zeros(1, 3, 2, 2),
        timestamps=(0.0,),
    )
    object.__setattr__(item, "length", np.int64(1))
    item.__post_init__()

    sequence = video_encoder([item], patch_encoder=patch_encoder)

    assert sequence.grid == (MediaGrid(1, 1, 1),)


def test_audio_helper_rejects_invalid_inputs_offsets_and_sources():
    encoder = AudioWindowEncoder(
        hidden_size=4,
        window_size=4,
        hop_size=2,
        sample_rate=16,
    )
    source = (MediaSource(0, 0, "audio-0"),)
    with pytest.raises(ValueError, match="non-empty"):
        encoder.from_waveforms([], sources=())
    for waveform in (
        torch.zeros(0),
        torch.zeros(1, 4),
        torch.zeros(2, 4),
    ):
        with pytest.raises(ValueError, match="1-D|non-empty"):
            encoder.from_waveforms([waveform], sources=source)
    with pytest.raises(TypeError, match="floating"):
        encoder.from_waveforms(
            [torch.zeros(4, dtype=torch.long)],
            sources=source,
        )
    with pytest.raises(ValueError, match="source"):
        encoder.from_waveforms([torch.zeros(4)], sources=())
    with pytest.raises(ValueError, match="unique"):
        encoder.from_waveforms(
            [torch.zeros(4), torch.zeros(4)],
            sources=(source[0], source[0]),
        )
    for offsets in ((), (-1.0,), (float("nan"),), (True,)):
        with pytest.raises((TypeError, ValueError), match="offset"):
            encoder.from_waveforms(
                [torch.zeros(4)],
                sources=source,
                timeline_offsets_seconds=offsets,
            )


def test_audio_forward_revalidates_shape_and_length():
    encoder = AudioWindowEncoder(
        hidden_size=4,
        window_size=4,
        hop_size=2,
        sample_rate=16,
    )
    stereo = corrupt(
        audio_item(torch.zeros(4)),
        tensor=torch.zeros(2, 4),
    )
    bad_length = corrupt(
        audio_item(torch.zeros(4)),
        length=3,
    )
    for item in (stereo, bad_length):
        with pytest.raises(ValueError):
            encoder([item])


def test_every_public_encoder_result_is_validated(monkeypatch):
    calls = []
    original_validate = MediaSequence.validate

    def validate(sequence):
        calls.append(sequence.modality)
        return original_validate(sequence)

    monkeypatch.setattr(MediaSequence, "validate", validate)
    patch_encoder = PatchVisionEncoder(
        in_channels=3,
        hidden_size=4,
        patch_size=2,
    )
    video_encoder = TemporalVideoEncoder(hidden_size=4)
    audio_encoder = AudioWindowEncoder(
        hidden_size=4,
        window_size=4,
        hop_size=2,
        sample_rate=16,
    )
    image = image_item(torch.zeros(3, 2, 2))
    video = video_item(
        torch.zeros(1, 3, 2, 2),
        timestamps=(0.0,),
    )
    audio = audio_item(torch.zeros(4))

    patch_encoder([image])
    patch_encoder.from_tensor_batch(
        torch.zeros(1, 3, 2, 2),
        sources=(MediaSource(0, 0, "image-helper"),),
    )
    video_encoder([video], patch_encoder=patch_encoder)
    audio_encoder([audio])
    audio_encoder.from_waveforms(
        [torch.zeros(4)],
        sources=(MediaSource(0, 0, "audio-helper"),),
    )

    assert calls == [
        MediaModality.IMAGE,
        MediaModality.IMAGE,
        MediaModality.VIDEO,
        MediaModality.AUDIO,
        MediaModality.AUDIO,
    ]
