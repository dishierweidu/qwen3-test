from __future__ import annotations

import math
import shutil
import subprocess
import wave
from pathlib import Path

import pytest
import torch
from PIL import Image

from qwen3_omni_pretrain.data.collators import MediaLoadError
from qwen3_omni_pretrain.multimodal import io as media_io
from qwen3_omni_pretrain.multimodal.io import (
    DecodedMedia,
    MediaRequest,
    StrictMediaLoader,
)
from qwen3_omni_pretrain.multimodal.modalities import MediaModality


def write_wave(path: Path, *, samples: int, rate: int) -> Path:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(b"\x00\x00" * samples)
    return path


def video_request(
    path: str,
    *,
    timestamps: tuple[float, ...] | None = None,
) -> MediaRequest:
    return MediaRequest(
        sample_id="video-sample",
        sample_index=0,
        item_index=0,
        source_id="video-0",
        modality=MediaModality.VIDEO,
        path=path,
        timestamps=timestamps,
    )


def media_request(modality: MediaModality) -> MediaRequest:
    return MediaRequest(
        sample_id="sample",
        sample_index=0,
        item_index=0,
        source_id=f"{modality.value}-0",
        modality=modality,
        path=f"{modality.value}.media",
    )


def write_three_frame_video(path: Path) -> Path:
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=4x2:r=2:d=1.5",
            "-frames:v",
            "3",
            "-c:v",
            "ffv1",
            str(path),
        ],
        check=False,
        capture_output=True,
        shell=False,
    )
    assert result.returncode == 0, result.stderr.decode(
        "utf-8",
        errors="replace",
    )
    return path


def write_rotated_asymmetric_video(path: Path) -> Path:
    source_path = path.with_name(f"{path.stem}-coded.mov")
    encode = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            (
                "color=c=red:s=8x4:r=1:d=1,"
                "drawbox=x=4:y=0:w=4:h=4:color=blue:t=fill"
            ),
            "-frames:v",
            "1",
            "-c:v",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            str(source_path),
        ],
        check=False,
        capture_output=True,
        shell=False,
    )
    assert encode.returncode == 0, encode.stderr.decode(
        "utf-8",
        errors="replace",
    )
    rotate = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(source_path),
            "-c",
            "copy",
            "-metadata:s:v:0",
            "rotate=90",
            str(path),
        ],
        check=False,
        capture_output=True,
        shell=False,
    )
    assert rotate.returncode == 0, rotate.stderr.decode(
        "utf-8",
        errors="replace",
    )
    return path


def test_audio_loader_preserves_true_length(tmp_path):
    path = write_wave(
        tmp_path / "tone.wav",
        samples=12_345,
        rate=16_000,
    )

    item = StrictMediaLoader().load(
        MediaRequest(
            sample_id="s0",
            sample_index=0,
            item_index=0,
            source_id="a0",
            modality=MediaModality.AUDIO,
            path=str(path),
        )
    )

    assert item.tensor.shape == (1, 12_345)
    assert item.length == 12_345


def test_referenced_video_without_decoder_is_fatal(monkeypatch):
    monkeypatch.setattr(media_io, "_ffmpeg_available", lambda: False)

    with pytest.raises(MediaLoadError, match="video"):
        StrictMediaLoader().load(video_request("missing.mp4"))


def test_image_loader_preserves_rgb_pixels_and_original_dimensions(tmp_path):
    path = tmp_path / "sample.png"
    Image.new("RGB", (5, 3), (64, 128, 255)).save(path)

    item = StrictMediaLoader().load(
        MediaRequest(
            sample_id="image-sample",
            sample_index=0,
            item_index=0,
            source_id="image-0",
            modality=MediaModality.IMAGE,
            path=str(path),
        )
    )

    assert item.tensor.shape == (3, 3, 5)
    assert item.tensor.dtype == torch.float32
    assert torch.allclose(
        item.tensor[:, 0, 0],
        torch.tensor([64.0, 128.0, 255.0]) / 255.0,
    )
    assert item.length == 1
    assert item.timestamps is None
    assert item.seconds_per_grid is None
    assert item.metadata["original_width"] == 5
    assert item.metadata["original_height"] == 3


def test_audio_loader_resamples_to_configured_rate_without_padding(tmp_path):
    path = write_wave(tmp_path / "low-rate.wav", samples=4_000, rate=8_000)

    item = StrictMediaLoader(sample_rate=16_000).load(
        MediaRequest(
            sample_id="audio-sample",
            sample_index=0,
            item_index=0,
            source_id="audio-0",
            modality=MediaModality.AUDIO,
            path=str(path),
        )
    )

    assert item.tensor.shape == (1, 8_000)
    assert item.length == 8_000
    assert item.metadata["sample_rate"] == 16_000
    assert item.metadata["source_sample_rate"] == 8_000
    assert item.timestamps is None
    assert item.seconds_per_grid is None


@pytest.mark.parametrize(
    "offset",
    [-0.01, float("inf"), float("-inf"), float("nan")],
)
def test_media_request_rejects_invalid_timeline_offsets(offset):
    with pytest.raises((TypeError, ValueError), match="timeline_offset"):
        MediaRequest(
            sample_id="sample",
            sample_index=0,
            item_index=0,
            source_id="image-0",
            modality=MediaModality.IMAGE,
            path="image.png",
            timeline_offset_seconds=offset,
        )


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="requires ffmpeg and ffprobe",
)
def test_video_loader_returns_three_frames_with_exact_uniform_timestamps(
    tmp_path,
):
    path = write_three_frame_video(tmp_path / "three-frames.mkv")

    item = StrictMediaLoader().load(video_request(str(path)))

    assert item.tensor.shape == (3, 3, 2, 4)
    assert item.length == 3
    assert item.timestamps is not None
    assert item.timestamps.tolist() == pytest.approx([0.0, 0.5, 1.0])
    assert item.seconds_per_grid == pytest.approx(0.5)
    assert item.metadata["width"] == 4
    assert item.metadata["height"] == 2
    assert item.metadata["time_base"]


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="requires ffmpeg and ffprobe",
)
def test_video_loader_keeps_irregular_explicit_timestamps_without_cadence(
    tmp_path,
):
    path = write_three_frame_video(tmp_path / "irregular.mkv")

    item = StrictMediaLoader().load(
        video_request(str(path), timestamps=(0.0, 0.5, 1.1))
    )

    assert item.timestamps.tolist() == pytest.approx([0.0, 0.5, 1.1])
    assert item.seconds_per_grid is None


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="requires ffmpeg and ffprobe",
)
def test_video_loader_keeps_coded_layout_when_rotation_metadata_is_present(
    tmp_path,
):
    path = write_rotated_asymmetric_video(tmp_path / "rotated.mov")

    item = StrictMediaLoader().load(video_request(str(path)))

    assert item.tensor.shape == (1, 3, 4, 8)
    left_pixel = item.tensor[0, :, 0, 0]
    right_pixel = item.tensor[0, :, 0, -1]
    assert left_pixel[0] > 0.8
    assert left_pixel[2] < 0.2
    assert right_pixel[2] > 0.8
    assert right_pixel[0] < 0.2


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="requires ffmpeg and ffprobe",
)
def test_video_timestamp_frame_count_mismatch_is_fatal(tmp_path):
    path = write_three_frame_video(tmp_path / "mismatch.mkv")

    with pytest.raises(MediaLoadError, match="timestamp"):
        StrictMediaLoader().load(
            video_request(str(path), timestamps=(0.0, 0.5))
        )


@pytest.mark.parametrize(
    ("modality", "filename"),
    [
        (MediaModality.IMAGE, "corrupt.png"),
        (MediaModality.AUDIO, "corrupt.wav"),
    ],
)
def test_corrupt_referenced_media_is_always_structured(
    tmp_path,
    modality,
    filename,
):
    path = tmp_path / filename
    path.write_bytes(b"not valid media")
    request = MediaRequest(
        sample_id="broken",
        sample_index=0,
        item_index=0,
        source_id="item-0",
        modality=modality,
        path=str(path),
    )

    with pytest.raises(MediaLoadError) as captured:
        StrictMediaLoader().load(request)

    assert captured.value.to_dict()["sample_id"] == "broken"
    assert captured.value.to_dict()["modality"] == modality.value
    assert captured.value.to_dict()["path"] == str(path)


def test_video_subprocesses_use_argument_lists_and_disable_shell(
    monkeypatch,
):
    calls = []

    class Result:
        returncode = 0
        stderr = b""

        def __init__(self, stdout):
            self.stdout = stdout

    def run(arguments, **kwargs):
        calls.append((arguments, kwargs))
        if arguments[0] == "ffprobe":
            return Result(
                b'{"streams":[{"width":1,"height":1,'
                b'"time_base":"1/1000"}],'
                b'"frames":[{"best_effort_timestamp_time":"0.0"}]}'
            )
        return Result(b"\x00\x00\x00")

    monkeypatch.setattr(media_io, "_ffmpeg_available", lambda: True)
    monkeypatch.setattr(media_io.subprocess, "run", run)

    item = StrictMediaLoader().load(video_request("odd;touch injected.mp4"))

    assert item.tensor.shape == (1, 3, 1, 1)
    assert len(calls) == 2
    for arguments, keyword_arguments in calls:
        assert isinstance(arguments, list)
        assert keyword_arguments["shell"] is False
    assert calls[0][0][-1] == "odd;touch injected.mp4"
    noautorotate_index = calls[1][0].index("-noautorotate")
    assert noautorotate_index < calls[1][0].index("-i")
    assert calls[1][0][calls[1][0].index("-i") + 1] == (
        "odd;touch injected.mp4"
    )


@pytest.mark.parametrize(
    "bad_timestamp",
    [float("nan"), float("inf"), -0.1],
)
def test_explicit_video_timestamps_must_be_finite_non_negative(
    bad_timestamp,
):
    with pytest.raises((TypeError, ValueError), match="timestamps"):
        video_request("clip.mp4", timestamps=(0.0, bad_timestamp))


def test_media_request_indices_and_identity_are_validated():
    with pytest.raises((TypeError, ValueError), match="sample_index"):
        MediaRequest(
            sample_id="sample",
            sample_index=-1,
            item_index=0,
            source_id="image-0",
            modality=MediaModality.IMAGE,
            path="image.png",
        )


@pytest.mark.parametrize("sample_rate", [True, 16_000.5, "16000"])
def test_strict_loader_rejects_non_integer_sample_rates(sample_rate):
    with pytest.raises(TypeError, match="sample_rate"):
        StrictMediaLoader(sample_rate=sample_rate)


def test_decoded_media_requires_float_tensor_and_mapping_metadata():
    request = media_request(MediaModality.IMAGE)

    with pytest.raises(TypeError, match="floating"):
        DecodedMedia(
            request=request,
            tensor=torch.zeros(3, 2, 2, dtype=torch.uint8),
            length=1,
            timestamps=None,
            seconds_per_grid=None,
            metadata={},
        )
    with pytest.raises(TypeError, match="metadata"):
        DecodedMedia(
            request=request,
            tensor=torch.zeros(3, 2, 2),
            length=1,
            timestamps=None,
            seconds_per_grid=None,
            metadata=[],
        )


@pytest.mark.parametrize(
    ("modality", "tensor", "length", "error"),
    [
        (MediaModality.IMAGE, torch.zeros(1, 3, 2, 2), 1, "image"),
        (MediaModality.IMAGE, torch.zeros(3, 2, 2), 2, "length"),
        (MediaModality.AUDIO, torch.zeros(2, 8), 8, "audio"),
        (MediaModality.AUDIO, torch.zeros(1, 8), 7, "length"),
        (MediaModality.VIDEO, torch.zeros(2, 2, 2, 3), 2, "video"),
        (MediaModality.VIDEO, torch.zeros(2, 3, 2, 2), 1, "length"),
    ],
)
def test_decoded_media_enforces_modality_shape_and_length(
    modality,
    tensor,
    length,
    error,
):
    timestamps = (
        torch.tensor([0.0, 0.5])
        if modality is MediaModality.VIDEO
        else None
    )
    with pytest.raises(ValueError, match=error):
        DecodedMedia(
            request=media_request(modality),
            tensor=tensor,
            length=length,
            timestamps=timestamps,
            seconds_per_grid=None,
            metadata={},
        )


@pytest.mark.parametrize(
    ("timestamps", "cadence", "error"),
    [
        (None, None, "timestamps"),
        (torch.tensor([0.0]), None, "timestamp"),
        (torch.tensor([0.0, float("nan")]), None, "finite"),
        (torch.tensor([0.5, 0.0]), None, "monotonic"),
        (torch.tensor([0.0, 0.5]), 0.0, "seconds_per_grid"),
        (torch.tensor([0.0, 0.5]), 0.25, "cadence"),
    ],
)
def test_decoded_video_validates_timestamps_and_explicit_cadence(
    timestamps,
    cadence,
    error,
):
    with pytest.raises((TypeError, ValueError), match=error):
        DecodedMedia(
            request=media_request(MediaModality.VIDEO),
            tensor=torch.zeros(2, 3, 2, 2),
            length=2,
            timestamps=timestamps,
            seconds_per_grid=cadence,
            metadata={},
        )


@pytest.mark.parametrize(
    "modality",
    [MediaModality.IMAGE, MediaModality.AUDIO],
)
def test_non_video_decoded_media_rejects_timestamps_and_cadence(modality):
    tensor = (
        torch.zeros(3, 2, 2)
        if modality is MediaModality.IMAGE
        else torch.zeros(1, 8)
    )
    length = 1 if modality is MediaModality.IMAGE else 8
    with pytest.raises(ValueError, match="timestamps"):
        DecodedMedia(
            request=media_request(modality),
            tensor=tensor,
            length=length,
            timestamps=torch.tensor([0.0]),
            seconds_per_grid=None,
            metadata={},
        )
    with pytest.raises(ValueError, match="seconds_per_grid"):
        DecodedMedia(
            request=media_request(modality),
            tensor=tensor,
            length=length,
            timestamps=None,
            seconds_per_grid=0.5,
            metadata={},
        )


@pytest.mark.parametrize(
    ("modality", "tensor", "length", "timestamps", "metadata", "error"),
    [
        (
            MediaModality.IMAGE,
            torch.zeros(3, 2, 3),
            1,
            None,
            {"original_width": 3},
            "original_height",
        ),
        (
            MediaModality.IMAGE,
            torch.zeros(3, 2, 3),
            1,
            None,
            {"original_width": 4, "original_height": 2},
            "original_width",
        ),
        (
            MediaModality.AUDIO,
            torch.zeros(1, 8),
            8,
            None,
            {"sample_rate": True},
            "sample_rate",
        ),
        (
            MediaModality.VIDEO,
            torch.zeros(2, 3, 2, 3),
            2,
            torch.tensor([0.0, 0.5]),
            {"width": 3},
            "height",
        ),
        (
            MediaModality.VIDEO,
            torch.zeros(2, 3, 2, 3),
            2,
            torch.tensor([0.0, 0.5]),
            {"width": 3, "height": 4},
            "height",
        ),
    ],
)
def test_decoded_media_requires_valid_modality_metadata(
    modality,
    tensor,
    length,
    timestamps,
    metadata,
    error,
):
    with pytest.raises(ValueError, match=error):
        DecodedMedia(
            request=media_request(modality),
            tensor=tensor,
            length=length,
            timestamps=timestamps,
            seconds_per_grid=None,
            metadata=metadata,
        )
