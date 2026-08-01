from __future__ import annotations

from dataclasses import dataclass
import json
import math
from numbers import Integral, Real
from pathlib import Path
import shutil
import subprocess
import wave
from typing import Mapping

import torch
import torchaudio
from PIL import Image

from qwen3_omni_pretrain.data.collators import MediaLoadError
from qwen3_omni_pretrain.multimodal.modalities import MediaModality


@dataclass(frozen=True)
class MediaRequest:
    sample_id: str
    sample_index: int
    item_index: int
    source_id: str
    modality: MediaModality
    path: str
    original_sample_index: int | None = None
    timeline_offset_seconds: float = 0.0
    timestamps: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        for name in ("sample_id", "source_id", "path"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string")
            if not value.strip():
                raise ValueError(f"{name} must not be empty")
        for name in ("sample_index", "item_index"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.original_sample_index is not None:
            value = self.original_sample_index
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    "original_sample_index must be an integer or None"
                )
            if value < 0:
                raise ValueError(
                    "original_sample_index must be non-negative"
                )
        if not isinstance(self.modality, MediaModality):
            raise TypeError("modality must be MediaModality")
        _validate_non_negative_number(
            self.timeline_offset_seconds,
            "timeline_offset_seconds",
        )
        if self.timestamps is not None:
            if not isinstance(self.timestamps, tuple):
                raise TypeError("timestamps must be a tuple or None")
            previous = -math.inf
            for value in self.timestamps:
                _validate_non_negative_number(value, "timestamps")
                numeric = float(value)
                if numeric < previous:
                    raise ValueError("timestamps must be monotonic")
                previous = numeric


@dataclass(frozen=True)
class DecodedMedia:
    request: MediaRequest
    tensor: torch.Tensor
    length: int
    timestamps: torch.Tensor | None
    seconds_per_grid: float | None
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.request, MediaRequest):
            raise TypeError("request must be a MediaRequest")
        if not isinstance(self.tensor, torch.Tensor):
            raise TypeError("tensor must be a torch.Tensor")
        if not torch.is_floating_point(self.tensor):
            raise TypeError("tensor must have a floating-point dtype")
        if self.tensor.numel() and not bool(
            torch.isfinite(self.tensor).all().item()
        ):
            raise ValueError("tensor values must be finite")
        if isinstance(self.length, bool) or not isinstance(
            self.length,
            Integral,
        ):
            raise TypeError("length must be an integer")
        if self.length <= 0:
            raise ValueError("length must be positive")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")

        modality = self.request.modality
        if modality is MediaModality.IMAGE:
            self._validate_image()
        elif modality is MediaModality.AUDIO:
            self._validate_audio()
        elif modality is MediaModality.VIDEO:
            self._validate_video()
        else:  # pragma: no cover - MediaRequest validates this first.
            raise ValueError(f"unsupported media modality: {modality!r}")

    def _validate_image(self) -> None:
        if (
            self.tensor.ndim != 3
            or self.tensor.shape[0] != 3
            or any(size <= 0 for size in self.tensor.shape)
        ):
            raise ValueError("image tensor must have shape [3, H, W]")
        if self.length != 1:
            raise ValueError("image length must equal 1")
        self._validate_non_video_timing("image")
        original_width = self._metadata_positive_integer(
            "original_width"
        )
        original_height = self._metadata_positive_integer(
            "original_height"
        )
        if original_width != self.tensor.shape[2]:
            raise ValueError(
                "image metadata original_width must match tensor width"
            )
        if original_height != self.tensor.shape[1]:
            raise ValueError(
                "image metadata original_height must match tensor height"
            )

    def _validate_audio(self) -> None:
        if (
            self.tensor.ndim != 2
            or self.tensor.shape[0] != 1
            or self.tensor.shape[1] <= 0
        ):
            raise ValueError("audio tensor must have shape [1, samples]")
        if self.length != self.tensor.shape[1]:
            raise ValueError(
                "audio length must equal the tensor sample dimension"
            )
        self._validate_non_video_timing("audio")
        self._metadata_positive_integer("sample_rate")

    def _validate_non_video_timing(self, modality: str) -> None:
        if self.timestamps is not None:
            raise ValueError(f"{modality} timestamps must be None")
        if self.seconds_per_grid is not None:
            raise ValueError(
                f"{modality} seconds_per_grid must be None"
            )

    def _metadata_positive_integer(self, name: str) -> int:
        value = self.metadata.get(name)
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(
                f"metadata {name} must be a positive integer"
            )
        numeric = int(value)
        if numeric <= 0:
            raise ValueError(
                f"metadata {name} must be a positive integer"
            )
        return numeric

    def _validate_video(self) -> None:
        if (
            self.tensor.ndim != 4
            or self.tensor.shape[1] != 3
            or any(size <= 0 for size in self.tensor.shape)
        ):
            raise ValueError(
                "video tensor must have shape [frames, 3, H, W]"
            )
        if self.length != self.tensor.shape[0]:
            raise ValueError(
                "video length must equal the tensor frame dimension"
            )
        if self.timestamps is None:
            raise ValueError("video timestamps must not be None")
        if not isinstance(self.timestamps, torch.Tensor):
            raise TypeError("video timestamps must be a torch.Tensor")
        if not torch.is_floating_point(self.timestamps):
            raise TypeError(
                "video timestamps must have a floating-point dtype"
            )
        if self.timestamps.ndim != 1:
            raise ValueError("video timestamps must be one-dimensional")
        if self.timestamps.numel() != self.length:
            raise ValueError(
                "video timestamp count must equal the frame count"
            )
        if not bool(torch.isfinite(self.timestamps).all().item()):
            raise ValueError("video timestamps must be finite")
        if bool((self.timestamps < 0).any().item()):
            raise ValueError("video timestamps must be non-negative")
        deltas = self.timestamps[1:] - self.timestamps[:-1]
        if deltas.numel() and bool((deltas < 0).any().item()):
            raise ValueError("video timestamps must be monotonic")

        cadence = self.seconds_per_grid
        if cadence is not None:
            if isinstance(cadence, bool) or not isinstance(cadence, Real):
                raise TypeError(
                    "video seconds_per_grid must be a real number or None"
                )
            cadence_value = float(cadence)
            if not math.isfinite(cadence_value) or cadence_value <= 0:
                raise ValueError(
                    "video seconds_per_grid must be finite and positive"
                )
            if deltas.numel() == 0:
                raise ValueError(
                    "video seconds_per_grid requires at least two timestamps"
                )
            expected = torch.full_like(deltas, cadence_value)
            if bool((deltas <= 0).any().item()) or not torch.allclose(
                deltas,
                expected,
                rtol=1e-6,
                atol=1e-6,
            ):
                raise ValueError(
                    "video timestamps do not match "
                    "seconds_per_grid cadence"
                )

        width = self._metadata_positive_integer("width")
        height = self._metadata_positive_integer("height")
        if width != self.tensor.shape[3]:
            raise ValueError("video metadata width must match tensor width")
        if height != self.tensor.shape[2]:
            raise ValueError("video metadata height must match tensor height")


def _validate_non_negative_number(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    if not math.isfinite(float(value)) or float(value) < 0:
        raise ValueError(f"{name} must be finite and non-negative")


def _ffmpeg_available() -> bool:
    return (
        shutil.which("ffmpeg") is not None
        and shutil.which("ffprobe") is not None
    )


class StrictMediaLoader:
    def __init__(self, *, sample_rate: int = 16_000) -> None:
        if isinstance(sample_rate, bool) or not isinstance(
            sample_rate,
            Integral,
        ):
            raise TypeError("sample_rate must be an integer")
        self.sample_rate = int(sample_rate)
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

    def load(self, item: MediaRequest) -> DecodedMedia:
        if not isinstance(item, MediaRequest):
            cause = TypeError("item must be a MediaRequest")
            raise MediaLoadError(
                modality="unknown",
                path="<unknown>",
                sample_id="<unknown>",
                cause=cause,
            ) from cause
        try:
            if item.modality is MediaModality.AUDIO:
                return self._load_audio(item)
            if item.modality is MediaModality.VIDEO:
                if not _ffmpeg_available():
                    raise RuntimeError(
                        "ffmpeg and ffprobe are required to decode video"
                    )
                return self._load_video(item)
            if item.modality is MediaModality.IMAGE:
                return self._load_image(item)
            raise ValueError(f"unsupported media modality: {item.modality!r}")
        except MediaLoadError:
            raise
        except Exception as cause:
            raise MediaLoadError(
                modality=item.modality.value,
                path=item.path,
                sample_id=item.sample_id,
                cause=cause,
            ) from cause

    @staticmethod
    def _load_image(item: MediaRequest) -> DecodedMedia:
        with Image.open(item.path) as source:
            original_width, original_height = source.size
            image = source.convert("RGB")
            payload = bytearray(image.tobytes())
        tensor = torch.frombuffer(payload, dtype=torch.uint8).clone()
        tensor = tensor.reshape(original_height, original_width, 3)
        tensor = (
            tensor.permute(2, 0, 1)
            .to(dtype=torch.float32)
            .div_(255.0)
            .contiguous()
        )
        return DecodedMedia(
            request=item,
            tensor=tensor,
            length=1,
            timestamps=None,
            seconds_per_grid=None,
            metadata={
                "original_width": original_width,
                "original_height": original_height,
            },
        )

    def _load_audio(self, item: MediaRequest) -> DecodedMedia:
        try:
            waveform, source_rate = self._load_pcm_wave(Path(item.path))
        except (wave.Error, EOFError):
            waveform, source_rate = torchaudio.load(Path(item.path))
        if waveform.ndim != 2 or waveform.shape[1] == 0:
            raise ValueError("audio decoder returned no samples")
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if int(source_rate) != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                int(source_rate),
                self.sample_rate,
            )
        waveform = waveform.to(dtype=torch.float32).contiguous()
        return DecodedMedia(
            request=item,
            tensor=waveform,
            length=int(waveform.shape[1]),
            timestamps=None,
            seconds_per_grid=None,
            metadata={
                "sample_rate": self.sample_rate,
                "source_sample_rate": int(source_rate),
            },
        )

    @staticmethod
    def _load_pcm_wave(path: Path) -> tuple[torch.Tensor, int]:
        with wave.open(str(path), "rb") as source:
            channel_count = source.getnchannels()
            sample_width = source.getsampwidth()
            sample_rate = source.getframerate()
            frame_count = source.getnframes()
            payload = source.readframes(frame_count)
        if channel_count <= 0 or frame_count <= 0:
            raise ValueError("audio decoder returned no samples")
        if sample_width != 2:
            raise wave.Error("only signed 16-bit PCM WAV is handled directly")
        values = torch.frombuffer(
            bytearray(payload),
            dtype=torch.int16,
        ).reshape(-1, channel_count)
        waveform = (
            values.transpose(0, 1)
            .to(dtype=torch.float32)
            .div_(32_768.0)
        )
        return waveform, int(sample_rate)

    @staticmethod
    def _run_process(arguments: list[str]) -> subprocess.CompletedProcess:
        result = subprocess.run(
            arguments,
            check=False,
            capture_output=True,
            shell=False,
        )
        if result.returncode != 0:
            detail = result.stderr.decode(
                "utf-8",
                errors="replace",
            ).strip()
            raise RuntimeError(
                f"{arguments[0]} failed with status "
                f"{result.returncode}: {detail}"
            )
        return result

    def _load_video(self, item: MediaRequest) -> DecodedMedia:
        probe = self._run_process(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                (
                    "stream=width,height,time_base:"
                    "frame=best_effort_timestamp_time,pkt_pts_time,pts_time"
                ),
                "-of",
                "json",
                item.path,
            ]
        )
        if not probe.stdout:
            raise ValueError("ffprobe returned empty output")
        try:
            description = json.loads(probe.stdout)
        except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as cause:
            raise ValueError("ffprobe returned invalid JSON") from cause
        streams = description.get("streams", [])
        if not streams:
            raise ValueError("ffprobe found no video stream")
        stream = streams[0]
        width = int(stream.get("width", 0))
        height = int(stream.get("height", 0))
        if width <= 0 or height <= 0:
            raise ValueError("ffprobe returned invalid video dimensions")
        time_base = stream.get("time_base")
        if not isinstance(time_base, str) or not time_base:
            raise ValueError("ffprobe returned no video time base")

        decoded = self._run_process(
            [
                "ffmpeg",
                "-v",
                "error",
                "-noautorotate",
                "-i",
                item.path,
                "-map",
                "0:v:0",
                "-vsync",
                "0",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ]
        )
        if not decoded.stdout:
            raise ValueError("ffmpeg returned empty video output")
        bytes_per_frame = width * height * 3
        if len(decoded.stdout) % bytes_per_frame != 0:
            raise ValueError(
                "ffmpeg output size does not match video dimensions"
            )
        frame_count = len(decoded.stdout) // bytes_per_frame
        if frame_count <= 0:
            raise ValueError("ffmpeg decoded no video frames")
        tensor = torch.frombuffer(
            bytearray(decoded.stdout),
            dtype=torch.uint8,
        )
        tensor = (
            tensor.reshape(frame_count, height, width, 3)
            .permute(0, 3, 1, 2)
            .to(dtype=torch.float32)
            .div_(255.0)
            .contiguous()
        )

        timestamp_values = (
            item.timestamps
            if item.timestamps is not None
            else self._probe_timestamps(description)
        )
        if len(timestamp_values) != frame_count:
            raise ValueError(
                "video timestamp/frame mismatch: "
                f"{len(timestamp_values)} timestamps for "
                f"{frame_count} frames"
            )
        timestamps = torch.tensor(
            timestamp_values,
            dtype=torch.float32,
        )
        seconds_per_grid = self._uniform_cadence(timestamp_values)
        return DecodedMedia(
            request=item,
            tensor=tensor,
            length=frame_count,
            timestamps=timestamps,
            seconds_per_grid=seconds_per_grid,
            metadata={
                "width": width,
                "height": height,
                "time_base": time_base,
            },
        )

    @staticmethod
    def _probe_timestamps(
        description: Mapping[str, object],
    ) -> tuple[float, ...]:
        values: list[float] = []
        frames = description.get("frames", [])
        if not isinstance(frames, list):
            raise ValueError("ffprobe returned invalid frame metadata")
        for frame in frames:
            if not isinstance(frame, dict):
                raise ValueError("ffprobe returned invalid frame metadata")
            raw_value = next(
                (
                    frame[name]
                    for name in (
                        "best_effort_timestamp_time",
                        "pkt_pts_time",
                        "pts_time",
                    )
                    if name in frame
                ),
                None,
            )
            if raw_value is None:
                raise ValueError("ffprobe frame has no timestamp")
            value = float(raw_value)
            _validate_non_negative_number(
                value,
                "video timestamps",
            )
            if values and value < values[-1]:
                raise ValueError("video timestamps must be monotonic")
            values.append(value)
        return tuple(values)

    @staticmethod
    def _uniform_cadence(
        timestamps: tuple[float, ...],
    ) -> float | None:
        if len(timestamps) < 2:
            return None
        cadence = float(timestamps[1]) - float(timestamps[0])
        if not math.isfinite(cadence) or cadence <= 0:
            return None
        for previous, current in zip(timestamps, timestamps[1:]):
            delta = float(current) - float(previous)
            if (
                not math.isfinite(delta)
                or delta <= 0
                or not math.isclose(
                    delta,
                    cadence,
                    rel_tol=1e-6,
                    abs_tol=1e-6,
                )
            ):
                return None
        return cadence


__all__ = [
    "DecodedMedia",
    "MediaRequest",
    "StrictMediaLoader",
]
