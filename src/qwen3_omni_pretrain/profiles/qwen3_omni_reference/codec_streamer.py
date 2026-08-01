from __future__ import annotations

from dataclasses import dataclass

import torch


_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def _request_id(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("request_id must be a non-empty string")
    return value


@dataclass(frozen=True)
class CodecOverlapState:
    request_id: str
    left_codes: torch.LongTensor | None
    emitted_frames: int
    finished: bool = False

    def __post_init__(self) -> None:
        _request_id(self.request_id)
        if type(self.emitted_frames) is not int or self.emitted_frames < 0:
            raise ValueError("emitted_frames must be a non-negative integer")
        if type(self.finished) is not bool:
            raise TypeError("finished must be a boolean")
        if self.left_codes is not None:
            if not isinstance(self.left_codes, torch.Tensor):
                raise TypeError("left_codes must be a tensor or None")
            if self.left_codes.dtype not in _INTEGER_DTYPES:
                raise TypeError("left_codes must have an integer dtype")
            if self.left_codes.ndim != 3 or self.left_codes.shape[1] != 16:
                raise ValueError("left_codes must have shape [B,16,T]")
            object.__setattr__(
                self,
                "left_codes",
                self.left_codes.detach().clone().to(torch.long),
            )

    @classmethod
    def empty(cls, request_id: str) -> CodecOverlapState:
        return cls(request_id=request_id, left_codes=None, emitted_frames=0)

    @property
    def batch_size(self) -> int | None:
        return None if self.left_codes is None else self.left_codes.shape[0]


@dataclass(frozen=True)
class WaveformChunk:
    waveform: torch.Tensor
    state: CodecOverlapState
    sample_rate: int
    final: bool

    def __post_init__(self) -> None:
        if not isinstance(self.waveform, torch.Tensor):
            raise TypeError("waveform must be a tensor")
        if self.waveform.ndim != 2 or self.waveform.shape[0] <= 0:
            raise ValueError("waveform must have shape [B,samples]")
        if not self.waveform.is_floating_point():
            raise TypeError("waveform must have a floating dtype")
        if not bool(torch.isfinite(self.waveform).all().item()):
            raise ValueError("waveform must contain finite values")
        if not isinstance(self.state, CodecOverlapState):
            raise TypeError("state must be CodecOverlapState")
        if type(self.sample_rate) is not int or self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if type(self.final) is not bool or self.final != self.state.finished:
            raise ValueError("WaveformChunk final flag must match state.finished")


class ReferenceCodecStreamer:
    implementation = "overlap-recompute"
    codec_incremental_output = True
    end_to_end_streaming = False

    def __init__(
        self,
        *,
        decoder: object,
        left_context_frames: int,
        samples_per_frame: int,
        sample_rate: int = 24_000,
    ) -> None:
        if not callable(getattr(decoder, "chunked_decode", None)):
            raise TypeError("decoder must define chunked_decode")
        for name, value in (
            ("left_context_frames", left_context_frames),
            ("samples_per_frame", samples_per_frame),
            ("sample_rate", sample_rate),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.decoder = decoder
        self.left_context_frames = left_context_frames
        self.samples_per_frame = samples_per_frame
        self.sample_rate = sample_rate

    @staticmethod
    def _validate_codes(codes: object) -> torch.Tensor:
        if not isinstance(codes, torch.Tensor):
            raise TypeError("codes must be a tensor")
        if codes.dtype not in _INTEGER_DTYPES:
            raise TypeError("codes must have an integer dtype")
        if codes.ndim != 3 or codes.shape[0] <= 0 or codes.shape[1] != 16:
            raise ValueError("codes must have shape [B,16,T]")
        if codes.shape[2] <= 0:
            raise ValueError("code chunks must contain at least one frame")
        return codes.to(torch.long)

    def push(
        self,
        *,
        codes: torch.LongTensor,
        state: CodecOverlapState,
        request_id: str,
        final: bool,
    ) -> WaveformChunk:
        codes = self._validate_codes(codes)
        if not isinstance(state, CodecOverlapState):
            raise TypeError("state must be CodecOverlapState")
        request_id = _request_id(request_id)
        if request_id != state.request_id:
            raise ValueError("codec state belongs to a different request")
        if state.finished:
            raise ValueError("final codec state cannot be reused")
        if type(final) is not bool:
            raise TypeError("final must be a boolean")
        if state.left_codes is not None:
            if state.left_codes.shape[0] != codes.shape[0]:
                raise ValueError("codec batch size changed across chunks")
            if state.left_codes.device != codes.device:
                raise ValueError("codec state and codes must share a device")
            combined = torch.cat((state.left_codes, codes), dim=2)
            context_frames = state.left_codes.shape[2]
        else:
            combined = codes
            context_frames = 0
        decoded = self.decoder.chunked_decode(combined)
        if not isinstance(decoded, torch.Tensor):
            raise TypeError("decoder.chunked_decode must return a tensor")
        if decoded.ndim == 3 and decoded.shape[1] == 1:
            decoded = decoded[:, 0]
        if decoded.ndim != 2 or decoded.shape[0] != codes.shape[0]:
            raise ValueError("decoded waveform must have shape [B,samples]")
        if not decoded.is_floating_point() or not bool(torch.isfinite(decoded).all().item()):
            raise ValueError("decoded waveform must contain finite floating values")
        drop_samples = context_frames * self.samples_per_frame
        expected_minimum = combined.shape[2] * self.samples_per_frame
        if decoded.shape[1] < expected_minimum or decoded.shape[1] < drop_samples:
            raise ValueError("decoder output is shorter than the configured frame rate")
        waveform = decoded[:, drop_samples:].detach().clone()
        retain = min(self.left_context_frames, combined.shape[2])
        next_state = CodecOverlapState(
            request_id=request_id,
            left_codes=combined[:, :, -retain:],
            emitted_frames=state.emitted_frames + codes.shape[2],
            finished=final,
        )
        return WaveformChunk(
            waveform=waveform,
            state=next_state,
            sample_rate=self.sample_rate,
            final=final,
        )


__all__ = [
    "CodecOverlapState",
    "ReferenceCodecStreamer",
    "WaveformChunk",
]
