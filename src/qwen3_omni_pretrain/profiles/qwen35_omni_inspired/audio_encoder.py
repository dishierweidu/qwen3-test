from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torchaudio
from torch import nn

from qwen3_omni_pretrain.multimodal.io import DecodedMedia
from qwen3_omni_pretrain.multimodal.modalities import MediaModality
from qwen3_omni_pretrain.multimodal.types import MediaSequence, MediaSource
from qwen3_omni_pretrain.profiles.qwen35_omni_inspired.configuration import (
    Qwen35AuTConfig,
)


def _conv_output_length(length: torch.Tensor) -> torch.Tensor:
    # Kernel 3, stride 2 and two samples of causal left padding.
    return torch.div(length + 1, 2, rounding_mode="floor")


def _mel_output_length(length: torch.Tensor) -> torch.Tensor:
    return torch.where(
        length >= 400,
        torch.div(length - 400, 160, rounding_mode="floor") + 1,
        torch.zeros_like(length),
    )


def _frontend_output_lengths(lengths: torch.Tensor) -> torch.Tensor:
    result = _mel_output_length(lengths)
    for _ in range(4):
        result = _conv_output_length(result)
    return result


@dataclass(frozen=True)
class Qwen35FrontendState:
    waveform_history: torch.Tensor
    emitted_frames: int
    final: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.waveform_history, torch.Tensor):
            raise TypeError("waveform_history must be a tensor")
        if self.waveform_history.ndim != 2 or self.waveform_history.shape[0] <= 0:
            raise ValueError("waveform_history must have shape [B, samples]")
        if not self.waveform_history.is_floating_point():
            raise TypeError("waveform_history must have a floating dtype")
        if type(self.emitted_frames) is not int or self.emitted_frames < 0:
            raise ValueError("emitted_frames must be a non-negative integer")
        if type(self.final) is not bool:
            raise TypeError("final must be a boolean")
        object.__setattr__(self, "waveform_history", self.waveform_history.detach().clone())

    @classmethod
    def empty(
        cls,
        *,
        batch_size: int = 1,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> Qwen35FrontendState:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return cls(torch.empty(batch_size, 0, device=device, dtype=dtype), 0)


@dataclass(frozen=True)
class Qwen35FrontendChunk:
    embeddings: torch.Tensor
    state: Qwen35FrontendState
    final: bool


class Qwen35MelFrontend(nn.Module):
    """Pinned center=False log-Mel extraction; it owns no Transformer state."""

    def __init__(self, config: Qwen35AuTConfig) -> None:
        super().__init__()
        self.config = config
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16_000,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=128,
            center=False,
            power=2.0,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if not isinstance(waveform, torch.Tensor):
            raise TypeError("waveform must be a tensor")
        if waveform.ndim != 2 or waveform.shape[0] <= 0:
            raise ValueError("waveform must have shape [B, samples]")
        if waveform.shape[1] < 400:
            return waveform.new_empty((waveform.shape[0], 128, 0))
        features = self.mel(waveform)
        return torch.log(features.clamp_min(1e-10))


class _CausalStride2ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=0,
        )
        self.activation = nn.GELU()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # Frequency is symmetrically padded; time is left-padded only. This
        # makes every emitted frame stable without inventing future samples.
        hidden = F.pad(hidden, (2, 0, 1, 1))
        return self.activation(self.conv(hidden))


class Qwen35ConvFrontend(nn.Module):
    """6.25 Hz Mel/Conv frontend with correctness-first chunk recomputation."""

    def __init__(self, config: Qwen35AuTConfig) -> None:
        super().__init__()
        self.config = config
        self.mel = Qwen35MelFrontend(config)
        channels = (1, 8, 16, 32, 64)
        self.conv_blocks = nn.ModuleList(
            _CausalStride2ConvBlock(channels[index], channels[index + 1])
            for index in range(4)
        )
        frequency_bins = config.num_mel_bins
        for _ in range(4):
            frequency_bins = (frequency_bins + 1) // 2
        self.input_projection = nn.Linear(
            channels[-1] * frequency_bins,
            config.hidden_size,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        features = self.mel(waveform).unsqueeze(1)
        if features.shape[-1] == 0:
            return waveform.new_empty(
                (waveform.shape[0], 0, self.config.hidden_size)
            )
        for block in self.conv_blocks:
            features = block(features)
        features = features.permute(0, 3, 1, 2).flatten(2)
        return self.input_projection(features)

    def push_chunk(
        self,
        waveform_chunk: torch.Tensor,
        *,
        state: Qwen35FrontendState,
        final: bool = False,
    ) -> Qwen35FrontendChunk:
        if not isinstance(state, Qwen35FrontendState):
            raise TypeError("state must be Qwen35FrontendState")
        if state.final:
            raise ValueError("final frontend state cannot be reused")
        if type(final) is not bool:
            raise TypeError("final must be a boolean")
        if not isinstance(waveform_chunk, torch.Tensor):
            raise TypeError("waveform_chunk must be a tensor")
        if (
            waveform_chunk.ndim != 2
            or waveform_chunk.shape[0] != state.waveform_history.shape[0]
        ):
            raise ValueError("waveform_chunk must have matching shape [B, samples]")
        if (
            waveform_chunk.dtype != state.waveform_history.dtype
            or waveform_chunk.device != state.waveform_history.device
        ):
            raise ValueError("waveform chunk and frontend state must share dtype/device")
        history = torch.cat((state.waveform_history, waveform_chunk), dim=1)
        complete = self(history)
        if state.emitted_frames > complete.shape[1]:
            raise ValueError("frontend state emitted_frames exceeds available output")
        newly_stable = complete[:, state.emitted_frames :]
        next_state = Qwen35FrontendState(
            waveform_history=history,
            emitted_frames=complete.shape[1],
            final=final,
        )
        return Qwen35FrontendChunk(newly_stable, next_state, final)


class Qwen35AuTEncoder(nn.Module):
    """Offline-only prototype AuT Transformer over the fixed 6.25 Hz frontend."""

    def __init__(
        self,
        config: Qwen35AuTConfig,
        *,
        backbone_hidden_size: int,
    ) -> None:
        super().__init__()
        if type(backbone_hidden_size) is not int or backbone_hidden_size <= 0:
            raise ValueError("backbone_hidden_size must be positive")
        self.config = config
        self.frontend = Qwen35ConvFrontend(config)
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.attention_heads,
            dim_feedforward=config.intermediate_size,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=config.encoder_layers,
            enable_nested_tensor=False,
        )
        self.projector = nn.Linear(config.hidden_size, backbone_hidden_size)

    def forward(
        self,
        waveform: torch.Tensor,
        *,
        lengths: torch.Tensor,
        sources: tuple[MediaSource, ...],
    ) -> MediaSequence:
        if not isinstance(waveform, torch.Tensor) or waveform.ndim != 2:
            raise ValueError("waveform must have shape [B, samples]")
        if not waveform.is_floating_point() or not bool(torch.isfinite(waveform).all().item()):
            raise ValueError("waveform must contain finite floating values")
        if not isinstance(lengths, torch.Tensor) or lengths.dtype is not torch.long:
            raise TypeError("lengths must have dtype torch.long")
        if lengths.shape != (waveform.shape[0],):
            raise ValueError("lengths must have shape [B]")
        if lengths.device != waveform.device:
            raise ValueError("lengths and waveform must share a device")
        if bool((lengths < 400).any().item()) or bool(
            (lengths > waveform.shape[1]).any().item()
        ):
            raise ValueError("audio lengths must be between 400 and padded samples")
        if len(sources) != waveform.shape[0] or any(
            not isinstance(source, MediaSource) for source in sources
        ):
            raise ValueError("one MediaSource is required per waveform row")

        hidden = self.frontend(waveform)
        output_lengths = _frontend_output_lengths(lengths)
        steps = hidden.shape[1]
        positions = torch.arange(steps, device=waveform.device).view(1, -1)
        valid_mask = positions < output_lengths.view(-1, 1)
        if not bool(valid_mask.any(dim=1).all().item()):
            raise ValueError("each audio row must produce at least one AuT token")
        hidden = self.transformer(
            hidden,
            src_key_padding_mask=~valid_mask,
        )
        hidden = self.projector(hidden)
        hidden = hidden.masked_fill(~valid_mask.unsqueeze(-1), 0)
        timestamps = (
            torch.arange(steps, device=waveform.device, dtype=torch.float32)
            / self.config.output_frame_hz
        ).view(1, -1).expand(waveform.shape[0], -1).clone()
        timestamps.masked_fill_(~valid_mask, 0)
        output = MediaSequence(
            embeddings=hidden,
            attention_mask=valid_mask,
            modality=MediaModality.AUDIO,
            sources=sources,
            grid=None,
            timestamps=timestamps,
            seconds_per_grid=None,
        )
        output.validate()
        return output


class Qwen35AudioSequenceAdapter(nn.Module):
    def __init__(self, encoder: Qwen35AuTEncoder) -> None:
        super().__init__()
        if not isinstance(encoder, Qwen35AuTEncoder):
            raise TypeError("encoder must be Qwen35AuTEncoder")
        self.encoder = encoder

    @property
    def projector(self) -> nn.Module:
        return self.encoder.projector

    @property
    def hidden_size(self) -> int:
        return self.encoder.projector.out_features

    def forward(self, items: tuple[DecodedMedia, ...]) -> MediaSequence:
        if not isinstance(items, tuple) or not items:
            raise ValueError("audio adapter requires a non-empty item tuple")
        for item in items:
            if not isinstance(item, DecodedMedia):
                raise TypeError("audio items must be DecodedMedia")
            if item.request.modality is not MediaModality.AUDIO:
                raise ValueError("Qwen35AudioSequenceAdapter accepts audio only")
            if item.metadata.get("sample_rate") != self.encoder.config.sample_rate:
                raise ValueError("decoded audio sample rate must be 16000 Hz")
        parameter = next(self.encoder.parameters())
        lengths = torch.tensor(
            [item.length for item in items],
            dtype=torch.long,
            device=parameter.device,
        )
        waveforms = torch.zeros(
            len(items),
            int(lengths.max().item()),
            dtype=parameter.dtype,
            device=parameter.device,
        )
        sources = []
        for row, item in enumerate(items):
            waveforms[row, : item.length] = item.tensor[0].to(
                device=parameter.device,
                dtype=parameter.dtype,
            )
            sources.append(
                MediaSource(
                    sample_index=item.request.sample_index,
                    item_index=item.request.item_index,
                    source_id=item.request.source_id,
                )
            )
        output = self.encoder(
            waveforms,
            lengths=lengths,
            sources=tuple(sources),
        )
        offsets = torch.tensor(
            [item.request.timeline_offset_seconds for item in items],
            dtype=output.timestamps.dtype,
            device=output.timestamps.device,
        ).view(-1, 1)
        timestamps = output.timestamps + offsets * output.attention_mask
        result = MediaSequence(
            embeddings=output.embeddings,
            attention_mask=output.attention_mask,
            modality=output.modality,
            sources=output.sources,
            grid=output.grid,
            timestamps=timestamps,
            seconds_per_grid=output.seconds_per_grid,
        )
        result.validate()
        return result


__all__ = [
    "Qwen35AudioSequenceAdapter",
    "Qwen35AuTEncoder",
    "Qwen35ConvFrontend",
    "Qwen35FrontendChunk",
    "Qwen35FrontendState",
    "Qwen35MelFrontend",
]
