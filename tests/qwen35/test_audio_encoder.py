from __future__ import annotations

import torch

from qwen3_omni_pretrain.multimodal.types import MediaSource
from qwen3_omni_pretrain.profiles.qwen35_omni_inspired.audio_encoder import (
    Qwen35AuTEncoder,
    Qwen35FrontendState,
)
from qwen3_omni_pretrain.profiles.qwen35_omni_inspired.configuration import (
    Qwen35AuTConfig,
)


def tiny_qwen35_audio_encoder() -> Qwen35AuTEncoder:
    return Qwen35AuTEncoder(
        Qwen35AuTConfig(
            hidden_size=32,
            encoder_layers=1,
            attention_heads=4,
            intermediate_size=64,
        ),
        backbone_hidden_size=24,
    )


def _sources(count: int) -> tuple[MediaSource, ...]:
    return tuple(MediaSource(index, 0, f"audio-{index}") for index in range(count))


def test_two_seconds_produces_about_twelve_or_thirteen_tokens():
    encoder = tiny_qwen35_audio_encoder().eval()
    output = encoder(
        torch.zeros(1, 32_000),
        lengths=torch.tensor([32_000]),
        sources=_sources(1),
    )
    assert int(output.attention_mask.sum()) in {12, 13}
    valid = output.timestamps[0, output.attention_mask[0]]
    torch.testing.assert_close(
        torch.diff(valid),
        torch.full_like(valid[1:], 0.16),
        atol=1e-6,
        rtol=0,
    )


def test_padded_audio_does_not_create_valid_tokens():
    encoder = tiny_qwen35_audio_encoder().eval()
    output = encoder(
        torch.zeros(2, 32_000),
        lengths=torch.tensor([16_000, 32_000]),
        sources=_sources(2),
    )
    assert output.attention_mask[0].sum() < output.attention_mask[1].sum()
    assert not output.attention_mask[0, int(output.attention_mask[0].sum()) :].any()


def test_valid_output_is_unchanged_by_right_padding():
    torch.manual_seed(3)
    encoder = tiny_qwen35_audio_encoder().eval()
    waveform = torch.randn(1, 16_000)
    alone = encoder(
        waveform,
        lengths=torch.tensor([16_000]),
        sources=(MediaSource(0, 0, "alone"),),
    )
    batched_waveform = torch.zeros(2, 32_000)
    batched_waveform[0, :16_000] = waveform[0]
    batched = encoder(
        batched_waveform,
        lengths=torch.tensor([16_000, 32_000]),
        sources=(MediaSource(0, 0, "alone"), MediaSource(1, 0, "long")),
    )
    count = int(alone.attention_mask.sum())
    torch.testing.assert_close(
        batched.embeddings[0, :count],
        alone.embeddings[0, :count],
        atol=1e-5,
        rtol=1e-5,
    )


def test_chunked_mel_conv_frontend_matches_offline_frames():
    torch.manual_seed(5)
    encoder = tiny_qwen35_audio_encoder().eval()
    waveform = torch.randn(1, 24_000)
    expected = encoder.frontend(waveform)
    state = Qwen35FrontendState.empty()
    pieces = []
    boundaries = (2_111, 7_333, 15_001, 24_000)
    start = 0
    for end in boundaries:
        chunk = encoder.frontend.push_chunk(
            waveform[:, start:end],
            state=state,
            final=end == boundaries[-1],
        )
        pieces.append(chunk.embeddings)
        state = chunk.state
        start = end
    actual = torch.cat(pieces, dim=1)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    assert state.final
    assert not hasattr(encoder, "push_chunk")


def test_audio_forward_backward_is_finite():
    torch.manual_seed(13)
    encoder = tiny_qwen35_audio_encoder().train()
    waveform = torch.randn(1, 8_000, requires_grad=True)
    output = encoder(
        waveform,
        lengths=torch.tensor([8_000]),
        sources=_sources(1),
    )
    loss = output.embeddings[output.attention_mask].square().mean()
    loss.backward()
    assert torch.isfinite(output.embeddings).all()
    assert waveform.grad is not None and torch.isfinite(waveform.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in encoder.parameters()
    )
