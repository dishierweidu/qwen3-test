from __future__ import annotations

import torch

from qwen3_omni_pretrain.profiles.qwen3_omni_reference.codec_streamer import (
    CodecOverlapState,
    ReferenceCodecStreamer,
)


class DeterministicFakeCode2Wav:
    def __init__(self, samples_per_frame: int = 4) -> None:
        self.samples_per_frame = samples_per_frame

    def chunked_decode(self, codes: torch.LongTensor) -> torch.Tensor:
        frames = codes.to(torch.float32).mean(dim=1)
        offsets = torch.arange(
            self.samples_per_frame,
            dtype=torch.float32,
            device=codes.device,
        ).view(1, 1, -1)
        return (frames.unsqueeze(-1) + offsets / 10).flatten(1)


def test_codec_stream_concatenation_matches_offline_decoder():
    decoder = DeterministicFakeCode2Wav(samples_per_frame=4)
    codes = torch.arange(16 * 10).view(1, 16, 10)
    streamer = ReferenceCodecStreamer(
        decoder=decoder,
        left_context_frames=2,
        samples_per_frame=4,
    )
    state = CodecOverlapState.empty(request_id="r1")
    outputs = []
    for chunk, final in (
        (codes[:, :, :4], False),
        (codes[:, :, 4:7], False),
        (codes[:, :, 7:], True),
    ):
        result = streamer.push(
            codes=chunk,
            state=state,
            request_id="r1",
            final=final,
        )
        outputs.append(result.waveform)
        state = result.state
    torch.testing.assert_close(
        torch.cat(outputs, dim=-1),
        decoder.chunked_decode(codes),
    )
    assert state.finished and state.emitted_frames == 10


def test_codec_stream_rejects_cross_request_and_final_reuse():
    decoder = DeterministicFakeCode2Wav()
    streamer = ReferenceCodecStreamer(
        decoder=decoder,
        left_context_frames=2,
        samples_per_frame=4,
    )
    codes = torch.zeros(1, 16, 2, dtype=torch.long)
    state = CodecOverlapState.empty("one")
    try:
        streamer.push(codes=codes, state=state, request_id="two", final=False)
    except ValueError as exc:
        assert "different request" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("cross-request codec state must fail")
    output = streamer.push(codes=codes, state=state, request_id="one", final=True)
    try:
        streamer.push(codes=codes, state=output.state, request_id="one", final=True)
    except ValueError as exc:
        assert "final" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("final codec state reuse must fail")
