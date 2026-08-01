from __future__ import annotations

import pytest
import torch

from qwen3_omni_pretrain.profiles.qwen3_omni_reference.codec_streamer import (
    CodecOverlapState,
    ReferenceCodecStreamer,
)
from qwen3_omni_pretrain.profiles.qwen35_omni_inspired.configuration import (
    PREDECESSOR_CODEC_MODEL_ID,
    PREDECESSOR_CODEC_PROVENANCE,
    PREDECESSOR_CODEC_REVISION,
    QWEN35_BACKBONE_REVISION,
    Qwen35InspiredConfig,
)
from qwen3_omni_pretrain.profiles.qwen35_omni_inspired.talker import (
    PredecessorCodecProxy,
    PredecessorMTPProxy,
    PrototypeCode2Wav,
    Qwen35InspiredTalker,
)


def _profile_config() -> Qwen35InspiredConfig:
    return Qwen35InspiredConfig(
        backbone_config={
            "model_type": "qwen3_5_moe_text",
            "vocab_size": 128,
            "hidden_size": 32,
            "num_hidden_layers": 4,
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
            "num_experts": 4,
            "num_experts_per_tok": 2,
        },
        audio_config={
            "hidden_size": 32,
            "encoder_layers": 1,
            "attention_heads": 4,
            "intermediate_size": 64,
        },
        aria_config={},
        codec_proxy={
            "source_model": PREDECESSOR_CODEC_MODEL_ID,
            "source_revision": PREDECESSOR_CODEC_REVISION,
            "provenance_label": PREDECESSOR_CODEC_PROVENANCE,
        },
        source_revision=QWEN35_BACKBONE_REVISION,
    )


def tiny_qwen35_inspired_talker() -> Qwen35InspiredTalker:
    pytest.importorskip("transformers", minversion="5.2.0")
    return Qwen35InspiredTalker(
        thinker_hidden_size=32,
        text_vocab_size=128,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_token=2,
        expert_intermediate_size=16,
        shared_intermediate_size=16,
        manifest=_profile_config().profile_manifest,
    )


def tiny_predecessor_codec_proxy() -> PredecessorCodecProxy:
    decoder = PrototypeCode2Wav(samples_per_frame=8)
    mtp = PredecessorMTPProxy(conditioning_size=32)
    return PredecessorCodecProxy(
        mtp_proxy=mtp,
        decoder=decoder,
        codec_streamer=ReferenceCodecStreamer(
            decoder=decoder,
            left_context_frames=2,
            samples_per_frame=8,
        ),
    )


def valid_proxy_codes(frames: int) -> torch.LongTensor:
    codes = torch.arange(16 * frames).view(1, 16, frames)
    codes[:, 0].remainder_(3072)
    codes[:, 1:].remainder_(2048)
    return codes


def test_talker_predicts_main_codebook_and_marks_proxy():
    torch.manual_seed(31)
    talker = tiny_qwen35_inspired_talker()
    output = talker(
        thinker_hidden=torch.randn(1, 4, 32),
        text_input_ids=torch.tensor([[3, 4, 5, 6]]),
        main_codes=torch.tensor([[1, 2, 3]]),
        labels=torch.tensor([[2, 3, 4]]),
    )
    assert output.main_code_logits.shape == (1, 3, 3072)
    assert output.loss is not None and torch.isfinite(output.loss)
    assert output.manifest.assumptions.count("predecessor-codec-proxy") == 1


def test_talker_is_causal_with_respect_to_future_main_codes():
    torch.manual_seed(37)
    talker = tiny_qwen35_inspired_talker().eval()
    hidden = torch.randn(1, 3, 32)
    text = torch.tensor([[3, 4]])
    first = talker(
        thinker_hidden=hidden,
        text_input_ids=text,
        main_codes=torch.tensor([[1, 2, 3]]),
    ).main_code_logits
    second = talker(
        thinker_hidden=hidden,
        text_input_ids=text,
        main_codes=torch.tensor([[1, 2, 99]]),
    ).main_code_logits
    torch.testing.assert_close(first[:, :2], second[:, :2], atol=1e-5, rtol=1e-5)


def test_talker_all_ignored_labels_do_not_create_nan_loss():
    torch.manual_seed(41)
    talker = tiny_qwen35_inspired_talker()
    output = talker(
        thinker_hidden=torch.randn(1, 2, 32),
        text_input_ids=torch.tensor([[3, 4]]),
        main_codes=torch.tensor([[1, 2]]),
        labels=torch.full((1, 2), -100, dtype=torch.long),
    )
    assert output.loss is None


def test_proxy_predicts_all_residual_codebooks_in_range():
    proxy = tiny_predecessor_codec_proxy()
    residual = proxy.predict_residual_codes(
        main_codes=torch.tensor([[1, 2, 3]]),
        conditioning=torch.randn(1, 3, 32),
    )
    assert residual.shape == (1, 15, 3)
    assert int(residual.min()) >= 0 and int(residual.max()) < 2048


def test_proxy_chunk_decode_matches_offline_decode():
    proxy = tiny_predecessor_codec_proxy()
    codes = valid_proxy_codes(frames=10)
    state = CodecOverlapState.empty(request_id="r1")
    chunks = []
    for part, final in (
        (codes[:, :, :4], False),
        (codes[:, :, 4:7], False),
        (codes[:, :, 7:], True),
    ):
        output = proxy.decode_code_chunk(
            codes=part,
            state=state,
            request_id="r1",
            final=final,
        )
        chunks.append(output.waveform)
        state = output.state
    torch.testing.assert_close(torch.cat(chunks, dim=-1), proxy.decode_codes(codes))
    assert output.final
