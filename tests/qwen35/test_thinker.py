from __future__ import annotations

import pytest
import torch
from torch import nn

pytest.importorskip("transformers", minversion="5.2.0")

from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

from qwen3_omni_pretrain.multimodal.encoders import (
    PatchVisionEncoder,
    TemporalVideoEncoder,
)
from qwen3_omni_pretrain.multimodal.prefill import MultimodalPrefillPipeline
from qwen3_omni_pretrain.multimodal.sequence_assembler import SequenceAssembler
from qwen3_omni_pretrain.multimodal.tokenization.schema import (
    ResolvedMultimodalTokens,
)
from qwen3_omni_pretrain.multimodal.types import PositionBatch
from qwen3_omni_pretrain.profiles.qwen35_omni_inspired.audio_encoder import (
    Qwen35AudioSequenceAdapter,
    Qwen35AuTEncoder,
)
from qwen3_omni_pretrain.profiles.qwen35_omni_inspired.configuration import (
    PREDECESSOR_CODEC_MODEL_ID,
    PREDECESSOR_CODEC_PROVENANCE,
    PREDECESSOR_CODEC_REVISION,
    QWEN35_BACKBONE_REVISION,
    Qwen35AuTConfig,
    Qwen35InspiredConfig,
)
from qwen3_omni_pretrain.profiles.qwen35_omni_inspired.thinker import (
    Qwen35InspiredThinker,
)
from qwen3_omni_pretrain.profiles.qwen35_omni_inspired.timestamp_alignment import (
    Qwen35TimestampExpansionPolicy,
    build_qwen35_position_builder,
)
from qwen3_omni_pretrain.runtime.protocols import (
    ModelDecodeInputs,
    ModelPrefillInputs,
)
from qwen3_omni_pretrain.runtime.state import StateOwner


class _Tokenizer:
    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        value = sum(text.encode("utf-8")) % 80 + 20
        return {"input_ids": [value]}


def tiny_public_config() -> Qwen3_5MoeTextConfig:
    return Qwen3_5MoeTextConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )


def tiny_profile_config(public: Qwen3_5MoeTextConfig) -> Qwen35InspiredConfig:
    return Qwen35InspiredConfig(
        backbone_config=public.to_dict(),
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


def tiny_pipeline(public: Qwen3_5MoeForCausalLM) -> MultimodalPrefillPipeline:
    audio = Qwen35AudioSequenceAdapter(
        Qwen35AuTEncoder(
            Qwen35AuTConfig(
                hidden_size=32,
                encoder_layers=1,
                attention_heads=4,
                intermediate_size=64,
            ),
            backbone_hidden_size=32,
        )
    )
    return MultimodalPrefillPipeline(
        tokens=ResolvedMultimodalTokens(
            image_pad=11,
            video_pad=12,
            audio_pad=13,
            vision_start=14,
            vision_end=15,
            audio_start=16,
            audio_end=17,
        ),
        image_encoder=PatchVisionEncoder(
            in_channels=3,
            hidden_size=32,
            patch_size=2,
        ),
        video_encoder=TemporalVideoEncoder(hidden_size=32),
        audio_encoder=audio,
        assembler=SequenceAssembler(),
        expansion_policy=Qwen35TimestampExpansionPolicy(tokenizer=_Tokenizer()),
        position_builder=build_qwen35_position_builder(),
        pad_token_id=0,
        joint_separator_token_ids=frozenset(),
        max_assembled_length=128,
    )


def tiny_thinker() -> Qwen35InspiredThinker:
    torch.manual_seed(23)
    config = tiny_public_config()
    public = Qwen3_5MoeForCausalLM(config)
    return Qwen35InspiredThinker.from_public_model(
        public,
        prefill_pipeline=tiny_pipeline(public),
        profile_config=tiny_profile_config(config),
    )


def _position(values: torch.Tensor) -> PositionBatch:
    return PositionBatch(
        position_ids=values.unsqueeze(0),
        rope_deltas=torch.zeros(
            values.shape[0], 1, dtype=values.dtype, device=values.device
        ),
        axis_names=("sequence",),
    )


def test_text_only_adapter_matches_public_backbone():
    torch.manual_seed(19)
    config = tiny_public_config()
    public = Qwen3_5MoeForCausalLM(config).eval()
    thinker = Qwen35InspiredThinker.from_public_model(
        public,
        prefill_pipeline=tiny_pipeline(public),
        profile_config=tiny_profile_config(config),
    ).eval()
    ids = torch.tensor([[3, 4, 5, 6]])
    expected = public(input_ids=ids, use_cache=False).logits
    actual = thinker(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        decoded_media=(),
        labels=None,
        request_id="r1",
        use_cache=False,
    )["logits"]
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_typed_cached_decode_matches_uncached_public_model():
    thinker = tiny_thinker().eval()
    owner = StateOwner.fresh("cache")
    prompt = torch.tensor([[3, 4, 5]])
    prefill = thinker.prefill(
        inputs=ModelPrefillInputs(
            input_ids=prompt,
            inputs_embeds=None,
            key_valid_mask=torch.ones_like(prompt, dtype=torch.bool),
            position_batch=_position(torch.arange(3).view(1, 3)),
        ),
        owner=owner,
        use_cache=True,
    )
    assert prefill.decoder_state is not None
    decoded = thinker.decode(
        inputs=ModelDecodeInputs(
            token_ids=torch.tensor([[6]]),
            current_key_valid_mask=torch.ones(1, 1, dtype=torch.bool),
            position_batch=_position(torch.tensor([[3]])),
            decoder_state=prefill.decoder_state,
        ),
        owner=owner,
    )
    expected = thinker.backbone(
        input_ids=torch.tensor([[3, 4, 5, 6]]),
        use_cache=False,
    ).logits[:, -1]
    torch.testing.assert_close(decoded.logits[:, -1], expected, atol=1e-5, rtol=1e-5)
    assert decoded.decoder_state is not None
    assert decoded.decoder_state.seen_tokens.tolist() == [4]
    assert decoded.decoder_state.owner == owner


def test_parameter_groups_are_disjoint_and_complete():
    thinker = tiny_thinker()
    groups = thinker.named_parameter_groups()
    ids = [id(parameter) for group in groups.values() for parameter in group]
    assert len(ids) == len(set(ids))
    assert set(ids) == {id(parameter) for parameter in thinker.parameters()}
    assert groups["thinker"]
    assert groups["vision_encoder"]
    assert groups["audio_encoder"]
    assert groups["projector"]


def test_cached_padding_rejected_before_public_backbone_call():
    thinker = tiny_thinker().eval()
    owner = StateOwner.fresh("padded")
    ids = torch.tensor([[3, 4], [5, 0]])
    mask = torch.tensor([[True, True], [True, False]])
    try:
        thinker.prefill(
            inputs=ModelPrefillInputs(
                input_ids=ids,
                inputs_embeds=None,
                key_valid_mask=mask,
                position_batch=_position(torch.tensor([[0, 1], [0, 0]])),
            ),
            owner=owner,
            use_cache=True,
        )
    except ValueError as exc:
        assert "equal-length" in str(exc) and "without padding" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("padded cached prefill must fail")
