from __future__ import annotations

import pytest
import torch

from qwen3_omni_pretrain.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeConfig,
)
from qwen3_omni_pretrain.models.qwen3_omni_moe.modeling_thinker_vision_audio import (
    Qwen3OmniMoeThinkerVisionAudioModel,
)
from qwen3_omni_pretrain.multimodal.types import PositionBatch
from qwen3_omni_pretrain.runtime import (
    DecoderState,
    LegacyMediaPrefillInputs,
    ModelDecodeInputs,
    ModelPrefillInputs,
    StateOwner,
)


def config() -> Qwen3OmniMoeConfig:
    return Qwen3OmniMoeConfig(
        vocab_size=32,
        thinker_config={
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 16,
            "use_moe": False,
            "use_flash_attention": False,
        },
    )


class CountingEncoder(torch.nn.Module):
    def __init__(self, hidden_size: int, value: float) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.value = value
        self.calls = 0
        self.batch_sizes = []

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        self.batch_sizes.append(values.shape[0])
        return torch.full(
            (values.shape[0], 1, self.hidden_size),
            self.value,
            dtype=values.dtype,
            device=values.device,
        )


def text_positions(values, mask: torch.Tensor) -> PositionBatch:
    position_ids = (
        values.detach().clone().to(dtype=torch.long)
        if isinstance(values, torch.Tensor)
        else torch.tensor(values, dtype=torch.long)
    )
    result = PositionBatch(
        position_ids=position_ids.unsqueeze(0),
        rope_deltas=torch.zeros(mask.shape[0], 1, dtype=torch.long),
        axis_names=("sequence",),
    )
    result.validate(mask)
    return result


def media_inputs(
    has_image: list[bool],
    has_audio: list[bool],
) -> LegacyMediaPrefillInputs:
    batch = len(has_image)
    return LegacyMediaPrefillInputs(
        pixel_values=(
            torch.zeros(batch, 3, 2, 2) if any(has_image) else None
        ),
        audio_values=(
            torch.zeros(batch, 8) if any(has_audio) else None
        ),
        has_image=torch.tensor(has_image, dtype=torch.bool),
        has_audio=torch.tensor(has_audio, dtype=torch.bool),
    )


def typed_prefill(
    ids: torch.Tensor,
    media: LegacyMediaPrefillInputs | None,
    *,
    mask: torch.Tensor | None = None,
) -> ModelPrefillInputs:
    if mask is None:
        mask = torch.ones_like(ids, dtype=torch.bool)
    values = torch.arange(ids.shape[1]).expand(ids.shape[0], -1)
    values = values.masked_fill(~mask, 0)
    return ModelPrefillInputs(
        input_ids=ids,
        inputs_embeds=None,
        key_valid_mask=mask,
        position_batch=text_positions(values, mask),
        media=media,
    )


@pytest.mark.parametrize(
    ("images", "audio", "expected"),
    [
        ([False], [False], (0, 0)),
        ([True], [False], (1, 0)),
        ([False], [True], (0, 1)),
        ([True], [True], (1, 1)),
    ],
)
def test_present_encoder_runs_once_and_absent_encoder_never_runs(
    images,
    audio,
    expected,
):
    model = Qwen3OmniMoeThinkerVisionAudioModel(config()).eval()
    model.vision_encoder = CountingEncoder(8, 1.0)
    model.audio_encoder = CountingEncoder(8, 2.0)
    owner = StateOwner.fresh("media")
    output = model.prefill(
        inputs=typed_prefill(
            torch.tensor([[3, 4]]),
            media_inputs(images, audio),
        ),
        owner=owner,
        use_cache=True,
    )
    assert (model.vision_encoder.calls, model.audio_encoder.calls) == expected
    state = output.decoder_state
    assert state is not None
    assert state.processed_media is not None
    assert state.processed_media.prefix_storage_length == 2
    assert state.position.key_valid_mask[:, :2].tolist() == [
        [images[0], audio[0]]
    ]

    decoded = model.decode(
        inputs=ModelDecodeInputs(
            token_ids=torch.tensor([[5]]),
            current_key_valid_mask=torch.ones(1, 1, dtype=torch.bool),
            position_batch=text_positions([[4]], torch.ones(1, 1, dtype=torch.bool)),
            decoder_state=state,
        ),
        owner=owner,
    )
    assert decoded.decoder_state is not None
    assert (model.vision_encoder.calls, model.audio_encoder.calls) == expected


def test_mixed_rows_encode_only_present_items_then_scatter_fixed_slots():
    model = Qwen3OmniMoeThinkerVisionAudioModel(config()).eval()
    model.vision_encoder = CountingEncoder(8, 1.0)
    model.audio_encoder = CountingEncoder(8, 2.0)
    output = model.prefill(
        inputs=typed_prefill(
            torch.tensor([[3], [4]]),
            media_inputs([True, False], [False, True]),
        ),
        owner=StateOwner.fresh("mixed"),
        use_cache=True,
    )
    assert model.vision_encoder.batch_sizes == [1]
    assert model.audio_encoder.batch_sizes == [1]
    assert output.decoder_state is not None
    assert output.decoder_state.position.key_valid_mask.tolist() == [
        [True, False, True],
        [False, True, True],
    ]
    assert output.decoder_state.position.cached.position_ids.tolist() == [
        [[0, 0, 2], [0, 1, 2]]
    ]


def test_cached_multimodal_decode_matches_uncached_full_request():
    torch.manual_seed(31)
    cached_model = Qwen3OmniMoeThinkerVisionAudioModel(config()).eval()
    cached_model.vision_encoder = CountingEncoder(8, 1.0)
    cached_model.audio_encoder = CountingEncoder(8, 2.0)
    reference_model = Qwen3OmniMoeThinkerVisionAudioModel(config()).eval()
    reference_model.load_state_dict(cached_model.state_dict(), strict=False)
    reference_model.vision_encoder = CountingEncoder(8, 1.0)
    reference_model.audio_encoder = CountingEncoder(8, 2.0)
    media = media_inputs([True], [True])
    owner = StateOwner.fresh("parity")
    first = cached_model.prefill(
        inputs=typed_prefill(torch.tensor([[3, 4]]), media),
        owner=owner,
        use_cache=True,
    )
    state = first.decoder_state
    assert state is not None
    second = cached_model.decode(
        inputs=ModelDecodeInputs(
            token_ids=torch.tensor([[5]]),
            current_key_valid_mask=torch.ones(1, 1, dtype=torch.bool),
            position_batch=text_positions([[4]], torch.ones(1, 1, dtype=torch.bool)),
            decoder_state=state,
        ),
        owner=owner,
    )
    full = reference_model.prefill(
        inputs=typed_prefill(torch.tensor([[3, 4, 5]]), media),
        owner=StateOwner.fresh("reference"),
        use_cache=False,
    )
    assert (
        second.logits[:, -1].float() - full.logits[:, -1].float()
    ).abs().max().item() <= 1e-5
    assert torch.equal(
        second.logits[:, -1].argmax(dim=-1),
        full.logits[:, -1].argmax(dim=-1),
    )
    assert cached_model.vision_encoder.calls == 1
    assert cached_model.audio_encoder.calls == 1


def test_decode_rejects_missing_processed_prefix_before_text_compute(monkeypatch):
    model = Qwen3OmniMoeThinkerVisionAudioModel(config()).eval()
    owner = StateOwner.fresh("partial")
    state = model.thinker.prefill(
        inputs=typed_prefill(torch.tensor([[3]]), None),
        owner=owner,
        use_cache=True,
    ).decoder_state
    assert state is not None
    calls = []
    monkeypatch.setattr(
        model.thinker.embed_tokens,
        "forward",
        lambda *_args, **_kwargs: calls.append("text"),
    )
    with pytest.raises(ValueError, match="processed media"):
        model.decode(
            inputs=ModelDecodeInputs(
                token_ids=torch.tensor([[4]]),
                current_key_valid_mask=torch.ones(1, 1, dtype=torch.bool),
                position_batch=text_positions(
                    [[1]],
                    torch.ones(1, 1, dtype=torch.bool),
                ),
                decoder_state=state,
            ),
            owner=owner,
        )
    assert calls == []


@pytest.mark.parametrize(
    "media",
    [
        lambda: LegacyMediaPrefillInputs(
            pixel_values=torch.zeros(1, 3, 2),
            audio_values=None,
            has_image=torch.tensor([True]),
            has_audio=torch.tensor([False]),
        ),
        lambda: LegacyMediaPrefillInputs(
            pixel_values=None,
            audio_values=torch.zeros(1, 1, 2, 2),
            has_image=torch.tensor([False]),
            has_audio=torch.tensor([True]),
        ),
    ],
)
def test_media_rank_rejection_precedes_all_encoders(media):
    model = Qwen3OmniMoeThinkerVisionAudioModel(config()).eval()
    model.vision_encoder = CountingEncoder(8, 1.0)
    model.audio_encoder = CountingEncoder(8, 2.0)
    with pytest.raises(ValueError, match="shape"):
        model.prefill(
            inputs=typed_prefill(torch.tensor([[3]]), media()),
            owner=StateOwner.fresh("bad-media"),
            use_cache=True,
        )
    assert model.vision_encoder.calls == 0
    assert model.audio_encoder.calls == 0
