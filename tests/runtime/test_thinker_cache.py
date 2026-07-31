from __future__ import annotations

import pytest
import torch

from qwen3_omni_pretrain.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeConfig,
)
from qwen3_omni_pretrain.models.qwen3_omni_moe.modeling_thinker_text import (
    Qwen3OmniMoeThinkerTextModel,
)
from qwen3_omni_pretrain.multimodal.types import PositionBatch
from qwen3_omni_pretrain.runtime import (
    CacheCapabilityError,
    CacheErrorCode,
    CacheCapableModel,
    DecoderState,
    ModelDecodeInputs,
    ModelPrefillInputs,
    StateOwner,
)


def tiny_config(
    *,
    layers: int = 2,
    max_positions: int = 16,
    use_deltanet: bool = False,
    deltanet_indices: str | None = None,
) -> Qwen3OmniMoeConfig:
    return Qwen3OmniMoeConfig(
        vocab_size=32,
        thinker_config={
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": layers,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": max_positions,
            "use_moe": False,
            "use_flash_attention": False,
            "use_deltanet": use_deltanet,
            "deltanet_layer_indices": deltanet_indices,
            "deltanet_num_heads": 4,
        },
    )


def positions(
    values,
    mask: torch.Tensor,
) -> PositionBatch:
    position_ids = (
        values.detach().clone().to(dtype=torch.long)
        if isinstance(values, torch.Tensor)
        else torch.tensor(values, dtype=torch.long)
    )
    if position_ids.ndim == 2:
        position_ids = position_ids.unsqueeze(0)
    result = PositionBatch(
        position_ids=position_ids,
        rope_deltas=torch.zeros(mask.shape[0], 1, dtype=torch.long),
        axis_names=("sequence",),
    )
    result.validate(mask)
    return result


def prefill_inputs(
    token_ids: torch.Tensor,
    mask: torch.Tensor | None = None,
    position_values=None,
) -> ModelPrefillInputs:
    if mask is None:
        mask = torch.ones_like(token_ids, dtype=torch.bool)
    if position_values is None:
        default = torch.arange(token_ids.shape[1]).expand(token_ids.shape[0], -1)
        default = default.masked_fill(~mask, 0)
        position_values = default.unsqueeze(0)
    return ModelPrefillInputs(
        input_ids=token_ids,
        inputs_embeds=None,
        key_valid_mask=mask,
        position_batch=positions(position_values, mask),
    )


def decode_inputs(
    state: DecoderState,
    token_ids: torch.Tensor,
    values,
    mask: torch.Tensor | None = None,
) -> ModelDecodeInputs:
    if mask is None:
        mask = torch.ones_like(token_ids, dtype=torch.bool)
    return ModelDecodeInputs(
        token_ids=token_ids,
        current_key_valid_mask=mask,
        position_batch=positions(values, mask),
        decoder_state=state,
    )


def test_typed_prefill_and_q1_q3_decode_match_uncached_full_sequence():
    torch.manual_seed(19)
    model = Qwen3OmniMoeThinkerTextModel(tiny_config()).eval()
    assert isinstance(model, CacheCapableModel)
    owner = StateOwner.fresh("parity")
    prompt = torch.tensor([[3, 4, 5]])
    first = model.prefill(
        inputs=prefill_inputs(prompt),
        owner=owner,
        use_cache=True,
    )
    state = first.decoder_state
    assert state is not None
    assert tuple(state.full_attention_kv) == (0, 1)
    assert state.seen_tokens.tolist() == [3]

    one = torch.tensor([[6]])
    second = model.decode(
        inputs=decode_inputs(state, one, [[3]]),
        owner=owner,
    )
    state = second.decoder_state
    assert state is not None
    three = torch.tensor([[7, 8, 9]])
    third = model.decode(
        inputs=decode_inputs(state, three, [[4, 5, 6]]),
        owner=owner,
    )

    complete = torch.cat((prompt, one, three), dim=1)
    reference = model(
        input_ids=complete,
        attention_mask=torch.ones_like(complete),
        position_ids=torch.arange(7).unsqueeze(0),
    )["logits"]
    cached = torch.cat((first.logits, second.logits, third.logits), dim=1)
    assert (cached.float() - reference.float()).abs().max().item() <= 1e-5
    assert torch.equal(cached.argmax(dim=-1), reference.argmax(dim=-1))
    assert third.decoder_state is not None
    assert third.decoder_state.seen_tokens.tolist() == [7]


def test_cached_decode_preserves_per_row_padding_and_explicit_position_gap():
    torch.manual_seed(23)
    model = Qwen3OmniMoeThinkerTextModel(tiny_config()).eval()
    owner = StateOwner.fresh("batch")
    prompt = torch.tensor([[3, 4, 5], [6, 7, 0]])
    prompt_mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
    first = model.prefill(
        inputs=prefill_inputs(
            prompt,
            prompt_mask,
            [[[0, 1, 2], [0, 1, 0]]],
        ),
        owner=owner,
        use_cache=True,
    )
    state = first.decoder_state
    assert state is not None
    current = torch.tensor([[8, 9, 10], [11, 0, 12]])
    current_mask = torch.tensor([[1, 1, 1], [1, 0, 1]], dtype=torch.bool)
    second = model.decode(
        inputs=decode_inputs(
            state,
            current,
            [[[5, 6, 7], [5, 0, 7]]],
            current_mask,
        ),
        owner=owner,
    )
    complete = torch.cat((prompt, current), dim=1)
    complete_mask = torch.cat((prompt_mask, current_mask), dim=1)
    complete_positions = torch.tensor(
        [[0, 1, 2, 5, 6, 7], [0, 1, 0, 5, 0, 7]]
    )
    reference = model(
        input_ids=complete,
        attention_mask=complete_mask,
        position_ids=complete_positions,
    )["logits"]
    assert (
        second.logits.float() - reference[:, -3:].float()
    ).abs().max().item() <= 1e-5
    assert second.decoder_state is not None
    assert second.decoder_state.seen_tokens.tolist() == [6, 4]
    cursor = second.decoder_state.position.continuation
    assert cursor.next_storage_position.tolist() == [8, 8]


@pytest.mark.parametrize("indices", ["0", None])
def test_explicit_and_implicit_deltanet_cache_reject_before_embedding(
    monkeypatch,
    indices,
):
    model = Qwen3OmniMoeThinkerTextModel(
        tiny_config(
            layers=4,
            use_deltanet=True,
            deltanet_indices=indices,
        )
    ).eval()
    calls = []
    monkeypatch.setattr(
        model.embed_tokens,
        "forward",
        lambda *_args, **_kwargs: calls.append("embedding"),
    )
    with pytest.raises(CacheCapabilityError) as captured:
        model.prefill(
            inputs=prefill_inputs(torch.tensor([[3, 4]])),
            owner=StateOwner.fresh("delta"),
            use_cache=True,
        )
    assert captured.value.code is CacheErrorCode.DELTA_NET_UNSUPPORTED
    assert calls == []


def test_owner_and_exact_layer_set_reject_before_embedding_or_attention(monkeypatch):
    model = Qwen3OmniMoeThinkerTextModel(tiny_config()).eval()
    owner = StateOwner.fresh("owned")
    first = model.prefill(
        inputs=prefill_inputs(torch.tensor([[3, 4]])),
        owner=owner,
        use_cache=True,
    )
    state = first.decoder_state
    assert state is not None
    calls = []
    monkeypatch.setattr(
        model.embed_tokens,
        "forward",
        lambda *_args, **_kwargs: calls.append("embedding"),
    )
    with pytest.raises(CacheCapabilityError) as captured:
        model.decode(
            inputs=decode_inputs(state, torch.tensor([[5]]), [[2]]),
            owner=StateOwner.fresh("owned"),
        )
    assert captured.value.code is CacheErrorCode.STATE_OWNER_MISMATCH
    assert calls == []

    malformed = DecoderState(
        owner=owner,
        seen_tokens=state.seen_tokens,
        position=state.position,
        full_attention_kv={0: state.full_attention_kv[0]},
        swa_kv={},
        processed_media=None,
        gdn_state=None,
        talker_state=None,
        mtp_state=None,
        codec_state=None,
    )
    with pytest.raises(ValueError, match="layer"):
        model.decode(
            inputs=decode_inputs(malformed, torch.tensor([[5]]), [[2]]),
            owner=owner,
        )
    assert calls == []


def test_cache_rejects_training_zero_valid_rows_and_context_overflow():
    model = Qwen3OmniMoeThinkerTextModel(tiny_config(max_positions=4))
    with pytest.raises(RuntimeError, match="eval"):
        model.prefill(
            inputs=prefill_inputs(torch.tensor([[3, 4]])),
            owner=StateOwner.fresh("training"),
            use_cache=True,
        )
    model.eval()
    with pytest.raises(ValueError, match="valid token"):
        model.prefill(
            inputs=prefill_inputs(
                torch.tensor([[0, 0]]),
                torch.zeros(1, 2, dtype=torch.bool),
                [[[0, 0]]],
            ),
            owner=StateOwner.fresh("empty"),
            use_cache=True,
        )
    with pytest.raises(CacheCapabilityError) as captured:
        model.prefill(
            inputs=prefill_inputs(torch.tensor([[1, 2, 3, 4, 5]])),
            owner=StateOwner.fresh("overflow"),
            use_cache=True,
        )
    assert captured.value.code is CacheErrorCode.CONTEXT_OVERFLOW


def test_layer_exception_never_mutates_committed_snapshot(monkeypatch):
    model = Qwen3OmniMoeThinkerTextModel(tiny_config()).eval()
    owner = StateOwner.fresh("atomic")
    first = model.prefill(
        inputs=prefill_inputs(torch.tensor([[3, 4]])),
        owner=owner,
        use_cache=True,
    )
    state = first.decoder_state
    assert state is not None
    fingerprint = state.full_attention_kv[0].key.clone()
    pointer = state.full_attention_kv[0].key.data_ptr()

    def fail(*_args, **_kwargs):
        raise RuntimeError("injected layer failure")

    monkeypatch.setattr(model.layers[1], "forward", fail)
    with pytest.raises(RuntimeError, match="injected"):
        model.decode(
            inputs=decode_inputs(state, torch.tensor([[5]]), [[2]]),
            owner=owner,
        )
    assert state.full_attention_kv[0].key.data_ptr() == pointer
    assert torch.equal(state.full_attention_kv[0].key, fingerprint)


def test_cache_capability_becomes_true_only_for_eval_all_attention_model():
    full = Qwen3OmniMoeThinkerTextModel(tiny_config()).eval()
    hybrid = Qwen3OmniMoeThinkerTextModel(
        tiny_config(layers=4, use_deltanet=True)
    ).eval()
    assert full.cache_support.incremental_decode_state is True
    assert hybrid.cache_support.incremental_decode_state is False
    assert full.cache_support.streaming_generation is False


def test_typed_no_cache_prefill_preserves_legacy_forward_logits_and_keys():
    torch.manual_seed(29)
    model = Qwen3OmniMoeThinkerTextModel(tiny_config()).eval()
    ids = torch.tensor([[3, 4, 5]])
    typed = model.prefill(
        inputs=prefill_inputs(ids),
        owner=StateOwner.fresh("uncached"),
        use_cache=False,
    )
    legacy = model(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        position_ids=torch.tensor([[0, 1, 2]]),
    )
    assert typed.decoder_state is None
    assert torch.equal(typed.logits, legacy["logits"])
    assert set(legacy) == {"logits", "loss"}
