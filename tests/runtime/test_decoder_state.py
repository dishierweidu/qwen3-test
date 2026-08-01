from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

import pytest
import torch

from qwen3_omni_pretrain.multimodal.types import PositionBatch
from qwen3_omni_pretrain.runtime import CacheCapabilityError, CacheErrorCode
from qwen3_omni_pretrain.runtime.protocols import (
    CacheCapableModel,
    CausalLMOutput,
    LegacyMediaPrefillInputs,
    ModelDecodeInputs,
    ModelPrefillInputs,
    legacy_position_ids,
)
from qwen3_omni_pretrain.runtime.state import (
    AttentionKV,
    DecoderPositionState,
    DecoderState,
    LegacyPositionCursor,
    LegacyProcessedPrefix,
    Qwen3DisjointPositionCursor,
    SlidingWindowKV,
    StateOwner,
    StatePartition,
)


def legacy_positions(
    values,
    *,
    mask: torch.Tensor | None = None,
    delta=None,
) -> tuple[PositionBatch, torch.Tensor]:
    position_ids = torch.tensor(values, dtype=torch.long)
    if position_ids.ndim == 2:
        position_ids = position_ids.unsqueeze(0)
    batch_size, sequence_length = position_ids.shape[1:]
    if mask is None:
        mask = torch.ones(batch_size, sequence_length, dtype=torch.bool)
    if delta is None:
        delta = torch.zeros(batch_size, 1, dtype=torch.long)
    result = PositionBatch(
        position_ids=position_ids,
        rope_deltas=delta,
        axis_names=("sequence",),
    )
    result.validate(mask)
    return result, mask


def qwen_positions() -> tuple[PositionBatch, torch.Tensor]:
    mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)
    result = PositionBatch(
        position_ids=torch.tensor(
            [
                [[0.0, 1.5, 1.0, 0.0]],
                [[0.0, 2.0, 0.0, 0.0]],
                [[0.0, 0.0, 3.0, 0.0]],
            ]
        ),
        rope_deltas=torch.tensor([[-2.5]]),
        axis_names=("temporal", "height", "width"),
    )
    result.validate(mask)
    return result, mask


def attention_kv(
    *,
    batch_size: int = 1,
    sequence_length: int = 3,
    key_dim: int = 4,
    value_dim: int = 5,
    mask: torch.Tensor | None = None,
    offset: float = 0.0,
) -> AttentionKV:
    key = torch.arange(
        batch_size * 2 * sequence_length * key_dim,
        dtype=torch.float32,
    ).reshape(batch_size, 2, sequence_length, key_dim)
    value = torch.arange(
        batch_size * 2 * sequence_length * value_dim,
        dtype=torch.float32,
    ).reshape(batch_size, 2, sequence_length, value_dim)
    if mask is None:
        mask = torch.ones(batch_size, sequence_length, dtype=torch.bool)
    return AttentionKV(
        key=key + offset,
        value=value + offset,
        key_valid_mask=mask,
    )


def populated_state(*, owner: StateOwner | None = None) -> DecoderState:
    owner = owner or StateOwner.fresh("request")
    positions, mask = legacy_positions([[0, 1, 2]])
    position = DecoderPositionState(
        cached=positions,
        key_valid_mask=mask,
        continuation=LegacyPositionCursor(torch.tensor([3])),
    )
    return DecoderState(
        owner=owner,
        seen_tokens=torch.tensor([3]),
        position=position,
        full_attention_kv={
            0: attention_kv(),
            1: attention_kv(offset=100.0),
        },
        swa_kv={},
        processed_media=None,
        gdn_state=None,
        talker_state=None,
        mtp_state=None,
        codec_state=None,
    )


def test_state_owner_uses_nonce_not_reusable_display_text():
    first = StateOwner.fresh("r1")
    concurrent = StateOwner.fresh("r1")
    later = StateOwner.fresh("r1")
    assert first.display_request_id == concurrent.display_request_id == "r1"
    assert len(first.nonce) >= 16
    assert len({first.nonce, concurrent.nonce, later.nonce}) == 3

    state = DecoderState.empty(first, batch_size=1)
    state.assert_owner(first)
    for wrong in (concurrent, later):
        with pytest.raises(CacheCapabilityError) as captured:
            state.assert_owner(wrong)
        assert captured.value.code is CacheErrorCode.STATE_OWNER_MISMATCH


@pytest.mark.parametrize(
    ("display", "nonce"),
    [
        ("", b"0" * 16),
        ("   ", b"0" * 16),
        (1, b"0" * 16),
        ("r", b"short"),
        ("r", bytearray(b"0" * 16)),
    ],
)
def test_state_owner_boundary_is_strict(display, nonce):
    with pytest.raises((TypeError, ValueError)):
        StateOwner(display, nonce)


def test_attention_kv_clones_detaches_and_allows_asymmetric_head_dims():
    source_key = torch.randn(1, 2, 3, 4, requires_grad=True)
    source_value = torch.randn(1, 2, 3, 6, requires_grad=True)
    source_mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    cache = AttentionKV(source_key, source_value, source_mask)

    assert cache.key.shape == (1, 2, 3, 4)
    assert cache.value.shape == (1, 2, 3, 6)
    assert cache.key.data_ptr() != source_key.data_ptr()
    assert cache.value.data_ptr() != source_value.data_ptr()
    assert cache.key_valid_mask.data_ptr() != source_mask.data_ptr()
    assert cache.key.requires_grad is False
    assert cache.key.grad_fn is None
    expected_key = cache.key.clone()
    expected_mask = cache.key_valid_mask.clone()
    with torch.no_grad():
        source_key.add_(100)
        source_mask.zero_()
    assert torch.equal(cache.key, expected_key)
    assert torch.equal(cache.key_valid_mask, expected_mask)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda key, value, mask: (key.squeeze(0), value, mask),
        lambda key, value, mask: (key, value[:, :1], mask),
        lambda key, value, mask: (key, value[:, :, :-1], mask),
        lambda key, value, mask: (key, value.double(), mask),
        lambda key, value, mask: (key, value, mask.long()),
        lambda key, value, mask: (key, value, mask[:, :-1]),
    ],
)
def test_attention_kv_rejects_malformed_coupled_tensors(mutation):
    key = torch.zeros(1, 2, 3, 4)
    value = torch.zeros(1, 2, 3, 5)
    mask = torch.ones(1, 3, dtype=torch.bool)
    key, value, mask = mutation(key, value, mask)
    with pytest.raises((TypeError, ValueError)):
        AttentionKV(key, value, mask)


def test_attention_append_is_non_aliasing_and_preserves_old_snapshot():
    old = attention_kv(sequence_length=2)
    old_fingerprint = old.key.clone()
    current_key = torch.full((1, 2, 1, 4), 10.0)
    current_value = torch.full((1, 2, 1, 5), 20.0)
    current_mask = torch.ones(1, 1, dtype=torch.bool)
    candidate = old.append(current_key, current_value, current_mask)

    assert candidate.key.shape[2] == 3
    assert torch.equal(old.key, old_fingerprint)
    assert candidate.key.data_ptr() != old.key.data_ptr()
    assert candidate.key.data_ptr() != current_key.data_ptr()
    current_key.zero_()
    assert torch.equal(candidate.key[:, :, -1], torch.full((1, 2, 4), 10.0))


def test_qwen_position_state_preserves_fractional_axes_and_negative_delta():
    positions, mask = qwen_positions()
    cursor = Qwen3DisjointPositionCursor(
        next_text_position=torch.tensor([4.5]),
        rope_deltas=torch.tensor([[-2.5]]),
        axis_names=("temporal", "height", "width"),
    )
    state = DecoderPositionState(positions, mask, cursor)

    assert torch.equal(state.cached.position_ids, positions.position_ids)
    assert state.cached.rope_deltas.tolist() == [[-2.5]]
    assert state.cached.position_ids.data_ptr() != positions.position_ids.data_ptr()
    assert state.key_valid_mask.data_ptr() != mask.data_ptr()
    assert state.cached.position_ids[0, 0, :3].tolist() == [0.0, 1.5, 1.0]


def test_position_state_append_updates_delta_and_does_not_alias_history():
    cached, mask = legacy_positions([[0, 1]])
    old = DecoderPositionState(
        cached,
        mask,
        LegacyPositionCursor(torch.tensor([2])),
    )
    current, current_mask = legacy_positions(
        [[5, 6]],
        delta=torch.tensor([[1]]),
    )
    candidate = old.append(
        current=current,
        current_key_valid_mask=current_mask,
        continuation=LegacyPositionCursor(torch.tensor([7])),
    )
    assert candidate.cached.position_ids.tolist() == [[[0, 1, 5, 6]]]
    assert candidate.cached.rope_deltas.tolist() == [[1]]
    assert candidate.continuation.next_storage_position.tolist() == [7]
    assert candidate.cached.position_ids.data_ptr() != old.cached.position_ids.data_ptr()
    assert old.cached.position_ids.tolist() == [[[0, 1]]]


@pytest.mark.parametrize(
    "position_factory",
    [
        lambda: PositionBatch(
            position_ids=torch.tensor([[[0.0, 1.5]]]),
            rope_deltas=torch.zeros(1, 1),
            axis_names=("sequence",),
        ),
        lambda: PositionBatch(
            position_ids=torch.zeros(3, 1, 2),
            rope_deltas=torch.zeros(1, 1),
            axis_names=("temporal", "height", "width"),
        ),
        lambda: PositionBatch(
            position_ids=torch.tensor([[[0, 8]]]),
            rope_deltas=torch.zeros(1, 1, dtype=torch.long),
            axis_names=("sequence",),
        ),
    ],
)
def test_legacy_position_adapter_rejects_fractional_three_axis_or_overflow(
    position_factory,
):
    calls = []
    with pytest.raises((TypeError, ValueError)):
        legacy_position_ids(
            position_factory(),
            torch.ones(1, 2, dtype=torch.bool),
            max_position_embeddings=8,
        )
        calls.append("qkv")
    assert calls == []


def test_legacy_position_adapter_accepts_integral_one_axis_with_padding():
    positions, mask = legacy_positions(
        [[3, 5, 0]],
        mask=torch.tensor([[1, 1, 0]], dtype=torch.bool),
    )
    adapted = legacy_position_ids(
        positions,
        mask,
        max_position_embeddings=6,
    )
    assert adapted.dtype is torch.long
    assert adapted.tolist() == [[3, 5, 0]]
    assert adapted.data_ptr() != positions.position_ids.data_ptr()


def test_legacy_cursor_tracks_storage_gaps_independently_of_valid_counts():
    cursor = LegacyPositionCursor(torch.tensor([2, 7]))
    advanced = cursor.advance(torch.tensor([3, 1]))
    assert cursor.next_storage_position.tolist() == [2, 7]
    assert advanced.next_storage_position.tolist() == [5, 8]
    assert advanced.next_storage_position.data_ptr() != cursor.next_storage_position.data_ptr()


def test_sliding_window_compacts_each_row_by_valid_tokens_after_every_append():
    first_mask = torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.bool)
    first_position = PositionBatch(
        position_ids=torch.tensor([[[0, 1, 2], [10, 0, 0]]]),
        rope_deltas=torch.zeros(2, 1, dtype=torch.long),
        axis_names=("sequence",),
    )
    cache = SlidingWindowKV(
        key=torch.tensor(
            [
                [[[0.0], [1.0], [2.0]]],
                [[[10.0], [0.0], [0.0]]],
            ]
        ),
        value=torch.tensor(
            [
                [[[100.0], [101.0], [102.0]]],
                [[[110.0], [0.0], [0.0]]],
            ]
        ),
        key_valid_mask=first_mask,
        position=first_position,
        window_size=3,
    )
    expected = [[0.0, 1.0, 2.0], [10.0]]

    chunks = (
        (
            torch.tensor([[[[3.0], [4.0]]], [[[11.0], [0.0]]]]),
            torch.tensor([[1, 1], [1, 0]], dtype=torch.bool),
            torch.tensor([[[3, 4], [11, 0]]]),
        ),
        (
            torch.tensor([[[[5.0]]], [[[12.0]]]]),
            torch.tensor([[1], [1]], dtype=torch.bool),
            torch.tensor([[[5], [12]]]),
        ),
    )
    for keys, masks, position_ids in chunks:
        values = keys + 100.0
        current_position = PositionBatch(
            position_ids=position_ids,
            rope_deltas=torch.zeros(2, 1, dtype=torch.long),
            axis_names=("sequence",),
        )
        cache = cache.append(
            key=keys,
            value=values,
            key_valid_mask=masks,
            position=current_position,
        )
        for row in range(2):
            expected[row].extend(
                keys[row, 0, :, 0].masked_select(masks[row]).tolist()
            )
            expected[row] = expected[row][-3:]
            actual = cache.key[row, 0, :, 0].masked_select(
                cache.key_valid_mask[row]
            )
            actual_positions = cache.position.position_ids[0, row].masked_select(
                cache.key_valid_mask[row]
            )
            assert actual.tolist() == expected[row]
            assert actual_positions.tolist() == [int(value) for value in expected[row]]
        assert cache.key.shape[2] == max(len(row) for row in expected)


def test_decoder_state_validates_exact_layer_set_and_common_history():
    state = populated_state()
    state.validate_full_attention_layers((0, 1))
    for expected in ((0,), (0, 1, 2), (1, 0)):
        with pytest.raises((TypeError, ValueError)):
            state.validate_full_attention_layers(expected)

    divergent = attention_kv(
        mask=torch.tensor([[1, 0, 1]], dtype=torch.bool),
    )
    with pytest.raises(ValueError, match="mask"):
        DecoderState(
            owner=state.owner,
            seen_tokens=torch.tensor([3]),
            position=state.position,
            full_attention_kv={0: state.full_attention_kv[0], 1: divergent},
            swa_kv={},
            processed_media=None,
            gdn_state=None,
            talker_state=None,
            mtp_state=None,
            codec_state=None,
        )


def test_decoder_state_derived_snapshots_do_not_alias_old_state():
    state = populated_state()
    advanced = state.advance_seen_tokens(torch.tensor([2]))
    assert state.seen_tokens.tolist() == [3]
    assert advanced.seen_tokens.tolist() == [5]
    assert advanced.seen_tokens.data_ptr() != state.seen_tokens.data_ptr()
    assert (
        advanced.full_attention_kv[0].key.data_ptr()
        != state.full_attention_kv[0].key.data_ptr()
    )
    advanced.full_attention_kv[0].key.zero_()
    assert bool((state.full_attention_kv[0].key != 0).any().item())
    assert advanced.owner == state.owner


def test_decoder_state_rejects_scalar_or_mismatched_seen_token_updates():
    state = populated_state()
    for delta in (
        1,
        True,
        torch.tensor(1),
        torch.tensor([1.0]),
        torch.tensor([-1]),
        torch.tensor([1, 2]),
    ):
        with pytest.raises((TypeError, ValueError)):
            state.advance_seen_tokens(delta)


def test_processed_prefix_is_typed_and_fixed_to_two_storage_slots():
    prefix = LegacyProcessedPrefix(
        has_image=(True, False),
        has_audio=(False, True),
        prefix_storage_length=2,
    )
    assert prefix.batch_size == 2
    assert prefix.clone_detached() == prefix
    for args in (
        ((True,), (False, True), 2),
        ((1,), (False,), 2),
        ((True,), (False,), 1),
    ):
        with pytest.raises((TypeError, ValueError)):
            LegacyProcessedPrefix(*args)


@dataclass(frozen=True)
class TinyPartition:
    tensor: torch.Tensor

    @property
    def batch_size(self) -> int:
        return self.tensor.shape[0]

    def clone_detached(self):
        return TinyPartition(self.tensor.detach().clone())

    def logical_tensor_bytes(self) -> int:
        return self.tensor.numel() * self.tensor.element_size()


def test_future_partitions_are_typed_cloned_and_batch_checked():
    assert isinstance(TinyPartition(torch.zeros(1, 2)), StatePartition)
    base = DecoderState.empty(StateOwner.fresh("r"), batch_size=1)
    with pytest.raises(TypeError, match="StatePartition"):
        DecoderState(
            owner=base.owner,
            seen_tokens=base.seen_tokens,
            position=None,
            full_attention_kv={},
            swa_kv={},
            processed_media=None,
            gdn_state=object(),
            talker_state=None,
            mtp_state=None,
            codec_state=None,
        )
    source = torch.ones(1, 2)
    state = DecoderState(
        owner=base.owner,
        seen_tokens=base.seen_tokens,
        position=None,
        full_attention_kv={},
        swa_kv={},
        processed_media=None,
        gdn_state=TinyPartition(source),
        talker_state=None,
        mtp_state=None,
        codec_state=None,
    )
    assert state.gdn_state is not None
    assert state.gdn_state.tensor.data_ptr() != source.data_ptr()


def test_state_byte_metrics_are_partitioned_and_dedupe_allocations():
    state = populated_state()
    assert state.logical_tensor_bytes() > 0
    assert state.logical_tensor_bytes("full_attention_kv") > 0
    assert state.unique_allocated_bytes() > 0
    assert state.unique_allocated_bytes() <= state.logical_tensor_bytes()
    with pytest.raises(ValueError, match="partition"):
        state.logical_tensor_bytes("unknown")


def test_legacy_media_prefill_union_is_strict():
    valid = LegacyMediaPrefillInputs(
        pixel_values=torch.zeros(2, 3, 2, 2),
        audio_values=None,
        has_image=torch.tensor([1, 0], dtype=torch.bool),
        has_audio=torch.tensor([0, 0], dtype=torch.bool),
    )
    assert valid.batch_size == 2
    assert valid.device == torch.device("cpu")

    invalid_values = (
        dict(
            pixel_values=None,
            audio_values=None,
            has_image=torch.tensor([1], dtype=torch.bool),
            has_audio=torch.tensor([0], dtype=torch.bool),
        ),
        dict(
            pixel_values=torch.zeros(1, 3, 2, 2),
            audio_values=None,
            has_image=torch.tensor([0], dtype=torch.bool),
            has_audio=torch.tensor([0], dtype=torch.bool),
        ),
        dict(
            pixel_values=None,
            audio_values=None,
            has_image=torch.tensor([1]),
            has_audio=torch.tensor([0], dtype=torch.bool),
        ),
    )
    for kwargs in invalid_values:
        with pytest.raises((TypeError, ValueError)):
            LegacyMediaPrefillInputs(**kwargs)


def test_model_prefill_inputs_require_one_payload_and_validate_position():
    positions, mask = legacy_positions([[0, 1]])
    valid = ModelPrefillInputs(
        input_ids=torch.tensor([[3, 4]]),
        inputs_embeds=None,
        key_valid_mask=mask,
        position_batch=positions,
        media=None,
    )
    assert valid.batch_size == 1
    assert valid.query_length == 2

    for ids, embeds in (
        (None, None),
        (torch.tensor([[3, 4]]), torch.zeros(1, 2, 4)),
    ):
        with pytest.raises(ValueError, match="exactly one"):
            ModelPrefillInputs(
                input_ids=ids,
                inputs_embeds=embeds,
                key_valid_mask=mask,
                position_batch=positions,
            )


def test_model_decode_inputs_couple_token_position_state_and_device():
    state = populated_state()
    current, mask = legacy_positions([[3]])
    inputs = ModelDecodeInputs(
        token_ids=torch.tensor([[7]]),
        current_key_valid_mask=mask,
        position_batch=current,
        decoder_state=state,
    )
    assert inputs.batch_size == 1
    assert inputs.query_length == 1
    with pytest.raises(ValueError, match="batch"):
        ModelDecodeInputs(
            token_ids=torch.tensor([[7], [8]]),
            current_key_valid_mask=torch.ones(2, 1, dtype=torch.bool),
            position_batch=PositionBatch(
                position_ids=torch.tensor([[[3], [3]]]),
                rope_deltas=torch.zeros(2, 1, dtype=torch.long),
                axis_names=("sequence",),
            ),
            decoder_state=state,
        )


def test_causal_output_validates_and_cache_protocol_is_runtime_checkable():
    state = populated_state()
    output = CausalLMOutput(
        logits=torch.zeros(1, 2, 8),
        loss=None,
        ce_loss=None,
        aux_loss=None,
        decoder_state=state,
        hidden_states=(torch.zeros(1, 2, 4),),
    )
    assert output.decoder_state is state

    class Model:
        def prefill(self, *, inputs, owner, use_cache):
            return output

        def decode(self, *, inputs, owner):
            return output

    assert isinstance(Model(), CacheCapableModel)

    with pytest.raises((TypeError, ValueError)):
        CausalLMOutput(
            logits=torch.zeros(1, 8),
            loss=None,
            ce_loss=None,
            aux_loss=None,
            decoder_state=None,
            hidden_states=None,
        )
