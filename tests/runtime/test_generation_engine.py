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
    CancellationToken,
    GenerationStep,
    GenerationExecutionError,
    GenerationRequest,
    LegacyGreedyPrefillDecodeEngine,
    ModelPrefillInputs,
)


def model(*, max_positions: int = 32) -> Qwen3OmniMoeThinkerTextModel:
    torch.manual_seed(37)
    config = Qwen3OmniMoeConfig(
        vocab_size=32,
        thinker_config={
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": max_positions,
            "use_moe": False,
            "use_flash_attention": False,
        },
    )
    return Qwen3OmniMoeThinkerTextModel(config).eval()


def inputs(
    ids: list[int],
    *,
    mask: list[bool] | None = None,
    positions: list[int] | None = None,
) -> ModelPrefillInputs:
    token_ids = torch.tensor([ids], dtype=torch.long)
    valid = torch.tensor(
        [mask if mask is not None else [True] * len(ids)],
        dtype=torch.bool,
    )
    values = (
        torch.arange(len(ids)).unsqueeze(0)
        if positions is None
        else torch.tensor([positions], dtype=torch.long)
    ).masked_fill(~valid, 0)
    position = PositionBatch(
        position_ids=values.unsqueeze(0),
        rope_deltas=torch.zeros(1, 1, dtype=torch.long),
        axis_names=("sequence",),
    )
    return ModelPrefillInputs(
        input_ids=token_ids,
        inputs_embeds=None,
        key_valid_mask=valid,
        position_batch=position,
    )


def request(
    *,
    max_new_tokens: int = 4,
    eos_token_id: int | None = None,
    cancellation: CancellationToken | None = None,
    num_beams: int = 1,
) -> GenerationRequest:
    return GenerationRequest(
        display_request_id="r1",
        prefill_inputs=inputs([3, 4, 5]),
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
        num_beams=num_beams,
        cancellation=cancellation,
    )


def uncached_tokens(
    thinker: Qwen3OmniMoeThinkerTextModel,
    count: int,
) -> torch.Tensor:
    generated = torch.tensor([[3, 4, 5]])
    with torch.inference_mode():
        for _ in range(count):
            output = thinker(
                input_ids=generated,
                attention_mask=torch.ones_like(generated),
            )
            token = output["logits"][:, -1].argmax(dim=-1, keepdim=True)
            generated = torch.cat((generated, token), dim=1)
    return generated[:, 3:]


def test_pending_token_engine_matches_uncached_and_uses_n_minus_one_decodes():
    thinker = model()
    result = LegacyGreedyPrefillDecodeEngine(thinker).generate(request())
    assert result.finish_reason == "length"
    assert result.prefill_calls == 1
    assert result.decode_calls == 3
    assert torch.equal(result.generated_ids, uncached_tokens(thinker, 4))
    assert result.checkpoint.pending_token_id is not None
    assert torch.equal(
        result.checkpoint.pending_token_id,
        result.generated_ids[:, -1:],
    )
    assert result.checkpoint.decoder_state.seen_tokens.tolist() == [6]


def test_one_token_and_prefill_eos_never_decode_and_eos_beats_length():
    thinker = model()
    first_token = int(uncached_tokens(thinker, 1).item())
    result = LegacyGreedyPrefillDecodeEngine(thinker).generate(
        request(max_new_tokens=1, eos_token_id=first_token)
    )
    assert result.finish_reason == "eos"
    assert result.prefill_calls == 1
    assert result.decode_calls == 0
    assert result.generated_ids.tolist() == [[first_token]]

    resumed = LegacyGreedyPrefillDecodeEngine(thinker).generate(
        request(max_new_tokens=4, eos_token_id=first_token),
        resume=result,
    )
    assert resumed.finish_reason == "eos"
    assert resumed.decode_calls == 0
    assert torch.equal(resumed.generated_ids, result.generated_ids)


def test_pause_after_every_call_and_resume_matches_uninterrupted():
    thinker = model()
    engine = LegacyGreedyPrefillDecodeEngine(thinker)
    expected = engine.generate(request())
    resumed = None
    owners = []
    while resumed is None or resumed.finish_reason == "cancelled":
        resumed = engine.generate(
            request(),
            resume=resumed,
            call_budget=1,
        )
        owners.append(resumed.checkpoint.decoder_state.owner.nonce)
    assert resumed.finish_reason == "length"
    assert torch.equal(resumed.generated_ids, expected.generated_ids)
    assert resumed.prefill_calls == 1
    assert resumed.decode_calls == 3
    assert len(set(owners)) == 1


def test_fresh_same_display_id_gets_a_new_nonce():
    engine = LegacyGreedyPrefillDecodeEngine(model())
    first = engine.generate(request(max_new_tokens=1))
    second = engine.generate(request(max_new_tokens=1))
    assert first.checkpoint.decoder_state.owner.display_request_id == "r1"
    assert second.checkpoint.decoder_state.owner.display_request_id == "r1"
    assert (
        first.checkpoint.decoder_state.owner.nonce
        != second.checkpoint.decoder_state.owner.nonce
    )


def test_generation_step_owns_non_aliasing_values_and_checks_device():
    result = LegacyGreedyPrefillDecodeEngine(model()).generate(
        request(max_new_tokens=1)
    )
    emitted = result.generated_ids[:, -1:].clone()
    finished = torch.tensor([False], dtype=torch.bool)
    step = GenerationStep(
        emitted_token_id=emitted,
        emitted_index=0,
        checkpoint=result.checkpoint,
        finished=finished,
    )
    emitted.zero_()
    finished.fill_(True)
    assert torch.equal(step.emitted_token_id, result.generated_ids[:, -1:])
    assert step.finished.tolist() == [False]
    assert (
        step.checkpoint.decoder_state.seen_tokens.data_ptr()
        != result.checkpoint.decoder_state.seen_tokens.data_ptr()
    )

    with pytest.raises(ValueError, match="device"):
        GenerationStep(
            emitted_token_id=torch.empty(
                (1, 1),
                dtype=torch.long,
                device="meta",
            ),
            emitted_index=0,
            checkpoint=result.checkpoint,
            finished=torch.tensor([False], dtype=torch.bool),
        )


@pytest.mark.parametrize("num_beams", [True, 0, 2, 1.0])
def test_invalid_beam_fails_before_any_model_call(num_beams):
    class NeverModel:
        calls = 0

        def prefill(self, **kwargs):
            self.calls += 1

        def decode(self, **kwargs):
            self.calls += 1

    fake = NeverModel()
    with pytest.raises(CacheCapabilityError) as captured:
        GenerationRequest(
            display_request_id="r",
            prefill_inputs=inputs([3]),
            max_new_tokens=1,
            eos_token_id=None,
            num_beams=num_beams,
        )
        LegacyGreedyPrefillDecodeEngine(fake).generate(request())
    assert captured.value.code is CacheErrorCode.BEAM_UNSUPPORTED
    assert fake.calls == 0


def test_right_padded_prompt_is_rejected_instead_of_sampling_padding():
    bad = GenerationRequest(
        display_request_id="padded",
        prefill_inputs=inputs([3, 0], mask=[True, False]),
        max_new_tokens=1,
        eos_token_id=None,
    )
    with pytest.raises(ValueError, match="final query"):
        LegacyGreedyPrefillDecodeEngine(model()).generate(bad)


def test_context_and_output_budget_rejects_before_embedding(monkeypatch):
    thinker = model()
    calls = []
    monkeypatch.setattr(
        thinker.embed_tokens,
        "forward",
        lambda *_args, **_kwargs: calls.append("embedding"),
    )
    with pytest.raises(CacheCapabilityError) as captured:
        LegacyGreedyPrefillDecodeEngine(thinker).generate(
            request(max_new_tokens=31)
        )
    assert captured.value.code is CacheErrorCode.CONTEXT_OVERFLOW
    assert calls == []


def test_position_gap_output_budget_rejects_before_embedding(monkeypatch):
    thinker = model(max_positions=8)
    engine = LegacyGreedyPrefillDecodeEngine(thinker)
    exact_boundary = GenerationRequest(
        display_request_id="position-boundary",
        prefill_inputs=inputs([3, 4], positions=[0, 6]),
        max_new_tokens=2,
        eos_token_id=None,
    )
    assert engine.generate(exact_boundary).finish_reason == "length"

    calls = []
    monkeypatch.setattr(
        thinker.embed_tokens,
        "forward",
        lambda *_args, **_kwargs: calls.append("embedding"),
    )
    overflow = GenerationRequest(
        display_request_id="position-overflow",
        prefill_inputs=inputs([3, 4], positions=[0, 6]),
        max_new_tokens=3,
        eos_token_id=None,
    )
    with pytest.raises(CacheCapabilityError) as captured:
        engine.generate(overflow)
    assert captured.value.code is CacheErrorCode.CONTEXT_OVERFLOW
    assert calls == []


def test_pre_cancelled_request_performs_zero_calls_and_can_resume():
    token = CancellationToken()
    token.cancel()
    engine = LegacyGreedyPrefillDecodeEngine(model())
    cancelled = engine.generate(request(cancellation=token))
    assert cancelled.finish_reason == "cancelled"
    assert cancelled.prefill_calls == 0
    assert cancelled.decode_calls == 0
    assert cancelled.generated_ids.shape == (1, 0)
    assert cancelled.checkpoint.pending_token_id is None

    resumed = engine.generate(request(max_new_tokens=1), resume=cancelled)
    assert resumed.finish_reason == "length"
    assert resumed.prefill_calls == 1


class HookedModel:
    def __init__(
        self,
        base,
        *,
        cancellation=None,
        cancel_on=None,
        fail_prefill=False,
        fail_decode=False,
    ):
        self.base = base
        self.cancellation = cancellation
        self.cancel_on = cancel_on
        self.fail_prefill = fail_prefill
        self.fail_decode = fail_decode
        self.prefill_calls = 0
        self.decode_calls = 0

    def prefill(self, **kwargs):
        self.prefill_calls += 1
        if self.fail_prefill:
            raise RuntimeError("prefill exploded")
        output = self.base.prefill(**kwargs)
        if self.cancellation is not None and self.cancel_on == "prefill":
            self.cancellation.cancel()
        return output

    def decode(self, **kwargs):
        self.decode_calls += 1
        if self.fail_decode:
            raise RuntimeError("decode exploded")
        output = self.base.decode(**kwargs)
        if self.cancellation is not None and self.cancel_on == "decode":
            self.cancellation.cancel()
        return output


def test_post_prefill_cancellation_discards_candidate_and_emission():
    token = CancellationToken()
    hooked = HookedModel(
        model(),
        cancellation=token,
        cancel_on="prefill",
    )
    result = LegacyGreedyPrefillDecodeEngine(hooked).generate(
        request(cancellation=token)
    )
    assert result.finish_reason == "cancelled"
    assert result.generated_ids.shape == (1, 0)
    assert result.prefill_calls == 1
    assert result.checkpoint.decoder_state.position is None


def test_prefill_exception_exposes_owned_empty_checkpoint():
    hooked = HookedModel(model(), fail_prefill=True)
    with pytest.raises(GenerationExecutionError) as captured:
        LegacyGreedyPrefillDecodeEngine(hooked).generate(request())
    error = captured.value
    assert isinstance(error.__cause__, RuntimeError)
    assert error.prefill_calls == 1
    assert error.decode_calls == 0
    assert error.checkpoint.pending_token_id is None
    assert error.checkpoint.decoder_state.position is None
    assert error.checkpoint.decoder_state.full_attention_kv == {}


def test_post_decode_cancellation_discards_candidate_and_resumes_exactly():
    token = CancellationToken()
    thinker = model()
    hooked = HookedModel(
        thinker,
        cancellation=token,
        cancel_on="decode",
    )
    engine = LegacyGreedyPrefillDecodeEngine(hooked)
    cancelled = engine.generate(request(cancellation=token))
    assert cancelled.finish_reason == "cancelled"
    assert cancelled.generated_ids.shape == (1, 1)
    assert cancelled.prefill_calls == 1
    assert cancelled.decode_calls == 1
    assert cancelled.checkpoint.decoder_state.seen_tokens.tolist() == [3]
    assert all(
        cache.sequence_length == 3
        for cache in cancelled.checkpoint.decoder_state.full_attention_kv.values()
    )

    resumed = LegacyGreedyPrefillDecodeEngine(thinker).generate(
        request(),
        resume=cancelled,
    )
    uninterrupted = LegacyGreedyPrefillDecodeEngine(thinker).generate(request())
    assert resumed.finish_reason == "length"
    assert torch.equal(resumed.generated_ids, uninterrupted.generated_ids)


def test_decode_exception_exposes_last_committed_checkpoint():
    hooked = HookedModel(model(), fail_decode=True)
    engine = LegacyGreedyPrefillDecodeEngine(hooked)
    with pytest.raises(GenerationExecutionError) as captured:
        engine.generate(request(max_new_tokens=3))
    error = captured.value
    assert isinstance(error.__cause__, RuntimeError)
    assert error.prefill_calls == 1
    assert error.decode_calls == 1
    assert error.checkpoint.pending_token_id is not None
    assert error.checkpoint.decoder_state.seen_tokens.tolist() == [3]
