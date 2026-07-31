from __future__ import annotations

from collections import deque
from dataclasses import fields

import pytest

from qwen3_omni_pretrain.profiles.qwen35_omni_inspired.aria import (
    AriaScheduler,
    AriaState,
    AriaTokenKind,
)
from qwen3_omni_pretrain.profiles.qwen35_omni_inspired.configuration import (
    AriaConfig,
)


TEXT_EOS = 99
SPEECH_EOS = 199


def tiny_aria_config(**changes) -> AriaConfig:
    raw = AriaConfig().to_dict()
    raw.update(changes)
    return AriaConfig(**raw)


def _drain(text_count: int, speech_count: int) -> AriaState:
    scheduler = AriaScheduler(tiny_aria_config())
    queues = {
        "text": deque([*range(text_count), TEXT_EOS]),
        "speech": deque([*range(speech_count), SPEECH_EOS]),
    }
    state = AriaState()
    for _ in range(text_count + speech_count + 2):
        allowed = scheduler.allowed_modalities(
            state,
            text_available=bool(queues["text"]),
            speech_available=bool(queues["speech"]),
        )
        assert allowed
        modality = allowed[0]
        token = queues[modality].popleft()
        state = scheduler.commit(
            state,
            modality,
            token_kind=(
                AriaTokenKind.EOS
                if token in {TEXT_EOS, SPEECH_EOS}
                else AriaTokenKind.CONTENT
            ),
        )
        if not state.text_eos_seen and not state.speech_eos_seen:
            assert (
                state.emitted_speech_tokens
                * scheduler.config.speech_tokens_per_text_den
                <= state.emitted_text_tokens
                * scheduler.config.speech_tokens_per_text_num
            )
    assert state.finished
    assert not queues["text"] and not queues["speech"]
    return state


def test_scheduler_uses_rate_target_without_future_totals():
    scheduler = AriaScheduler(
        AriaConfig(
            speech_tokens_per_text_num=12,
            speech_tokens_per_text_den=5,
            text_first=True,
            tie_break="text",
        )
    )
    state = AriaState()
    assert all("total_" not in field.name for field in fields(state))
    assert scheduler.allowed_modalities(
        state,
        text_available=True,
        speech_available=True,
    )[0] == "text"
    finished = _drain(2, 6)
    assert finished.text_eos_seen and finished.speech_eos_seen


def test_one_eos_does_not_finish_or_block_the_other_stream():
    scheduler = AriaScheduler(tiny_aria_config())
    state = scheduler.commit(
        AriaState(),
        "text",
        token_kind=AriaTokenKind.EOS,
    )
    assert not state.finished
    assert scheduler.allowed_modalities(
        state,
        text_available=False,
        speech_available=True,
    ) == ("speech",)
    state = scheduler.commit(
        state,
        "speech",
        token_kind=AriaTokenKind.EOS,
    )
    assert state.finished


def test_snapshot_contains_no_oracle_future_lengths():
    scheduler = AriaScheduler(tiny_aria_config())
    snapshot = scheduler.snapshot(AriaState())
    assert set(snapshot) == {
        "emitted_text_tokens",
        "emitted_speech_tokens",
        "text_eos_seen",
        "speech_eos_seen",
        "step",
    }
    assert scheduler.restore(snapshot) == AriaState()


@pytest.mark.parametrize("text_count", range(9))
@pytest.mark.parametrize("speech_count", range(9))
def test_all_small_finite_streams_terminate(text_count, speech_count):
    _drain(text_count, speech_count)


def test_snapshot_restore_preserves_decisions():
    scheduler = AriaScheduler(tiny_aria_config(tie_break="speech"))
    state = scheduler.commit(
        AriaState(), "text", token_kind=AriaTokenKind.CONTENT
    )
    restored = scheduler.restore(scheduler.snapshot(state))
    assert scheduler.allowed_modalities(
        state, text_available=True, speech_available=True
    ) == scheduler.allowed_modalities(
        restored, text_available=True, speech_available=True
    )


def test_count_flags_and_padding_rules_are_exact():
    uncounted = AriaScheduler(tiny_aria_config())
    with pytest.raises(ValueError, match="uncounted.*padding"):
        uncounted.commit(
            AriaState(), "text", token_kind=AriaTokenKind.PADDING
        )
    counted = AriaScheduler(
        tiny_aria_config(count_eos=True, count_padding=True)
    )
    state = counted.commit(
        AriaState(), "text", token_kind=AriaTokenKind.PADDING
    )
    assert state.emitted_text_tokens == 1
    state = counted.commit(state, "text", token_kind=AriaTokenKind.EOS)
    assert state.emitted_text_tokens == 2 and state.text_eos_seen


@pytest.mark.parametrize(
    "changes",
    [
        {"speech_tokens_per_text_num": 0},
        {"speech_tokens_per_text_den": -1},
        {"speech_tokens_per_text_num": True},
        {"text_first": False},
        {"tie_break": "random"},
    ],
)
def test_invalid_aria_configuration_fails(changes):
    with pytest.raises((TypeError, ValueError)):
        tiny_aria_config(**changes)


def test_invalid_snapshot_and_post_eos_commit_fail():
    scheduler = AriaScheduler(tiny_aria_config())
    with pytest.raises(ValueError, match="unknown"):
        scheduler.restore({**scheduler.snapshot(AriaState()), "total_text": 3})
    finished = AriaState(
        text_eos_seen=True,
        speech_eos_seen=True,
        step=2,
    )
    with pytest.raises(ValueError, match="both"):
        scheduler.commit(finished, "text", token_kind=AriaTokenKind.CONTENT)
