from __future__ import annotations

import torch

from qwen3_omni_pretrain.profiles.qwen35_omni_inspired.aria import (
    AriaScheduler,
    AriaState,
)
from qwen3_omni_pretrain.profiles.qwen35_omni_inspired.configuration import (
    AriaConfig,
)
from qwen3_omni_pretrain.profiles.qwen35_omni_inspired.runtime import (
    AriaStepOutput,
    AriaWaitOutput,
    select_interleaved_token,
)


def _logits(token: int, vocab: int = 8) -> torch.Tensor:
    result = torch.full((1, vocab), -10.0)
    result[0, token] = 10.0
    return result


def test_selection_obeys_online_scheduler_and_dual_eos():
    scheduler = AriaScheduler(AriaConfig())
    state = AriaState()
    first = select_interleaved_token(
        text_logits=_logits(3),
        speech_logits=_logits(4),
        text_eos_token_id=1,
        speech_eos_token_id=2,
        state=state,
        scheduler=scheduler,
    )
    assert isinstance(first, AriaStepOutput) and first.modality == "text"
    text_eos = select_interleaved_token(
        text_logits=_logits(1),
        speech_logits=None,
        text_eos_token_id=1,
        speech_eos_token_id=2,
        state=first.state,
        scheduler=scheduler,
    )
    assert isinstance(text_eos, AriaStepOutput)
    assert text_eos.state.text_eos_seen and not text_eos.state.finished
    speech_eos = select_interleaved_token(
        text_logits=None,
        speech_logits=_logits(2),
        text_eos_token_id=1,
        speech_eos_token_id=2,
        state=text_eos.state,
        scheduler=scheduler,
    )
    assert isinstance(speech_eos, AriaStepOutput)
    assert speech_eos.state.finished


def test_no_allowed_modality_returns_wait_without_state_change():
    scheduler = AriaScheduler(AriaConfig())
    state = AriaState()
    output = select_interleaved_token(
        text_logits=None,
        speech_logits=_logits(3),
        text_eos_token_id=1,
        speech_eos_token_id=2,
        state=state,
        scheduler=scheduler,
    )
    assert isinstance(output, AriaWaitOutput)
    assert output.state == state
