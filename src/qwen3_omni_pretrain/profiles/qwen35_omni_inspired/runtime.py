from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import torch
from torch import nn

from qwen3_omni_pretrain.profiles.qwen35_omni_inspired.aria import (
    AriaScheduler,
    AriaState,
    AriaTokenKind,
)
from qwen3_omni_pretrain.profiles.qwen35_omni_inspired.talker import (
    PredecessorCodecProxy,
    Qwen35InspiredTalker,
)
from qwen3_omni_pretrain.profiles.qwen35_omni_inspired.thinker import (
    Qwen35InspiredThinker,
    unique_parameters_by_identity,
)
from qwen3_omni_pretrain.runtime.protocols import (
    CausalLMOutput,
    ModelDecodeInputs,
    ModelPrefillInputs,
)
from qwen3_omni_pretrain.runtime.state import StateOwner


@dataclass(frozen=True)
class AriaStepOutput:
    modality: Literal["text", "speech"]
    token_id: torch.LongTensor
    state: AriaState


@dataclass(frozen=True)
class AriaWaitOutput:
    state: AriaState
    reason: Literal["no_allowed_available_modality"] = (
        "no_allowed_available_modality"
    )


def select_interleaved_token(
    *,
    text_logits: torch.Tensor | None,
    speech_logits: torch.Tensor | None,
    text_eos_token_id: int,
    speech_eos_token_id: int,
    state: AriaState,
    scheduler: AriaScheduler,
) -> AriaStepOutput | AriaWaitOutput:
    if type(text_eos_token_id) is not int or text_eos_token_id < 0:
        raise ValueError("text_eos_token_id must be non-negative")
    if type(speech_eos_token_id) is not int or speech_eos_token_id < 0:
        raise ValueError("speech_eos_token_id must be non-negative")
    for name, logits in (("text_logits", text_logits), ("speech_logits", speech_logits)):
        if logits is not None:
            if not isinstance(logits, torch.Tensor) or logits.ndim < 1:
                raise ValueError(f"{name} must be a logits tensor or None")
            if not logits.is_floating_point() or not bool(torch.isfinite(logits).all().item()):
                raise ValueError(f"{name} must contain finite floating logits")
    allowed = scheduler.allowed_modalities(
        state,
        text_available=text_logits is not None,
        speech_available=speech_logits is not None,
    )
    if not allowed:
        return AriaWaitOutput(state)
    modality = allowed[0]
    logits = text_logits if modality == "text" else speech_logits
    assert logits is not None
    token_id = logits.argmax(dim=-1).to(torch.long)
    if token_id.numel() != 1:
        raise ValueError("ARIA token selection currently requires batch size one")
    eos_id = text_eos_token_id if modality == "text" else speech_eos_token_id
    next_state = scheduler.commit(
        state,
        modality,
        token_kind=(
            AriaTokenKind.EOS
            if int(token_id.item()) == eos_id
            else AriaTokenKind.CONTENT
        ),
    )
    return AriaStepOutput(modality, token_id, next_state)


class Qwen35InspiredRuntime(nn.Module):
    def __init__(
        self,
        *,
        thinker: Qwen35InspiredThinker,
        aria_scheduler: AriaScheduler,
        talker: Qwen35InspiredTalker,
        codec_proxy: PredecessorCodecProxy,
    ) -> None:
        super().__init__()
        if not isinstance(thinker, Qwen35InspiredThinker):
            raise TypeError("thinker must be Qwen35InspiredThinker")
        if not isinstance(aria_scheduler, AriaScheduler):
            raise TypeError("aria_scheduler must be AriaScheduler")
        if not isinstance(talker, Qwen35InspiredTalker):
            raise TypeError("talker must be Qwen35InspiredTalker")
        if not isinstance(codec_proxy, PredecessorCodecProxy):
            raise TypeError("codec_proxy must be PredecessorCodecProxy")
        self.thinker = thinker
        self.aria_scheduler = aria_scheduler
        self.talker = talker
        self.codec_proxy = codec_proxy
        self.config = thinker.config

    @property
    def mtp(self) -> nn.Module:
        return self.codec_proxy.mtp_proxy

    def forward(self, *args, **kwargs):
        return self.thinker(*args, **kwargs)

    def prefill(
        self,
        *,
        inputs: ModelPrefillInputs,
        owner: StateOwner,
        use_cache: bool,
    ) -> CausalLMOutput:
        return self.thinker.prefill(inputs=inputs, owner=owner, use_cache=use_cache)

    def decode(
        self,
        *,
        inputs: ModelDecodeInputs,
        owner: StateOwner,
    ) -> CausalLMOutput:
        return self.thinker.decode(inputs=inputs, owner=owner)

    def select_interleaved_token(self, **kwargs) -> AriaStepOutput | AriaWaitOutput:
        return select_interleaved_token(scheduler=self.aria_scheduler, **kwargs)

    def named_parameter_groups(self) -> Mapping[str, tuple[nn.Parameter, ...]]:
        groups = dict(self.thinker.named_parameter_groups())
        mtp_ids = {id(parameter) for parameter in self.mtp.parameters()}
        codec_ids = {
            id(parameter)
            for parameter in self.codec_proxy.decoder.parameters()
        }
        groups["talker"] = unique_parameters_by_identity(self.talker.parameters())
        groups["mtp"] = unique_parameters_by_identity(
            parameter for parameter in self.parameters() if id(parameter) in mtp_ids
        )
        groups["codec"] = unique_parameters_by_identity(
            parameter for parameter in self.parameters() if id(parameter) in codec_ids
        )
        identities = [id(parameter) for values in groups.values() for parameter in values]
        if len(identities) != len(set(identities)):
            raise ValueError("runtime parameter groups must be pairwise disjoint")
        if set(identities) != {id(parameter) for parameter in self.parameters()}:
            raise ValueError("runtime parameter groups must cover every parameter exactly once")
        return groups


__all__ = [
    "AriaStepOutput",
    "AriaWaitOutput",
    "Qwen35InspiredRuntime",
    "select_interleaved_token",
]
