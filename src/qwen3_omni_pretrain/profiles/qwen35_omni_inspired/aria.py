from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Literal, Mapping

from qwen3_omni_pretrain.profiles.qwen35_omni_inspired.configuration import (
    AriaConfig,
)


AriaModality = Literal["text", "speech"]


class AriaTokenKind(str, Enum):
    CONTENT = "content"
    EOS = "eos"
    PADDING = "padding"


@dataclass(frozen=True)
class AriaState:
    emitted_text_tokens: int = 0
    emitted_speech_tokens: int = 0
    text_eos_seen: bool = False
    speech_eos_seen: bool = False
    step: int = 0

    def __post_init__(self) -> None:
        for name in (
            "emitted_text_tokens",
            "emitted_speech_tokens",
            "step",
        ):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be a plain integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if type(self.text_eos_seen) is not bool or type(self.speech_eos_seen) is not bool:
            raise TypeError("ARIA EOS flags must be booleans")
        if self.step < self.emitted_text_tokens + self.emitted_speech_tokens:
            raise ValueError("ARIA step cannot be below counted token events")
        if self.step < int(self.text_eos_seen) + int(self.speech_eos_seen):
            raise ValueError("ARIA EOS flags contradict step")

    @property
    def finished(self) -> bool:
        return self.text_eos_seen and self.speech_eos_seen


class AriaScheduler:
    def __init__(self, config: AriaConfig) -> None:
        if not isinstance(config, AriaConfig):
            raise TypeError("config must be AriaConfig")
        self.config = config

    def _validate_state(self, state: AriaState) -> None:
        if not isinstance(state, AriaState):
            raise TypeError("state must be AriaState")
        state.__post_init__()
        if (
            not state.text_eos_seen
            and not state.speech_eos_seen
            and state.emitted_speech_tokens
            * self.config.speech_tokens_per_text_den
            > state.emitted_text_tokens
            * self.config.speech_tokens_per_text_num
        ):
            raise ValueError("ARIA state violates the active speech:text rate envelope")

    def _sort_by_integer_rate_error(
        self,
        state: AriaState,
        candidates: list[AriaModality],
    ) -> tuple[AriaModality, ...]:
        if len(candidates) <= 1:
            return tuple(candidates)
        num = self.config.speech_tokens_per_text_num
        den = self.config.speech_tokens_per_text_den
        errors = {
            "text": abs(
                state.emitted_speech_tokens * den
                - (state.emitted_text_tokens + 1) * num
            ),
            "speech": abs(
                (state.emitted_speech_tokens + 1) * den
                - state.emitted_text_tokens * num
            ),
        }
        tie_order = {
            self.config.tie_break: 0,
            "speech" if self.config.tie_break == "text" else "text": 1,
        }
        return tuple(
            sorted(candidates, key=lambda value: (errors[value], tie_order[value]))
        )

    def allowed_modalities(
        self,
        state: AriaState,
        *,
        text_available: bool,
        speech_available: bool,
    ) -> tuple[AriaModality, ...]:
        self._validate_state(state)
        if type(text_available) is not bool or type(speech_available) is not bool:
            raise TypeError("ARIA availability flags must be booleans")
        if state.finished:
            return ()
        candidates: list[AriaModality] = []
        if text_available and not state.text_eos_seen:
            candidates.append("text")
        if speech_available and not state.speech_eos_seen:
            speech_within_rate = (
                (state.emitted_speech_tokens + 1)
                * self.config.speech_tokens_per_text_den
                <= state.emitted_text_tokens
                * self.config.speech_tokens_per_text_num
            )
            if state.text_eos_seen or speech_within_rate:
                candidates.append("speech")
        if state.speech_eos_seen:
            return tuple(value for value in candidates if value == "text")
        if state.text_eos_seen:
            return tuple(value for value in candidates if value == "speech")
        if state.step == 0 and self.config.text_first and "text" in candidates:
            return ("text",) + tuple(value for value in candidates if value != "text")
        return self._sort_by_integer_rate_error(state, candidates)

    def commit(
        self,
        state: AriaState,
        modality: AriaModality,
        *,
        token_kind: AriaTokenKind,
    ) -> AriaState:
        self._validate_state(state)
        if state.finished:
            raise ValueError("cannot commit after both ARIA streams reached EOS")
        if modality not in {"text", "speech"}:
            raise ValueError("ARIA modality must be text or speech")
        if not isinstance(token_kind, AriaTokenKind):
            raise TypeError("token_kind must be AriaTokenKind")
        already_eos = state.text_eos_seen if modality == "text" else state.speech_eos_seen
        if already_eos:
            raise ValueError(f"cannot commit {modality} after its EOS")
        if token_kind is AriaTokenKind.PADDING and not self.config.count_padding:
            raise ValueError("uncounted ARIA padding is rejected")

        counted = (
            token_kind is AriaTokenKind.CONTENT
            or (token_kind is AriaTokenKind.EOS and self.config.count_eos)
            or (token_kind is AriaTokenKind.PADDING and self.config.count_padding)
        )
        text_count = state.emitted_text_tokens
        speech_count = state.emitted_speech_tokens
        if counted and modality == "text":
            text_count += 1
        elif counted:
            if (
                not state.text_eos_seen
                and not state.speech_eos_seen
                and (speech_count + 1)
                * self.config.speech_tokens_per_text_den
                > text_count * self.config.speech_tokens_per_text_num
            ):
                raise ValueError("counted speech token exceeds the active ARIA rate envelope")
            speech_count += 1
        text_eos = state.text_eos_seen or (
            modality == "text" and token_kind is AriaTokenKind.EOS
        )
        speech_eos = state.speech_eos_seen or (
            modality == "speech" and token_kind is AriaTokenKind.EOS
        )
        return AriaState(
            emitted_text_tokens=text_count,
            emitted_speech_tokens=speech_count,
            text_eos_seen=text_eos,
            speech_eos_seen=speech_eos,
            step=state.step + 1,
        )

    def snapshot(self, state: AriaState) -> dict[str, int | bool]:
        self._validate_state(state)
        return asdict(state)

    def restore(self, snapshot: Mapping[str, object]) -> AriaState:
        if not isinstance(snapshot, Mapping):
            raise TypeError("ARIA snapshot must be a mapping")
        expected = {
            "emitted_text_tokens",
            "emitted_speech_tokens",
            "text_eos_seen",
            "speech_eos_seen",
            "step",
        }
        missing = expected - set(snapshot)
        unknown = set(snapshot) - expected
        if missing or unknown:
            raise ValueError(
                f"invalid ARIA snapshot keys: missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}"
            )
        for name in ("emitted_text_tokens", "emitted_speech_tokens", "step"):
            if type(snapshot[name]) is not int:
                raise TypeError(f"ARIA snapshot {name} must be a plain integer")
        for name in ("text_eos_seen", "speech_eos_seen"):
            if type(snapshot[name]) is not bool:
                raise TypeError(f"ARIA snapshot {name} must be a boolean")
        state = AriaState(**dict(snapshot))
        self._validate_state(state)
        return state


__all__ = [
    "AriaModality",
    "AriaScheduler",
    "AriaState",
    "AriaTokenKind",
]
