"""Correct greedy and sampled speculative verification with state repair."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch

from qwen3_omni_pretrain.runtime.state import DecoderState


@dataclass(frozen=True)
class DraftProposal:
    token_ids: torch.LongTensor
    probabilities: tuple[torch.Tensor, ...]
    state: DecoderState

    def __post_init__(self) -> None:
        if (
            not isinstance(self.token_ids, torch.Tensor)
            or self.token_ids.dtype is not torch.long
            or self.token_ids.ndim != 1
            or self.token_ids.numel() == 0
        ):
            raise ValueError("draft token_ids must have non-empty long shape [N]")
        if bool((self.token_ids < 0).any().item()):
            raise ValueError("draft token IDs must be non-negative")
        if not isinstance(self.probabilities, tuple) or len(
            self.probabilities
        ) != self.token_ids.numel():
            raise ValueError("one draft probability vector is required per token")
        vocabulary_size: int | None = None
        cloned_probabilities: list[torch.Tensor] = []
        for probability in self.probabilities:
            normalized = _normalize_distribution(probability, name="draft")
            if vocabulary_size is None:
                vocabulary_size = normalized.numel()
            elif normalized.numel() != vocabulary_size:
                raise ValueError("draft probability vectors must share vocabulary")
            cloned_probabilities.append(normalized.detach().clone())
        if vocabulary_size is not None and bool(
            (self.token_ids >= vocabulary_size).any().item()
        ):
            raise ValueError("draft token ID exceeds probability vocabulary")
        if not isinstance(self.state, DecoderState) or self.state.batch_size != 1:
            raise ValueError("draft state must be a single-row DecoderState")
        object.__setattr__(self, "token_ids", self.token_ids.detach().clone())
        object.__setattr__(self, "probabilities", tuple(cloned_probabilities))
        object.__setattr__(self, "state", self.state.clone_detached())


@dataclass(frozen=True)
class SpeculativeStepOutput:
    committed_token_ids: torch.LongTensor
    decoder_state: DecoderState
    proposed: int
    accepted: int
    rejected: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.committed_token_ids, torch.Tensor)
            or self.committed_token_ids.dtype is not torch.long
            or self.committed_token_ids.ndim != 1
            or self.committed_token_ids.numel() == 0
        ):
            raise ValueError("committed_token_ids must have non-empty long shape [N]")
        if not isinstance(self.decoder_state, DecoderState):
            raise TypeError("decoder_state must be DecoderState")
        for field in ("proposed", "accepted", "rejected"):
            value = getattr(self, field)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")
        if self.accepted > self.proposed or self.rejected not in {0, 1}:
            raise ValueError("speculative acceptance counters are inconsistent")
        object.__setattr__(
            self,
            "committed_token_ids",
            self.committed_token_ids.detach().clone(),
        )
        object.__setattr__(
            self, "decoder_state", self.decoder_state.clone_detached()
        )


class SpeculativeTarget(Protocol):
    def score_proposal(
        self,
        *,
        state: DecoderState,
        proposed_token_ids: torch.LongTensor,
    ) -> tuple[torch.Tensor, ...]:
        """Return N+1 target next-token distributions."""
        ...

    def replay(
        self,
        *,
        state: DecoderState,
        token_ids: torch.LongTensor,
    ) -> DecoderState:
        ...


class SpeculativeDraft(Protocol):
    def propose(
        self,
        *,
        state: DecoderState,
        max_draft_tokens: int,
    ) -> DraftProposal:
        ...


def _normalize_distribution(
    probability: torch.Tensor, *, name: str
) -> torch.Tensor:
    if (
        not isinstance(probability, torch.Tensor)
        or probability.ndim != 1
        or probability.numel() == 0
        or not probability.is_floating_point()
    ):
        raise ValueError(f"{name} probability must have floating shape [V]")
    values = probability.float()
    if not bool(torch.isfinite(values).all().item()) or bool(
        (values < 0).any().item()
    ):
        raise ValueError(f"{name} probability must be finite and non-negative")
    total = values.sum()
    if not bool((total > 0).item()):
        raise ValueError(f"{name} probability must have positive mass")
    return values / total


def _target_distributions(
    values: tuple[torch.Tensor, ...],
    *,
    proposed: int,
    vocabulary_size: int,
) -> tuple[torch.Tensor, ...]:
    if not isinstance(values, tuple) or len(values) != proposed + 1:
        raise ValueError("target must return one distribution per draft plus one")
    normalized = tuple(
        _normalize_distribution(value, name="target") for value in values
    )
    if any(value.numel() != vocabulary_size for value in normalized):
        raise ValueError("target and draft probability vocabularies differ")
    return normalized


def _repair_state(
    *,
    committed_state: DecoderState,
    token_ids: torch.Tensor,
    replay: Callable[..., DecoderState],
) -> DecoderState:
    repaired = replay(state=committed_state, token_ids=token_ids)
    if not isinstance(repaired, DecoderState):
        raise TypeError("target replay must return DecoderState")
    repaired.assert_owner(committed_state.owner)
    expected_seen = committed_state.seen_tokens + token_ids.numel()
    if not torch.equal(repaired.seen_tokens, expected_seen):
        raise ValueError("target replay did not advance every committed token")
    return repaired


def _validate_verification_inputs(
    *,
    proposal: DraftProposal,
    target_probabilities: tuple[torch.Tensor, ...],
    committed_state: DecoderState,
) -> tuple[torch.Tensor, ...]:
    if not isinstance(proposal, DraftProposal):
        raise TypeError("proposal must be DraftProposal")
    if not isinstance(committed_state, DecoderState) or committed_state.batch_size != 1:
        raise ValueError("committed_state must be a single-row DecoderState")
    proposal.state.assert_owner(committed_state.owner)
    return _target_distributions(
        target_probabilities,
        proposed=proposal.token_ids.numel(),
        vocabulary_size=proposal.probabilities[0].numel(),
    )


def verify_greedy_proposal(
    *,
    proposal: DraftProposal,
    target_probabilities: tuple[torch.Tensor, ...],
    committed_state: DecoderState,
    replay: Callable[..., DecoderState],
) -> SpeculativeStepOutput:
    """Accept a matching prefix, then replay target correction or bonus."""

    targets = _validate_verification_inputs(
        proposal=proposal,
        target_probabilities=target_probabilities,
        committed_state=committed_state,
    )
    accepted = 0
    committed: list[int] = []
    rejected = 0
    for index, draft_token in enumerate(proposal.token_ids.tolist()):
        target_token = int(torch.argmax(targets[index]).item())
        if draft_token != target_token:
            committed.append(target_token)
            rejected = 1
            break
        committed.append(draft_token)
        accepted += 1
    if rejected == 0:
        committed.append(int(torch.argmax(targets[-1]).item()))
    committed_ids = torch.tensor(
        committed,
        dtype=torch.long,
        device=proposal.token_ids.device,
    )
    repaired = _repair_state(
        committed_state=committed_state,
        token_ids=committed_ids,
        replay=replay,
    )
    return SpeculativeStepOutput(
        committed_token_ids=committed_ids,
        decoder_state=repaired,
        proposed=proposal.token_ids.numel(),
        accepted=accepted,
        rejected=rejected,
    )


def _sample(
    probability: torch.Tensor,
    *,
    generator: torch.Generator | None,
) -> int:
    if generator is not None and generator.device.type != "cpu":
        raise ValueError(
            "speculative verification requires a CPU generator for "
            "device-independent correction sampling"
        )
    return int(
        torch.multinomial(
            probability.detach().float().cpu(),
            1,
            generator=generator,
        ).item()
    )


def verify_sampled_proposal(
    *,
    proposal: DraftProposal,
    target_probabilities: tuple[torch.Tensor, ...],
    committed_state: DecoderState,
    replay: Callable[..., DecoderState],
    generator: torch.Generator | None = None,
) -> SpeculativeStepOutput:
    """Apply min(1,p/q), residual correction, and immutable replay.

    If a draft assigns zero probability to its own token, a target-positive
    token is accepted (the p/q limit is infinite); when both are zero it is
    rejected.  If numerical subtraction leaves zero residual mass, the target
    distribution is the documented fallback.
    """

    targets = _validate_verification_inputs(
        proposal=proposal,
        target_probabilities=target_probabilities,
        committed_state=committed_state,
    )
    accepted = 0
    rejected = 0
    committed: list[int] = []
    for index, draft_token in enumerate(proposal.token_ids.tolist()):
        target = targets[index]
        draft = proposal.probabilities[index].to(target.device)
        p_x = float(target[draft_token].item())
        q_x = float(draft[draft_token].item())
        if q_x == 0.0:
            acceptance = 1.0 if p_x > 0.0 else 0.0
        else:
            acceptance = min(1.0, p_x / q_x)
        draw = float(
            torch.rand((), generator=generator, device="cpu").item()
        )
        if draw < acceptance:
            committed.append(draft_token)
            accepted += 1
            continue
        residual = torch.clamp(target - draft, min=0.0)
        if not bool((residual.sum() > 0).item()):
            residual = target
        else:
            residual = residual / residual.sum()
        committed.append(_sample(residual, generator=generator))
        rejected = 1
        break
    if rejected == 0:
        committed.append(_sample(targets[-1], generator=generator))
    committed_ids = torch.tensor(
        committed,
        dtype=torch.long,
        device=proposal.token_ids.device,
    )
    repaired = _repair_state(
        committed_state=committed_state,
        token_ids=committed_ids,
        replay=replay,
    )
    return SpeculativeStepOutput(
        committed_token_ids=committed_ids,
        decoder_state=repaired,
        proposed=proposal.token_ids.numel(),
        accepted=accepted,
        rejected=rejected,
    )


def _proposal(
    *,
    draft: SpeculativeDraft,
    state: DecoderState,
    max_draft_tokens: int,
) -> DraftProposal:
    if type(max_draft_tokens) is not int or max_draft_tokens <= 0:
        raise ValueError("max_draft_tokens must be a positive integer")
    proposal = draft.propose(
        state=state.clone_detached(),
        max_draft_tokens=max_draft_tokens,
    )
    if not isinstance(proposal, DraftProposal):
        raise TypeError("draft.propose must return DraftProposal")
    if proposal.token_ids.numel() > max_draft_tokens:
        raise ValueError("draft returned more than max_draft_tokens")
    return proposal


def speculative_step_greedy(
    *,
    target: SpeculativeTarget,
    draft: SpeculativeDraft,
    state: DecoderState,
    max_draft_tokens: int,
) -> SpeculativeStepOutput:
    proposal = _proposal(
        draft=draft,
        state=state,
        max_draft_tokens=max_draft_tokens,
    )
    scores = target.score_proposal(
        state=state.clone_detached(),
        proposed_token_ids=proposal.token_ids,
    )
    return verify_greedy_proposal(
        proposal=proposal,
        target_probabilities=scores,
        committed_state=state,
        replay=target.replay,
    )


def speculative_step_sampled(
    *,
    target: SpeculativeTarget,
    draft: SpeculativeDraft,
    state: DecoderState,
    max_draft_tokens: int,
    generator: torch.Generator | None = None,
) -> SpeculativeStepOutput:
    proposal = _proposal(
        draft=draft,
        state=state,
        max_draft_tokens=max_draft_tokens,
    )
    scores = target.score_proposal(
        state=state.clone_detached(),
        proposed_token_ids=proposal.token_ids,
    )
    return verify_sampled_proposal(
        proposal=proposal,
        target_probabilities=scores,
        committed_state=state,
        replay=target.replay,
        generator=generator,
    )


__all__ = [
    "DraftProposal",
    "SpeculativeDraft",
    "SpeculativeStepOutput",
    "SpeculativeTarget",
    "speculative_step_greedy",
    "speculative_step_sampled",
    "verify_greedy_proposal",
    "verify_sampled_proposal",
]
