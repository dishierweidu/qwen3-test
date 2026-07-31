from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from qwen3_omni_pretrain.generation.speculative import (
    DraftProposal,
    speculative_step_greedy,
    speculative_step_sampled,
    verify_sampled_proposal,
)
from qwen3_omni_pretrain.runtime.capabilities import CacheCapabilityError
from qwen3_omni_pretrain.runtime.state import DecoderState, StateOwner


def _categorical(token: int, vocab_size: int = 12) -> torch.Tensor:
    probability = torch.zeros(vocab_size)
    probability[token] = 1.0
    return probability


def _state(*, seen_tokens: int = 4, owner: StateOwner | None = None):
    return DecoderState.empty(
        owner or StateOwner.fresh("speculative-test"), batch_size=1
    )._derive(seen_tokens=torch.tensor([seen_tokens], dtype=torch.long))


@dataclass
class ScriptedDraft:
    tokens: list[int]
    probabilities: tuple[torch.Tensor, ...] | None = None

    def propose(self, *, state, max_draft_tokens):
        tokens = self.tokens[:max_draft_tokens]
        probabilities = self.probabilities or tuple(
            _categorical(token) for token in tokens
        )
        draft_state = state.advance_seen_tokens(
            torch.tensor([len(tokens)], dtype=torch.long)
        )
        return DraftProposal(
            token_ids=torch.tensor(tokens, dtype=torch.long),
            probabilities=probabilities[: len(tokens)],
            state=draft_state,
        )


@dataclass
class ScriptedTarget:
    distributions: tuple[torch.Tensor, ...]

    def score_proposal(self, *, state, proposed_token_ids):
        assert state.batch_size == 1
        return self.distributions[: proposed_token_ids.numel() + 1]

    def replay(self, *, state, token_ids):
        return state.advance_seen_tokens(
            torch.tensor([token_ids.numel()], dtype=torch.long)
        )


def test_greedy_rejection_restores_committed_state():
    target = ScriptedTarget(
        tuple(_categorical(token) for token in [5, 6, 9, 10])
    )
    draft = ScriptedDraft(tokens=[5, 7, 8])
    initial = _state(seen_tokens=4)

    result = speculative_step_greedy(
        target=target,
        draft=draft,
        state=initial,
        max_draft_tokens=3,
    )

    assert result.committed_token_ids.tolist() == [5, 6]
    assert result.accepted == 1
    assert result.rejected == 1
    assert result.proposed == 3
    assert result.decoder_state.seen_tokens.tolist() == [6]
    assert initial.seen_tokens.tolist() == [4]


def test_greedy_all_accepted_adds_target_bonus_token():
    target = ScriptedTarget(
        tuple(_categorical(token) for token in [2, 3, 4])
    )
    result = speculative_step_greedy(
        target=target,
        draft=ScriptedDraft(tokens=[2, 3]),
        state=_state(seen_tokens=1),
        max_draft_tokens=2,
    )
    assert result.committed_token_ids.tolist() == [2, 3, 4]
    assert (result.proposed, result.accepted, result.rejected) == (2, 2, 0)
    assert result.decoder_state.seen_tokens.tolist() == [4]


def test_sampled_forced_rejection_uses_normalized_p_minus_q():
    # For draft token 0: acceptance=0.25/0.5=0.5. On rejection,
    # clamp(p-q) has all mass at token 1.
    p = torch.tensor([0.25, 0.75])
    q = torch.tensor([0.50, 0.50])
    proposal = ScriptedDraft(tokens=[0], probabilities=(q,)).propose(
        state=_state(), max_draft_tokens=1
    )
    generator = torch.Generator().manual_seed(1)  # first draw > 0.5
    result = verify_sampled_proposal(
        proposal=proposal,
        target_probabilities=(p, p),
        committed_state=_state(owner=proposal.state.owner),
        replay=ScriptedTarget((p, p)).replay,
        generator=generator,
    )
    assert result.committed_token_ids.tolist() == [1]
    assert (result.accepted, result.rejected) == (0, 1)


def test_seeded_sampled_acceptance_matches_analytic_probability():
    p = torch.tensor([0.25, 0.75])
    q = torch.tensor([0.50, 0.50])
    initial = _state()
    proposal = ScriptedDraft(tokens=[0], probabilities=(q,)).propose(
        state=initial, max_draft_tokens=1
    )
    target = ScriptedTarget((p, p))
    generator = torch.Generator().manual_seed(1234)
    accepted = 0
    trials = 4_000
    for _ in range(trials):
        result = verify_sampled_proposal(
            proposal=proposal,
            target_probabilities=(p, p),
            committed_state=initial,
            replay=target.replay,
            generator=generator,
        )
        accepted += result.accepted
    assert accepted / trials == pytest.approx(0.5, abs=0.03)


def test_sampled_all_accepted_adds_one_target_draw():
    target = ScriptedTarget(
        (_categorical(3), _categorical(4), _categorical(5))
    )
    result = speculative_step_sampled(
        target=target,
        draft=ScriptedDraft(tokens=[3, 4]),
        state=_state(),
        max_draft_tokens=2,
        generator=torch.Generator().manual_seed(9),
    )
    assert result.committed_token_ids.tolist() == [3, 4, 5]
    assert result.decoder_state.seen_tokens.tolist() == [7]


def test_repaired_state_keeps_committed_owner_and_rejects_foreign_draft():
    initial = _state()
    foreign = _state(owner=StateOwner.fresh("foreign"))
    proposal = DraftProposal(
        token_ids=torch.tensor([1]),
        probabilities=(_categorical(1),),
        state=foreign.advance_seen_tokens(torch.tensor([1])),
    )
    with pytest.raises(CacheCapabilityError, match="STATE_OWNER_MISMATCH"):
        verify_sampled_proposal(
            proposal=proposal,
            target_probabilities=(_categorical(1), _categorical(2)),
            committed_state=initial,
            replay=ScriptedTarget((_categorical(1), _categorical(2))).replay,
        )


def test_returned_state_is_replay_not_overadvanced_draft_state():
    initial = _state(seen_tokens=10)
    proposal = DraftProposal(
        token_ids=torch.tensor([1, 2, 3]),
        probabilities=tuple(_categorical(token) for token in [1, 2, 3]),
        state=initial.advance_seen_tokens(torch.tensor([20])),
    )
    target = ScriptedTarget(
        tuple(_categorical(token) for token in [1, 9, 9, 9])
    )
    result = speculative_step_greedy(
        target=target,
        draft=ScriptedDraft(
            tokens=[1, 2, 3], probabilities=proposal.probabilities
        ),
        state=initial,
        max_draft_tokens=3,
    )
    assert result.committed_token_ids.tolist() == [1, 9]
    assert result.decoder_state.seen_tokens.tolist() == [12]
    assert proposal.state.seen_tokens.tolist() == [30]
