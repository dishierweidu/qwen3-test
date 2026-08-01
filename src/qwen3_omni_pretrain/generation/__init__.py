"""Generation helpers that are independent of architecture-specific models."""

from .speculative import (
    DraftProposal,
    SpeculativeStepOutput,
    speculative_step_greedy,
    speculative_step_sampled,
    verify_greedy_proposal,
    verify_sampled_proposal,
)

__all__ = [
    "DraftProposal",
    "SpeculativeStepOutput",
    "speculative_step_greedy",
    "speculative_step_sampled",
    "verify_greedy_proposal",
    "verify_sampled_proposal",
]
