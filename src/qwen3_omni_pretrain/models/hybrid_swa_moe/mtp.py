"""Teacher-forced multi-token prediction heads."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class MtpTrainingOutput:
    logits: tuple[torch.Tensor, ...]
    losses: tuple[torch.Tensor, ...]
    loss: torch.Tensor | None
    target_offsets: tuple[int, ...]
    valid_token_counts: tuple[int, ...]


class _PredictorHead(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, eps: float) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=eps)
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        projected = hidden_states + F.silu(
            self.projection(self.norm(hidden_states))
        )
        return self.lm_head(projected)


class MultiTokenPredictor(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        vocab_size: int,
        num_predictors: int,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or vocab_size <= 0:
            raise ValueError("MTP dimensions must be positive")
        if type(num_predictors) is not int or not 1 <= num_predictors <= 3:
            raise ValueError("num_predictors must be in [1, 3]")
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_predictors = num_predictors
        self.predictors = nn.ModuleList(
            _PredictorHead(hidden_size, vocab_size, rms_norm_eps)
            for _ in range(num_predictors)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        labels: torch.LongTensor | None = None,
    ) -> MtpTrainingOutput:
        if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [B, T, H]")
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError("hidden size does not match MTP")
        if labels is not None and (
            not isinstance(labels, torch.Tensor)
            or labels.dtype is not torch.long
            or labels.shape != hidden_states.shape[:2]
        ):
            raise ValueError("labels must have dtype long and shape [B, T]")

        sequence_length = hidden_states.shape[1]
        logits: list[torch.Tensor] = []
        losses: list[torch.Tensor] = []
        offsets: list[int] = []
        valid_counts: list[int] = []
        for predictor_index, predictor in enumerate(self.predictors):
            offset = predictor_index + 2
            offsets.append(offset)
            usable = max(0, sequence_length - offset)
            predictor_logits = predictor(hidden_states[:, :usable])
            logits.append(predictor_logits)
            if labels is None or usable == 0:
                valid_counts.append(0)
                continue
            targets = labels[:, offset:]
            valid_count = int((targets != -100).sum().item())
            valid_counts.append(valid_count)
            if valid_count == 0:
                continue
            loss = F.cross_entropy(
                predictor_logits.reshape(-1, self.vocab_size).float(),
                targets.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            ) / valid_count
            losses.append(loss)
        aggregate = torch.stack(losses).mean() if losses else None
        return MtpTrainingOutput(
            logits=tuple(logits),
            losses=tuple(losses),
            loss=aggregate,
            target_offsets=tuple(offsets),
            valid_token_counts=tuple(valid_counts),
        )


__all__ = ["MtpTrainingOutput", "MultiTokenPredictor"]
