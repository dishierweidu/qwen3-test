from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class ExpertMLP(nn.Module):
    """Two-layer SiLU expert used by the routed MoE block."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Qwen3OmniMoeMLP(nn.Module):
    """
    Sparse top-k routed experts plus a Switch-style load-balancing loss.

    The enclosing decoder layer already executes ``shared_mlp`` on every
    token. This module intentionally contains routed experts only; enabling a
    second internal shared expert would double-count the dense FFN path.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        *,
        use_shared_expert: bool = False,
        shared_intermediate_size: int | None = None,
        router_init_std: float = 1e-3,
        router_normalize_init: bool = True,
        renormalize_topk: bool = True,
    ) -> None:
        super().__init__()
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if use_shared_expert:
            raise ValueError(
                "Qwen3OmniMoeMLP must not create an internal shared expert: "
                "the decoder shared_mlp already supplies the shared dense path. "
                "Set use_shared_expert=false in the model config."
            )
        if shared_intermediate_size is not None:
            raise ValueError(
                "shared_intermediate_size is invalid when the decoder shared_mlp "
                "owns the shared dense path"
            )

        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.num_experts = int(num_experts)
        self.num_experts_per_tok = max(
            1, min(int(num_experts_per_tok), self.num_experts)
        )
        self.renormalize_topk = bool(renormalize_topk)

        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        with torch.no_grad():
            self.gate.weight.normal_(mean=0.0, std=float(router_init_std))
            if router_normalize_init:
                norms = torch.norm(self.gate.weight, dim=-1, keepdim=True)
                self.gate.weight.div_(norms + 1e-6)

        self.experts = nn.ModuleList(
            [
                ExpertMLP(self.hidden_size, self.intermediate_size)
                for _ in range(self.num_experts)
            ]
        )
        # Kept for compatibility with parameter-inspection code and old callers.
        self.shared_expert = None
        self.register_buffer(
            "_nonfinite_diagnostic", torch.tensor(False), persistent=False
        )
        self._nonfinite_diagnostic_reason = (
            "MoE router probabilities or auxiliary loss are non-finite"
        )

    def _dispatch_tokens(
        self, gate_probs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if gate_probs.dim() != 2 or gate_probs.size(-1) != self.num_experts:
            raise ValueError(
                "gate_probs must have shape [num_tokens, num_experts]"
            )
        num_tokens = gate_probs.size(0)
        topk_values, topk_indices = gate_probs.topk(
            k=self.num_experts_per_tok, dim=-1
        )
        if self.renormalize_topk:
            denominator = topk_values.sum(dim=-1, keepdim=True)
            # Keep this branch free of rank-local numerical exceptions. Any
            # NaN/Inf propagates into model outputs and is handled by the
            # trainer's synchronized non-finite check.
            topk_values = topk_values / denominator

        token_indices = (
            torch.arange(num_tokens, device=gate_probs.device)
            .unsqueeze(1)
            .expand(num_tokens, self.num_experts_per_tok)
            .reshape(-1)
        )
        return (
            token_indices,
            topk_indices.reshape(-1),
            topk_values.reshape(-1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_size = x.shape
        if hidden_size != self.hidden_size:
            raise ValueError(
                f"expected hidden size {self.hidden_size}, got {hidden_size}"
            )
        x_flat = x.reshape(batch_size * sequence_length, hidden_size)
        gate_logits = self.gate(x_flat)
        gate_probs = torch.softmax(gate_logits, dim=-1)
        token_indices, expert_indices, scores = self._dispatch_tokens(gate_probs)

        importance = gate_probs.mean(dim=0)
        load = torch.zeros(
            self.num_experts,
            device=x_flat.device,
            dtype=gate_probs.dtype,
        )
        load.index_add_(
            0,
            expert_indices,
            torch.ones_like(expert_indices, dtype=gate_probs.dtype),
        )
        load = load / max(1, expert_indices.numel())
        aux_loss = (importance * load).sum() * self.num_experts
        self._nonfinite_diagnostic = (
            (~torch.isfinite(gate_probs).all()) | (~torch.isfinite(aux_loss))
        ).detach()
        y_flat = torch.zeros_like(x_flat)
        for expert_id, expert in enumerate(self.experts):
            selected = expert_indices == expert_id
            has_tokens = bool(selected.any().item())
            if has_tokens:
                selected_token_indices = token_indices[selected]
                selected_scores = scores[selected]
                expert_input = x_flat[selected_token_indices]
            else:
                # DeepSpeed ZeRO-3 ranks must execute every expert consistently.
                selected_token_indices = None
                selected_scores = None
                expert_input = x_flat[:1]

            expert_output = expert(expert_input)
            if has_tokens:
                weighted_output = expert_output * selected_scores.unsqueeze(-1)
                y_flat.index_add_(
                    0, selected_token_indices, weighted_output
                )

        return y_flat.view(batch_size, sequence_length, hidden_size), aux_loss
