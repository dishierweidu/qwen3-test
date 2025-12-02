# src/qwen3_omni_pretrain/models/qwen3_omni_moe/modules/moe.py

from typing import Tuple

import torch
import torch.nn as nn


class ExpertMLP(nn.Module):
    """单个 Expert，用最简单的 SiLU-MLP。"""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Qwen3OmniMoeMLP(nn.Module):
    """
    稀疏 Top-k MoE MLP + 负载均衡 aux loss。

    - Router: Linear(H -> E)，softmax 后对每个 token 选 top-k expert
    - 只对被选中的 (token, expert) 做前向，其他 expert 不算
    - aux_loss: Switch/Mixtral 风格的负载均衡项
      importance_i = mean(g_i)        # gate 概率在所有 token 上的均值
      load_i       = fraction(token i 被路由到 expert i)
      aux_loss = E * sum_i importance_i * load_i  ~ O(1)
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_tok = max(1, min(num_experts_per_tok, num_experts))

        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [ExpertMLP(hidden_size, intermediate_size) for _ in range(num_experts)]
        )

    def _dispatch_tokens(
        self,
        gate_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        根据 gate_probs 做 top-k，返回：
        - token_indices_flat: [Nsel] 被选中的 token 在 flatten 后的 index
        - expert_indices_flat: [Nsel] 对应的 expert id
        - scores_flat: [Nsel] 对应的 gate 权重
        """
        nt, E = gate_probs.shape
        k = self.num_experts_per_tok

        # top-k: [NT, k]
        topk_vals, topk_idx = gate_probs.topk(k=k, dim=-1)

        # 展平
        topk_vals_flat = topk_vals.reshape(-1)  # [NT * k]
        topk_idx_flat = topk_idx.reshape(-1)    # [NT * k]

        # 每个 (token, expert) 的 token 索引
        token_indices = (
            torch.arange(nt, device=gate_probs.device)
            .unsqueeze(1)
            .expand(nt, k)
            .reshape(-1)
        )  # [NT * k]

        return token_indices, topk_idx_flat, topk_vals_flat

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, T, H]
        return:
          - y: [B, T, H]
          - aux_loss: 标量 Tensor，用于负载均衡
        """
        B, T, H = x.shape
        nt = B * T
        x_flat = x.view(nt, H)  # [NT, H]

        # router logits & probs
        gate_logits = self.gate(x_flat)                     # [NT, E]
        gate_probs = torch.softmax(gate_logits, dim=-1)     # [NT, E]

        # 计算 top-k token 分派
        token_idx_flat, expert_idx_flat, scores_flat = self._dispatch_tokens(gate_probs)

        # ----- 负载均衡 aux loss -----
        E = self.num_experts
        # importance_i: gate 概率在所有 token 上的均值
        importance = gate_probs.mean(dim=0)  # [E]

        # load_i: 实际被分配到 expert i 的 (token, expert) 对占比
        load = torch.zeros(E, device=x_flat.device, dtype=gate_probs.dtype)
        ones = torch.ones_like(expert_idx_flat, dtype=gate_probs.dtype)
        load.index_add_(0, expert_idx_flat, ones)
        load = load / expert_idx_flat.numel()  # 归一化到 [0,1], sum(load) = 1

        aux_loss = (importance * load).sum() * E  # ~ O(1)

        # ----- 稀疏前向：只算被选中的 (token, expert) -----
        y_flat = torch.zeros_like(x_flat)  # [NT, H]

        for e_id, expert in enumerate(self.experts):
            # 找到属于该 expert 的 (token, gate) 条目
            mask = (expert_idx_flat == e_id)  # [NT * k]
            if not mask.any():
                continue

            sel_token_idx = token_idx_flat[mask]  # [N_sel]
            sel_scores = scores_flat[mask]        # [N_sel]

            x_sel = x_flat[sel_token_idx]         # [N_sel, H]
            out_sel = expert(x_sel)               # [N_sel, H]

            # gate 权重加权，然后 scatter 回 y_flat
            weighted_out = out_sel * sel_scores.unsqueeze(-1)  # [N_sel, H]

            # 一个 token 可能被多个 expert 选中，因此用 index_add_ 累加
            y_flat.index_add_(0, sel_token_idx, weighted_out)

        return y_flat.view(B, T, H), aux_loss
