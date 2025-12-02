# src/qwen3_omni_pretrain/models/qwen3_omni_moe/modules/moe.py

from typing import List

import torch
import torch.nn as nn


class ExpertMLP(nn.Module):
    """单个 Expert，用最简单的 SiLU-MLP。后续可以换成 SwiGLU。"""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Qwen3OmniMoeMLP(nn.Module):
    """
    一个简单的 Top-k MoE MLP：
    - Router：Linear(hidden_size -> num_experts)，softmax 后做 top-k
    - Expert：多个独立的 ExpertMLP
    - 组合：对 top-k expert 输出按 gate 权重做加权和

    注意：目前实现是“稠密 MoE”，每个 forward 会对所有 expert 计算一遍输出，
    然后只用 top-k 的结果做组合，也就是说计算量是完整 MoE 的 ~num_experts 倍。
    先保证结构对齐，后面再优化成真正稀疏路由。
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, H]
        return: [B, T, H]
        """
        B, T, H = x.shape
        nt = B * T
        x_flat = x.view(nt, H)  # [NT, H]

        # router logits & probs
        gate_logits = self.gate(x_flat)          # [NT, E]
        gate_probs = torch.softmax(gate_logits, dim=-1)  # [NT, E]

        # top-k gating
        k = self.num_experts_per_tok
        topk_vals, topk_idx = gate_probs.topk(k=k, dim=-1)  # [NT, k]

        # 计算所有 expert 输出（稠密版本，先保证正确性）
        all_outs: List[torch.Tensor] = []
        for expert in self.experts:
            all_outs.append(expert(x_flat))      # 每个: [NT, H]
        all_outs = torch.stack(all_outs, dim=1)  # [NT, E, H]

        # 按 top-k 索引挑出对应 expert 的输出
        expanded_idx = topk_idx.unsqueeze(-1).expand(-1, k, H)  # [NT, k, H]
        topk_outs = all_outs.gather(dim=1, index=expanded_idx)  # [NT, k, H]

        # gate 权重加权平均
        weighted = (topk_outs * topk_vals.unsqueeze(-1)).sum(dim=1)  # [NT, H]

        return weighted.view(B, T, H)
