import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from .config import ModelConfig
from .ercd import EnhancedRecursiveContextDistiller


class DenseBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x


class SparseMoEBlock(nn.Module):
    def __init__(self, d_model: int, n_experts: int = 64, top_k: int = 8):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model)
            ) for _ in range(n_experts)
        ])
        self.gate = nn.Linear(d_model, n_experts)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, top_k: Optional[int] = None) -> torch.Tensor:
        k = top_k if top_k is not None else self.top_k
        batch_size, seq_len, d_model = x.shape
        gate_logits = self.gate(x)
        gate_scores = F.softmax(gate_logits, dim=-1)
        top_scores, top_indices = torch.topk(gate_scores, k, dim=-1)
        top_scores = top_scores / top_scores.sum(dim=-1, keepdim=True)
        output = torch.zeros_like(x)
        for expert_idx in range(self.n_experts):
            mask = (top_indices == expert_idx).any(dim=-1)
            if not mask.any():
                continue
            expert_input = x[mask]
            expert_output = self.experts[expert_idx](expert_input)
            output[mask] += expert_output * 0.5
        output = self.norm(x + output)
        return output


class OmniMeshSuperBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.dense = DenseBlock(config.d_model, config.n_heads)
        self.moe = SparseMoEBlock(config.d_model, config.n_experts, config.top_k_experts)
        self.ercd = EnhancedRecursiveContextDistiller(config.d_model, config.ercd_summary_len, config.ercd_memory_size)
        self.should_distill = True

    def forward(self, x: torch.Tensor, atsg_state: Optional[Dict] = None, training: bool = True) -> torch.Tensor:
        x = self.dense(x)
        top_k = atsg_state.get('topk', 8) if atsg_state else 8
        x = self.moe(x, top_k=top_k)
        if training and self.should_distill and x.size(1) > 4096:
            chunk = x[:, -4096:, :]
            summary = self.ercd.distill_chunk(chunk)
            x = torch.cat([summary, x], dim=1)
        return x


class OmniMeshBackbone(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.blocks = nn.ModuleList([OmniMeshSuperBlock(config) for _ in range(config.n_layers)])
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor, atsg_state: Optional[Dict] = None, training: bool = True) -> torch.Tensor:
        for i, block in enumerate(self.blocks):
            block.should_distill = (i % 4 == 0)
            x = block(x, atsg_state, training)
        return self.norm(x)
