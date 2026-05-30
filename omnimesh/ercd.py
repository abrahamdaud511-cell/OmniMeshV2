import torch
import torch.nn as nn
from typing import Dict, Optional


class EnhancedRecursiveContextDistiller(nn.Module):
    """
    Memampatkan konteks panjang menjadi summary dan mempertahankan memory bank
    dengan pruning berdasarkan relevance score.
    """
    def __init__(self, d_model: int, summary_len: int = 4, max_memory_size: int = 10000):
        super().__init__()
        self.summary_query = nn.Parameter(torch.randn(summary_len, d_model))
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        self.relevance_scorer = nn.Linear(d_model, 1)
        self.memory_bank = []
        self.max_memory = max_memory_size
        self.d_model = d_model

    def distill_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        batch_size = chunk.size(0)
        query = self.summary_query.unsqueeze(0).expand(batch_size, -1, -1)
        summary, _ = self.cross_attn(query, chunk, chunk)
        score = torch.sigmoid(self.relevance_scorer(summary.mean(dim=1))).mean().item()
        self.memory_bank.append((summary.detach(), score))
        if len(self.memory_bank) > self.max_memory:
            self.memory_bank.sort(key=lambda x: x[1])
            self.memory_bank.pop(0)
        return summary

    def retrieve_context(self, query: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.memory_bank:
            return None
        summaries = torch.cat([s for s, _ in self.memory_bank], dim=1)
        attended, _ = self.cross_attn(query, summaries, summaries)
        return attended

    def get_memory_stats(self) -> Dict:
        return {
            'memory_size': len(self.memory_bank),
            'avg_relevance': sum(s for _, s in self.memory_bank) / max(1, len(self.memory_bank))
        }
