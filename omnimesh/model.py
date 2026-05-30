import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
from typing import Dict, Optional
from .config import ModelConfig
from .udie import UniversalDataIngestionEngine
from .backbone import OmniMeshBackbone
from .safety import ConstitutionalSafetyRouterV2
from .atsg import AdaptiveTrainingStabilityGovernor


class OmniMeshV2(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.udie = UniversalDataIngestionEngine(vocab_size=config.vocab_size)
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.backbone = OmniMeshBackbone(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)
        self.safety_router = ConstitutionalSafetyRouterV2(backbone=self)
        self.atsg = AdaptiveTrainingStabilityGovernor(config)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, inputs: Dict, mode: str = 'train') -> torch.Tensor:
        if 'file_path' in inputs:
            tokens = self.udie.ingest_file(inputs['file_path'])
        elif 'tokens' in inputs:
            tokens = inputs['tokens']
        else:
            raise ValueError("Input must contain either 'tokens' or 'file_path'")
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        batch_size, seq_len = tokens.shape
        token_emb = self.token_embedding(tokens)
        positions = torch.arange(0, seq_len, device=tokens.device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)
        x = token_emb + pos_emb
        atsg_state = None
        if mode == 'train' and self.atsg:
            atsg_state = self.atsg.get_training_params()
        x = self.backbone(x, atsg_state, training=(mode == 'train'))
        logits = self.lm_head(x)
        return logits

    def generate(self, prompt: str = None, file_path: str = None,
                 max_new_tokens: int = 512, temperature: float = 0.7,
                 top_k: int = 50, use_safety: bool = True) -> str:
        self.eval()
        if file_path:
            tokens = self.udie.ingest_file(file_path)
        else:
            tokens = self.udie.text_tokenizer.encode(prompt)
        input_ids = tokens.unsqueeze(0)
        generated = []
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.forward({'tokens': input_ids}, mode='infer')
                next_logits = logits[0, -1, :] / temperature
                if top_k > 0:
                    indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                    next_logits[indices_to_remove] = float('-inf')
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                if next_token.item() == 3:  # EOS token
                    break
                generated.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                if input_ids.size(1) > self.config.max_seq_len:
                    break
        output = self.udie.text_tokenizer.decode(torch.tensor(generated))
        if use_safety:
            output = self.safety_router.check_and_revise(output, context=prompt or file_path or "")
        return output

    def train_with_file(self, file_path: str, epochs: int = 5, learning_rate: float = 3e-4):
        self.train()
        tokens = self.udie.ingest_file(file_path)
        optimizer = optim.AdamW(self.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            total_loss = 0
            for step in range(10):  # Simplified for demo
                optimizer.zero_grad()
                logits = self.forward({'tokens': tokens.unsqueeze(0)}, mode='train')
                loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), tokens[1:])
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/10:.4f}")
        return self
