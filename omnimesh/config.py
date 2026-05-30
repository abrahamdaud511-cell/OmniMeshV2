from dataclasses import dataclass
import os

@dataclass
class ModelConfig:
    d_model: int = 1024
    n_heads: int = 16
    n_layers: int = 48
    vocab_size: int = 256000
    max_seq_len: int = 8192
    n_experts: int = 64
    top_k_experts: int = 8
    expert_capacity: float = 1.25
    ercd_summary_len: int = 4
    ercd_memory_size: int = 10000
    batch_size: int = 32
    learning_rate: float = 3e-4
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 2000
    weight_decay: float = 0.01
    cpu_threshold: float = 80.0
    gpu_mem_threshold: float = 0.9
    temp_threshold: float = 82.0
    model_dir: str = "./models/omnimesh_v2"
    log_dir: str = "./logs"
    data_dir: str = "./data"
    
    def __post_init__(self):
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
