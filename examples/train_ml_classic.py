#!/usr/bin/env python
from omnimesh.trainers import MLClassicTrainer
from omnimesh.config import ModelConfig

config = ModelConfig()
trainer = MLClassicTrainer(data_dir="./data", model_dir=config.model_dir)
trainer.train()
