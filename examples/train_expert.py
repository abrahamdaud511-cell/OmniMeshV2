#!/usr/bin/env python
from omnimesh.trainers import ExpertTrainer
from omnimesh.config import ModelConfig

config = ModelConfig()
trainer = ExpertTrainer(model_dir=config.model_dir)
trainer.train(dataset_name="indonlu", subset="smsa")
