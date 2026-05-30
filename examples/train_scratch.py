#!/usr/bin/env python
import pandas as pd
import glob
from omnimesh.trainers import ScratchTrainer
from omnimesh.config import ModelConfig

config = ModelConfig()
texts, labels = [], []
for file_path in glob.glob("./data/*.csv"):
    df = pd.read_csv(file_path)
    if 'text' in df and 'label' in df:
        texts.extend(df['text'].astype(str).tolist())
        labels.extend(df['label'].tolist())

# Konversi label ke integer
unique = list(set(labels))
label_map = {lbl:i for i,lbl in enumerate(unique)}
labels = [label_map[l] for l in labels]

trainer = ScratchTrainer(config)
trainer.train(texts, labels, epochs=5)
