import os
import glob
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
from typing import List, Tuple
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from .config import ModelConfig
from .atsg import AdaptiveTrainingStabilityGovernor


class DataLoaderWithThrottle:
    def __init__(self, dataset, batch_size, shuffle=True, throttle_level=1.0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.throttle_level = throttle_level
        self._data_iter = None

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            import random
            random.shuffle(indices)
        self._data_iter = (self.dataset[i] for i in indices)
        return self

    def __next__(self):
        if self.throttle_level < 1.0:
            time.sleep(0.01 * (1 - self.throttle_level))
        batch = []
        for _ in range(int(self.batch_size * self.throttle_level)):
            try:
                batch.append(next(self._data_iter))
            except StopIteration:
                break
        if not batch:
            raise StopIteration
        return torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch])

    def set_throttle(self, level):
        self.throttle_level = level


class MLClassicTrainer:
    def __init__(self, data_dir: str, model_dir: str):
        self.data_dir = data_dir
        self.model_dir = model_dir

    def load_data(self) -> Tuple[List[str], List]:
        all_texts = []
        all_labels = []
        for file_path in glob.glob(os.path.join(self.data_dir, "*.csv")):
            try:
                df = pd.read_csv(file_path)
                if 'text' in df.columns and 'label' in df.columns:
                    all_texts.extend(df['text'].astype(str).tolist())
                    all_labels.extend(df['label'].tolist())
                else:
                    print(f"Skipping {file_path}: missing text/label columns")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        return all_texts, all_labels

    def train(self):
        print("📊 Training ML Classic (TF-IDF + Logistic Regression)...")
        texts, labels = self.load_data()
        if not texts:
            raise ValueError("No valid data found in data directory")
        label_map = {}
        if isinstance(labels[0], str):
            unique_labels = list(set(labels))
            label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
            labels = [label_map[lbl] for lbl in labels]
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
            ('clf', LogisticRegression(max_iter=200, C=1.0))
        ])
        pipeline.fit(texts, labels)
        preds = pipeline.predict(texts)
        acc = accuracy_score(labels, preds)
        print(f"✅ ML Classic Accuracy: {acc:.4f}")
        os.makedirs(self.model_dir, exist_ok=True)
        model_path = os.path.join(self.model_dir, "ml_classic.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump({'pipeline': pipeline, 'label_map': label_map}, f)
        report = classification_report(labels, preds, target_names=list(label_map.keys()) if label_map else None)
        with open(os.path.join(self.model_dir, "classification_report.txt"), 'w') as f:
            f.write(report)
        return pipeline, label_map

    def predict(self, text: str, pipeline=None) -> str:
        if pipeline is None:
            with open(os.path.join(self.model_dir, "ml_classic.pkl"), 'rb') as f:
                data = pickle.load(f)
                pipeline = data['pipeline']
                label_map = data.get('label_map', {})
        pred = pipeline.predict([text])[0]
        if label_map:
            inv_map = {v: k for k, v in label_map.items()}
            return inv_map.get(pred, str(pred))
        return str(pred)


class ExpertTrainer:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir

    def train(self, dataset_name: str = "indonlu", subset: str = "smsa"):
        print("🚀 Expert Mode: Fine-tuning DistilBERT...")
        model_name = "cahya/distilbert-base-indonesian"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
        dataset = load_dataset(dataset_name, subset)
        def tokenize(batch):
            return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)
        dataset = dataset.map(tokenize, batched=True)
        dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        training_args = TrainingArguments(
            output_dir=os.path.join(self.model_dir, "expert_checkpoints"),
            num_train_epochs=2,
            per_device_train_batch_size=16,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            logging_steps=50,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            tokenizer=tokenizer,
        )
        trainer.train()
        final_path = os.path.join(self.model_dir, "model_expert_final")
        model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        print(f"✅ Expert model saved to {final_path}")
        return model, tokenizer


class ScratchTrainer:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.atsg = AdaptiveTrainingStabilityGovernor(config)

    def train(self, texts: List[str], labels: List[int], epochs: int = 5):
        print("🧠 Scratch Mode: Training transformer from scratch with ATSG...")
        counter = Counter()
        for text in texts:
            counter.update(text.lower().split())
        vocab = {word: idx+2 for idx, (word, _) in enumerate(counter.most_common(5000))}
        vocab["<PAD>"] = 0
        vocab["<UNK>"] = 1
        vocab_size = len(vocab)
        max_len = 128
        def encode(text):
            tokens = text.lower().split()[:max_len]
            ids = [vocab.get(t, 1) for t in tokens]
            ids += [0] * (max_len - len(ids))
            return ids
        X = [encode(t) for t in texts]
        y = labels
        X = torch.tensor(X, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        class MiniTransformerClf(nn.Module):
            def __init__(self, vocab_size, embed_dim=128, num_heads=4, num_layers=3, num_classes=3, max_len=128):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
                self.pos_embed = nn.Parameter(torch.randn(1, max_len, embed_dim))
                encoder_layer = nn.TransformerEncoderLayer(embed_dim, num_heads, batch_first=True)
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
                self.fc = nn.Linear(embed_dim, num_classes)
            def forward(self, x):
                emb = self.embedding(x) + self.pos_embed[:, :x.size(1), :]
                out = self.transformer(emb)
                out = out.mean(dim=1)
                return self.fc(out)
        model = MiniTransformerClf(vocab_size, num_classes=len(set(y.tolist())))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        self.atsg.start()
        dataset = TensorDataset(X, y)
        for epoch in range(epochs):
            params = self.atsg.get_training_params()
            batch_size = params['batch_size']
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            model.train()
            total_loss = 0
            for i, (xb, yb) in enumerate(loader):
                if i % 50 == 0:
                    status, cpu, mem = self._get_system_status()
                    if status == "berhenti":
                        self._wait_until_safe()
                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Batch size: {batch_size}")
        self.atsg.stop()
        os.makedirs(self.config.model_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(self.config.model_dir, "model_scratch.pth"))
        with open(os.path.join(self.config.model_dir, "vocab.pkl"), 'wb') as f:
            pickle.dump(vocab, f)
        print("✅ Scratch model saved")
        return model, vocab

    def _get_system_status(self):
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=1)
            mem = psutil.virtual_memory().percent
        except:
            cpu = mem = 50
        if cpu > 90 or mem > 90:
            return "berhenti", cpu, mem
        elif cpu > 80 or mem > 80:
            return "hati-hati", cpu, mem
        return "normal", cpu, mem

    def _wait_until_safe(self):
        print("⏸️ System overloaded. Training paused...")
        while True:
            status, cpu, mem = self._get_system_status()
            if status != "berhenti":
                print(f"✅ System recovered (CPU: {cpu}%, RAM: {mem}%). Resuming...")
                break
            time.sleep(3)
