import os
import glob
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import torch
import psutil
from .config import ModelConfig
from .model import OmniMeshV2
from .trainers import MLClassicTrainer, ExpertTrainer, ScratchTrainer

try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False


class OmniMeshGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("OmniMeshV2 - AI Universal Trainer")
        self.root.geometry("1000x700")
        self.config = ModelConfig()
        self.model = None
        self.is_training = False
        self._setup_ui()
        self._update_load_display()

    def _setup_ui(self):
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_frame = tk.LabelFrame(main_frame, text="🎓 Training", padx=5, pady=5)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        tk.Label(left_frame, text="Training Mode:").pack(anchor=tk.W)
        self.mode_var = tk.StringVar(value="ml_classic")
        modes = [
            ("ML Classic (TF-IDF + LogReg)", "ml_classic"),
            ("Expert (Fine-tune DistilBERT)", "expert"),
            ("Scratch (Transformer from zero)", "scratch")
        ]
        for text, mode in modes:
            tk.Radiobutton(left_frame, text=text, variable=self.mode_var, value=mode).pack(anchor=tk.W)

        tk.Label(left_frame, text="Data Directory:").pack(anchor=tk.W, pady=(10,0))
        self.data_dir_var = tk.StringVar(value="./data")
        data_frame = tk.Frame(left_frame)
        data_frame.pack(fill=tk.X)
        tk.Entry(data_frame, textvariable=self.data_dir_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(data_frame, text="Browse", command=self._browse_data_dir).pack(side=tk.RIGHT)

        self.train_btn = tk.Button(left_frame, text="▶️ Start Training", command=self._start_training,
                                   bg="#4CAF50", fg="white", font=("Arial", 12, "bold"))
        self.train_btn.pack(pady=10, fill=tk.X)

        self.progress = ttk.Progressbar(left_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=5)

        self.status_var = tk.StringVar(value="Ready")
        tk.Label(left_frame, textvariable=self.status_var, fg="blue").pack(anchor=tk.W)

        monitor_frame = tk.LabelFrame(left_frame, text="🖥️ System Monitor", padx=5, pady=5)
        monitor_frame.pack(fill=tk.X, pady=10)
        self.load_var = tk.StringVar(value="CPU: -%, RAM: -%, GPU: -%")
        tk.Label(monitor_frame, textvariable=self.load_var).pack()

        right_frame = tk.LabelFrame(main_frame, text="💬 Inference", padx=5, pady=5)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        tk.Label(right_frame, text="Context File (PDF, code, text):").pack(anchor=tk.W)
        file_frame = tk.Frame(right_frame)
        file_frame.pack(fill=tk.X)
        self.context_file_var = tk.StringVar()
        tk.Entry(file_frame, textvariable=self.context_file_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(file_frame, text="Browse", command=self._browse_context_file).pack(side=tk.RIGHT)

        tk.Label(right_frame, text="Prompt:").pack(anchor=tk.W, pady=(10,0))
        self.prompt_text = tk.Text(right_frame, height=5, width=50)
        self.prompt_text.pack(fill=tk.X, pady=5)

        self.generate_btn = tk.Button(right_frame, text="✨ Generate", command=self._generate,
                                      bg="#2196F3", fg="white", font=("Arial", 10, "bold"))
        self.generate_btn.pack(pady=5)

        tk.Label(right_frame, text="Output:").pack(anchor=tk.W)
        self.output_text = tk.Text(right_frame, height=12, width=50, wrap=tk.WORD)
        self.output_text.pack(fill=tk.BOTH, expand=True, pady=5)

        tk.Button(right_frame, text="Load Model", command=self._load_model).pack(pady=5)

    def _browse_data_dir(self):
        dir_path = filedialog.askdirectory()
        if dir_path:
            self.data_dir_var.set(dir_path)

    def _browse_context_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("All supported", "*.txt *.pdf *.py *.js *.html *.csv"), ("All files", "*.*")]
        )
        if file_path:
            self.context_file_var.set(file_path)

    def _update_load_display(self):
        try:
            cpu = psutil.cpu_percent()
            mem = psutil.virtual_memory().percent
            gpu = "N/A"
            if HAS_GPUTIL:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = f"{gpus[0].load*100:.0f}%"
            self.load_var.set(f"CPU: {cpu:.0f}%, RAM: {mem:.0f}%, GPU: {gpu}")
        except:
            pass
        self.root.after(3000, self._update_load_display)

    def _start_training(self):
        if self.is_training:
            return
        mode = self.mode_var.get()
        data_dir = self.data_dir_var.get()
        self.is_training = True
        self.train_btn.config(state=tk.DISABLED)
        self.progress.start()
        self.status_var.set(f"Training in progress ({mode})...")

        def train_thread():
            try:
                if mode == "ml_classic":
                    trainer = MLClassicTrainer(data_dir, self.config.model_dir)
                    trainer.train()
                    self.root.after(0, self._on_training_done, "ML Classic training completed!")
                elif mode == "expert":
                    trainer = ExpertTrainer(self.config.model_dir)
                    trainer.train()
                    self.root.after(0, self._on_training_done, "Expert training completed!")
                elif mode == "scratch":
                    texts, labels = [], []
                    for file_path in glob.glob(os.path.join(data_dir, "*.csv")):
                        df = pd.read_csv(file_path)
                        if 'text' in df.columns and 'label' in df.columns:
                            texts.extend(df['text'].astype(str).tolist())
                            labels.extend(df['label'].tolist())
                    if not texts:
                        raise ValueError("No valid CSV files with 'text' and 'label' columns")
                    unique_labels = list(set(labels))
                    label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
                    labels = [label_map[lbl] for lbl in labels]
                    trainer = ScratchTrainer(self.config)
                    trainer.train(texts, labels)
                    self.root.after(0, self._on_training_done, "Scratch training completed!")
            except Exception as e:
                self.root.after(0, self._on_training_error, str(e))

        threading.Thread(target=train_thread, daemon=True).start()

    def _on_training_done(self, message):
        self.progress.stop()
        self.train_btn.config(state=tk.NORMAL)
        self.is_training = False
        self.status_var.set(message)
        messagebox.showinfo("Success", message)

    def _on_training_error(self, error):
        self.progress.stop()
        self.train_btn.config(state=tk.NORMAL)
        self.is_training = False
        self.status_var.set(f"Error: {error}")
        messagebox.showerror("Training Error", error)

    def _load_model(self):
        try:
            self.model = OmniMeshV2(self.config)
            weights_path = os.path.join(self.config.model_dir, "omnimesh_v2.pth")
            if os.path.exists(weights_path):
                self.model.load_state_dict(torch.load(weights_path, map_location='cpu'))
            self.status_var.set("Model loaded successfully")
            messagebox.showinfo("Success", "Model loaded!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")

    def _generate(self):
        if self.model is None:
            messagebox.showwarning("Warning", "Please load a model first")
            return
        prompt = self.prompt_text.get("1.0", tk.END).strip()
        context_file = self.context_file_var.get()
        if not prompt and not context_file:
            messagebox.showwarning("Warning", "Enter a prompt or select a context file")
            return
        self.generate_btn.config(state=tk.DISABLED)
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, "Generating...")

        def generate_thread():
            try:
                if context_file:
                    response = self.model.generate(file_path=context_file, prompt=prompt if prompt else None)
                else:
                    response = self.model.generate(prompt=prompt)
                self.root.after(0, self._on_generation_done, response)
            except Exception as e:
                self.root.after(0, self._on_generation_error, str(e))

        threading.Thread(target=generate_thread, daemon=True).start()

    def _on_generation_done(self, response):
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, response)
        self.generate_btn.config(state=tk.NORMAL)

    def _on_generation_error(self, error):
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, f"Error: {error}")
        self.generate_btn.config(state=tk.NORMAL)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="OmniMeshV2 - Universal AI Model")
    parser.add_argument("command", nargs="?", default="gui",
                        choices=["gui", "train", "infer", "watch"],
                        help="Command to execute")
    parser.add_argument("--mode", choices=["ml_classic", "expert", "scratch"], default="ml_classic")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--file", help="File for inference")
    parser.add_argument("--prompt", help="Prompt for inference")
    args = parser.parse_args()
    config = ModelConfig()
    config.data_dir = args.data_dir
    if args.command == "gui":
        root = tk.Tk()
        app = OmniMeshGUI(root)
        root.mainloop()
    elif args.command == "train":
        if args.mode == "ml_classic":
            trainer = MLClassicTrainer(args.data_dir, config.model_dir)
            trainer.train()
        elif args.mode == "expert":
            trainer = ExpertTrainer(config.model_dir)
            trainer.train()
        elif args.mode == "scratch":
            texts, labels = [], []
            for file_path in glob.glob(os.path.join(args.data_dir, "*.csv")):
                df = pd.read_csv(file_path)
                if 'text' in df.columns and 'label' in df.columns:
                    texts.extend(df['text'].astype(str).tolist())
                    labels.extend(df['label'].tolist())
            if not texts:
                print("No valid data found")
                return
            unique_labels = list(set(labels))
            label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
            labels = [label_map[lbl] for lbl in labels]
            trainer = ScratchTrainer(config)
            trainer.train(texts, labels)
    elif args.command == "infer":
        model = OmniMeshV2(config)
        weights_path = os.path.join(config.model_dir, "omnimesh_v2.pth")
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        if args.file:
            response = model.generate(file_path=args.file, prompt=args.prompt or "")
        else:
            response = model.generate(prompt=args.prompt or "Hello, how can I help?")
        print("\n" + "="*50)
        print("RESPONSE:")
        print("="*50)
        print(response)
    elif args.command == "watch":
        print("👀 Watching for changes in data directory...")
        last_hash = None
        def get_dir_hash():
            import hashlib
            hasher = hashlib.md5()
            for file_path in sorted(glob.glob(os.path.join(args.data_dir, "*.csv"))):
                with open(file_path, 'rb') as f:
                    hasher.update(f.read())
            return hasher.hexdigest()
        try:
            while True:
                current_hash = get_dir_hash()
                if current_hash != last_hash and last_hash is not None:
                    print("\n🔄 Data changed! Retraining ML Classic...")
                    trainer = MLClassicTrainer(args.data_dir, config.model_dir)
                    trainer.train()
                last_hash = current_hash
                time.sleep(5)
        except KeyboardInterrupt:
            print("\nWatch stopped.")


if __name__ == "__main__":
    main()
