import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os

class ActivationExtractor:
    def __init__(self, model_name="EleutherAI/pythia-1b", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading {model_name} on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16
        )
        self.model.to(self.device)
        self.model.eval()

        self.num_layers = self.model.config.num_hidden_layers
        print(f"Loaded. Layers: {self.num_layers}, Hidden dim: {self.model.config.hidden_size}")

    def extract(self, texts, batch_size=8, max_length=128):
        """Returns {layer_idx: np.array of shape (n_samples, hidden_dim)}"""
        all_acts = {i: [] for i in range(self.num_layers)}

        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting"):
            batch = texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(
                    **encoded,
                    output_hidden_states=True  # returns all layer hidden states
                )

            # outputs.hidden_states: tuple of (num_layers+1) tensors
            # index 0 = embedding layer, 1..N = transformer layers
            # each tensor shape: [batch, seq_len, hidden_dim]
            for layer_idx in range(self.num_layers):
                hidden = outputs.hidden_states[layer_idx + 1][:, -1, :]
                all_acts[layer_idx].append(
                    hidden.detach().cpu().float().numpy()
                )

        return {i: np.vstack(all_acts[i]) for i in range(self.num_layers)}

    def save(self, activations, labels, save_dir="results/activations"):
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "labels.npy"), labels)
        for layer_idx, acts in activations.items():
            np.save(os.path.join(save_dir, f"layer_{layer_idx}.npy"), acts)
        print(f"Saved {len(activations)} layers → {save_dir}")