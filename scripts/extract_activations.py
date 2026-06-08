import pandas as pd
import numpy as np
import sys
sys.path.append(".")
from src.activation_extractor import ActivationExtractor

if __name__ == "__main__":
    train = pd.read_csv("/teamspace/studios/this_studio/data/data/data/raw/herg_train_text.csv")
    texts  = train['text'].tolist()
    labels = train['Y'].values.astype(int)

    print(f"Samples: {len(texts)}")
    print(f"Label split — 1 (blocks hERG): {labels.sum()}, 0 (safe): {(labels==0).sum()}")

    extractor = ActivationExtractor("EleutherAI/pythia-1b")
    activations = extractor.extract(texts, batch_size=8)
    extractor.save(activations, labels)