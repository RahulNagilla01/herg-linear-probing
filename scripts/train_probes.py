import numpy as np
import os, json, sys
sys.path.append(".")
from src.probes import LogisticProbe, DifferenceOfMeansProbe

def run_probing(activations_dir="results/activations", num_layers=16):
    labels = np.load(os.path.join(activations_dir, "labels.npy"))
    lr_accs, dom_accs = [], []

    for layer_idx in range(num_layers):
        X = np.load(os.path.join(activations_dir, f"layer_{layer_idx}.npy"))

        lr_acc  = LogisticProbe().fit_evaluate(X, labels)
        dom_acc = DifferenceOfMeansProbe().fit_evaluate(X, labels)

        lr_accs.append(lr_acc)
        dom_accs.append(dom_acc)

        print(f"Layer {layer_idx:2d} | LR: {lr_acc:.3f} | DoM: {dom_acc:.3f}")

    results = {"lr": lr_accs, "dom": dom_accs}
    os.makedirs("results", exist_ok=True)
    with open("results/metrics.json", "w") as f:
        json.dump(results, f)

    print("\nSaved → results/metrics.json")
    return results

if __name__ == "__main__":
    run_probing()