import numpy as np
import json, os, sys
import matplotlib.pyplot as plt
sys.path.append(".")
from src.probes import LogisticProbe, DifferenceOfMeansProbe

def run_permutation_test(
    activations_dir="results/activations",
    num_layers=16,
    n_shuffles=10
):
    labels = np.load(os.path.join(activations_dir, "labels.npy"))

    # Load real results
    with open("results/metrics.json") as f:
        real = json.load(f)

    print("Running permutation test (shuffled labels)...\n")

    shuffled_lr  = {i: [] for i in range(num_layers)}
    shuffled_dom = {i: [] for i in range(num_layers)}

    for run in range(n_shuffles):
        shuffled_labels = labels.copy()
        np.random.shuffle(shuffled_labels)

        for layer_idx in range(num_layers):
            X = np.load(
                os.path.join(activations_dir, f"layer_{layer_idx}.npy")
            )
            lr_acc  = LogisticProbe().fit_evaluate(X, shuffled_labels)
            dom_acc = DifferenceOfMeansProbe().fit_evaluate(X, shuffled_labels)
            shuffled_lr[layer_idx].append(lr_acc)
            shuffled_dom[layer_idx].append(dom_acc)

        print(f"  Shuffle {run+1}/{n_shuffles} done")

    # Mean + std across shuffles
    lr_shuf_mean  = [np.mean(shuffled_lr[i])  for i in range(num_layers)]
    lr_shuf_std   = [np.std(shuffled_lr[i])   for i in range(num_layers)]
    dom_shuf_mean = [np.mean(shuffled_dom[i]) for i in range(num_layers)]

    layers = list(range(num_layers))

    # ── Print comparison ──────────────────────────────
    print(f"\n{'Layer':>6} | {'LR Real':>8} | {'LR Shuffled':>11} | {'Gap':>6}")
    print("-" * 42)
    for i in layers:
        gap = real["lr"][i] - lr_shuf_mean[i]
        print(f"{i:>6} | {real['lr'][i]:>8.3f} | "
              f"{lr_shuf_mean[i]:>11.3f} | {gap:>+6.3f}")

    # ── Plot ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(layers, real["lr"],  marker='o', color='#2196F3',
            linewidth=2, label='LR — Real labels')
    ax.plot(layers, real["dom"], marker='s', color='#FF5722',
            linewidth=2, linestyle='--', label='DoM — Real labels')

    ax.plot(layers, lr_shuf_mean, marker='o', color='#90CAF9',
            linewidth=2, linestyle=':', label='LR — Shuffled labels')
    ax.fill_between(
        layers,
        np.array(lr_shuf_mean) - np.array(lr_shuf_std),
        np.array(lr_shuf_mean) + np.array(lr_shuf_std),
        alpha=0.15, color='#2196F3'
    )
    ax.plot(layers, dom_shuf_mean, marker='s', color='#FFCCBC',
            linewidth=2, linestyle=':', label='DoM — Shuffled labels')

    ax.axhline(0.5, color='gray', linewidth=0.8,
               linestyle=':', label='Random baseline')
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(
        'Shortcut Control Test — Real vs Shuffled Labels\n'
        'Pythia-1B | hERG Channel Blocking',
        fontsize=12
    )
    ax.set_xticks(layers)
    ax.set_ylim(0.40, 0.90)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/figures/shortcut_test.png', dpi=150)
    print("\nSaved → results/figures/shortcut_test.png")

    # Save
    os.makedirs("results", exist_ok=True)
    with open("results/shortcut_metrics.json", "w") as f:
        json.dump({
            "lr_shuffled_mean": lr_shuf_mean,
            "lr_shuffled_std":  lr_shuf_std,
            "dom_shuffled_mean": dom_shuf_mean
        }, f)

if __name__ == "__main__":
    run_permutation_test()